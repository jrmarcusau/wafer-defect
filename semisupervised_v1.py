#!/usr/bin/env python3
"""
semi_dual_raw256.py
────────────────────
Semi-supervised dual-stream Conv-AE + DEC *with seed labels*.

Folder layout it expects
------------------------
data/
  Label_Scratch/          (2-10 PNGs each)
  Label_BrokenLine/
  Label_Smudged/
  …
  Unlabeled/              (~2 000 PNGs)

What the script does
--------------------
1.  Dual-stream encoder (polar & cart) – **raw gray + CoordConv only**.
2.  Reconstruction loss  (cart + polar)   ➜ keeps latent informative.
3.  DEC clustering on the **fused 256-D latent** with **K = 20**.
4.  **Cross-entropy on seed images**: a linear classifier (num_seed_labels)
    taps the fused latent; its loss (λ = 0.1) pulls same-label
    seeds together without forcing global clusters to match 1-for-1.
5.  Every 10 epochs: recompute K-means centroids, plot loss / cluster hist /
    t-SNE, save checkpoints, top-9 & random-9 montages, and
    copy originals into `type_<k>/`.
6.  After final epoch:
        • if soft-assignment confidence < τ (= 0.55)
          → assign to one of **Weak-{0…3}** buckets (by quartile).
        • else keep cluster id 0-19.

Outputs → <out_dir>/
    checkpoints/ckpt_###.pth
    figures/loss_curve.png • hist_###.png • tsne_###.png
    summaries/top9/, random9/
    type_<id>/  (id = 0-19 or Weak-0…3)

Run
----
python semi_dual_raw256.py \
       --data_dir data \
       --out_dir  outputs/semisup_raw
"""

# ───────────────────────── Imports ──────────────────────────
import os, random, argparse, shutil, cv2, numpy as np
from glob import glob
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.utils as vutils

# ─────────────────── Hyper-parameters ───────────────────────
IMG_SIZE          = 256
Z_STREAM          = 256
Z_FUSED           = 256
K_CLUSTERS        = 20
BATCH_SIZE        = 32
WARMUP_EPOCHS     = 5
TOTAL_EPOCHS      = 100
UPDATE_INT        = 10
TSNE_INT          = 10
GAMMA_DEC         = 0.1          # weight of KL loss
LAMBDA_CE         = 0.1          # weight of supervised CE
LR                = 1e-3
SEED              = 42
TOP_K, RAND_K     = 9, 9
CONF_STRONG_TAU   = 0.55         # ≥τ ⇒ strong cluster 0-19

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP     = DEVICE.type == "cuda"
scaler  = torch.cuda.amp.GradScaler(enabled=AMP)
autocast= torch.cuda.amp.autocast

# ───────────────────── Dataset ───────────────────────────────
class WaferSemi(Dataset):
    def __init__(self, root: str):
        self.seed_paths, self.seed_labels = [], []
        self.unlab_paths = []
        label_folders = sorted([d for d in os.listdir(root)
                                if d.startswith("Label_") and os.path.isdir(os.path.join(root,d))])
        self.label2idx = {name: i for i, name in enumerate(label_folders)}
        # collect labelled
        for name, idx in self.label2idx.items():
            for p in glob(os.path.join(root, name, "*.png")) + glob(os.path.join(root, name, "*.PNG")):
                self.seed_paths.append(p); self.seed_labels.append(idx)
        # collect unlabeled
        self.unlab_paths = (glob(os.path.join(root, "Unlabeled", "*.png")) +
                            glob(os.path.join(root, "Unlabeled", "*.PNG")))
        if not self.seed_paths or not self.unlab_paths:
            raise RuntimeError("Seed or unlabeled folders missing / empty.")
        self.paths = self.seed_paths + self.unlab_paths
        self.is_seed = np.array([1]*len(self.seed_paths) + [0]*len(self.unlab_paths), bool)
        self.seed_labels = np.array(self.seed_labels + [-1]*len(self.unlab_paths))

    def __len__(self): return len(self.paths)

    @staticmethod
    def _coord():
        xs = torch.linspace(-1,1,IMG_SIZE).view(1,1,-1).expand(1,IMG_SIZE,IMG_SIZE)
        ys = torch.linspace(-1,1,IMG_SIZE).view(1,-1,1).expand(1,IMG_SIZE,IMG_SIZE)
        return torch.cat([xs,ys],0)

    def __getitem__(self, idx):
        p = self.paths[idx]
        g = np.array(Image.open(p).convert("L"))
        # polar
        pol = cv2.warpPolar(g,(IMG_SIZE,IMG_SIZE),
                            (g.shape[1]//2,g.shape[0]//2),
                            g.shape[0]//2, cv2.WARP_POLAR_LINEAR)
        pol = cv2.rotate(pol, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # cart
        car = cv2.resize(g,(IMG_SIZE,IMG_SIZE),cv2.INTER_LINEAR)
        coord = self._coord()
        xp = torch.cat([torch.from_numpy(pol/255.).unsqueeze(0).float(), coord], 0)
        xc = torch.cat([torch.from_numpy(car/255.).unsqueeze(0).float(), coord], 0)
        return xp, xc, torch.tensor(self.is_seed[idx]), torch.tensor(self.seed_labels[idx]), idx, p

# ─────────────────── Model components ───────────────────────
class StreamEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,32,3,1,1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,1,1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,1,1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128*32*32, Z_STREAM))
    def forward(self,x): return self.net(x)

class StreamDec(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(Z_STREAM,128*32*32),
            nn.Unflatten(1,(128,32,32)),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128,64,3,1,1), nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64,32,3,1,1), nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32,1,3,1,1), nn.Sigmoid())
    def forward(self,z): return self.net(z)

class SemiDual(nn.Module):
    def __init__(self, n_seed_labels: int):
        super().__init__()
        self.enc_pol, self.enc_cart = StreamEnc(), StreamEnc()
        self.dec_pol, self.dec_cart = StreamDec(), StreamDec()
        self.fuse = nn.Sequential(
            nn.Linear(2*Z_STREAM, 512), nn.ReLU(),
            nn.Linear(512, Z_FUSED))
        self.head = nn.Linear(Z_FUSED, n_seed_labels)  # classifier head

    def forward(self, xp, xc):
        zp, zc = self.enc_pol(xp), self.enc_cart(xc)
        z = self.fuse(torch.cat([zc, zp], 1))
        rec_p = self.dec_pol(zp)
        rec_c = self.dec_cart(zc)
        logit = self.head(z)
        return z, rec_p, rec_c, logit

# ─────────── DEC helpers ───────────
def soft_assign(z, cent, alpha=1.0):
    d2 = torch.cdist(z, cent).pow(2)
    q = (1.0 + d2/alpha).pow(-(alpha+1)/2)
    return q / q.sum(1, keepdim=True)
def target_dist(q):
    w = (q**2) / q.sum(0)
    return (w.t() / w.sum(1)).t()

# ─────────── Diagnostics helpers ───────────
def plot_loss(figdir, e, rp, rc, kl, ce):
    plt.figure()
    plt.plot(e,rp,label='rec_pol'); plt.plot(e,rc,label='rec_cart')
    plt.plot(e,kl,label='KL'); plt.plot(e,ce,label='CE')
    plt.legend(); plt.savefig(os.path.join(figdir,'loss_curve.png')); plt.close()
def plot_hist(figdir, ep, labels, k):
    plt.figure(); plt.bar(range(k), np.bincount(labels, minlength=k))
    plt.savefig(os.path.join(figdir,f'hist_{ep:03d}.png')); plt.close()
def tsne_fig(figdir, ep, feats, labels):
    try:
        emb = TSNE(2, init='pca', learning_rate='auto',
                   perplexity=min(30,max(5,len(labels)//20)), random_state=SEED).fit_transform(feats)
        plt.figure(figsize=(6,5)); sc=plt.scatter(emb[:,0],emb[:,1],c=labels,s=6,cmap='tab20'); plt.colorbar(sc)
        plt.savefig(os.path.join(figdir,f'tsne_{ep:03d}.png')); plt.close()
    except Exception as e: print('[t-SNE warn]',e)
def montage(sel, out, k):
    tf=T.Resize((IMG_SIZE,IMG_SIZE))
    imgs=[T.ToTensor()(tf(Image.open(p).convert('RGB'))) for p in sel]
    while len(imgs)<k: imgs.append(torch.zeros(3,IMG_SIZE,IMG_SIZE))
    vutils.save_image(vutils.make_grid(imgs,nrow=int(np.ceil(np.sqrt(k))),padding=2), out)

# ─────────── Training ───────────
def train(data_root: str, out_root: str):
    # dirs
    figdir = f'{out_root}/figures'; ckptdir=f'{out_root}/checkpoints'
    topdir = f'{out_root}/summaries/top9'; rnddir=f'{out_root}/summaries/random9'
    for d in (figdir, ckptdir, topdir, rnddir): os.makedirs(d, exist_ok=True)

    # data
    ds = WaferSemi(data_root)
    n_seed = len(ds.label2idx)
    loader = DataLoader(ds, BATCH_SIZE, True, num_workers=4,
                        pin_memory=torch.cuda.is_available())
    model = SemiDual(n_seed).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    mse   = nn.MSELoss()
    ce    = nn.CrossEntropyLoss()

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    centers = tgt_q = None
    e_log, rp_log, rc_log, kl_log, ce_log = [], [], [], [], []

    for ep in range(1, TOTAL_EPOCHS+1):
        model.train(); s_rp=s_rc=s_kl=s_ce=n=0
        for xp,xc,is_seed,label,idcs,_ in tqdm(loader,desc=f'Ep{ep}/{TOTAL_EPOCHS}'):
            xp,xc,is_seed,label = xp.to(DEVICE), xc.to(DEVICE), is_seed.to(DEVICE), label.to(DEVICE)
            with autocast(enabled=AMP):
                z, rp, rc, logit = model(xp, xc)
                l_rp = mse(rp, xp[:, :1]); l_rc = mse(rc, xc[:, :1])
                l_ce = ce(logit[is_seed.bool()], label[is_seed.bool()]) if is_seed.any() else torch.zeros((),device=DEVICE)
                if ep > WARMUP_EPOCHS and centers is not None:
                    q = soft_assign(z, centers)
                    p = tgt_q[idcs].to(DEVICE)
                    l_kl = (p*torch.log(p/(q+1e-8))).sum(1).mean()
                else: l_kl = torch.zeros((),device=DEVICE)
                loss = l_rp + l_rc + GAMMA_DEC*l_kl + LAMBDA_CE*l_ce
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            s_rp+=l_rp.item(); s_rc+=l_rc.item(); s_kl+=l_kl.item(); s_ce+=l_ce.item(); n+=1
        rp,rc,kl,cel = s_rp/n, s_rc/n, s_kl/n, s_ce/n
        print(f'Ep{ep:03d}  recP {rp:.3f}  recC {rc:.3f}  KL {kl:.3f}  CE {cel:.3f}')
        e_log.append(ep); rp_log.append(rp); rc_log.append(rc); kl_log.append(kl); ce_log.append(cel)

        # ─── diagnostics every UPDATE_INT ───
        if ep%UPDATE_INT==0 or ep==TOTAL_EPOCHS:
            model.eval(); feats=[]; paths=[]; 
            with torch.no_grad():
                for xp,xc,_,_,_,pth in DataLoader(ds,BATCH_SIZE,False):
                    z,_,_,_ = model(xp.to(DEVICE), xc.to(DEVICE))
                    feats.append(z.cpu())
                    paths.extend(pth)
            feats=torch.cat(feats)
            km   = KMeans(K_CLUSTERS, n_init=20, random_state=SEED).fit(feats)
            centers=torch.tensor(km.cluster_centers_,device=DEVICE,dtype=torch.float)
            tgt_q=target_dist(soft_assign(feats.to(DEVICE),centers).cpu())
            cluster_id = km.labels_
            conf = np.max(soft_assign(feats.to(DEVICE),centers).cpu().numpy(),1)

            # strong/weak split
            strong_mask = conf >= CONF_STRONG_TAU
            weak_bins   = np.digitize(conf, np.quantile(conf[~strong_mask], [0.25,0.5,0.75]))  # 0-3
            final_labels = cluster_id.copy()
            final_labels[~strong_mask] = K_CLUSTERS + weak_bins[~strong_mask]   # Weak-0…3 ids 20-23

            # plots
            plot_loss(figdir,e_log,rp_log,rc_log,kl_log,ce_log)
            plot_hist(figdir,ep,cluster_id,K_CLUSTERS)
            if ep%TSNE_INT==0: tsne_fig(figdir,ep,feats.numpy(),final_labels)

            # summaries
            for k in range(K_CLUSTERS):
                idx=np.where(cluster_id==k)[0]
                if len(idx)==0: continue
                dist=np.linalg.norm(feats[idx].numpy()-centers[k].cpu().numpy(),axis=1)
                montage([paths[i] for i in idx[np.argsort(dist)][:TOP_K]], f'{topdir}/cluster_{k}.png', TOP_K)
                montage([paths[i] for i in np.random.choice(idx,min(len(idx),RAND_K),False)],
                        f'{rnddir}/cluster_{k}.png', RAND_K)
            # buckets
            for src,lbl in zip(paths,final_labels):
                dst=f'{out_root}/type_{lbl}'; os.makedirs(dst,exist_ok=True)
                shutil.copy2(src,f'{dst}/{os.path.basename(src)}')

            torch.save({'epoch':ep,
                        'model_state':model.state_dict(),
                        'centers':centers.cpu()},
                       f'{ckptdir}/ckpt_{ep:03d}.pth')
    print("✔ Training complete")

# ──────────── CLI ────────────
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--data_dir", default='data/Op3176_DefectMap_Labeled')
    ap.add_argument("--out_dir",  default='outputs/sup_v1')
    args=ap.parse_args()
    train(args.data_dir, args.out_dir)
