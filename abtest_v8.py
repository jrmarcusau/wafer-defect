#!/usr/bin/env python3
"""
ab_dual_raw256.py
──────────────────
Dual-stream Conv-AE (cart + polar) with DEC clustering — raw-only A/B variant.

• Inputs per stream: 1×gray + 2×CoordConv  → 3 channels.
• A-run (default): first conv 3×3, dilation 1.
• B-run (--dilated): same conv but dilation 2 (receptive field 5×5).

Diagnostics every 10 epochs:
    figures/loss_curve.png
    figures/cluster_hist_<ep>.png
    figures/tsne_<ep>.png
    summaries/topK & randomK montage grids
    type_<k>/ buckets
    checkpoints/ckpt_<ep>.pth
"""

# ───────────────────────── imports ─────────────────────────
import os, random, argparse, shutil, cv2, numpy as np, matplotlib.pyplot as plt
from glob import glob
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.utils as vutils

# ────────────────── hyper-parameters ───────────────────────
IMG_SIZE        = 256
Z_STREAM        = 256
Z_FUSED         = 256
NUM_CLUSTERS    = 30
BATCH_SIZE      = 32
WARMUP_EPOCHS   = 5           # early DEC
TOTAL_EPOCHS    = 100
UPDATE_INT      = 10
TSNE_INT        = 10
GAMMA           = 0.1
LR              = 1e-3
SEED            = 42
TOP_K, RAND_K   = 9, 9

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP     = DEVICE.type == "cuda"
scaler  = torch.cuda.amp.GradScaler(enabled=AMP)
autocast= torch.cuda.amp.autocast

# ───────────────────── dataset ─────────────────────────────
class WaferRaw(Dataset):
    def __init__(self, root: str):
        self.paths = sorted(glob(os.path.join(root, "*.png")) +
                            glob(os.path.join(root, "*.PNG")))
        if not self.paths:
            raise RuntimeError(f"No PNGs in {root}")

    def __len__(self): return len(self.paths)

    @staticmethod
    def _coord():
        xs = torch.linspace(-1,1,IMG_SIZE).view(1,1,-1).expand(1,IMG_SIZE,IMG_SIZE)
        ys = torch.linspace(-1,1,IMG_SIZE).view(1,-1,1).expand(1,IMG_SIZE,IMG_SIZE)
        return torch.cat([xs,ys],0)

    def __getitem__(self, idx):
        p   = self.paths[idx]
        g   = np.array(Image.open(p).convert("L"))

        # polar 256×256
        pol = cv2.warpPolar(
            g, (IMG_SIZE,IMG_SIZE),
            (g.shape[1]//2, g.shape[0]//2),
            g.shape[0]//2, cv2.WARP_POLAR_LINEAR)
        pol = cv2.rotate(pol, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # cart 256×256
        car = cv2.resize(g, (IMG_SIZE, IMG_SIZE), cv2.INTER_LINEAR)

        coord = self._coord()
        xp = torch.cat([torch.from_numpy(pol/255.).unsqueeze(0).float(), coord], 0)  # 3×H×W
        xc = torch.cat([torch.from_numpy(car/255.).unsqueeze(0).float(), coord], 0)
        return xp, xc, idx, p

# ───────────────────── model parts ─────────────────────────
class StreamEnc(nn.Module):
    def __init__(self, dilated: bool = False):
        super().__init__()
        d = 2 if dilated else 1
        p = 2 if dilated else 1          # keep output size
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=p, dilation=d), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64,128,3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128*32*32, Z_STREAM))
    def forward(self,x): return self.net(x)

class StreamDec(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(Z_STREAM, 128*32*32),
            nn.Unflatten(1,(128,32,32)),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128,64,3,1,1), nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64,32,3,1,1), nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32,1,3,1,1), nn.Sigmoid())   # 1-channel recon
    def forward(self,z): return self.net(z)

class DualAE(nn.Module):
    def __init__(self, dilated: bool):
        super().__init__()
        self.enc_pol  = StreamEnc(dilated)
        self.enc_cart = StreamEnc(dilated)
        self.dec_pol  = StreamDec()
        self.dec_cart = StreamDec()
        self.fuse = nn.Sequential(
            nn.Linear(2*Z_STREAM, 512), nn.ReLU(),
            nn.Linear(512, Z_FUSED))
    def forward(self,xp,xc):
        zp, zc = self.enc_pol(xp), self.enc_cart(xc)
        z = self.fuse(torch.cat([zc, zp],1))
        return z, self.dec_pol(zp), self.dec_cart(zc)

# DEC helpers
def soft_assign(z,c,a=1.): 
    d2=torch.cdist(z,c).pow(2)
    q=(1+d2/a).pow(-(a+1)/2)
    return q/q.sum(1,keepdim=True)

def target_dist(q): 
    w=(q**2)/q.sum(0)
    return (w.t()/w.sum(1)).t()

# plot helpers
def plot_loss(figdir, e, rp, rc, kl):
    plt.figure(); plt.plot(e,rp,label='rec_pol'); plt.plot(e,rc,label='rec_cart'); plt.plot(e,kl,label='KL'); plt.legend()
    plt.savefig(os.path.join(figdir,'loss_curve.png')); plt.close()
def plot_hist(figdir, ep, lab, k):
    plt.figure(); plt.bar(range(k),np.bincount(lab,minlength=k)); plt.savefig(os.path.join(figdir,f'hist_{ep:03d}.png')); plt.close()
def tsne_plot(figdir, ep, f, l):
    try:
        emb=TSNE(2,init='pca',learning_rate='auto',perplexity=min(30,max(5,len(l)//20)),random_state=SEED).fit_transform(f)
        plt.figure(figsize=(6,5)); sc=plt.scatter(emb[:,0],emb[:,1],c=l,s=6,cmap='tab20'); plt.colorbar(sc)
        plt.savefig(os.path.join(figdir,f'tsne_{ep:03d}.png')); plt.close()
    except Exception as e: print('[tsne]',e)
def montage(sel, out, k):
    tf=T.Resize((IMG_SIZE,IMG_SIZE)); imgs=[T.ToTensor()(tf(Image.open(p).convert('RGB'))) for p in sel]
    while len(imgs)<k: imgs.append(torch.zeros(3,IMG_SIZE,IMG_SIZE))
    vutils.save_image(vutils.make_grid(imgs,nrow=int(np.ceil(np.sqrt(k))),padding=2), out)

# ────────────────── training loop ────────────────────────
def train(data_root: str, out_root: str, dilated: bool):
    # dirs
    os.makedirs(out_root,exist_ok=True)
    figdir=f'{out_root}/figures'; ckptdir=f'{out_root}/checkpoints'
    topdir=f'{out_root}/summaries/topK'; rnddir=f'{out_root}/summaries/randomK'
    for d in (figdir, ckptdir, topdir, rnddir): os.makedirs(d,exist_ok=True)

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    ds = WaferRaw(data_root)
    loader = DataLoader(ds, BATCH_SIZE, True, num_workers=4,
                        pin_memory=torch.cuda.is_available())
    model = DualAE(dilated).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    mse   = nn.MSELoss()

    centers = tgt_q = None
    e_log=rpp=rcc=klp=[]

    for ep in range(1, TOTAL_EPOCHS+1):
        model.train(); sp=sc=sk=n=0
        for xp,xc,idx,_ in tqdm(loader,desc=f'Ep{ep}/{TOTAL_EPOCHS}'):
            xp,xc = xp.to(DEVICE), xc.to(DEVICE)
            with autocast(enabled=AMP):
                z,rp,rc = model(xp,xc)
                l_rp = mse(rp, xp[:, :1])
                l_rc = mse(rc, xc[:, :1])
                if ep>WARMUP_EPOCHS and centers is not None:
                    q = soft_assign(z, centers)
                    p = tgt_q[idx].to(DEVICE)
                    l_kl = (p*torch.log(p/(q+1e-8))).sum(1).mean()
                else: l_kl=torch.zeros((),device=DEVICE)
                loss = l_rp + l_rc + GAMMA*l_kl
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            sp+=l_rp.item(); sc+=l_rc.item(); sk+=l_kl.item(); n+=1
        rp,rc,kl = sp/n, sc/n, sk/n
        print(f'Ep{ep:03d}  rec_pol {rp:.4f}  rec_cart {rc:.4f}  KL {kl:.4f}')
        e_log.append(ep); rpp.append(rp); rcc.append(rc); klp.append(kl)

        if ep%UPDATE_INT==0 or ep==TOTAL_EPOCHS:
            model.eval(); feats=[]; paths=[]
            with torch.no_grad():
                for xp,xc,_,pth in DataLoader(ds,BATCH_SIZE,False,num_workers=4):
                    z,_,_ = model(xp.to(DEVICE), xc.to(DEVICE))
                    feats.append(z.cpu()); paths.extend(pth)
            feats=torch.cat(feats)
            km = KMeans(NUM_CLUSTERS,n_init=20,random_state=SEED).fit(feats)
            centers=torch.tensor(km.cluster_centers_,device=DEVICE)
            tgt_q=target_dist(soft_assign(feats.to(DEVICE),centers).cpu())
            labels=km.labels_

            plot_loss(figdir,e_log,rpp,rcc,klp)
            plot_hist(figdir,ep,labels,NUM_CLUSTERS)
            if ep%TSNE_INT==0: tsne_plot(figdir,ep,feats.numpy(),labels)

            # summaries
            for k in range(NUM_CLUSTERS):
                idx=np.where(labels==k)[0]
                if len(idx)==0: continue
                dist=np.linalg.norm(feats[idx].numpy()-centers[k].cpu().numpy(),axis=1)
                montage([paths[i] for i in idx[np.argsort(dist)][:TOP_K]],
                        f'{topdir}/cluster_{k}.png', TOP_K)
                montage([paths[i] for i in np.random.choice(idx,min(len(idx),RAND_K),False)],
                        f'{rnddir}/cluster_{k}.png', RAND_K)
            # buckets
            for src,lbl in zip(paths,labels):
                dst=f'{out_root}/type_{lbl}'; os.makedirs(dst,exist_ok=True)
                shutil.copy2(src,f'{dst}/{os.path.basename(src)}')

            torch.save({'epoch':ep,'model_state':model.state_dict(),'centers':centers.cpu()},
                       f'{ckptdir}/ckpt_{ep:03d}.pth')

    print('✔ done')

# ─────────────────────── CLI ──────────────────────────────
if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument("--data_dir", default='data/Op3176_DefectMap')
    ap.add_argument("--out_dir",  default='outputs/v8_abtest')
    ap.add_argument('--dilated',action='store_true',help='Use dilation=2 in first conv (B-run)')
    args=ap.parse_args()
    train(args.data_dir, args.out_dir, args.dilated)
