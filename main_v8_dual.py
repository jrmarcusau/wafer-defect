#!/usr/bin/env python3
"""
main_v8_256_dual_full.py
────────────────────────
Dual-stream Conv-AE (cartesian + polar) with DEC clustering **and full diagnostics**.

• Two independent Conv-AE encoders *and* decoders (one per stream).
• Latent fusion: concat [z_cart, z_polar] → Linear 512 → ReLU → Linear 256.
• Optional per-stream HOG blocks (`USE_HOG = True` by default).
• Loss = MSE_cart + MSE_polar + γ·KL  (with γ = 0.1).
• Diagnostics written to `<out_dir>/figures/` every `UPDATE_INTERVAL` epochs:
      - loss_curve.png
      - cluster_hist_<ep>.png
      - tsne_<ep>.png
• At the same interval it saves:
      - checkpoints/ckpt_<ep>.pth
      - summaries/topK/cluster_<k>.png
      - summaries/randomK/cluster_<k>.png
      - type_<k>/  (copies of wafer PNGs)

Run
----
python main_v8_256_dual_full.py \
        --data_dir data/Op3176_DefectMap \
        --out_dir  outputs/v8_dual_full
"""

# ───────────────────────────── imports ────────────────────────────
import os, random, argparse, csv, cv2, numpy as np, matplotlib.pyplot as plt
from glob import glob
from PIL import Image
from tqdm import tqdm
from skimage.feature import hog
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.utils as vutils

# ────────────────────── hyper-parameters / config ─────────────────
IMG_SIZE        = 256
Z_DIM_STREAM    = 256            # latent per stream
Z_DIM_FUSED     = 256            # after fusion
NUM_CLUSTERS    = 30
BATCH_SIZE      = 32
WARMUP_EPOCHS   = 20
TOTAL_EPOCHS    = 100
UPDATE_INTERVAL = 10             # recompute K-means, diagnostics
TSNE_INTERVAL   = 10
GAMMA           = 0.1
LR              = 1e-3
SEED            = 42
USE_HOG         = True
TOP_K           = 9
RAND_K          = 9

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP     = DEVICE.type == "cuda"
scaler  = torch.amp.GradScaler(enabled=AMP)

# ───────────────────────────── dataset ────────────────────────────
class WaferDual(Dataset):
    def __init__(self, root: str) -> None:
        self.paths = sorted(glob(os.path.join(root, "*.png")) +
                            glob(os.path.join(root, "*.PNG")))
        if not self.paths:
            raise RuntimeError(f"No PNGs in {root}")
        dummy = np.zeros((IMG_SIZE, IMG_SIZE), np.uint8)
        self.hog_dim = hog(
            dummy,
            orientations=9,
            pixels_per_cell=(16, 16),
            cells_per_block=(1, 1),
            feature_vector=True        # ← explicit keyword
        ).shape[0]

    def __len__(self): return len(self.paths)

    @staticmethod
    def _coord():
        xs = torch.linspace(-1, 1, IMG_SIZE).view(1, 1, -1).expand(1, IMG_SIZE, IMG_SIZE)
        ys = torch.linspace(-1, 1, IMG_SIZE).view(1, -1, 1).expand(1, IMG_SIZE, IMG_SIZE)
        return torch.cat([xs, ys], 0)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        gray = np.array(Image.open(path).convert("L"))

        # ── polar branch ───────────────────────────────────────────
        polar = cv2.warpPolar(
            gray, (IMG_SIZE, IMG_SIZE),
            (gray.shape[1] // 2, gray.shape[0] // 2),
            maxRadius=gray.shape[0] // 2,
            flags=cv2.WARP_POLAR_LINEAR
        )
        polar = cv2.rotate(polar, cv2.ROTATE_90_COUNTERCLOCKWISE)
        edge_p = cv2.Canny(polar, 50, 150)
        hog_p = hog(
            polar,
            orientations=9,
            pixels_per_cell=(16, 16),
            cells_per_block=(1, 1),
            feature_vector=True
        ).astype(np.float32) if USE_HOG else np.empty(0)

        # ── cart branch ───────────────────────────────────────────
        cart = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), cv2.INTER_LINEAR)
        edge_c = cv2.Canny(cart, 50, 150)
        hog_c = hog(
            cart,
            orientations=9,
            pixels_per_cell=(16, 16),
            cells_per_block=(1, 1),
            feature_vector=True
        ).astype(np.float32) if USE_HOG else np.empty(0)

        coord = self._coord()
        x_polar = torch.cat([
            torch.from_numpy(polar / 255.).unsqueeze(0).float(),
            torch.from_numpy(edge_p / 255.).unsqueeze(0).float(),
            coord
        ], 0)
        x_cart = torch.cat([
            torch.from_numpy(cart / 255.).unsqueeze(0).float(),
            torch.from_numpy(edge_c / 255.).unsqueeze(0).float(),
            coord
        ], 0)

        return (x_polar,
                x_cart,
                torch.from_numpy(hog_p.astype(np.float32)),
                torch.from_numpy(hog_c.astype(np.float32)),
                idx,
                path)

# ──────────────────────────── model ───────────────────────────────
class StreamEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                              # 256 → 128
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                              # 128 →  64
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                              #  64 →  32
            nn.Flatten(),
            nn.Linear(128 * 32 * 32, Z_DIM_STREAM),
        )

    def forward(self, x): return self.net(x)


class StreamDec(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(Z_DIM_STREAM, 128 * 32 * 32),
            nn.Unflatten(1, (128, 32, 32)),
            nn.Upsample(scale_factor=2),                  # 32 → 64
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2),                  # 64 → 128
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2),                  # 128 → 256
            nn.Conv2d(32, 2, 3, padding=1), nn.Sigmoid(),
        )

    def forward(self, z): return self.net(z)


class DualAE(nn.Module):
    def __init__(self, hog_dim: int):
        super().__init__()
        self.enc_pol, self.enc_cart = StreamEnc(), StreamEnc()
        self.dec_pol, self.dec_cart = StreamDec(), StreamDec()
        in_dim = 2 * Z_DIM_STREAM + (2 * hog_dim if USE_HOG else 0)
        self.fuse = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(),
            nn.Linear(512, Z_DIM_FUSED)
        )

    def forward(self, xp, xc, hp, hc):
        zp = self.enc_pol(xp)
        zc = self.enc_cart(xc)
        cat = [zc, zp] + ([hp, hc] if USE_HOG else [])
        z_fused = self.fuse(torch.cat(cat, 1))
        rec_p = self.dec_pol(zp)
        rec_c = self.dec_cart(zc)
        return z_fused, rec_p, rec_c

# ───────────────────── DEC helper functions ───────────────────────
def soft_assign(z, centers, alpha: float = 1.0):
    d2 = torch.cdist(z, centers).pow(2)
    q = (1.0 + d2 / alpha).pow(-(alpha + 1) / 2)
    return q / q.sum(1, keepdim=True)

def target_distribution(q):
    w = (q ** 2) / q.sum(0)
    return (w.t() / w.sum(1)).t()

# ─────────────────────── diagnostics utils ────────────────────────
def save_loss_curve(fig_dir, ep, e_list, rp, rc, kl):
    plt.figure()
    plt.plot(e_list, rp, label="recon_pol")
    plt.plot(e_list, rc, label="recon_cart")
    plt.plot(e_list, kl, label="KL")
    plt.xlabel("epoch"); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "loss_curve.png"))
    plt.close()

def save_cluster_hist(fig_dir, ep, assigns, k):
    plt.figure()
    plt.bar(range(k), np.bincount(assigns, minlength=k))
    plt.xlabel("cluster"); plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"cluster_hist_{ep:03d}.png"))
    plt.close()

def save_tsne(fig_dir, ep, feats, labels):
    try:
        tsne = TSNE(2, init="pca", learning_rate="auto",
                    perplexity=min(30, max(5, len(labels) // 20)),
                    random_state=SEED)
        emb2 = tsne.fit_transform(feats)
        plt.figure(figsize=(6, 5))
        sc = plt.scatter(emb2[:, 0], emb2[:, 1], c=labels, s=6, cmap="tab20")
        plt.colorbar(sc); plt.title("t-SNE")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"tsne_{ep:03d}.png"))
        plt.close()
    except Exception as e:
        print("[warn] t-SNE failed:", e)

def montage(paths, out_file, k):
    tf = T.Resize((IMG_SIZE, IMG_SIZE))
    imgs = [T.ToTensor()(tf(Image.open(p).convert("RGB"))) for p in paths]
    while len(imgs) < k:
        imgs.append(torch.zeros(3, IMG_SIZE, IMG_SIZE))
    grid = vutils.make_grid(imgs, nrow=int(np.ceil(np.sqrt(k))), padding=2)
    vutils.save_image(grid, out_file)

# ─────────────────────────── training ─────────────────────────────
def train(data_root: str, out_root: str):
    # dirs
    os.makedirs(out_root, exist_ok=True)
    fig_dir = os.path.join(out_root, "figures");      os.makedirs(fig_dir, exist_ok=True)
    ckpt_dir= os.path.join(out_root, "checkpoints");  os.makedirs(ckpt_dir, exist_ok=True)
    sum_top = os.path.join(out_root, "summaries", "topK");   os.makedirs(sum_top, exist_ok=True)
    sum_rnd = os.path.join(out_root, "summaries", "randomK");os.makedirs(sum_rnd, exist_ok=True)

    # reproducibility
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    ds = WaferDual(data_root)
    loader = DataLoader(ds, BATCH_SIZE, True, num_workers=4,
                        pin_memory=torch.cuda.is_available(),
                        persistent_workers=True, prefetch_factor=2)
    model = DualAE(ds.hog_dim).to(DEVICE)
    criterion = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    centers = tgt_q = None
    ep_log, rec_p_log, rec_c_log, kl_log = [], [], [], []

    for ep in range(1, TOTAL_EPOCHS + 1):
        model.train()
        sum_rp = sum_rc = sum_kl = n = 0

        for xp, xc, hp, hc, idx, paths in tqdm(loader, desc=f"Epoch {ep}/{TOTAL_EPOCHS}"):
            xp, xc = xp.to(DEVICE), xc.to(DEVICE)
            hp, hc = hp.to(DEVICE), hc.to(DEVICE)

            with torch.amp.autocast("cuda", enabled=AMP):
                z, rec_p, rec_c = model(xp, xc, hp, hc)
                loss_rp = criterion(rec_p, xp[:, :2])
                loss_rc = criterion(rec_c, xc[:, :2])

                if ep > WARMUP_EPOCHS and centers is not None:
                    q = soft_assign(z, centers)
                    p = tgt_q[idx].to(DEVICE)
                    loss_kl = (p * torch.log(p / (q + 1e-8))).sum(1).mean()
                else:
                    loss_kl = torch.zeros((), device=DEVICE)

                loss = loss_rp + loss_rc + GAMMA * loss_kl

            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optim); scaler.update()

            sum_rp += loss_rp.item(); sum_rc += loss_rc.item(); sum_kl += loss_kl.item(); n += 1

        avg_rp = sum_rp / n; avg_rc = sum_rc / n; avg_kl = sum_kl / n
        print(f"Ep{ep:03d}  recon_p {avg_rp:.4f}  recon_c {avg_rc:.4f}  KL {avg_kl:.4f}")

        ep_log.append(ep); rec_p_log.append(avg_rp); rec_c_log.append(avg_rc); kl_log.append(avg_kl)

        # ── diagnostics & checkpoint every UPDATE_INTERVAL ─────────
        if ep % UPDATE_INTERVAL == 0 or ep == TOTAL_EPOCHS:
            # encode full set once
            model.eval(); feats = []; all_paths=[]
            with torch.no_grad():
                for xp, xc, hp, hc, _, pth in DataLoader(ds, BATCH_SIZE, False,
                                                         num_workers=4, pin_memory=False):
                    z,_ ,_ = model(xp.to(DEVICE), xc.to(DEVICE),
                                   hp.to(DEVICE), hc.to(DEVICE))
                    feats.append(z.cpu()); all_paths.extend(pth)
            feats = torch.cat(feats)
            # K-means update
            km = KMeans(NUM_CLUSTERS, n_init=20, random_state=SEED).fit(feats)
            centers = torch.tensor(
                km.cluster_centers_,        # numpy float64
                device=DEVICE,
                dtype=torch.float           # ← force float32
            )
            tgt_q = target_distribution(soft_assign(feats.to(DEVICE), centers).cpu())
            assigns = km.labels_

            # plots
            save_loss_curve(fig_dir, ep, ep_log, rec_p_log, rec_c_log, kl_log)
            save_cluster_hist(fig_dir, ep, assigns, NUM_CLUSTERS)
            if ep % TSNE_INTERVAL == 0:
                save_tsne(fig_dir, ep, feats.numpy(), assigns)

            # topK / randomK montages
            for k in range(NUM_CLUSTERS):
                idx = np.where(assigns == k)[0]
                if len(idx) == 0: continue
                # top-K by distance to center
                d = np.linalg.norm(feats[idx].numpy() - centers[k].cpu().numpy(), axis=1)
                top = idx[np.argsort(d)][:TOP_K]
                montage([all_paths[i] for i in top],
                        os.path.join(sum_top, f"cluster_{k}.png"), TOP_K)
                rnd = np.random.choice(idx, min(len(idx), RAND_K), replace=False)
                montage([all_paths[i] for i in rnd],
                        os.path.join(sum_rnd, f"cluster_{k}.png"), RAND_K)
                # copy originals into buckets
            for src, cid in zip(all_paths, assigns):
                dst = os.path.join(out_root, f"type_{cid}")
                os.makedirs(dst, exist_ok=True)
                shutil.copy2(src, os.path.join(dst, os.path.basename(src)))

            torch.save({
                "epoch": ep,
                "model_state": model.state_dict(),
                "centers": centers.cpu(),
            }, os.path.join(ckpt_dir, f"ckpt_{ep:03d}.pth"))

    print("✔ training complete")


# ───────────────────────────── CLI ────────────────────────────────
if __name__ == "__main__":
    import shutil
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default='data/Op3176_DefectMap')
    ap.add_argument("--out_dir",  default='outputs/v8_dual')
    args = ap.parse_args()
    train(args.data_dir, args.out_dir)
