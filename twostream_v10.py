#!/usr/bin/env python3
"""
main_v8_twostream.py — Two-stream Conv-AE (Polar + Cartesian) + DEC clustering
- Polar AE input (4×256×256): [polar_gray, polar_edge, x, y], recon on first 2
- Cartesian AE input (4×256×256): [cart_gray, cart_edge_256, x, y], recon on first 2
- DEC features (equal-weight by block L2): z_polar, HOG_polar, z_cart, HOG_cart, line_stats
- KMeans updates + KL target same cadence as v8
"""

import os
import cv2
import numpy as np
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from skimage.feature import hog
from sklearn.cluster import KMeans

# ────────────────────────────────────────────────────
#  Hyperparameters (kept close to v8)
# ────────────────────────────────────────────────────
DATA_DIR        = "data/Op3176_DefectMap"
OUTPUT_DIR      = "outputs/v10"
IMG_SIZE        = 256          # ← now 256×256
EDGE_SIZE       = 512          # build cartesian edges at high-res, then max-pool to 256
BATCH_SIZE      = 16           # 256×256 is heavier; adjust as VRAM allows
Z_DIM           = 256          # latent per AE (polar and cart); pure latent (no HOG baked in)
NUM_CLUSTERS    = 20
WARMUP_EPOCHS   = 20
TOTAL_EPOCHS    = 100
UPDATE_INTERVAL = 10           # recompute centroids every 10 epochs
GAMMA           = 0.1          # weight of clustering loss
LR              = 1e-3
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED            = 42

# Edge params (consistent, fast)
CANNY_LOW       = 20
CANNY_HIGH      = 60
MORPH_LEN       = 9            # length of oriented line SEs for directional morphology
USE_LSD         = False        # LSD can be slow; keep off by default

# HOG at 256 => 8×8 cells with 32×32 = 576 dims (stable)
HOG_ORI         = 9
HOG_PPC         = (32, 32)
HOG_CPB         = (1, 1)

# I/O
os.makedirs(os.path.join(OUTPUT_DIR, "checkpoints"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "figures"), exist_ok=True)

# ────────────────────────────────────────────────────
#  Utilities
# ────────────────────────────────────────────────────
def makedirs(path):
    os.makedirs(path, exist_ok=True)

def save_checkpoint(state, epoch):
    fn = os.path.join(OUTPUT_DIR, "checkpoints", f"ckpt_{epoch:03d}.pth")
    torch.save(state, fn)

def plot_losses(epoch_list, recon_list, cluster_list):
    plt.figure()
    plt.plot(epoch_list, recon_list, label="Reconstruction loss (polar+cart)")
    plt.plot(epoch_list, cluster_list, label="Clustering loss (KL)")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "figures", "loss_curve.png"))
    plt.close()

def plot_cluster_hist(assignments, epoch):
    plt.figure()
    counts = np.bincount(assignments, minlength=NUM_CLUSTERS)
    plt.bar(range(NUM_CLUSTERS), counts)
    plt.xlabel("Cluster")
    plt.ylabel("Count")
    plt.title(f"Epoch {epoch}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "figures", f"cluster_hist_{epoch:03d}.png"))
    plt.close()

def l2norm(t, eps=1e-8):
    return t / (t.norm(dim=1, keepdim=True) + eps)

# ────────────────────────────────────────────────────
#  Edge helpers (Cartesian 512 → 256 fused edges)
# ────────────────────────────────────────────────────
def directional_morph_edges(gray_u8, length=MORPH_LEN):
    L = max(3, int(length))
    k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1))
    k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, L))
    k_d1 = np.eye(L, dtype=np.uint8)
    k_d2 = np.fliplr(k_d1)

    edges = []
    for k in (k_h, k_v, k_d1, k_d2):
        dil = cv2.dilate(gray_u8, k)
        ero = cv2.erode(gray_u8, k)
        edges.append(cv2.absdiff(dil, ero))
    return np.maximum.reduce(edges)

def lsd_mask(gray_u8, min_len_frac=0.06, thickness=1):
    H, W = gray_u8.shape
    try:
        lsd = cv2.createLineSegmentDetector(_refine=cv2.LSD_REFINE_STD)
    except Exception:
        return np.zeros_like(gray_u8, dtype=np.uint8)
    lines, _, _, _ = lsd.detect(gray_u8)
    if lines is None:
        return np.zeros_like(gray_u8, dtype=np.uint8)
    min_len = (H + W) * 0.5 * float(min_len_frac)
    mask = np.zeros_like(gray_u8, dtype=np.uint8)
    for ln in lines:
        x1, y1, x2, y2 = ln[0]
        if np.hypot(x2 - x1, y2 - y1) >= min_len:
            cv2.line(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, thickness, cv2.LINE_AA)
    return mask

def downsample_max(bin_u8, src_size, dst_size):
    f = src_size // dst_size
    assert src_size % dst_size == 0, "EDGE_SIZE must be a multiple of IMG_SIZE"
    e = (bin_u8 > 0).astype(np.uint8)
    e = e[:dst_size*f, :dst_size*f]
    e = e.reshape(dst_size, f, dst_size, f).max(axis=(1,3))
    return (e * 255).astype(np.uint8)

def build_cart_edge_256(src_gray_u8):
    """Make 512×512 fused edges (Canny + directional morph [+LSD]) and max-pool to 256×256."""
    cart512 = cv2.resize(src_gray_u8, (EDGE_SIZE, EDGE_SIZE), interpolation=cv2.INTER_LINEAR)
    canny = cv2.Canny(cart512, CANNY_LOW, CANNY_HIGH)
    morph = directional_morph_edges(cart512, length=MORPH_LEN)
    if USE_LSD:
        lines = lsd_mask(cart512, min_len_frac=0.06, thickness=1)
        fused512 = np.maximum.reduce([canny, morph, lines])
    else:
        fused512 = np.maximum(canny, morph)
    edge256 = downsample_max(fused512, EDGE_SIZE, IMG_SIZE)
    return edge256

def cart_hough_line_features(edge_u8, min_line_len=None, max_gap=10):
    """Tiny 4-D vector: [count, total_len_norm, mean_angle, std_angle]."""
    H, W = edge_u8.shape
    if min_line_len is None:
        min_line_len = max(H, W) // 8
    lines = cv2.HoughLinesP(edge_u8, 1, np.pi/360, threshold=30,
                            minLineLength=int(min_line_len), maxLineGap=int(max_gap))
    if lines is None:
        return np.array([0, 0, 0, 0], dtype=np.float32)
    Ls, angs = [], []
    for x1, y1, x2, y2 in lines[:,0]:
        dx, dy = (x2 - x1), (y2 - y1)
        Ls.append(np.hypot(dx, dy))
        angs.append(np.arctan2(dy, dx))
    Ls = np.array(Ls, dtype=np.float32)
    angs = np.array(angs, dtype=np.float32)
    return np.array([
        float(len(Ls)),
        float(Ls.sum() / (H + W + 1e-6)),
        float(angs.mean()) if len(angs)>0 else 0.0,
        float(angs.std())  if len(angs)>1 else 0.0
    ], dtype=np.float32)

# ────────────────────────────────────────────────────
#  Dataset (two-stream outputs)
# ────────────────────────────────────────────────────
class WaferDataset(Dataset):
    def __init__(self, root, img_size=IMG_SIZE):
        np.random.seed(SEED)
        self.img_paths = sorted(glob(os.path.join(root, "*.PNG")) +
                                glob(os.path.join(root, "*.png")))
        assert len(self.img_paths) > 0, "No images found!"
        self.img_size = img_size

        # Precompute HOG dims @256 with (32,32) cells
        dummy = np.zeros((img_size, img_size), dtype=np.uint8)
        self.hog_dim = hog(dummy,
                           orientations=HOG_ORI,
                           pixels_per_cell=HOG_PPC,
                           cells_per_block=HOG_CPB,
                           feature_vector=True).shape[0]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        pil = Image.open(path).convert("L")
        arr = np.array(pil)  # original grayscale

        # ---------- Polar branch (256) ----------
        polar = cv2.warpPolar(
            arr,
            (self.img_size, self.img_size),
            center=(arr.shape[1]//2, arr.shape[0]//2),
            maxRadius=arr.shape[0]//2,
            flags=cv2.WARP_POLAR_LINEAR
        )
        polar = cv2.rotate(polar, cv2.ROTATE_90_COUNTERCLOCKWISE)
        polar = polar.astype(np.uint8)

        edge_polar = cv2.Canny(polar, CANNY_LOW, CANNY_HIGH)

        hog_polar = hog(polar,
                        orientations=HOG_ORI,
                        pixels_per_cell=HOG_PPC,
                        cells_per_block=HOG_CPB,
                        feature_vector=True).astype(np.float32)

        polar_t = torch.from_numpy(polar/255.0).unsqueeze(0).float()    # 1×H×W
        edgep_t = torch.from_numpy(edge_polar/255.0).unsqueeze(0).float()

        xs = torch.linspace(-1,1,self.img_size).view(1,1,-1).expand(1,self.img_size,self.img_size)
        ys = torch.linspace(-1,1,self.img_size).view(1,-1,1).expand(1,self.img_size,self.img_size)
        coord = torch.cat([xs, ys], dim=0)   # 2×H×W

        x_polar = torch.cat([polar_t, edgep_t, coord], dim=0)  # 4×H×W

        # ---------- Cartesian branch (256) ----------
        cart = cv2.resize(arr, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        edge_cart256 = build_cart_edge_256(arr)  # uses 512→256 fused edges

        hog_cart = hog(cart,
                       orientations=HOG_ORI,
                       pixels_per_cell=HOG_PPC,
                       cells_per_block=HOG_CPB,
                       feature_vector=True).astype(np.float32)

        # tiny line stats on 256 edge map
        line_stats = cart_hough_line_features(edge_cart256, min_line_len=self.img_size//8, max_gap=10)

        cart_t  = torch.from_numpy(cart/255.0).unsqueeze(0).float()
        edgec_t = torch.from_numpy((edge_cart256>0).astype(np.float32)).unsqueeze(0)

        x_cart = torch.cat([cart_t, edgec_t, coord], dim=0)     # 4×H×W

        return x_polar, x_cart, torch.from_numpy(hog_polar), torch.from_numpy(hog_cart), torch.from_numpy(line_stats), idx

# ────────────────────────────────────────────────────
#  Model: Conv-AE (dynamic for IMG_SIZE=256)
# ────────────────────────────────────────────────────
class ConvAE(nn.Module):
    def __init__(self, z_dim, img_size=IMG_SIZE):
        super().__init__()
        self.img_size = img_size
        self.enc = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                               # 256→128
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                               # 128→64
            nn.Flatten(),
            nn.LazyLinear(z_dim)                           # infer in_features at first forward
        )
        # Decoder: compute fan_out from IMG_SIZE (two pools → size//4)
        fH = fW = img_size // 4
        self.dec = nn.Sequential(
            nn.Linear(z_dim, 64 * fH * fW),
            nn.Unflatten(1, (64, fH, fW)),
            nn.Upsample(scale_factor=2, mode='nearest'),   # 64→128
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),   # 128→256
            nn.Conv2d(32, 2, 3, padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        z = self.enc(x)
        x_rec = self.dec(z)
        return z, x_rec

# ────────────────────────────────────────────────────
#  DEC helpers
# ────────────────────────────────────────────────────
def soft_assign(z, centers, alpha=1.0):
    dist_sq = torch.cdist(z, centers) ** 2
    q = (1.0 + dist_sq/alpha).pow(-(alpha+1)/2)
    return q / q.sum(dim=1, keepdim=True)

def target_distribution(q):
    w = (q**2) / q.sum(0)
    return (w.t() / w.sum(1)).t()

# ────────────────────────────────────────────────────
#  Training loop
# ────────────────────────────────────────────────────
def train():
    # dirs / seeds
    torch.manual_seed(SEED); np.random.seed(SEED)

    ds = WaferDataset(DATA_DIR, IMG_SIZE)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    # Two AEs
    ae_polar = ConvAE(z_dim=Z_DIM, img_size=IMG_SIZE).to(DEVICE)
    ae_cart  = ConvAE(z_dim=Z_DIM, img_size=IMG_SIZE).to(DEVICE)

    mse_fn    = nn.MSELoss()
    params    = list(ae_polar.parameters()) + list(ae_cart.parameters())
    optimizer = optim.Adam(params, lr=LR)

    centers, target_q = None, None
    epoch_list, recon_list, clust_list = [], [], []

    for epoch in range(1, TOTAL_EPOCHS+1):
        ae_polar.train(); ae_cart.train()
        running_recon, running_clust, n = 0.0, 0.0, 0

        # ── Update KMeans centers (same cadence, now with two latents + two HOGs + line stats)
        if epoch > WARMUP_EPOCHS and epoch % UPDATE_INTERVAL == 1:
            print(f"\nUpdating KMeans centers at epoch {epoch-1}...")
            ae_polar.eval(); ae_cart.eval()
            all_feats, all_idxs = [], []

            with torch.no_grad():
                eval_loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
                for x_pol, x_car, hog_pol, hog_car, line_vec, idxs in eval_loader:
                    x_pol = x_pol.to(DEVICE); x_car = x_car.to(DEVICE)
                    z_p, _ = ae_polar(x_pol)
                    z_c, _ = ae_cart(x_car)

                    hp = hog_pol.to(DEVICE).float()
                    hc = hog_car.to(DEVICE).float()
                    lf = line_vec.to(DEVICE).float()

                    feats = torch.cat([
                        l2norm(z_p), l2norm(hp), l2norm(z_c), l2norm(hc), l2norm(lf)
                    ], dim=1)
                    all_feats.append(feats.cpu())
                    all_idxs.append(idxs)

            all_feats = torch.cat(all_feats, dim=0)

            # KMeans
            km = KMeans(n_clusters=NUM_CLUSTERS, n_init=20, random_state=SEED)
            assigns = km.fit_predict(all_feats.numpy())
            centers = torch.tensor(km.cluster_centers_, device=DEVICE, dtype=torch.float)

            # target distribution
            q_all = soft_assign(all_feats.to(DEVICE), centers).cpu()
            target_q = target_distribution(q_all)

            # visuals (cluster histogram)
            plot_cluster_hist(assigns, epoch-1)
            # (no checkpoint here; we'll save at end of epoch multiples of UPDATE_INTERVAL)
            ae_polar.train(); ae_cart.train()

        # ── One epoch
        for x_pol, x_car, hog_pol, hog_car, line_vec, idxs in tqdm(loader, desc=f"Epoch {epoch}/{TOTAL_EPOCHS}"):
            x_pol = x_pol.to(DEVICE); x_car = x_car.to(DEVICE)
            z_p, xrec_p = ae_polar(x_pol)
            z_c, xrec_c = ae_cart(x_car)

            # Recon on channel 0..1 for each stream
            loss_recon_p = mse_fn(xrec_p, x_pol[:, :2, :, :])
            loss_recon_c = mse_fn(xrec_c, x_car[:, :2, :, :])
            loss_recon   = loss_recon_p + loss_recon_c

            # Clustering loss
            if epoch > WARMUP_EPOCHS and centers is not None:
                hp = hog_pol.to(DEVICE).float()
                hc = hog_car.to(DEVICE).float()
                lf = line_vec.to(DEVICE).float()

                feats   = torch.cat([l2norm(z_p), l2norm(hp), l2norm(z_c), l2norm(hc), l2norm(lf)], dim=1)
                q_batch = soft_assign(feats, centers)
                p_batch = target_q[idxs].to(DEVICE)
                loss_clust = (p_batch * torch.log(p_batch/(q_batch+1e-8))).sum(1).mean()
            else:
                loss_clust = torch.tensor(0.0, device=DEVICE)

            loss = loss_recon + GAMMA * loss_clust

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_recon += loss_recon.item()
            running_clust += (loss_clust.item() if epoch > WARMUP_EPOCHS else 0.0)
            n += 1

        avg_recon = running_recon / max(1, n)
        avg_clust = running_clust / max(1, n)
        print(f"Epoch {epoch:03d} | Recon: {avg_recon:.4f} | Clust: {avg_clust:.4f}")
        epoch_list.append(epoch); recon_list.append(avg_recon); clust_list.append(avg_clust)

        # ── Save curves & checkpoint every UPDATE_INTERVAL epochs (10,20,30,...)
        if epoch % UPDATE_INTERVAL == 0 or epoch == TOTAL_EPOCHS:
            plot_losses(epoch_list, recon_list, clust_list)
            save_checkpoint({
                'epoch': epoch,
                'polar_state': ae_polar.state_dict(),
                'cart_state': ae_cart.state_dict(),
                'opt_state': optimizer.state_dict(),
                'cluster_centers': centers.cpu() if centers is not None else None
            }, epoch)

    print("▶ Training complete.")

if __name__ == "__main__":
    train()
