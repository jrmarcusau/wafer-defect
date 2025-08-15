#!/usr/bin/env python3
"""
main_v8_256_memsafe.py — Conv-AE + DEC clustering @ 256×256 (GPU-optimised, memory-safe)

Key points
----------
• Takes 256×256 images (no information lost).
• Adds a third down-sampling step so the fully-connected layer is modest in size.
• AE bottleneck stays at Z_DIM (256); we only append HOG features outside the AE.
• Mixed-precision, fast DataLoader, and async H2D copies included.
"""

# ───────────────────────────── Imports ─────────────────────────────
import os
from glob import glob
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.feature import hog

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans

# ────────────────────── Hyper-parameters / config ──────────────────
DATA_DIR        = "data/Op3176_DefectMap"
OUTPUT_DIR      = "outputs/v8_256_memsafe"

IMG_SIZE        = 256          # input resolution
BATCH_SIZE      = 32
Z_DIM           = 256          # AE latent size
NUM_CLUSTERS    = 20

WARMUP_EPOCHS   = 20           # epochs of pure AE training
TOTAL_EPOCHS    = 100
UPDATE_INTERVAL = 10           # re-compute K-means every n epochs

GAMMA           = 0.1          # weight of clustering loss
LR              = 1e-3
SEED            = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mixed precision helpers
AMP    = DEVICE.type == "cuda"
scaler = torch.cuda.amp.GradScaler(enabled=AMP)
try:
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
except Exception:
    pass
try:
    torch.set_float32_matmul_precision("high")   # PyTorch ≥ 2.0
except Exception:
    pass

# ────────────────────────── Utilities ──────────────────────────────
def makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_ckpt(state: dict, epoch: int) -> None:
    path = os.path.join(OUTPUT_DIR, "checkpoints", f"ckpt_{epoch:03d}.pth")
    torch.save(state, path)

def plot_losses(epochs, recon, clust) -> None:
    plt.figure()
    plt.plot(epochs, recon,  label="reconstruction")
    plt.plot(epochs, clust,  label="clustering")
    plt.xlabel("epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "figures", "loss_curve.png"))
    plt.close()

def plot_hist(assignments, epoch) -> None:
    plt.figure()
    counts = np.bincount(assignments, minlength=NUM_CLUSTERS)
    plt.bar(range(NUM_CLUSTERS), counts)
    plt.xlabel("cluster")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "figures",
                             f"cluster_hist_{epoch:03d}.png"))
    plt.close()

# ───────────────────────────── Dataset ─────────────────────────────
class WaferDataset(Dataset):
    def __init__(self, root: str, img_size: int) -> None:
        np.random.seed(SEED)

        self.paths = sorted(
            glob(os.path.join(root, "*.png")) + glob(os.path.join(root, "*.PNG"))
        )
        if not self.paths:
            raise RuntimeError("No images found!")

        self.img_size = img_size

        # Pre-compute HOG length
        dummy = np.zeros((img_size, img_size), dtype=np.uint8)
        self.hog_dim = hog(
            dummy,
            orientations=9,
            pixels_per_cell=(16, 16),
            cells_per_block=(1, 1),
            feature_vector=True,
        ).shape[0]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        # 1) load grayscale
        arr = np.array(Image.open(self.paths[idx]).convert("L"))

        # 2) polar warp → 256×256
        polar = cv2.warpPolar(
            arr,
            (self.img_size, self.img_size),
            (arr.shape[1] // 2, arr.shape[0] // 2),
            maxRadius=arr.shape[0] // 2,
            flags=cv2.WARP_POLAR_LINEAR,
        )
        polar = cv2.rotate(polar, cv2.ROTATE_90_COUNTERCLOCKWISE).astype(np.uint8)

        # 3) edge channel
        edge = cv2.Canny(polar, 50, 150)

        # 4) HOG vector
        hog_vec = hog(
            polar,
            orientations=9,
            pixels_per_cell=(16, 16),
            cells_per_block=(1, 1),
            feature_vector=True,
        ).astype(np.float32)

        # 5) tensorise
        p = torch.from_numpy(polar / 255.0).unsqueeze(0).float()      # 1×H×W
        e = torch.from_numpy(edge  / 255.0).unsqueeze(0).float()      # 1×H×W

        # 6) coord channels
        xs = torch.linspace(-1, 1, self.img_size).view(1, 1, -1).expand(1, self.img_size, self.img_size)
        ys = torch.linspace(-1, 1, self.img_size).view(1, -1, 1).expand(1, self.img_size, self.img_size)
        coord = torch.cat([xs, ys], dim=0)                            # 2×H×W

        x = torch.cat([p, e, coord], dim=0)                           # 4×H×W
        return x, torch.from_numpy(hog_vec), idx

# ───────────────────────────── Model ──────────────────────────────
class ConvAE(nn.Module):
    """256×256 → 128 → 64 → 32 feature map (memory-friendly)"""

    def __init__(self, z_dim: int) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(4,   32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                       # 256 → 128
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                       # 128 → 64
            nn.Conv2d(64,128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                       #  64 → 32
            nn.Flatten(),
            nn.Linear(128 * 32 * 32, z_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 128 * 32 * 32),
            nn.Unflatten(1, (128, 32, 32)),
            nn.Upsample(scale_factor=2),           # 32 → 64
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2),           # 64 → 128
            nn.Conv2d(64,  32, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2),           # 128 → 256
            nn.Conv2d(32,   2, 3, padding=1), nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return z, x_rec

# ───────────────────── DEC helper functions ───────────────────────
def soft_assign(z, centers, alpha: float = 1.0):
    centers = centers.to(z.dtype)
    dist2 = torch.cdist(z, centers) ** 2
    q = (1.0 + dist2 / alpha).pow(-(alpha + 1) / 2)
    return q / q.sum(dim=1, keepdim=True)

def target_distribution(q):
    weight = (q ** 2) / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

# ───────────────────────────── Train ──────────────────────────────
def train():
    # Setup
    makedirs(os.path.join(OUTPUT_DIR, "checkpoints"))
    makedirs(os.path.join(OUTPUT_DIR, "figures"))

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    ds = WaferDataset(DATA_DIR, IMG_SIZE)
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    model = ConvAE(Z_DIM).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    centers, target_q = None, None
    epochs, recon_hist, clust_hist = [], [], []

    # ───────────── main loop ─────────────
    for epoch in range(1, TOTAL_EPOCHS + 1):
        model.train()
        recon_running, clust_running, n_batches = 0.0, 0.0, 0

        # ----- K-means update -----
        if epoch > WARMUP_EPOCHS and epoch % UPDATE_INTERVAL == 1:
            print(f"\n[Epoch {epoch-1}] updating K-means centres …")
            model.eval()
            all_feats = []

            eval_loader = DataLoader(
                ds,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=8,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=2,
            )

            with torch.no_grad():
                for x, hog_vec, _ in eval_loader:
                    x       = x.to(DEVICE, non_blocking=True)
                    hog_vec = hog_vec.to(DEVICE, non_blocking=True)

                    with torch.cuda.amp.autocast(enabled=AMP):
                        z, _ = model(x)
                    all_feats.append(torch.cat([z.float().cpu(), hog_vec.cpu()], dim=1))

            all_feats = torch.cat(all_feats, dim=0)
            kmeans = KMeans(NUM_CLUSTERS, n_init=20, random_state=SEED).fit(all_feats)
            centers = torch.tensor(kmeans.cluster_centers_,
                                   device=DEVICE, dtype=torch.float)

            q_all = soft_assign(all_feats.to(DEVICE), centers).float().cpu()
            target_q = target_distribution(q_all)

            plot_hist(kmeans.labels_, epoch - 1)
            save_ckpt(
                {
                    "epoch": epoch - 1,
                    "model_state": model.state_dict(),
                    "opt_state": optimizer.state_dict(),
                    "centers": centers.cpu(),
                },
                epoch - 1,
            )
            model.train()

        # ----- training epoch -----
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{TOTAL_EPOCHS}")
        for x, hog_vec, idx in pbar:
            x       = x.to(DEVICE, non_blocking=True)
            hog_vec = hog_vec.to(DEVICE, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=AMP):
                z, x_rec   = model(x)
                loss_recon = criterion(x_rec, x[:, :2])     # only polar+edge

                if epoch > WARMUP_EPOCHS and centers is not None:
                    feat      = torch.cat([z, hog_vec], dim=1)
                    q_batch   = soft_assign(feat, centers)
                    p_batch   = target_q[idx].to(DEVICE, non_blocking=True)
                    loss_clust= (p_batch * torch.log(p_batch / (q_batch + 1e-8))).sum(1).mean()
                else:
                    loss_clust = torch.zeros((), device=DEVICE)

                loss = loss_recon + GAMMA * loss_clust

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            recon_running += loss_recon.item()
            clust_running += loss_clust.item()
            n_batches     += 1

        # Logging
        avg_recon = recon_running / n_batches
        avg_clust = clust_running / n_batches
        print(f"Epoch {epoch:03d} │ recon {avg_recon:.4f} │ cluster {avg_clust:.4f}")

        epochs.append(epoch)
        recon_hist.append(avg_recon)
        clust_hist.append(avg_clust)

        if epoch % UPDATE_INTERVAL == 0 or epoch == TOTAL_EPOCHS:
            plot_losses(epochs, recon_hist, clust_hist)
            save_ckpt(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "opt_state": optimizer.state_dict(),
                    "centers": centers.cpu() if centers is not None else None,
                },
                epoch,
            )

    print("✔ Training complete.")


if __name__ == "__main__":
    train()
