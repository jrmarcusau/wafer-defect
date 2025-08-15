#!/usr/bin/env python3
"""
main_v9.py  —  Conv-AE + DEC clustering with CoordConv, Polar warp, HOG & 256-D latent
               Extended warm-up, more frequent updates, gamma ramp, up to 30 clusters
"""

import os
import cv2
import time
import numpy as np
from glob import glob
from PIL import Image
from skimage.feature import hog
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.utils as vutils

from sklearn.cluster import KMeans

# ────────────────────────────────────────────────────
#  Hyperparameters
# ────────────────────────────────────────────────────
DATA_DIR        = "data/Op3176_DefectMap"
OUTPUT_DIR      = "outputs/v9"
IMG_SIZE        = 128
BATCH_SIZE      = 32
Z_DIM           = 256
NUM_CLUSTERS    = 30
WARMUP_EPOCHS   = 30
TOTAL_EPOCHS    = 150
UPDATE_INTERVAL = 5     # recompute centroids every 5 epochs
GAMMA           = 0.1   # max weight of clustering loss
GAMMA_RAMP_EPOCHS = 20  # ramp clustering weight over these DEC epochs
LR              = 1e-3
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED            = 42

# ────────────────────────────────────────────────────
#  Utility functions
# ────────────────────────────────────────────────────
def makedirs(path):
    os.makedirs(path, exist_ok=True)

def save_checkpoint(state, epoch):
    fn = os.path.join(OUTPUT_DIR, "checkpoints", f"ckpt_{epoch:03d}.pth")
    torch.save(state, fn)

def plot_losses(epoch_list, recon_list, cluster_list):
    plt.figure()
    plt.plot(epoch_list, recon_list, label="Reconstruction loss")
    plt.plot(epoch_list, cluster_list, label="Clustering loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "figures", "loss_curve.png"))
    plt.close()

def plot_cluster_hist(assigns, epoch):
    plt.figure()
    counts = np.bincount(assigns, minlength=NUM_CLUSTERS)
    plt.bar(range(NUM_CLUSTERS), counts)
    plt.xlabel("Cluster")
    plt.ylabel("Count")
    plt.title(f"Epoch {epoch}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "figures", f"cluster_hist_{epoch:03d}.png"))
    plt.close()

# ────────────────────────────────────────────────────
#  Dataset: polar-warp + edge + coord + HOG
# ────────────────────────────────────────────────────
class WaferDataset(Dataset):
    def __init__(self, root, img_size):
        np.random.seed(SEED)
        self.img_paths = sorted(glob(os.path.join(root, "*.PNG")) +
                                glob(os.path.join(root, "*.png")))
        assert self.img_paths, f"No images in {root}"
        self.img_size = img_size

        # Precompute HOG dimension
        dummy = np.zeros((img_size, img_size), dtype=np.uint8)
        self.hog_dim = hog(
            dummy,
            orientations=9,
            pixels_per_cell=(16,16),
            cells_per_block=(1,1),
            feature_vector=True
        ).shape[0]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        pil  = Image.open(path).convert("L")
        arr  = np.array(pil)

        # polar warp + rotate
        polar = cv2.warpPolar(
            arr,
            (self.img_size, self.img_size),
            center=(arr.shape[1]//2, arr.shape[0]//2),
            maxRadius=arr.shape[0]//2,
            flags=cv2.WARP_POLAR_LINEAR
        )
        polar = cv2.rotate(polar, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # edge channel
        edge = cv2.Canny(polar, 50, 150)

        # HOG features
        hog_vec = hog(
            polar,
            orientations=9,
            pixels_per_cell=(16,16),
            cells_per_block=(1,1),
            feature_vector=True
        ).astype(np.float32)

        # to tensors
        polar_t = torch.from_numpy(polar/255.0).unsqueeze(0).float()
        edge_t  = torch.from_numpy(edge/255.0).unsqueeze(0).float()

        # CoordConv channels
        xs = torch.linspace(-1,1,self.img_size).view(1,1,-1) \
               .expand(1,self.img_size,self.img_size)
        ys = torch.linspace(-1,1,self.img_size).view(1,-1,1) \
               .expand(1,self.img_size,self.img_size)
        coord  = torch.cat([xs, ys], dim=0).float()

        # final input
        x = torch.cat([polar_t, edge_t, coord], dim=0)  # 4×H×W

        return x, torch.from_numpy(hog_vec), idx

# ────────────────────────────────────────────────────
#  Model: Conv-AE with 4→256 latent + hog concat
# ────────────────────────────────────────────────────
class ConvAE(nn.Module):
    def __init__(self, z_dim, hog_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4,  32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                  # 128→64
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                  # 64→32
            nn.Flatten(),
            nn.Linear(64*32*32, z_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim + hog_dim, 64*32*32),
            nn.Unflatten(1, (64,32,32)),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 2, 3, padding=1),  nn.Sigmoid()
        )

    def forward(self, x, hog_vec):
        z = self.encoder(x)
        z_full = torch.cat([z, hog_vec], dim=1)
        x_rec  = self.decoder(z_full)
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
    makedirs(os.path.join(OUTPUT_DIR, "checkpoints"))
    makedirs(os.path.join(OUTPUT_DIR, "figures"))

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    ds     = WaferDataset(DATA_DIR, IMG_SIZE)
    loader = DataLoader(ds,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        num_workers=4,
                        pin_memory=True)

    model = ConvAE(Z_DIM, ds.hog_dim).to(DEVICE)
    mse_fn    = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    centers, target_q = None, None
    epoch_list, recon_list, clust_list = [], [], []

    for epoch in range(1, TOTAL_EPOCHS + 1):
        model.train()
        running_recon, running_clust = 0.0, 0.0
        batch_count = 0

        # update cluster centers
        if epoch > WARMUP_EPOCHS and (epoch - WARMUP_EPOCHS - 1) % UPDATE_INTERVAL == 0:
            print(f"\n↻ Updating centers at epoch {epoch-1} …")
            model.eval()
            all_feats, all_idxs = [], []

            with torch.no_grad():
                for x, hog_vec, idxs in DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False):
                    x = x.to(DEVICE)
                    z, _ = model(x, hog_vec.to(DEVICE))
                    feats = torch.cat([z, hog_vec.to(DEVICE)], dim=1)
                    all_feats.append(feats.cpu())
            all_feats = torch.cat(all_feats, dim=0)

            km = KMeans(n_clusters=NUM_CLUSTERS, n_init=20, random_state=SEED)
            assigns = km.fit_predict(all_feats.numpy())
            centers = torch.tensor(km.cluster_centers_, device=DEVICE, dtype=torch.float)

            q_all    = soft_assign(all_feats.to(DEVICE), centers).cpu()
            target_q = target_distribution(q_all)

            plot_cluster_hist(assigns, epoch-1)
            save_checkpoint({
                'epoch': epoch-1,
                'model_state': model.state_dict(),
                'opt_state': optimizer.state_dict(),
                'cluster_centers': centers.cpu()
            }, epoch-1)
            model.train()

        # one training epoch
        for x, hog_vec, idxs in tqdm(loader, desc=f"Epoch {epoch}/{TOTAL_EPOCHS}"):
            x = x.to(DEVICE)
            hog_batch = hog_vec.to(DEVICE)
            z, x_rec = model(x, hog_batch)

            # recon loss on first 2 channels
            loss_recon = mse_fn(x_rec, x[:, :2, :, :])

            # determine clustering weight
            if epoch > WARMUP_EPOCHS:
                rel_epoch = epoch - WARMUP_EPOCHS
                ramp = min(rel_epoch / GAMMA_RAMP_EPOCHS, 1.0)
                gamma = GAMMA * ramp

                feats   = torch.cat([z, hog_batch], dim=1)
                q_batch = soft_assign(feats, centers)
                p_batch = target_q[idxs].to(DEVICE)
                loss_clust = (p_batch * torch.log(p_batch/(q_batch + 1e-8))).sum(1).mean()
            else:
                gamma = 0.0
                loss_clust = torch.tensor(0.0, device=DEVICE)

            loss = loss_recon + gamma * loss_clust
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_recon += loss_recon.item()
            running_clust += loss_clust.item()
            batch_count   += 1

        avg_recon = running_recon / batch_count
        avg_clust = running_clust / batch_count
        print(f"Epoch {epoch:03d} | Recon: {avg_recon:.4f} | Clust: {avg_clust:.4f}")

        epoch_list.append(epoch)
        recon_list.append(avg_recon)
        clust_list.append(avg_clust)

        if epoch % UPDATE_INTERVAL == 0 or epoch == TOTAL_EPOCHS:
            plot_losses(epoch_list, recon_list, clust_list)
            save_checkpoint({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'opt_state': optimizer.state_dict(),
                'cluster_centers': centers.cpu() if centers is not None else None
            }, epoch)

    print("▶ Training complete.")

if __name__ == "__main__":
    train()
