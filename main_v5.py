#!/usr/bin/env python3
"""
main_v5.py  —  Conv-Autoencoder + DEC clustering on 128×128 wafer masks
Includes TQDM progress bars and visual saves of cluster samples every update interval
"""

import os
import sys
import time
import numpy as np
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.utils as vutils

from sklearn.cluster import KMeans

# --------------- hyperparameters ---------------
DATA_DIR         = "data/Op3176_DefectMap"
OUTPUT_DIR       = "outputs/v5"
IMG_SIZE         = 128
BATCH_SIZE       = 32
Z_DIM            = 128
NUM_CLUSTERS     = 10
WARMUP_EPOCHS    = 20
TOTAL_EPOCHS     = 100
UPDATE_INTERVAL  = 10    # recompute centroids & target every 10 epochs
GAMMA            = 0.1   # weight of clustering loss
LR               = 1e-3
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED             = 42

# --------------- utility functions ---------------
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

# --------------- cluster sample visualization ---------------
def save_cluster_samples(ds, assignments, epoch, n_per_cluster=5):
    """
    Save a grid of sample images for each cluster.
    Outputs a PNG of shape (NUM_CLUSTERS rows × n_per_cluster cols).
    """
    dest_dir = os.path.join(OUTPUT_DIR, "figures", "cluster_imgs")
    makedirs(dest_dir)
    imgs_per_cluster = []
    for k in range(NUM_CLUSTERS):
        idxs = np.where(assignments == k)[0][:n_per_cluster]
        samples = []
        for idx in idxs:
            path = ds.paths[idx]
            img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            samples.append(T.ToTensor()(img))
        # pad if fewer than n_per_cluster
        while len(samples) < n_per_cluster:
            samples.append(torch.zeros(3, IMG_SIZE, IMG_SIZE))
        imgs_per_cluster.extend(samples)

    grid = vutils.make_grid(imgs_per_cluster, nrow=n_per_cluster, padding=2)
    fn = os.path.join(dest_dir, f"cluster_imgs_{epoch:03d}.png")
    vutils.save_image(grid, fn)

# --------------- dataset ---------------
class WaferMaskDS(Dataset):
    def __init__(self, root, img_size):
        self.paths = sorted(glob(os.path.join(root, "*.PNG")) + glob(os.path.join(root, "*.png")))
        self.tf = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),            # [0,1]
        ])
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("L")
        x = self.tf(img)
        return x, idx

# --------------- model ---------------
class ConvAutoencoder(nn.Module):
    def __init__(self, z_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 128→64
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 64→32
            nn.Flatten(),
            nn.Linear(64*32*32, z_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 64*32*32),
            nn.Unflatten(1, (64,32,32)),
            nn.Upsample(scale_factor=2), nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2), nn.Conv2d(32, 1, 3, padding=1),  nn.Sigmoid()
        )
    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return z, x_rec

# --------------- DEC helpers ---------------
def soft_assign(z, cluster_centers, alpha=1.0):
    dist_sq = torch.cdist(z, cluster_centers)**2
    q = (1.0 + dist_sq / alpha).pow(-(alpha+1)/2)
    return (q / q.sum(dim=1, keepdim=True))

def target_distribution(q):
    weight = (q**2) / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

# --------------- training pipeline ---------------
def train():
    # prepare dirs
    makedirs(os.path.join(OUTPUT_DIR, "checkpoints"))
    makedirs(os.path.join(OUTPUT_DIR, "figures"))

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    ds = WaferMaskDS(DATA_DIR, IMG_SIZE)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    print(f"Found {len(ds)} images → training on {DEVICE}")

    model = ConvAutoencoder(z_dim=Z_DIM).to(DEVICE)
    recons_loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    epoch_list, recon_list, cluster_list = [], [], []
    cluster_centers, target_q = None, None

    for epoch in range(1, TOTAL_EPOCHS+1):
        model.train()
        running_recon, running_clus, cnt = 0.0, 0.0, 0

        # update clusters & save visuals
        if epoch > WARMUP_EPOCHS and epoch % UPDATE_INTERVAL == 1:
            print(f"\n↻ Updating cluster centers at epoch {epoch-1}...")
            model.eval()
            all_z = []
            with torch.no_grad():
                for x, _ in DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False):
                    z, _ = model(x.to(DEVICE))
                    all_z.append(z.cpu())
            all_z = torch.cat(all_z, dim=0)

            km = KMeans(n_clusters=NUM_CLUSTERS, n_init=20, random_state=SEED)
            assignments = km.fit_predict(all_z.numpy())
            cluster_centers = torch.tensor(km.cluster_centers_, device=DEVICE, dtype=torch.float)
            q_all = soft_assign(all_z.to(DEVICE), cluster_centers)
            target_q = target_distribution(q_all).cpu()

            # save histograms and sample grids
            plot_cluster_hist(assignments, epoch-1)
            save_cluster_samples(ds, assignments, epoch-1, n_per_cluster=5)
            save_checkpoint({
                "epoch": epoch-1,
                "model_state": model.state_dict(),
                "opt_state": optimizer.state_dict(),
                "cluster_centers": cluster_centers.cpu(),
            }, epoch-1)
            model.train()

        # training loop with tqdm
        for x, idx in tqdm(loader, desc=f"Epoch {epoch}/{TOTAL_EPOCHS}"):
            x = x.to(DEVICE)
            z, x_rec = model(x)
            loss_recon = recons_loss_fn(x_rec, x)
            if epoch > WARMUP_EPOCHS:
                q_batch = soft_assign(z, cluster_centers)
                p_batch = target_q[idx].to(DEVICE)
                loss_clust = torch.sum(p_batch * torch.log(p_batch / (q_batch + 1e-8)), dim=1).mean()
            else:
                loss_clust = torch.tensor(0.0, device=DEVICE)

            loss = loss_recon + GAMMA * loss_clust
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_recon += loss_recon.item()
            running_clus  += loss_clust.item() if epoch>WARMUP_EPOCHS else 0.0
            cnt += 1

        avg_recon = running_recon / cnt
        avg_clus  = running_clus / cnt if epoch>WARMUP_EPOCHS else 0.0
        print(f"Epoch {epoch:03d} | Recon: {avg_recon:.4f} | Clust: {avg_clus:.4f}")
        epoch_list.append(epoch)
        recon_list.append(avg_recon)
        cluster_list.append(avg_clus)

        if epoch % UPDATE_INTERVAL == 0 or epoch==TOTAL_EPOCHS:
            plot_losses(epoch_list, recon_list, cluster_list)
            save_checkpoint({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "opt_state": optimizer.state_dict(),
                "cluster_centers": cluster_centers.cpu() if cluster_centers is not None else None,
            }, epoch)

    print("\n▶︎ Training complete.")

if __name__ == "__main__":
    train()
