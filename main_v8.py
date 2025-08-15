#!/usr/bin/env python3
"""
main_v8.py  —  Conv-AE + DEC clustering with CoordConv, Polar warp, HOG & 256-D latent
"""

import os
import time
import cv2
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
OUTPUT_DIR      = "outputs/v8"
IMG_SIZE        = 128
BATCH_SIZE      = 32
Z_DIM           = 256
NUM_CLUSTERS    = 20
WARMUP_EPOCHS   = 20
TOTAL_EPOCHS    = 100
UPDATE_INTERVAL = 10    # recompute centroids every 10 epochs
GAMMA           = 0.1   # weight of clustering loss
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

# ────────────────────────────────────────────────────
#  Dataset: polar-warp + edge + coord channels + HOG
# ────────────────────────────────────────────────────
class WaferDataset(Dataset):
    def __init__(self, root, img_size):
        np.random.seed(SEED)
        self.img_paths = sorted(glob(os.path.join(root, "*.PNG")) +
                                glob(os.path.join(root, "*.png")))
        assert len(self.img_paths) > 0, "No images found!"
        self.img_size = img_size

        # Precompute HOG dimension by running once on zeros
        dummy = np.zeros((img_size, img_size), dtype=np.uint8)
        self.hog_dim = hog(dummy,
                           orientations=9,
                           pixels_per_cell=(16,16),
                           cells_per_block=(1,1),
                           feature_vector=True).shape[0]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]

        # 1) load grayscale image
        pil = Image.open(path).convert("L")
        arr = np.array(pil)

        # 2) polar warp
        polar = cv2.warpPolar(
            arr,
            (self.img_size, self.img_size),
            center=(arr.shape[1]//2, arr.shape[0]//2),
            maxRadius=arr.shape[0]//2,
            flags=cv2.WARP_POLAR_LINEAR
        )
        polar = cv2.rotate(polar, cv2.ROTATE_90_COUNTERCLOCKWISE)
        polar = polar.astype(np.uint8)

        # 3) edge channel (Canny)
        edge = cv2.Canny(polar, 50, 150)

        # 4) HOG features on polar mask
        hog_vec = hog(polar,
                      orientations=9,
                      pixels_per_cell=(16,16),
                      cells_per_block=(1,1),
                      feature_vector=True)
        hog_vec = hog_vec.astype(np.float32)

        # 5) to torch tensors
        polar_t = torch.from_numpy(polar/255.0).unsqueeze(0).float()   # 1×H×W
        edge_t  = torch.from_numpy(edge/255.0).unsqueeze(0).float()   # 1×H×W

        # 6) coord channels
        xs = torch.linspace(-1,1,self.img_size).view(1,1,-1) \
               .expand(1,self.img_size,self.img_size)
        ys = torch.linspace(-1,1,self.img_size).view(1,-1,1) \
               .expand(1,self.img_size,self.img_size)
        coord = torch.cat([xs, ys], dim=0)   # 2×H×W

        # 7) final input tensor
        x = torch.cat([polar_t, edge_t, coord], dim=0)   # 4×H×W

        return x, torch.from_numpy(hog_vec), idx

# ────────────────────────────────────────────────────
#  Model: Conv-AE with 4-channel input, 2-channel decoder
# ────────────────────────────────────────────────────
class ConvAE(nn.Module):
    def __init__(self, z_dim):
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
            nn.Linear(z_dim, 64*32*32),
            nn.Unflatten(1, (64,32,32)),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64,32,3,padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 2,3,padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        z     = self.encoder(x)
        x_rec = self.decoder(z)
        return z, x_rec

# ────────────────────────────────────────────────────
#  DEC clustering helpers
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
    # make dirs
    makedirs(os.path.join(OUTPUT_DIR, "checkpoints"))
    makedirs(os.path.join(OUTPUT_DIR, "figures"))

    # seeds
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # data
    ds    = WaferDataset(DATA_DIR, IMG_SIZE)
    loader= DataLoader(ds,
                       batch_size=BATCH_SIZE,
                       shuffle=True,
                       num_workers=4,
                       pin_memory=True)

    # model, loss, optimizer
    model     = ConvAE(z_dim=Z_DIM + ds.hog_dim).to(DEVICE)
    mse_fn    = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # placeholders
    centers, target_q = None, None
    epoch_list, recon_list, clust_list = [], [], []

    for epoch in range(1, TOTAL_EPOCHS+1):
        model.train()
        running_recon, running_clust, n=0,0,0

        # update cluster centers
        if epoch > WARMUP_EPOCHS and epoch % UPDATE_INTERVAL == 1:
            print(f"\nUpdating KMeans centers at epoch {epoch-1}...")
            model.eval()
            all_feats, all_idxs = [], []

            with torch.no_grad():
                for x, hog_vec, idxs in DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False):
                    x = x.to(DEVICE)
                    z, _ = model(x)                         # B×(Z+H)
                    # concatenate hog
                    hog_batch = hog_vec.to(DEVICE)
                    feats = torch.cat([z, hog_batch], dim=1) # B×(Z+H)
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

            # visuals
            plot_cluster_hist(assigns, epoch-1)
            save_checkpoint({
                'epoch':epoch-1,
                'model_state':model.state_dict(),
                'opt_state':optimizer.state_dict(),
                'cluster_centers':centers.cpu()
            }, epoch-1)
            model.train()

        # one epoch
        for x, hog_vec, idxs in tqdm(loader, desc=f"Epoch {epoch}/{TOTAL_EPOCHS}"):
            x = x.to(DEVICE)
            z, x_rec = model(x)

            # reconstruction on first 2 channels only
            loss_recon = mse_fn(x_rec, x[:, :2, :, :])

            # clustering loss
            if epoch > WARMUP_EPOCHS:
                hog_batch = hog_vec.to(DEVICE)
                feats     = torch.cat([z, hog_batch], dim=1)
                q_batch   = soft_assign(feats, centers)
                p_batch   = target_q[idxs].to(DEVICE)
                loss_clust= (p_batch * torch.log(p_batch/(q_batch+1e-8))).sum(1).mean()
            else:
                loss_clust = torch.tensor(0.0, device=DEVICE)

            loss = loss_recon + GAMMA * loss_clust
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_recon += loss_recon.item()
            running_clust += loss_clust.item() if epoch>WARMUP_EPOCHS else 0.0
            n += 1

        # end epoch logging
        avg_recon = running_recon / n
        avg_clust = running_clust / n
        print(f"Epoch {epoch:03d} | Recon: {avg_recon:.4f} | Clust: {avg_clust:.4f}")
        epoch_list.append(epoch)
        recon_list.append(avg_recon)
        clust_list.append(avg_clust)

        if epoch % UPDATE_INTERVAL == 0 or epoch==TOTAL_EPOCHS:
            plot_losses(epoch_list, recon_list, clust_list)
            save_checkpoint({
                'epoch':epoch,
                'model_state':model.state_dict(),
                'opt_state':optimizer.state_dict(),
                'cluster_centers':centers.cpu() if centers is not None else None
            }, epoch)

    print("▶ Training complete.")

if __name__ == "__main__":
    train()
