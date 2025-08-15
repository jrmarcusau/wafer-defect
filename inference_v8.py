#!/usr/bin/env python3
"""
inference_v8.py — Inference + summaries for main_v8 DEC model
"""

import os
import argparse
import shutil
from glob import glob

import cv2
import numpy as np
from PIL import Image
from skimage.feature import hog

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.utils as vutils

# ────────────────────────────────────────────────────
# Constants (must match main_v8)
# ────────────────────────────────────────────────────
IMG_SIZE     = 128
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED         = 42

# ────────────────────────────────────────────────────
# Dataset: polar + edge + coord + HOG exactly like main_v8
# ────────────────────────────────────────────────────
class WaferDatasetV8(Dataset):
    def __init__(self, root):
        self.paths = sorted(glob(os.path.join(root, "*.PNG")) +
                            glob(os.path.join(root, "*.png")))
        if not self.paths:
            raise RuntimeError(f"No images found under {root}!")
        # Precompute HOG dimension
        dummy = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        self.hog_dim = hog(
            dummy,
            orientations=9,
            pixels_per_cell=(16,16),
            cells_per_block=(1,1),
            feature_vector=True
        ).shape[0]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img  = Image.open(path).convert("L")
        arr  = np.array(img)

        # 1) polar warp + rotate so angle is horizontal
        polar = cv2.warpPolar(
            arr,
            (IMG_SIZE, IMG_SIZE),
            center=(arr.shape[1]//2, arr.shape[0]//2),
            maxRadius=arr.shape[0]//2,
            flags=cv2.WARP_POLAR_LINEAR
        )
        polar = cv2.rotate(polar, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # 2) edge channel
        edge = cv2.Canny(polar, 50, 150)

        # 3) hog feature vector
        hog_vec = hog(
            polar,
            orientations=9,
            pixels_per_cell=(16,16),
            cells_per_block=(1,1),
            feature_vector=True
        ).astype(np.float32)

        # 4) to torch tensors
        polar_t = torch.from_numpy(polar/255.0).unsqueeze(0).float()
        edge_t  = torch.from_numpy(edge/255.0).unsqueeze(0).float()

        # 5) CoordConv channels
        xs = torch.linspace(-1,1,IMG_SIZE).view(1,1,-1) \
               .expand(1,IMG_SIZE,IMG_SIZE)
        ys = torch.linspace(-1,1,IMG_SIZE).view(1,-1,1) \
               .expand(1,IMG_SIZE,IMG_SIZE)
        coord = torch.cat([xs, ys], dim=0).float()

        # 6) final 4-channel input
        x = torch.cat([polar_t, edge_t, coord], dim=0)

        return x, torch.from_numpy(hog_vec), idx, path

# ────────────────────────────────────────────────────
# Conv-AE encoder matching main_v8.py (input 4-ch, latent=z_dim)
# ────────────────────────────────────────────────────
class ConvAEEncoder(nn.Module):
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

    def forward(self, x):
        return self.encoder(x)

# ────────────────────────────────────────────────────
# soft‐assignment for DEC
# ────────────────────────────────────────────────────
def soft_assign(z, centers, alpha=1.0):
    # z: (N,D), centers: (K,D)
    dist_sq = torch.cdist(z, centers) ** 2
    q = (1.0 + dist_sq/alpha).pow(-(alpha+1)/2)
    return q / q.sum(dim=1, keepdim=True)

# ────────────────────────────────────────────────────
# Main inference
# ────────────────────────────────────────────────────
def main(args):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # 1) Load cluster centers & z_dim from checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=DEVICE)
    centers = ckpt["cluster_centers"]  # CPU or CUDA tensor
    if centers is None:
        raise RuntimeError("No cluster_centers in checkpoint!")
    centers = centers.to(DEVICE)

    # 2) Build dataset + model
    ds = WaferDatasetV8(args.input_dir)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)
    z_dim_total = centers.size(1) - ds.hog_dim  # total dim = z_dim + hog
    print(f"Reconstructed z_dim = {z_dim_total}, hog_dim = {ds.hog_dim}")

    model = ConvAEEncoder(z_dim_total).to(DEVICE)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()

    # 3) Extract all embeddings + paths
    all_feats, all_paths = [], []
    all_qs = []
    with torch.no_grad():
        for x, hog_vec, idxs, paths in loader:
            x = x.to(DEVICE)
            z = model(x)                        # B × z_dim
            hog_b = hog_vec.to(DEVICE)         # B × hog_dim
            feats = torch.cat([z, hog_b], dim=1)  # B × (z_dim+hog_dim)
            all_feats.append(feats.cpu())
            all_paths.extend(paths)
    all_feats = torch.cat(all_feats, dim=0)  # N × D

    # 4) Soft‐assign to clusters
    print("Computing soft assignments …")
    feats_device = all_feats.to(DEVICE)
    q_all = soft_assign(feats_device, centers)  # N × K
    q_all = q_all.cpu().numpy()
    assigns    = np.argmax(q_all, axis=1)
    confidences= np.max(q_all, axis=1)

    # 5) Prepare output dirs
    for k in range(centers.size(0)):
        os.makedirs(os.path.join(args.output_dir, f"type_{k}"), exist_ok=True)
    summary_dir = os.path.join(args.output_dir, "summaries")
    os.makedirs(summary_dir, exist_ok=True)

    # 6) Copy every wafer into its assigned bucket
    print("Copying images into type_*/ …")
    for path, cid in zip(all_paths, assigns):
        dst = os.path.join(args.output_dir, f"type_{cid}", os.path.basename(path))
        shutil.copy(path, dst)

    # 7) Generate summary slides (top_k by confidence)
    print(f"Making summary slides (top {args.top_k}) …")
    for k in range(centers.size(0)):
        idxs = np.where(assigns == k)[0]
        # sort by confidence desc
        idxs = idxs[np.argsort(-confidences[idxs])][: args.top_k]

        imgs = []
        for i in idxs:
            img = Image.open(all_paths[i]).convert("RGB")
            tf  = T.Resize((IMG_SIZE,IMG_SIZE))
            imgs.append(T.ToTensor()(tf(img)))
        # pad if fewer
        while len(imgs) < args.top_k:
            imgs.append(torch.zeros(3, IMG_SIZE, IMG_SIZE))
        grid = vutils.make_grid(imgs, nrow=int(np.ceil(np.sqrt(args.top_k))), padding=2)
        vutils.save_image(
            grid,
            os.path.join(summary_dir, f"summary_cluster_{k}.png")
        )

    print("Done. Buckets in", args.output_dir)
    print("Summaries in", summary_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir",    required=True,
                   help="Path to wafer PNGs")
    p.add_argument("--checkpoint",   required=True,
                   help="main_v8.py checkpoint (ckpt_*.pth)")
    p.add_argument("--output_dir",   required=True,
                   help="Where to copy type_*/ and summaries/")
    p.add_argument("--top_k",        type=int, default=9,
                   help="How many images per cluster in summary")
    args = p.parse_args()
    main(args)
