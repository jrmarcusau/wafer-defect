#!/usr/bin/env python3
"""
inference_v9.py — Inference + top/random summaries for main_v9 DEC model
"""

import os
import argparse
import shutil
import random
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
# Constants  — must match main_v9.py
# ────────────────────────────────────────────────────
IMG_SIZE   = 128
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED       = 42

# ────────────────────────────────────────────────────
# Dataset: polar warp + edge + CoordConv + HOG
# ────────────────────────────────────────────────────
class WaferDatasetV9(Dataset):
    def __init__(self, root):
        self.paths = sorted(glob(os.path.join(root, "*.PNG")) +
                            glob(os.path.join(root, "*.png")))
        if not self.paths:
            raise RuntimeError(f"No images found under {root}")
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

        # 1) Polar warp + rotate so angle maps horizontally
        polar = cv2.warpPolar(
            arr,
            (IMG_SIZE, IMG_SIZE),
            center=(arr.shape[1]//2, arr.shape[0]//2),
            maxRadius=arr.shape[0]//2,
            flags=cv2.WARP_POLAR_LINEAR
        )
        polar = cv2.rotate(polar, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # 2) Edge channel (Canny)
        edge = cv2.Canny(polar, 50, 150)

        # 3) HOG descriptor on polar mask
        hog_vec = hog(
            polar,
            orientations=9,
            pixels_per_cell=(16,16),
            cells_per_block=(1,1),
            feature_vector=True
        ).astype(np.float32)

        # 4) To torch tensors
        polar_t = torch.from_numpy(polar/255.0).unsqueeze(0).float()
        edge_t  = torch.from_numpy(edge/255.0).unsqueeze(0).float()

        # 5) CoordConv channels (x, y in [-1,1])
        xs = torch.linspace(-1,1,IMG_SIZE).view(1,1,-1) \
               .expand(1,IMG_SIZE,IMG_SIZE)
        ys = torch.linspace(-1,1,IMG_SIZE).view(1,-1,1) \
               .expand(1,IMG_SIZE,IMG_SIZE)
        coord = torch.cat([xs, ys], dim=0).float()

        # 6) Final input: 4×H×W
        x = torch.cat([polar_t, edge_t, coord], dim=0)

        return x, torch.from_numpy(hog_vec), idx, path

# ────────────────────────────────────────────────────
# Conv-AE definition (encoder + decoder) — same as main_v9
# ────────────────────────────────────────────────────
class ConvAE(nn.Module):
    def __init__(self, z_dim, hog_dim):
        super().__init__()
        # encoder: 4→z_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(4,  32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*32*32, z_dim)
        )
        # decoder: (z+hog)→2
        self.decoder = nn.Sequential(
            nn.Linear(z_dim + hog_dim, 64*32*32),
            nn.Unflatten(1, (64,32,32)),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64,32,3,padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 2,3,padding=1),  nn.Sigmoid()
        )

    def forward(self, x, hog_vec):
        z      = self.encoder(x)
        z_full = torch.cat([z, hog_vec], dim=1)
        x_rec  = self.decoder(z_full)
        return z, x_rec

# ────────────────────────────────────────────────────
# DEC soft assignment
# ────────────────────────────────────────────────────
def soft_assign(feats, centers, alpha=1.0):
    dist_sq = torch.cdist(feats, centers)**2
    q = (1.0 + dist_sq/alpha).pow(-(alpha+1)/2)
    return q / q.sum(dim=1, keepdim=True)

# ────────────────────────────────────────────────────
# Main inference
# ────────────────────────────────────────────────────
def main(args):
    # reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=DEVICE)
    centers = ckpt["cluster_centers"]
    if centers is None:
        raise RuntimeError("No cluster_centers found in checkpoint")
    centers = centers.to(DEVICE)

    # build dataset + loader
    ds     = WaferDatasetV9(args.input_dir)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)
    z_dim  = centers.size(1) - ds.hog_dim
    K      = centers.size(0)
    print(f"Detected z_dim={z_dim}, hog_dim={ds.hog_dim}, K={K}")

    # load model
    model = ConvAE(z_dim, ds.hog_dim).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # collect embeddings + confidences + paths
    all_feats, all_paths = [], []
    for x, hog_vec, idxs, paths in loader:
        x_batch   = x.to(DEVICE)
        hog_batch = hog_vec.to(DEVICE)
        with torch.no_grad():
            z, _ = model(x_batch, hog_batch)
        feats = torch.cat([z, hog_batch], dim=1)
        all_feats.append(feats.cpu())
        all_paths.extend(paths)
    all_feats = torch.cat(all_feats, dim=0)  # N×D

    # compute soft assignments
    print("Computing soft assignments …")
    q_all = soft_assign(all_feats.to(DEVICE), centers).cpu().numpy()  # N×K
    assigns     = np.argmax(q_all, axis=1)
    confidences = np.max(q_all, axis=1)

    # prepare output dirs
    for k in range(K):
        os.makedirs(os.path.join(args.output_dir, f"type_{k}"), exist_ok=True)
    top_dir    = os.path.join(args.output_dir, "summaries", "top9")
    rand_dir   = os.path.join(args.output_dir, "summaries", "random9")
    os.makedirs(top_dir,  exist_ok=True)
    os.makedirs(rand_dir, exist_ok=True)

    # copy every wafer to its bucket
    print("Copying wafers into type_*/ …")
    for path, cid in zip(all_paths, assigns):
        dst = os.path.join(args.output_dir, f"type_{cid}", os.path.basename(path))
        shutil.copy(path, dst)

    # build summaries
    print(f"Building summaries (top {args.top_k} & random {args.random_k}) …")
    tf_resize = T.Resize((IMG_SIZE, IMG_SIZE))
    for k in range(K):
        idxs = np.where(assigns == k)[0]
        if len(idxs) == 0:
            continue

        # Top-k by confidence
        top_idxs = idxs[np.argsort(-confidences[idxs])][: args.top_k]
        top_imgs = []
        for i in top_idxs:
            img = Image.open(all_paths[i]).convert("RGB")
            top_imgs.append(T.ToTensor()(tf_resize(img)))
        while len(top_imgs) < args.top_k:
            top_imgs.append(torch.zeros(3,IMG_SIZE,IMG_SIZE))
        grid_top = vutils.make_grid(top_imgs, nrow=int(np.ceil(np.sqrt(args.top_k))), padding=2)
        vutils.save_image(grid_top, os.path.join(top_dir, f"cluster_{k}.png"))

        # Random-k
        rand_sample = random.sample(list(idxs), min(len(idxs), args.random_k))
        rand_imgs = []
        for i in rand_sample:
            img = Image.open(all_paths[i]).convert("RGB")
            rand_imgs.append(T.ToTensor()(tf_resize(img)))
        while len(rand_imgs) < args.random_k:
            rand_imgs.append(torch.zeros(3,IMG_SIZE,IMG_SIZE))
        grid_rand = vutils.make_grid(rand_imgs, nrow=int(np.ceil(np.sqrt(args.random_k))), padding=2)
        vutils.save_image(grid_rand, os.path.join(rand_dir, f"cluster_{k}.png"))

    print("Done.")
    print("• Buckets: ", args.output_dir)
    print("• Summaries top9: ", top_dir)
    print("• Summaries random9: ", rand_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",   required=True,
                        help="Folder of wafer .PNG masks")
    parser.add_argument("--checkpoint",  required=True,
                        help="Path to v9 checkpoint (ckpt_*.pth)")
    parser.add_argument("--output_dir",  required=True,
                        help="Where to write type_*/ and summaries/")
    parser.add_argument("--top_k",       type=int, default=9,
                        help="How many top images per cluster")
    parser.add_argument("--random_k",    type=int, default=9,
                        help="How many random images per cluster")
    args = parser.parse_args()
    main(args)
