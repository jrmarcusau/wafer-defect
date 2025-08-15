#!/usr/bin/env python3
"""
inference_sup_v4.py
────────────────────
Inference-only script for v4 models (no weak buckets).
Forces every input image into one of 30 clusters (0–29).

Inputs
------
  • --data_dir  : flat folder of unlabeled PNGs
  • --ckpt      : trained checkpoint with model_state + centers
  • --out_dir   : output root; creates type_* and summaries/{top9,random9}

Outputs
-------
  <out_dir>/
    ├─ type_0 ... type_29/
    └─ summaries/
         ├─ top9/cluster_<id>.png
         └─ random9/cluster_<id>.png
"""

# ───────────────────────── Imports ──────────────────────────
from __future__ import annotations

import argparse
import os
import shutil
from glob import glob
from typing import List

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, Resize

# ───────────────────── Global hyper-parameters ─────────────────────
IMG_SIZE   = 256
MAIN_BATCH = 32

Z_STREAM   = 256
Z_FUSED    = 256
K_CLUSTERS = 30

TOP_K  = 9
RAND_K = 9

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def to_four_channel(arr_u8: np.ndarray) -> torch.Tensor:
    """
    Convert an H×W uint8 image to a 4×H×W torch.Tensor:
      [gray, |sobel_x|, coord_x, coord_y], all normalized [0,1].
    """
    gray = cv2.resize(arr_u8, (IMG_SIZE, IMG_SIZE), cv2.INTER_LINEAR)

    sob = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sob = cv2.convertScaleAbs(np.abs(sob))

    coords = np.linspace(-1.0, 1.0, IMG_SIZE, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(coords, coords)

    stacked = np.stack([
        gray.astype(np.float32) / 255.0,
        sob.astype(np.float32)  / 255.0,
        grid_x,
        grid_y
    ], axis=0)

    return torch.from_numpy(stacked.astype(np.float32))


def ensure_dirs(*dirs: str) -> None:
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def save_montage(img_paths: List[str], out_file: str, k: int) -> None:
    """Save a K-image grid without duplicates, padding with blanks if needed."""
    tf_resize = Resize((IMG_SIZE, IMG_SIZE))
    imgs: List[torch.Tensor] = []
    seen = set()

    for p in img_paths:
        if p not in seen:
            seen.add(p)
            imgs.append(ToTensor()(tf_resize(Image.open(p).convert("RGB"))))
        if len(imgs) == k:
            break

    while len(imgs) < k:
        imgs.append(torch.zeros(3, IMG_SIZE, IMG_SIZE))

    grid = vutils.make_grid(imgs[:k], nrow=int(np.ceil(np.sqrt(k))), padding=2)
    vutils.save_image(grid, out_file)


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class RawDataset(Dataset):
    def __init__(self, image_paths: List[str]) -> None:
        self.paths = image_paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        arr = cv2.imread(self.paths[idx], cv2.IMREAD_GRAYSCALE)
        x = to_four_channel(arr)
        return x, idx, self.paths[idx]


# -----------------------------------------------------------------------------
# Model (must match training architecture)
# -----------------------------------------------------------------------------
class StreamEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 256→128
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 128→64
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64→32
            nn.Flatten(),
            nn.Linear(128 * 32 * 32, Z_STREAM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StreamDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(Z_STREAM, 128 * 32 * 32),
            nn.Unflatten(1, (128, 32, 32)),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class SemiDualNet(nn.Module):
    """Dual-stream AE + supervised head (head size inferred from checkpoint)."""

    def __init__(self, n_seed_classes: int) -> None:
        super().__init__()
        self.enc_pol = StreamEncoder()
        self.enc_cart = StreamEncoder()
        self.dec_pol = StreamDecoder()
        self.dec_cart = StreamDecoder()

        self.fuse = nn.Sequential(
            nn.Linear(2 * Z_STREAM, 512),
            nn.ReLU(),
            nn.Linear(512, Z_FUSED),
        )
        self.head = nn.Linear(Z_FUSED, n_seed_classes)

    def forward(self, x_polar: torch.Tensor, x_cart: torch.Tensor):
        z_p = self.enc_pol(x_polar)
        z_c = self.enc_cart(x_cart)
        fused = self.fuse(torch.cat([z_c, z_p], 1))
        rec_p = self.dec_pol(z_p)
        rec_c = self.dec_cart(z_c)
        logits = self.head(fused)
        return fused, rec_p, rec_c, logits


# -----------------------------------------------------------------------------
# DEC helper functions
# -----------------------------------------------------------------------------
def soft_assign(z: torch.Tensor, centers: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Student-t similarity used in DEC."""
    dist2 = torch.cdist(z, centers).pow(2)
    q = (1.0 + dist2 / alpha).pow(-(alpha + 1) / 2)
    return q / q.sum(dim=1, keepdim=True)


# -----------------------------------------------------------------------------
# Main inference
# -----------------------------------------------------------------------------
def run_inference(data_dir: str, out_dir: str, ckpt_path: str) -> None:
    # Gather images
    img_paths = sorted(glob(os.path.join(data_dir, "*.png")) + glob(os.path.join(data_dir, "*.PNG")))
    assert len(img_paths) > 0, f"No PNG images found in {data_dir}"

    # Prepare outputs
    top9_dir = os.path.join(out_dir, "summaries", "top9")
    rnd9_dir = os.path.join(out_dir, "summaries", "random9")
    ensure_dirs(out_dir, top9_dir, rnd9_dir)
    for k in range(K_CLUSTERS):
        ensure_dirs(os.path.join(out_dir, f"type_{k}"))

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_state = ckpt["model_state"]
    centers = ckpt["centers"].to(DEVICE).float()  # [30, 256]

    # Infer head size and rebuild model
    n_seed_classes = model_state["head.weight"].shape[0]
    model = SemiDualNet(n_seed_classes=n_seed_classes).to(DEVICE)
    model.load_state_dict(model_state, strict=True)
    model.eval()

    # Dataset / loader
    dataset = RawDataset(img_paths)
    loader = DataLoader(
        dataset,
        batch_size=MAIN_BATCH,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    feats_list: List[torch.Tensor] = []
    paths_list: List[str] = []

    with torch.no_grad():
        for xb, _, pths in loader:
            xb = xb.to(DEVICE)
            z, _, _, _ = model(xb, xb)
            feats_list.append(z.detach().cpu())
            paths_list.extend(list(pths))

    feats = torch.cat(feats_list, dim=0).float()             # [N, 256]
    q = soft_assign(feats.to(DEVICE), centers).cpu()          # [N, 30]
    cluster_id = q.argmax(dim=1).numpy()                      # force 0–29

    # Copy images into type_<id> folders
    for p, lbl in zip(paths_list, cluster_id):
        dst_dir = os.path.join(out_dir, f"type_{int(lbl)}")
        dst = os.path.join(dst_dir, os.path.basename(p))
        if p != dst:
            try:
                shutil.copy2(p, dst)
            except Exception:
                pass

    # Build summaries: top9 by distance-to-center + random9
    for k in range(K_CLUSTERS):
        idx = np.where(cluster_id == k)[0]
        if idx.size == 0:
            continue

        # distance to center k
        dist = np.linalg.norm(feats[idx].numpy() - centers[k].detach().cpu().numpy(), axis=1)
        top_order = idx[np.argsort(dist)]

        save_montage([paths_list[i] for i in top_order[:TOP_K]],
                     os.path.join(top9_dir, f"cluster_{k}.png"),
                     TOP_K)

        rnd_sel = np.random.choice(idx, min(len(idx), RAND_K), replace=False)
        save_montage([paths_list[i] for i in rnd_sel],
                     os.path.join(rnd9_dir, f"cluster_{k}.png"),
                     RAND_K)

    # Console summary
    print("─" * 60)
    print(f"Total images: {len(paths_list)}")
    for k in range(K_CLUSTERS):
        print(f"type_{k}: {int(np.sum(cluster_id == k))}")
    print("Summaries saved to:", os.path.join(out_dir, "summaries"))
    print("Done.")


# -----------------------------------------------------------------------------
# CLI entry
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/Op3176_DefectMap")
    parser.add_argument("--ckpt",     default="outputs/sup_v4/checkpoints/ckpt_100.pth")
    parser.add_argument("--out_dir",  default="outputs/sup_v4/results2")
    args = parser.parse_args()

    run_inference(args.data_dir, args.out_dir, args.ckpt)


if __name__ == "__main__":
    main()
