#!/usr/bin/env python3
"""
inference_sup_v3.py
────────────────────
Inference-only script for clustering a raw wafer set into:
  • 30 strong clusters (ids 0–29)
  • 4 weak buckets (ids 30–33)
  • noise (very low-confidence items)

Inputs
------
  • Raw images: data/Op3176_DefectMap/*.png
  • Trained model + DEC centers: outputs/sup_v3/checkpoints/ckpt_100.pth

Outputs
-------
  • outputs/sup_v3/results/
        ├─ type_0 ... type_33/
        ├─ type_noise/
        └─ summaries/top9/cluster_<id>.png   (and cluster_noise.png)
"""

# ───────────────────────── Imports ──────────────────────────
from __future__ import annotations

import argparse
import os
from glob import glob
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils
from PIL import Image
from torchvision.transforms import Resize, ToTensor
from torch.utils.data import DataLoader, Dataset

# ───────────────────── Global hyper-parameters ─────────────────────
IMG_SIZE = 256

MAIN_BATCH = 32

Z_STREAM = 256
Z_FUSED = 256
K_CLUSTERS = 30
N_WEAK = 4                    # Weak-0…Weak-3  → ids 30-33

TAU_STRONG = 0.65             # same as training v3
NOISE_Q_THRESH = 0.10         # heuristic: ultra-low DEC confidence → noise

TOP_K = 9

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

    stacked = np.stack(
        [
            gray.astype(np.float32) / 255.0,
            sob.astype(np.float32) / 255.0,
            grid_x,
            grid_y,
        ],
        axis=0,
    )

    return torch.from_numpy(stacked)


def ensure_dirs(*dirs: str | Path) -> None:
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def save_montage(img_paths: List[str], out_file: str, k: int) -> None:
    """Top-K grid, de-duplicated, padded with blanks if needed."""
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
    grid = vutils.make_grid(imgs, nrow=int(np.ceil(np.sqrt(k))), padding=2)
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
# Model (must match the training architecture exactly)
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
# DEC helpers
# -----------------------------------------------------------------------------
def soft_assign(z: torch.Tensor, centers: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Student-t similarity used in DEC."""
    dist2 = torch.cdist(z, centers).pow(2)
    q = (1.0 + dist2 / alpha).pow(-(alpha + 1) / 2)
    return q / q.sum(dim=1, keepdim=True)


# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------
def run_inference(data_dir: str, out_dir: str, ckpt_path: str) -> None:
    # Gather images (png/PNG)
    img_paths = sorted(glob(os.path.join(data_dir, "*.png")) + glob(os.path.join(data_dir, "*.PNG")))
    assert len(img_paths) > 0, f"No PNG images found in {data_dir}"

    # Output directories
    top9_dir = os.path.join(out_dir, "summaries", "top9")
    ensure_dirs(out_dir, top9_dir)
    for k in range(K_CLUSTERS + N_WEAK):
        ensure_dirs(os.path.join(out_dir, f"type_{k}"))
    ensure_dirs(os.path.join(out_dir, "type_noise"))

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_state = ckpt["model_state"]
    centers = ckpt["centers"].to(DEVICE).float()

    # Infer head size from checkpoint
    head_weight = model_state["head.weight"]
    n_seed_classes = head_weight.shape[0]

    # Rebuild model and load weights
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

    # Storage for features and bookkeeping
    feats_list: List[torch.Tensor] = []
    rec_err_list: List[float] = []
    paths_list: List[str] = []

    with torch.no_grad():
        for x, _, pths in loader:
            x = x.to(DEVICE)
            z, rec_p, rec_c, _ = model(x, x)

            # Use polar decoder recon error as a quick consistency proxy
            rec_err = torch.mean((rec_p - x[:, :1]) ** 2, dim=[1, 2, 3])

            feats_list.append(z.detach().cpu())
            rec_err_list.extend(rec_err.detach().cpu().tolist())
            paths_list.extend(list(pths))

    feats = torch.cat(feats_list, dim=0).float()
    rec_err = np.asarray(rec_err_list, dtype=np.float32)

    # DEC soft assignment to saved centers
    q = soft_assign(feats.to(DEVICE), centers).cpu().numpy()
    conf = q.max(axis=1)
    cid = q.argmax(axis=1)

    # Heuristic noise: ultra-low confidence to any center
    is_noise = conf < NOISE_Q_THRESH

    # Strong vs weak buckets (only for non-noise)
    strong_mask = (conf >= TAU_STRONG) & (~is_noise)
    weak_mask = (~strong_mask) & (~is_noise)

    final = cid.copy()

    if np.any(weak_mask):
        # Quantile bins on weak subset only
        qcuts = np.quantile(conf[weak_mask], [0.25, 0.5, 0.75])
        weak_bin = np.digitize(conf[weak_mask], qcuts)
        final[weak_mask] = K_CLUSTERS + weak_bin

    # Copy images to folders
    for p, lbl, nflag in zip(paths_list, final, is_noise):
        if nflag:
            dst_dir = os.path.join(out_dir, "type_noise")
        else:
            dst_dir = os.path.join(out_dir, f"type_{int(lbl)}")
        ensure_dirs(dst_dir)
        shutil_path = os.path.join(dst_dir, os.path.basename(p))
        if p != shutil_path:
            try:
                # Copy (not move) to keep raw intact
                from shutil import copy2
                copy2(p, shutil_path)
            except Exception:
                pass

    # Build top9 summaries for each bucket 0–33 and noise
    # For 0–29 and 30–33: rank by L2 distance to their assigned center
    # For noise: rank by lowest confidence (most "noisy" first)
    top_map = {}
    for k in range(K_CLUSTERS + N_WEAK):
        idx = np.where((final == k) & (~is_noise))[0]
        if idx.size == 0:
            continue
        center_k = centers[k if k < K_CLUSTERS else (k - K_CLUSTERS)].detach().cpu().numpy() \
                   if k < K_CLUSTERS else centers[cid[idx]].detach().cpu().numpy()
        # For weak buckets, there is no dedicated center saved; approximate with the
        # per-item assigned strong center for distance ranking.
        if k < K_CLUSTERS:
            dist = np.linalg.norm(feats[idx].numpy() - centers[k].detach().cpu().numpy(), axis=1)
        else:
            dist = np.linalg.norm(
                feats[idx].numpy() - centers[cid[idx]].detach().cpu().numpy(),
                axis=1,
            )
        order = idx[np.argsort(dist)]
        top_map[k] = [paths_list[i] for i in order[:TOP_K]]

    for k, paths_k in top_map.items():
        out_png = os.path.join(top9_dir, f"cluster_{k}.png")
        save_montage(paths_k, out_png, TOP_K)

    # Noise summary (pick the K lowest-confidence samples)
    idx_noise = np.where(is_noise)[0]
    if idx_noise.size > 0:
        order_noise = idx_noise[np.argsort(conf[idx_noise])]
        top_noise = [paths_list[i] for i in order_noise[:TOP_K]]
        save_montage(top_noise, os.path.join(top9_dir, "cluster_noise.png"), TOP_K)

    # Console summary
    counts = {}
    for k in range(K_CLUSTERS + N_WEAK):
        counts[k] = int(np.sum((final == k) & (~is_noise)))
    n_noise = int(np.sum(is_noise))

    print("─" * 60)
    print(f"Total images: {len(paths_list)}")
    print(f"Noise: {n_noise}")
    for k in range(K_CLUSTERS + N_WEAK):
        print(f"type_{k}: {counts[k]}")
    print("Top9 summaries saved to:", top9_dir)
    print("Done.")


# -----------------------------------------------------------------------------
# CLI entry
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/Op3176_DefectMap")
    parser.add_argument("--out_dir",  default="outputs/sup_v3/results")
    parser.add_argument("--ckpt",     default="outputs/sup_v3/checkpoints/ckpt_100.pth")
    args = parser.parse_args()

    run_inference(args.data_dir, args.out_dir, args.ckpt)


if __name__ == "__main__":
    main()
