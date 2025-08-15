#!/usr/bin/env python3
"""
inference_edge_dbscan_v7.py — Inference-only DBSCAN clustering on Conv-AE embeddings (v7 adapted)

This script:
 1. Loads the Conv-AE encoder (mask+edge) from a DEC checkpoint.
 2. Embeds all wafer masks in the input directory (mask + edge channels).
 3. Runs DBSCAN on normalized embeddings (cosine metric) to find arbitrarily many tight clusters.
 4. Copies clustered images (labels ≥ 0) into `outputs/type_labels/type_<cluster_id>/`.
 5. Generates summary slides of the top-K most frequent images per cluster.

Usage:
    python inference_edge_dbscan_v7.py \
      --input data/Op3176_DefectMap \
      --checkpoint outputs/v5/checkpoints/ckpt_100.pth \
      --output outputs/v5/dbscan_results \
      --eps 0.1 \
      --min_samples 5 \
      --top_k 9
"""
import os
import argparse
import shutil
import torch
import numpy as np
from glob import glob
from PIL import Image, ImageFilter
import torchvision.transforms as T
import torchvision.utils as vutils
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
import torch.nn as nn

# ----------------------------------------------------------------
# Constants for image size and device
# These should match your training configuration
IMG_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------------
# Define ConvAutoencoder matching v7 (2-channel input)
class ConvAutoencoder(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 128→64
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 64→32
            nn.Flatten(), nn.Linear(64*32*32, z_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 64*32*32), nn.Unflatten(1, (64,32,32)),
            nn.Upsample(scale_factor=2), nn.Conv2d(64,32,3,padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2), nn.Conv2d(32,2,3,padding=1), nn.Sigmoid()
        )
    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return z, x_rec

# ----------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------
parser = argparse.ArgumentParser(description='DBSCAN clustering on Conv-AE embeddings')
parser.add_argument('--input',       type=str, required=True, help='Directory of wafer masks')
parser.add_argument('--checkpoint',  type=str, required=True, help='Path to DEC checkpoint .pth')
parser.add_argument('--output',      type=str, default='outputs/v5/dbscan_results', help='Root output directory')
parser.add_argument('--eps',         type=float, default=0.1, help='DBSCAN eps for cosine metric')
parser.add_argument('--min_samples', type=int,   default=5,   help='DBSCAN min_samples')
parser.add_argument('--top_k',       type=int,   default=9,   help='Top-K images per cluster for summary')
args = parser.parse_args()

INPUT_DIR   = args.input
CKPT_PATH   = args.checkpoint
OUTPUT_ROOT = args.output
EPS         = args.eps
MIN_SAMP    = args.min_samples
TOP_K       = args.top_k

# ----------------------------------------------------------------
# Prepare output dirs
# ----------------------------------------------------------------
clusters_dir = os.path.join(OUTPUT_ROOT, 'type_labels')
os.makedirs(clusters_dir, exist_ok=True)
summary_dir = os.path.join(OUTPUT_ROOT, 'summaries')
os.makedirs(summary_dir, exist_ok=True)

# ----------------------------------------------------------------
# Load encoder
# ----------------------------------------------------------------
print(f"Loading encoder from checkpoint: {CKPT_PATH}")
ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
z_dim = ckpt['cluster_centers'].shape[1] if 'cluster_centers' in ckpt else ckpt['model_state']['encoder.6.weight'].shape[0]
model = ConvAutoencoder(z_dim=z_dim).to(DEVICE)
model.load_state_dict(ckpt['model_state'])
model.eval()
encoder = model.encoder

# ----------------------------------------------------------------
# Transforms for mask+edge
# ----------------------------------------------------------------
resize = T.Resize((IMG_SIZE, IMG_SIZE))
totensor = T.ToTensor()

def load_edge_tensor(path):
    img = Image.open(path).convert('L')
    mask = resize(img)
    mask_t = totensor(mask)
    edges = mask.filter(ImageFilter.FIND_EDGES)
    edges = resize(edges)
    edges_t = totensor(edges)
    return torch.cat([mask_t, edges_t], dim=0).unsqueeze(0)

# ----------------------------------------------------------------
# Embed all images
# ----------------------------------------------------------------
paths = sorted(glob(os.path.join(INPUT_DIR, '*.[Pp][Nn][Gg]')))
print(f"Embedding {len(paths)} images...")
embs = []
for p in paths:
    x = load_edge_tensor(p).to(DEVICE)
    with torch.no_grad():
        z, _ = model(x)
    embs.append(z.cpu().numpy().reshape(-1))
embs = np.vstack(embs)
# normalize for cosine DBSCAN
embs_norm = normalize(embs)

# ----------------------------------------------------------------
# DBSCAN clustering
# ----------------------------------------------------------------
print(f"Running DBSCAN(eps={EPS}, min_samples={MIN_SAMP})...")
db = DBSCAN(eps=EPS, min_samples=MIN_SAMP, metric='cosine')
labels = db.fit_predict(embs_norm)
unique_labels = sorted({l for l in labels if l >= 0})
print(f"Discovered {len(unique_labels)} clusters: {unique_labels}")

# ----------------------------------------------------------------
# Assign and copy images
# ----------------------------------------------------------------
results = []
for p, lbl in zip(paths, labels):
    if lbl < 0:
        continue
    dest = os.path.join(clusters_dir, f"type_{lbl}")
    os.makedirs(dest, exist_ok=True)
    shutil.copy(p, dest)
    results.append((p, lbl))
print(f"Assigned {len(results)} images to clusters")

# ----------------------------------------------------------------
# Generate summary slides
# ----------------------------------------------------------------
print("Generating summary slides...")
for lbl in unique_labels:
    members = [p for (p, l) in results if l == lbl]
    selected = members[:TOP_K]
    imgs = []
    for p in selected:
        img = Image.open(p).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
        imgs.append(totensor(img))
    while len(imgs) < TOP_K:
        imgs.append(torch.zeros(3, IMG_SIZE, IMG_SIZE))
    grid = vutils.make_grid(imgs, nrow=int(np.sqrt(TOP_K)), padding=2)
    fname = os.path.join(summary_dir, f"summary_cluster_{lbl}.png")
    vutils.save_image(grid, fname)
print(f"Summary slides saved to {summary_dir}")
