#!/usr/bin/env python3
"""
inference_edge_v8.py — Inference-only fresh KMeans clustering on embeddings

This script:
 1. Loads the trained encoder from a DEC checkpoint (mask+edge autoencoder).
 2. Embeds all wafer masks in the input directory.
 3. Runs fresh KMeans with K=num_clusters on the embeddings.
 4. Computes cosine similarity to each cluster center and assigns only if similarity ≥ threshold.
 5. Copies high-confidence images into outputs/type_<cluster_id>/
 6. Generates summary slides top-K per cluster.

Usage:
    python inference_edge_v8.py \
      --input data/Op3176_DefectMap \
      --checkpoint outputs/v5/checkpoints/ckpt_100.pth \
      --output outputs/v5/results \
      --num_clusters 20 \
      --sim_threshold 0.9 \
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
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

# import encoder
from main_v7 import ConvAutoencoder, IMG_SIZE, DEVICE, Z_DIM

# ----------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='data/Op3176_DefectMap', help='Input directory')
parser.add_argument('--checkpoint', type=str, default='outputs/v7/checkpoints/ckpt_100.pth', help='Path to DEC checkpoint')
parser.add_argument('--output', type=str, default='outputs/v7/results', help='Root output directory')
parser.add_argument('--num_clusters', type=int, default=20, help='Number of clusters')
parser.add_argument('--sim_threshold', type=float, default=0.9, help='Cosine similarity threshold to include image')
parser.add_argument('--top_k', type=int, default=9, help='Top-K images per cluster for summary')
args = parser.parse_args()

INPUT_DIR   = args.input
CKPT_PATH   = args.checkpoint
OUTPUT_ROOT = args.output
K           = args.num_clusters
SIM_THR     = args.sim_threshold
TOP_K       = args.top_k

# ----------------------------------------------------------------
# Load encoder
# ----------------------------------------------------------------
print(f"Loading DEC checkpoint from {CKPT_PATH}...")
ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
model = ConvAutoencoder(z_dim=Z_DIM).to(DEVICE)
model.load_state_dict(ckpt['model_state'])
model.eval()
# we'll only use the encoder

# ----------------------------------------------------------------
# Prepare transforms
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
emb_list = []
for p in paths:
    x = load_edge_tensor(p).to(DEVICE)
    with torch.no_grad():
        z, _ = model(x)
    emb_list.append(z.cpu().numpy().reshape(-1))
embs = np.vstack(emb_list)  # (N, Z_DIM)
# normalize for cosine similarity
norm_embs = normalize(embs)

# ----------------------------------------------------------------
# Fresh KMeans clustering
# ----------------------------------------------------------------
print(f"Clustering into {K} clusters (KMeans)...")
km = KMeans(n_clusters=K, n_init=20, random_state=42)
assignments = km.fit_predict(norm_embs)
centers = normalize(km.cluster_centers_)

# ----------------------------------------------------------------
# Assign high-similarity images
# ----------------------------------------------------------------
print(f"Assigning with cosine sim ≥ {SIM_THR}...")
results = []  # (path, cid, sim)
sim_matrix = norm_embs.dot(centers.T)  # (N,K)
for idx, p in enumerate(paths):
    cid = int(assignments[idx])
    sim = float(sim_matrix[idx, cid])
    if sim >= SIM_THR:
        dest = os.path.join(OUTPUT_ROOT, f"type_{cid}")
        os.makedirs(dest, exist_ok=True)
        shutil.copy(p, dest)
        results.append((p, cid, sim))
print(f"Kept {len(results)} images above threshold {SIM_THR}")

# ----------------------------------------------------------------
# Summary slides
# ----------------------------------------------------------------
print("Generating summary slides...")
summary_dir = os.path.join(OUTPUT_ROOT, 'summaries')
os.makedirs(summary_dir, exist_ok=True)
for cid in range(K):
    entries = [(p, sim) for (p, c, sim) in results if c == cid]
    if not entries:
        continue
    entries.sort(key=lambda x: x[1], reverse=True)
    top = entries[:TOP_K]
    imgs = []
    for (path, _) in top:
        img = Image.open(path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
        imgs.append(totensor(img))
    while len(imgs) < TOP_K:
        imgs.append(torch.zeros(3, IMG_SIZE, IMG_SIZE))
    grid = vutils.make_grid(imgs, nrow=int(np.sqrt(TOP_K)), padding=2)
    fn = os.path.join(summary_dir, f"summary_cluster_{cid}.png")
    vutils.save_image(grid, fn)
print(f"Summary slides saved to {summary_dir}")
