#!/usr/bin/env python3
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

# ─── your pipeline imports ────────────────────────────────────
from semisupervised_v2 import (
    collect_paths,
    SemiDataset,
    SemiDualNet,
    soft_assign,
)

# ─── USER SETTINGS ────────────────────────────────────────────
DATA_DIR   = "data/Op3176_DefectMap_Labeled"
OUT_DIR    = "outputs/sup_v2"
CKPT_FILE  = os.path.join(OUT_DIR, "checkpoints", "ckpt_100.pth")
BATCH_SIZE = 32
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── 1) Load model + centers ──────────────────────────────────
ckpt    = torch.load(CKPT_FILE, map_location=DEVICE)
centers = ckpt["centers"].to(DEVICE)

# ─── 2) Build data + model ───────────────────────────────────
seed_paths, seed_lbls, unl_paths, _, n_seed = collect_paths(DATA_DIR)
dataset = SemiDataset(seed_paths, seed_lbls, unl_paths)
loader  = DataLoader(dataset, BATCH_SIZE, shuffle=False, pin_memory=True)

model = SemiDualNet(n_seed).to(DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()

# ─── 3) Embed everything, get hard‐assigns & seed flags ───────
feats = []
seed_flags = []
with torch.no_grad():
    for x, is_seed, _, _, _ in loader:
        x = x.to(DEVICE)
        z, _, _, _ = model(x, x)
        feats.append(z.cpu().numpy())
        seed_flags.extend(is_seed.numpy())

feats      = np.vstack(feats)  # (N, Z)
seed_flags = np.array(seed_flags, bool)

q       = soft_assign(torch.from_numpy(feats).to(DEVICE), centers)
cids    = q.argmax(1).cpu().numpy()  # values 0–33

# ─── 4) t-SNE on all points ───────────────────────────────────
tsne2d = TSNE(
    n_components=2, init="pca", learning_rate="auto",
    perplexity=30, random_state=42
).fit_transform(feats)

# ─── 5) Build 29‐color palette for clusters 1–29 ───────────────
c20  = plt.cm.tab20(np.linspace(0,1,20))     # 20 colors
c20b = plt.cm.tab20b(np.linspace(0,1,9))     # 9 more
palette_29 = np.vstack([c20, c20b])          # shape (29,4)

# ─── 6) Plot ─────────────────────────────────────────────────
plt.figure(figsize=(8,6))

# A) clusters 1–29
mask_1_29 = (cids>=1)&(cids<=29)
labels_1_29 = cids[mask_1_29] - 1  # shift to 0–28

#   – non-seed circles
m_ns = mask_1_29 & ~seed_flags
colors_ns = palette_29[cids[m_ns]-1]
plt.scatter(
    tsne2d[m_ns,0], tsne2d[m_ns,1],
    c=colors_ns, marker='o', s=20, alpha=0.7,
    label="unlabeled (1–29)"
)

#   – seed squares
m_s = mask_1_29 & seed_flags
colors_s = palette_29[cids[m_s]-1]
plt.scatter(
    tsne2d[m_s,0], tsne2d[m_s,1],
    c=colors_s, marker='s', s=50,
    edgecolor='k', linewidth=0.8, label="seed (1–29)"
)

# B) buckets 30–33 as big black triangles
mask_buck = (cids>=30)&(cids<=33)
plt.scatter(
    tsne2d[mask_buck,0], tsne2d[mask_buck,1],
    c='k', marker='^', s=60, alpha=0.9,
    label="unclassified (30–33)"
)

# ─── 7) Colorbar for clusters 1–29 only ───────────────────────
# create a dummy ScalarMappable
from matplotlib.colors import ListedColormap, BoundaryNorm
cmap29 = ListedColormap(palette_29)
norm29 = BoundaryNorm(np.arange(0,30)-0.5, cmap29.N)
sm = plt.cm.ScalarMappable(cmap=cmap29, norm=norm29)
sm.set_array([])
cb = plt.colorbar(sm, ticks=np.arange(29)+0.5)
cb.set_ticklabels(np.arange(1,30))
cb.set_label("Cluster ID")

plt.legend(loc='best', scatterpoints=1)
plt.title("t-SNE of All Points: Clusters 1–29 & Buckets 30–33")
plt.tight_layout()

# ─── 8) Save ─────────────────────────────────────────────────
out = os.path.join(OUT_DIR, "figures", "tsne_all_distinct.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=150)
print(f"Saved ➔ {out}")
