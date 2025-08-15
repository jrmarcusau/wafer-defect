#!/usr/bin/env python3
"""
patch_cluster_inference.py — Unsupervised clustering of localized defects via patch extraction

This script:
 1. Scans wafer defect masks in `data/Op3176_WaferMap2`
 2. Binarizes and finds connected components (defect patches)
 3. Extracts each patch and embeds it with an ImageNet-pretrained EfficientNet-B0 backbone
 4. Clusters patch embeddings into K groups (e.g. 10) via K-means
 5. For each cluster, saves patch crops under `outputs/v5/results/patch_clusters/cluster_<id>/`
 6. Optionally tags parent images if they contain any cluster-
     to separate images by presence of localized defects

Images without any patch above size threshold can be left unassigned.
"""
import os
from glob import glob
import shutil
import torch
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from skimage.measure import label, regionprops
import torchvision.transforms as T
from torchvision.models import efficientnet_b0

# --------------- parameters ---------------
INPUT_DIR       = "data/Op3176_DefectMap2"
OUTPUT_ROOT     = "outputs/v6/results/patch_clusters"
IMG_SIZE        = 128               # patch resize
MIN_AREA        = 50                # minimum pixels for a patch
BATCH_SIZE      = 32
NUM_CLUSTERS    = 10
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------- model setup ---------------
# Load EfficientNet-B0 and use its features + avgpool
backbone = efficientnet_b0(pretrained=True).to(DEVICE)
# remove classifier
backbone.classifier = torch.nn.Identity()
backbone.eval()

# Preprocessing for patches
preprocess = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),        # single-channel -> we'll replicate
    T.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet
                 std=[0.229, 0.224, 0.225])
])

# --------------- gather patches ---------------
patchs = []  # list of (feature_vector, (img_path, region_index, PIL_image))
print("Scanning images and extracting patches...")
for img_path in sorted(glob(os.path.join(INPUT_DIR, "*.PNG")) + glob(os.path.join(INPUT_DIR, "*.png"))):
    mask = np.array(Image.open(img_path).convert("L"))
    # binarize >0
    bin_mask = (mask > 0).astype(np.uint8)
    lbl = label(bin_mask)
    props = regionprops(lbl)
    for i, prop in enumerate(props):
        if prop.area < MIN_AREA:
            continue
        minr, minc, maxr, maxc = prop.bbox
        crop = bin_mask[minr:maxr, minc:maxc] * 255
        # convert to PIL RGB
        patch = Image.fromarray(crop).convert("RGB")
        # store metadata for later save
        patchs.append((img_path, i, patch))

print(f"Extracted {len(patchs)} patches above area {MIN_AREA}")

# --------------- compute embeddings in batches ---------------
print("Computing embeddings...")
embs = []
meta = []
with torch.no_grad():
    for batch_start in range(0, len(patchs), BATCH_SIZE):
        batch = patchs[batch_start:batch_start + BATCH_SIZE]
        imgs = []
        for (img_path, idx, pil) in batch:
            x = preprocess(pil)
            # replicate gray to 3 channels if needed
            if x.size(0) == 1:
                x = x.repeat(3,1,1)
            imgs.append(x)
            meta.append((img_path, idx, pil))
        x_batch = torch.stack(imgs).to(DEVICE)
        feats = backbone(x_batch)  # (B, 1280)
        embs.append(feats.cpu())
embs = torch.cat(embs, dim=0).numpy()

# --------------- cluster embeddings ---------------
print(f"Clustering into {NUM_CLUSTERS} clusters...")
km = KMeans(n_clusters=NUM_CLUSTERS, n_init=20, random_state=42)
labels = km.fit_predict(embs)

# --------------- save patch crops per cluster ---------------
print("Saving clustered patches...")
for k in range(NUM_CLUSTERS):
    dest = os.path.join(OUTPUT_ROOT, f"cluster_{k}")
    os.makedirs(dest, exist_ok=True)

for (lbl, (img_path, idx, pil)) in zip(labels, meta):
    fname = f"{os.path.basename(img_path).rsplit('.',1)[0]}_patch_{idx}.png"
    pil.save(os.path.join(OUTPUT_ROOT, f"cluster_{lbl}", fname))

# --------------- optional: save images by presence ---------------
print("Tagging parent images with detected clusters...")
for img_path in sorted(set(m[0] for m in meta)):
    # find all patches for this image
    idxs = [i for i, m in enumerate(meta) if m[0] == img_path]
    img_labels = set(int(labels[i]) for i in idxs)
    for k in img_labels:
        dest_dir = os.path.join(OUTPUT_ROOT, f"type_{k}")
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy(img_path, os.path.join(dest_dir, os.path.basename(img_path)))

print("Done. Check outputs/v6/results/patch_clusters and type_* folders.")
