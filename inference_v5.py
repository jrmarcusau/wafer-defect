#!/usr/bin/env python3
"""
inference_v5.py — Load trained DEC model and assign wafer images to cluster folders

Usage:
    python inference_v5.py 

This script loads the checkpoint at epoch 100, runs inference on all masks in 
`data/Op3176_WaferMap2`, and copies each image into 
`outputs/v5/results/type_<cluster_id>/`.
"""

import os
import shutil
import torch
from glob import glob
from PIL import Image
import torchvision.transforms as T

# --------------------------------------------------------------------
# Import model definition and DEC helper (assumes main_v5.py is in PYTHONPATH)
# --------------------------------------------------------------------
from main_v5 import ConvAutoencoder, soft_assign, Z_DIM, IMG_SIZE, DEVICE

# --------------------------------------------------------------------
# Paths and parameters
# --------------------------------------------------------------------
CHECKPOINT_PATH = "outputs/v7/checkpoints/ckpt_100.pth"
INPUT_DIR       = "data/Op3176_DefectMap"
OUTPUT_ROOT     = "outputs/v7/results"
BATCH_SIZE      = 1      # process one image at a time
NUM_CLUSTERS     = 10
ALPHA           = 1.0    # same alpha used in training

# --------------------------------------------------------------------
# Utility
# --------------------------------------------------------------------

def makedirs(path):
    os.makedirs(path, exist_ok=True)

# --------------------------------------------------------------------
# Load model and checkpoint
# --------------------------------------------------------------------
print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model = ConvAutoencoder(z_dim=Z_DIM).to(DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()
cluster_centers = torch.tensor(ckpt["cluster_centers"], device=DEVICE, dtype=torch.float)

# --------------------------------------------------------------------
# Prepare transform and dataset list
# --------------------------------------------------------------------
transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),    # 1×H×W in [0,1]
])
paths = sorted(glob(os.path.join(INPUT_DIR, "*.PNG")) + glob(os.path.join(INPUT_DIR, "*.png")))
print(f"Found {len(paths)} images for inference")

# --------------------------------------------------------------------
# Inference loop
# --------------------------------------------------------------------
for img_path in paths:
    # load & preprocess
    img = Image.open(img_path).convert("L")
    x = transform(img).unsqueeze(0).to(DEVICE)  # 1×1×H×W

    # embed
    with torch.no_grad():
        z, _ = model(x)
        # compute soft assignments, then pick highest-prob cluster
        q = soft_assign(z, cluster_centers, alpha=ALPHA)
        cluster_id = int(q.argmax(dim=1).item())

    # prepare output
    dest_dir = os.path.join(OUTPUT_ROOT, f"type_{cluster_id}")
    makedirs(dest_dir)
    # copy original image
    fname = os.path.basename(img_path)
    dest_path = os.path.join(dest_dir, fname)
    shutil.copy(img_path, dest_path)

print("Inference complete. Images saved under outputs/v5/results/type_<cluster_id>/")
