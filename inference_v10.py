#!/usr/bin/env python3
"""
inference_v8_twostream.py — Inference + diagnostics for main_v8_twostream.py

Outputs (like v8 + extras):
- type_*/ : copies of original images grouped by predicted cluster
- summaries/summary_cluster_*.png : top-K confident samples per cluster
- diagnostics/:
    cluster_sizes.png
    confidence_hist.png
    tsne_feats.png           (needs sklearn)
    block_norms.png          (avg pre-normalization L2 of each feature block)
    (optional) overlays/     (cartesian edge overlays for top/bottom samples)

ASSUMPTIONS (must match training):
- IMG_SIZE = 256
- HOG: orientations=9, pixels_per_cell=(32,32), cells_per_block=(1,1)  -> 576 dims
- Feature concat order for DEC: [L2(z_polar), L2(hog_polar), L2(z_cart), L2(hog_cart), L2(line_stats)]
- Checkpoint saved keys: 'polar_state', 'cart_state', 'cluster_centers'
"""

import os
import argparse
import shutil
from glob import glob

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.utils as vutils

from skimage.feature import hog
from sklearn.manifold import TSNE

# ────────────────────────────────────────────────────
# Constants (must match main_v8_twostream)
# ────────────────────────────────────────────────────
IMG_SIZE     = 256
EDGE_SIZE    = 512
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED         = 42

# Edge params (match training defaults)
CANNY_LOW    = 20
CANNY_HIGH   = 60
MORPH_LEN    = 9
USE_LSD      = False

# HOG at 256 -> 8x8x9 = 576
HOG_ORI      = 9
HOG_PPC      = (32, 32)
HOG_CPB      = (1, 1)

# ────────────────────────────────────────────────────
# Helpers (must mirror training)
# ────────────────────────────────────────────────────
def l2norm(t, eps=1e-8):
    return t / (t.norm(dim=1, keepdim=True) + eps)

def directional_morph_edges(gray_u8, length=MORPH_LEN):
    L = max(3, int(length))
    k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1))
    k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, L))
    k_d1 = np.eye(L, dtype=np.uint8)
    k_d2 = np.fliplr(k_d1)
    edges = []
    for k in (k_h, k_v, k_d1, k_d2):
        dil = cv2.dilate(gray_u8, k)
        ero = cv2.erode(gray_u8, k)
        edges.append(cv2.absdiff(dil, ero))
    return np.maximum.reduce(edges)

def lsd_mask(gray_u8, min_len_frac=0.06, thickness=1):
    H, W = gray_u8.shape
    try:
        lsd = cv2.createLineSegmentDetector(_refine=cv2.LSD_REFINE_STD)
    except Exception:
        return np.zeros_like(gray_u8, dtype=np.uint8)
    lines, _, _, _ = lsd.detect(gray_u8)
    if lines is None:
        return np.zeros_like(gray_u8, dtype=np.uint8)
    min_len = (H + W) * 0.5 * float(min_len_frac)
    mask = np.zeros_like(gray_u8, dtype=np.uint8)
    for ln in lines:
        x1, y1, x2, y2 = ln[0]
        if np.hypot(x2 - x1, y2 - y1) >= min_len:
            cv2.line(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, thickness, cv2.LINE_AA)
    return mask

def downsample_max(bin_u8, src_size, dst_size):
    f = src_size // dst_size
    assert src_size % dst_size == 0
    e = (bin_u8 > 0).astype(np.uint8)
    e = e[:dst_size*f, :dst_size*f]
    e = e.reshape(dst_size, f, dst_size, f).max(axis=(1,3))
    return (e * 255).astype(np.uint8)

def build_cart_edge_256(src_gray_u8):
    cart512 = cv2.resize(src_gray_u8, (EDGE_SIZE, EDGE_SIZE), interpolation=cv2.INTER_LINEAR)
    canny = cv2.Canny(cart512, CANNY_LOW, CANNY_HIGH)
    morph = directional_morph_edges(cart512, length=MORPH_LEN)
    if USE_LSD:
        lines = lsd_mask(cart512, min_len_frac=0.06, thickness=1)
        fused512 = np.maximum.reduce([canny, morph, lines])
    else:
        fused512 = np.maximum(canny, morph)
    edge256 = downsample_max(fused512, EDGE_SIZE, IMG_SIZE)
    return edge256

def cart_hough_line_features(edge_u8, min_line_len=None, max_gap=10):
    H, W = edge_u8.shape
    if min_line_len is None:
        min_line_len = max(H, W) // 8
    lines = cv2.HoughLinesP(edge_u8, 1, np.pi/360, threshold=30,
                            minLineLength=int(min_line_len), maxLineGap=int(max_gap))
    if lines is None:
        return np.array([0, 0, 0, 0], dtype=np.float32)
    Ls, angs = [], []
    for x1, y1, x2, y2 in lines[:,0]:
        dx, dy = (x2 - x1), (y2 - y1)
        Ls.append(np.hypot(dx, dy))
        angs.append(np.arctan2(dy, dx))
    Ls = np.array(Ls, dtype=np.float32)
    angs = np.array(angs, dtype=np.float32)
    return np.array([
        float(len(Ls)),
        float(Ls.sum() / (H + W + 1e-6)),
        float(angs.mean()) if len(angs)>0 else 0.0,
        float(angs.std())  if len(angs)>1 else 0.0
    ], dtype=np.float32)

def soft_assign(z, centers, alpha=1.0):
    dist_sq = torch.cdist(z, centers) ** 2
    q = (1.0 + dist_sq/alpha).pow(-(alpha+1)/2)
    return q / q.sum(dim=1, keepdim=True)

def overlay_edges(base_gray, edge_mask, color=(0,0,255), alpha=0.85):
    base = base_gray
    if base.ndim == 2:
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    color_img = np.zeros_like(base); color_img[:] = color
    mask = (edge_mask > 0).astype(np.uint8) * 255
    blend_full = cv2.addWeighted(base, 1.0 - alpha, color_img, alpha, 0.0)
    return np.where(mask[..., None] > 0, blend_full, base)

# ────────────────────────────────────────────────────
# Dataset (two-stream, inference)
# ────────────────────────────────────────────────────
class WaferDatasetTwoStream(Dataset):
    def __init__(self, root):
        self.paths = sorted(glob(os.path.join(root, "*.PNG")) +
                            glob(os.path.join(root, "*.png")))
        if not self.paths:
            raise RuntimeError(f"No images found under {root}!")

        dummy = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        self.hog_dim = hog(dummy,
                           orientations=HOG_ORI,
                           pixels_per_cell=HOG_PPC,
                           cells_per_block=HOG_CPB,
                           feature_vector=True).shape[0]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img  = Image.open(path).convert("L")
        arr  = np.array(img)  # original grayscale

        # ----- Polar (256) -----
        polar = cv2.warpPolar(
            arr, (IMG_SIZE, IMG_SIZE),
            center=(arr.shape[1]//2, arr.shape[0]//2),
            maxRadius=arr.shape[0]//2,
            flags=cv2.WARP_POLAR_LINEAR
        )
        polar = cv2.rotate(polar, cv2.ROTATE_90_COUNTERCLOCKWISE).astype(np.uint8)
        edge_polar = cv2.Canny(polar, CANNY_LOW, CANNY_HIGH)

        hog_polar = hog(polar,
                        orientations=HOG_ORI,
                        pixels_per_cell=HOG_PPC,
                        cells_per_block=HOG_CPB,
                        feature_vector=True).astype(np.float32)

        polar_t = torch.from_numpy(polar/255.0).unsqueeze(0).float()
        edgep_t = torch.from_numpy(edge_polar/255.0).unsqueeze(0).float()

        xs = torch.linspace(-1,1,IMG_SIZE).view(1,1,-1).expand(1,IMG_SIZE,IMG_SIZE)
        ys = torch.linspace(-1,1,IMG_SIZE).view(1,-1,1).expand(1,IMG_SIZE,IMG_SIZE)
        coord = torch.cat([xs, ys], dim=0).float()

        x_polar = torch.cat([polar_t, edgep_t, coord], dim=0)  # 4×H×W

        # ----- Cartesian (256) -----
        cart = cv2.resize(arr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        edge_cart256 = build_cart_edge_256(arr)                # 512→256 fused

        hog_cart = hog(cart,
                       orientations=HOG_ORI,
                       pixels_per_cell=HOG_PPC,
                       cells_per_block=HOG_CPB,
                       feature_vector=True).astype(np.float32)

        line_stats = cart_hough_line_features(edge_cart256, min_line_len=IMG_SIZE//8, max_gap=10)

        cart_t  = torch.from_numpy(cart/255.0).unsqueeze(0).float()
        edgec_t = torch.from_numpy((edge_cart256>0).astype(np.float32)).unsqueeze(0)

        x_cart = torch.cat([cart_t, edgec_t, coord], dim=0)    # 4×H×W

        return x_polar, x_cart, torch.from_numpy(hog_polar), torch.from_numpy(hog_cart), \
               torch.from_numpy(line_stats), idx, path, cart, edge_cart256

# ────────────────────────────────────────────────────
# Encoders (match training arch; we forward full AE to get z)
# ────────────────────────────────────────────────────
class ConvAE(nn.Module):
    def __init__(self, z_dim, img_size=IMG_SIZE):
        super().__init__()
        self.img_size = img_size
        self.enc = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                               # 256→128
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                               # 128→64
            nn.Flatten(),
            nn.LazyLinear(z_dim)
        )
        fH = fW = img_size // 4
        self.dec = nn.Sequential(
            nn.Linear(z_dim, 64 * fH * fW),
            nn.Unflatten(1, (64, fH, fW)),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, 2, 3, padding=1), nn.Sigmoid()
        )
    def forward(self, x):
        z = self.enc(x)
        x_rec = self.dec(z)
        return z, x_rec

# ────────────────────────────────────────────────────
# Main inference
# ────────────────────────────────────────────────────
def main(args):
    torch.manual_seed(SEED); np.random.seed(SEED)

    # 1) Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=DEVICE)
    centers = ckpt.get("cluster_centers", None)
    if centers is None:
        raise RuntimeError("No cluster_centers in checkpoint!")
    centers = centers.to(DEVICE)
    K, D = centers.shape
    print(f"Centers: K={K}, feat_dim={D}")

    # 2) Dataset / loaders
    ds = WaferDatasetTwoStream(args.input_dir)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    hog_dim = ds.hog_dim                    # 576
    line_dim = 4
    # Recover z_dim from centers width: D = z_p(=Z) + hog_p(=576) + z_c(=Z) + hog_c(=576) + line(=4)
    z_dim = (D - (hog_dim*2 + line_dim)) // 2
    assert z_dim > 0 and (2*z_dim + 2*hog_dim + line_dim) == D, "Center feature width doesn't match expected blocks"
    print(f"Recovered z_dim per stream = {z_dim}")

    # 3) Build encoders and load weights
    ae_polar = ConvAE(z_dim=z_dim, img_size=IMG_SIZE).to(DEVICE)
    ae_cart  = ConvAE(z_dim=z_dim, img_size=IMG_SIZE).to(DEVICE)
    ae_polar.load_state_dict(ckpt["polar_state"], strict=True)
    ae_cart.load_state_dict(ckpt["cart_state"], strict=True)
    ae_polar.eval(); ae_cart.eval()

    # 4) Inference: extract features, soft-assign
    all_feats, all_paths, all_assigns, all_conf = [], [], [], []
    block_norms = []   # [||z_p||, ||hog_p||, ||z_c||, ||hog_c||, ||line||] per sample
    edge_cover = []    # proportion of edge pixels in cart edge (debug)
    with torch.no_grad():
        for x_pol, x_car, hog_pol, hog_car, line_vec, idxs, paths, cart256, edgec256 in loader:
            x_pol = x_pol.to(DEVICE); x_car = x_car.to(DEVICE)
            z_p, _ = ae_polar(x_pol)
            z_c, _ = ae_cart(x_car)

            hp = hog_pol.to(DEVICE).float()
            hc = hog_car.to(DEVICE).float()
            lf = line_vec.to(DEVICE).float()

            # record pre-normalization norms (diagnostics)
            bn = torch.stack([
                z_p.norm(dim=1),
                hp.norm(dim=1),
                z_c.norm(dim=1),
                hc.norm(dim=1),
                lf.norm(dim=1)
            ], dim=1).cpu().numpy()
            block_norms.append(bn)

            feats = torch.cat([
                l2norm(z_p), l2norm(hp), l2norm(z_c), l2norm(hc), l2norm(lf)
            ], dim=1)

            q = soft_assign(feats, centers)          # (B, K)
            a = torch.argmax(q, dim=1)
            c = torch.max(q, dim=1).values

            all_feats.append(feats.cpu())
            all_paths.extend(paths)
            all_assigns.append(a.cpu().numpy())
            all_conf.append(c.cpu().numpy())

            # edge coverage ratio
            for e in edgec256.numpy():
                edge_cover.append(float((e > 0).mean()))

    all_feats = torch.cat(all_feats, dim=0).numpy()
    assigns   = np.concatenate(all_assigns, axis=0)
    conf      = np.concatenate(all_conf, axis=0)
    block_norms = np.concatenate(block_norms, axis=0)  # N × 5
    N = len(all_paths)
    print(f"Inferred {N} samples.")

    # 5) Prepare output dirs
    os.makedirs(args.output_dir, exist_ok=True)
    for k in range(K):
        os.makedirs(os.path.join(args.output_dir, f"type_{k}"), exist_ok=True)
    summary_dir = os.path.join(args.output_dir, "summaries")
    diag_dir    = os.path.join(args.output_dir, "diagnostics")
    os.makedirs(summary_dir, exist_ok=True)
    os.makedirs(diag_dir, exist_ok=True)

    # 6) Copy images into buckets
    print("Copying images into type_*/ …")
    for path, cid in zip(all_paths, assigns):
        dst = os.path.join(args.output_dir, f"type_{cid}", os.path.basename(path))
        shutil.copy(path, dst)

    # 7) Summaries (top-K by confidence, like v8) + optional bottom-K overlays
    print(f"Making summary slides (top {args.top_k}) …")
    for k in range(K):
        idxs = np.where(assigns == k)[0]
        if len(idxs) == 0:
            # still make an empty grid
            grid = torch.zeros(3, IMG_SIZE, IMG_SIZE)
            vutils.save_image(grid, os.path.join(summary_dir, f"summary_cluster_{k}.png"))
            continue

        idxs_top = idxs[np.argsort(-conf[idxs])][: args.top_k]
        imgs = []
        for i in idxs_top:
            img = Image.open(all_paths[i]).convert("RGB")
            tf  = T.Resize((IMG_SIZE, IMG_SIZE))
            imgs.append(T.ToTensor()(tf(img)))
        while len(imgs) < args.top_k:
            imgs.append(torch.zeros(3, IMG_SIZE, IMG_SIZE))
        grid = vutils.make_grid(imgs, nrow=int(np.ceil(np.sqrt(args.top_k))), padding=2)
        vutils.save_image(grid, os.path.join(summary_dir, f"summary_cluster_{k}.png"))

    # 8) Diagnostics
    # 8a) Cluster sizes
    counts = np.bincount(assigns, minlength=K)
    plt.figure()
    plt.bar(range(K), counts)
    plt.xlabel("Cluster"); plt.ylabel("Count"); plt.title("Cluster sizes")
    plt.tight_layout(); plt.savefig(os.path.join(diag_dir, "cluster_sizes.png")); plt.close()

    # 8b) Confidence histogram
    plt.figure()
    plt.hist(conf, bins=30, range=(0,1))
    plt.xlabel("Max soft-assignment (confidence)"); plt.ylabel("Frequency")
    plt.title("Assignment confidence")
    plt.tight_layout(); plt.savefig(os.path.join(diag_dir, "confidence_hist.png")); plt.close()

    # 8c) t-SNE of features (2D)
    try:
        tsne = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=min(30, max(5, N//20)), random_state=SEED)
        emb2 = tsne.fit_transform(all_feats)
        plt.figure(figsize=(6,5))
        sc = plt.scatter(emb2[:,0], emb2[:,1], c=assigns, s=8, cmap="tab20")
        plt.colorbar(sc, label="cluster")
        plt.title("t-SNE of DEC features")
        plt.tight_layout(); plt.savefig(os.path.join(diag_dir, "tsne_feats.png")); plt.close()
    except Exception as e:
        print("[warn] t-SNE failed:", e)

    # 8d) Block norms (mean ± std)
    labels = ["||z_p||", "||hog_p||", "||z_c||", "||hog_c||", "||line||"]
    m = block_norms.mean(axis=0); s = block_norms.std(axis=0)
    plt.figure()
    plt.bar(range(len(labels)), m, yerr=s, capsize=3)
    plt.xticks(range(len(labels)), labels, rotation=0)
    plt.title("Pre-normalization block norms (mean ± std)")
    plt.tight_layout(); plt.savefig(os.path.join(diag_dir, "block_norms.png")); plt.close()

    # 8e) Edge coverage (what fraction of 256 pixels are 'edge')
    plt.figure()
    plt.hist(edge_cover, bins=30, range=(0,1))
    plt.xlabel("Cartesian edge coverage (fraction of pixels)"); plt.ylabel("Frequency")
    plt.title("Edge coverage distribution (cart 512→256)")
    plt.tight_layout(); plt.savefig(os.path.join(diag_dir, "edge_coverage_hist.png")); plt.close()

    # 8f) (Optional) save overlays for top/bottom confidence samples per cluster
    if args.save_overlays:
        ov_dir = os.path.join(diag_dir, "overlays"); os.makedirs(ov_dir, exist_ok=True)
        print("Saving overlays for extremes …")
        # rebuild a tiny loader that also gives cart/edge arrays (already provided by dataset)
        for k in range(K):
            idxs = np.where(assigns == k)[0]
            if len(idxs) == 0: continue
            idxs_top = idxs[np.argsort(-conf[idxs])][: min(3, len(idxs))]
            idxs_low = idxs[np.argsort(+conf[idxs])][: min(3, len(idxs))]
            for tag, sel in [("top", idxs_top), ("low", idxs_low)]:
                for i in sel:
                    cart = cv2.imread(all_paths[i], cv2.IMREAD_GRAYSCALE)
                    cart256 = cv2.resize(cart, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
                    edge256 = build_cart_edge_256(cart)
                    ov = overlay_edges(cart256, edge256, (0,0,255), 0.85)
                    outp = os.path.join(ov_dir, f"c{k}_{tag}_{os.path.basename(all_paths[i])}")
                    cv2.imwrite(outp, ov)

    print("Done.")
    print("Buckets:", args.output_dir)
    print("Summaries:", os.path.join(args.output_dir, "summaries"))
    print("Diagnostics:", os.path.join(args.output_dir, "diagnostics"))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir",  required=True, help="Path to wafer PNGs")
    p.add_argument("--checkpoint", required=True, help="main_v8_twostream.py checkpoint (ckpt_*.pth)")
    p.add_argument("--output_dir", required=True, help="Where to write type_*/, summaries/, diagnostics/")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--top_k", type=int, default=9)
    p.add_argument("--save_overlays", action="store_true", help="Also save cart edge overlays for extremes")
    args = p.parse_args()
    main(args)
