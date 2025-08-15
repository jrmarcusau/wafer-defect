#!/usr/bin/env python3
"""
edge_experiment4.py

Goal:
- Stay in CARTESIAN space (no polar).
- Compute edges at 512x512 to preserve thin lines.
- Fuse Canny + directional morphological edges (+ optional LSD).
- Max-pool edges down to 256x256.
- Save visuals AND a tensor that mirrors your AE input channels in cartesian:
    [ gray_256, edge_256, xcoord, ycoord ]  -> shape (4, 256, 256), float32

Usage:
  python edge_experiment4.py path/to/image.png \
    --out experiment/edge4 --base_size 256 --edge_size 512 \
    --canny_pct 90 --canny_ratio 3.0 --morph_len 9 --use_lsd 1 --save_pt 1
"""

import os
import cv2
import argparse
import numpy as np

# ----------------------------- utils -----------------------------

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def save_gray(path, img):
    img8 = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite(path, img8)

def to_color(gray):
    if len(gray.shape) == 2:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return gray

def overlay_edges(base_gray, edge_mask, color=(0,0,255), alpha=0.85):
    base = to_color(base_gray)
    color_img = np.zeros_like(base); color_img[:] = color
    mask = (edge_mask > 0).astype(np.uint8) * 255
    blend_full = cv2.addWeighted(base, 1.0 - alpha, color_img, alpha, 0.0)
    return np.where(mask[..., None] > 0, blend_full, base)

def downsample_max(bin_u8, src_size, dst_size):
    """Max-pool downsample: keep a low-res pixel ON if any hi-res pixel in its block is ON."""
    f = src_size // dst_size
    assert src_size % dst_size == 0, "edge_size must be a multiple of base_size"
    e = (bin_u8 > 0).astype(np.uint8)
    e = e[:dst_size*f, :dst_size*f]
    e = e.reshape(dst_size, f, dst_size, f).max(axis=(1,3))
    return (e * 255).astype(np.uint8)

def coord_channels(size):
    xs = np.linspace(-1, 1, size, dtype=np.float32)
    ys = np.linspace(-1, 1, size, dtype=np.float32)
    x = np.tile(xs[None, :], (size, 1))
    y = np.tile(ys[:, None], (1, size))
    # visuals for sanity
    xvis = ((x + 1) * 127.5).astype(np.uint8)
    yvis = ((y + 1) * 127.5).astype(np.uint8)
    return x, y, xvis, yvis

# ----------------------------- edges -----------------------------

def auto_canny(img_u8, pct=90, ratio=3.0, ksize=3):
    gx = cv2.Sobel(img_u8, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(img_u8, cv2.CV_32F, 0, 1, ksize=ksize)
    mag = cv2.magnitude(gx, gy)
    hi = float(np.percentile(mag, pct))
    lo = hi / max(1.0, ratio)
    hi = max(0, min(255, int(round(hi))))
    lo = max(0, min(255, int(round(lo))))
    return cv2.Canny(img_u8, lo, hi)

def directional_morph_edges(gray_u8, length=9):
    """
    Morphological 'gradient' along oriented line elements:
    returns max over {0°, 45°, 90°, 135°}.
    """
    L = max(3, int(length))  # odd preferred
    k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1))
    k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, L))
    # Diagonals via custom kernels
    k_d1 = np.eye(L, dtype=np.uint8)
    k_d2 = np.fliplr(k_d1)

    edges = []
    for k in (k_h, k_v, k_d1, k_d2):
        dil = cv2.dilate(gray_u8, k)
        ero = cv2.erode(gray_u8, k)
        edges.append(cv2.absdiff(dil, ero))
    return np.maximum.reduce(edges)

def lsd_mask(gray_u8, min_len_frac=0.06, thickness=1):
    """
    Optional: use OpenCV LSD (if available) to get long line segments.
    Returns a 0/255 mask with drawn segments.
    """
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

# ----------------------------- main -----------------------------

def main():
    ap = argparse.ArgumentParser("Cartesian 512-edge → 256-edge channel + tensor")
    ap.add_argument("input_image", help="Path to input image (grayscale or RGB)")
    ap.add_argument("--out", default="experiment/edge4", help="Output directory")
    ap.add_argument("--base_size", type=int, default=256, help="Target size for input/tensors (e.g., 256)")
    ap.add_argument("--edge_size", type=int, default=512, help="High-res size for edge detection (multiple of base)")
    ap.add_argument("--canny_pct", type=float, default=90.0, help="Percentile for auto Canny high threshold")
    ap.add_argument("--canny_ratio", type=float, default=3.0, help="High/Low ratio for Canny")
    ap.add_argument("--morph_len", type=int, default=9, help="Length of line SEs for directional morph edges")
    ap.add_argument("--blur", type=float, default=0.0, help="Optional Gaussian blur sigma BEFORE edges (0=off)")
    ap.add_argument("--use_lsd", type=int, default=1, help="Use LSD line mask if available (1=yes)")
    ap.add_argument("--save_pt", type=int, default=1, help="Save PyTorch .pt tensors if torch available")
    args = ap.parse_args()

    ensure_dir(args.out)

    # Load grayscale
    src = cv2.imread(args.input_image, cv2.IMREAD_GRAYSCALE)
    if src is None:
        raise FileNotFoundError(args.input_image)

    # Prepare 512 for edges, 256 for input view
    edge_size = int(args.edge_size)
    base_size = int(args.base_size)
    if edge_size % base_size != 0:
        raise ValueError("--edge_size must be a multiple of --base_size (e.g., 512 vs 256)")

    cart512 = cv2.resize(src, (edge_size, edge_size), interpolation=cv2.INTER_LINEAR)
    cart256 = cv2.resize(src, (base_size, base_size), interpolation=cv2.INTER_LINEAR)

    save_gray(os.path.join(args.out, f"cart_{edge_size}.png"), cart512)
    save_gray(os.path.join(args.out, f"cart_{base_size}.png"), cart256)

    work512 = cart512.copy()
    if args.blur > 0:
        k = max(3, int(2 * round(3 * args.blur) + 1))
        work512 = cv2.GaussianBlur(work512, (k, k), args.blur)

    # --- Edges at 512 ---
    canny512 = auto_canny(work512, pct=args.canny_pct, ratio=args.canny_ratio, ksize=3)
    morph512 = directional_morph_edges(work512, length=args.morph_len)
    if args.use_lsd:
        lsd512 = lsd_mask(work512, min_len_frac=0.06, thickness=1)
    else:
        lsd512 = np.zeros_like(work512, dtype=np.uint8)

    # Fuse (be permissive): take per-pixel max
    edge512 = np.maximum.reduce([canny512, morph512, lsd512])

    # Visuals on 512
    save_gray(os.path.join(args.out, "edge512_canny.png"), canny512)
    save_gray(os.path.join(args.out, "edge512_morph.png"), morph512)
    if args.use_lsd:
        save_gray(os.path.join(args.out, "edge512_lsd.png"), lsd512)
    save_gray(os.path.join(args.out, "edge512_fused.png"), edge512)
    save_gray(os.path.join(args.out, "overlay512_fused.png"), overlay_edges(cart512, edge512, (0,0,255), 0.85))

    # --- Down to 256 for the edge channel (max-pool preserves thin stuff) ---
    edge256 = downsample_max(edge512, edge_size, base_size)
    save_gray(os.path.join(args.out, "edge256_channel.png"), edge256)
    save_gray(os.path.join(args.out, "overlay256_channel.png"), overlay_edges(cart256, edge256, (0,0,255), 0.85))

    # --- Build the 4×256×256 input tensor in CARTESIAN space ---
    gray_ch = (cart256.astype(np.float32) / 255.0)[None, ...]    # (1,H,W)
    edge_ch = ((edge256 > 0).astype(np.float32))[None, ...]      # (1,H,W)
    xmap, ymap, xvis, yvis = coord_channels(base_size)           # (-1..1)
    x_ch = xmap[None, ...]
    y_ch = ymap[None, ...]

    tensor = np.concatenate([gray_ch, edge_ch, x_ch, y_ch], axis=0).astype(np.float32)  # (4,H,W)

    # Save tensor(s)
    np.save(os.path.join(args.out, "input_tensor_cartesian_4x256x256.npy"), tensor)
    save_gray(os.path.join(args.out, "coord_x_vis.png"), xvis)
    save_gray(os.path.join(args.out, "coord_y_vis.png"), yvis)

    # Also save edge-only tensor if you want a single channel
    np.save(os.path.join(args.out, "edge_channel_tensor_1x256x256.npy"), edge_ch)

    # Optional: save .pt if torch is present
    if args.save_pt:
        try:
            import torch
            torch.save(torch.from_numpy(tensor), os.path.join(args.out, "input_tensor_cartesian_4x256x256.pt"))
            torch.save(torch.from_numpy(edge_ch), os.path.join(args.out, "edge_channel_tensor_1x256x256.pt"))
        except Exception as e:
            print("[warn] torch save failed:", e)

    # tiny readme
    with open(os.path.join(args.out, "README.txt"), "w") as f:
        f.write(
            "Files:\n"
            f"- cart_{edge_size}.png : grayscale at {edge_size}\n"
            f"- cart_{base_size}.png : grayscale at {base_size}\n"
            "- edge512_canny/morph/(lsd).png : component edges at 512\n"
            "- edge512_fused.png, overlay512_fused.png : fused edges at 512\n"
            "- edge256_channel.png, overlay256_channel.png : DOWN-SAMPLED edge (what the model would see)\n"
            "- coord_x_vis.png, coord_y_vis.png : coord channels visualization\n"
            "- input_tensor_cartesian_4x256x256.npy/.pt : tensor [gray, edge, x, y]\n"
            "- edge_channel_tensor_1x256x256.npy/.pt : edge-only tensor\n"
        )

    print("Saved to:", os.path.abspath(args.out))

if __name__ == "__main__":
    main()
