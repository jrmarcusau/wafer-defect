#!/usr/bin/env python3
"""
edge_experiment_v2_expand.py

Same as your v2, plus an optional "expand (dilate) k×k" step BEFORE resizing
to help 1px features survive downsampling.

Example:
  python edge_experiment_v2_expand.py data/Op3176_DefectMap/image.png \
      --output_dir experiment/edge2 --img_size 128 \
      --canny_low 20 --canny_high 60 \
      --expand_k 4 --expand_iter 1 --expand_thresh -1 --expand_dark 0 \
      --morph_ksize 3
"""

import os
import cv2
import argparse
import numpy as np


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_gray(path: str, img: np.ndarray):
    img8 = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite(path, img8)

def sobel_mag(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    mag = cv2.magnitude(gx, gy)
    return cv2.convertScaleAbs(mag)

def scharr_mag(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    return cv2.convertScaleAbs(mag)

def laplacian_edges(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
    return cv2.convertScaleAbs(lap)

def morph_gradient(bin_img: np.ndarray, ksize: int = 3) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    dil = cv2.dilate(bin_img, kernel)
    ero = cv2.erode(bin_img, kernel)
    return cv2.absdiff(dil, ero)

def draw_contours_from_binary(bin_img: np.ndarray, thickness: int = 1) -> np.ndarray:
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    edge = np.zeros_like(bin_img)
    cv2.drawContours(edge, contours, -1, color=255, thickness=thickness)
    return edge


def main():
    parser = argparse.ArgumentParser(description="Edge detection experiment on a wafer image (v2 + expand)")
    parser.add_argument("input_image", help="Path to input wafer image (grayscale or RGB)")
    parser.add_argument("--output_dir", default="experiment/edge3", help="Directory to save outputs")
    parser.add_argument("--img_size", type=int, default=128, help="Downsample target size (H=W=img_size)")
    parser.add_argument("--canny_low", type=int, default=50, help="Canny low threshold")
    parser.add_argument("--canny_high", type=int, default=150, help="Canny high threshold")
    parser.add_argument("--blur", type=float, default=0.0,
                        help="Optional Gaussian blur sigma before edges (0 = off).")
    parser.add_argument("--morph_ksize", type=int, default=3, help="Kernel size for morphological gradient")
    parser.add_argument("--binary_thresh", type=int, default=0,
                        help="If >0, threshold value (0-255) to make a binary mask for morph/contours.")

    # NEW: expand (dilate) options
    parser.add_argument("--expand_k", type=int, default=0,
                        help="If >0, expand (dilate) with k×k kernel BEFORE resizing (e.g., 3,4,5). 0 = off.")
    parser.add_argument("--expand_iter", type=int, default=1, help="Dilation iterations for expand step.")
    parser.add_argument("--expand_thresh", type=int, default=-1,
                        help="Threshold for expand mask. -1 = Otsu (auto). Otherwise use 0..255.")
    parser.add_argument("--expand_dark", type=int, default=0,
                        help="Set 1 if defects are darker than background (uses THRESH_BINARY_INV).")
    args = parser.parse_args()

    ensure_dir(args.output_dir)

    # 1) Load grayscale
    src = cv2.imread(args.input_image, cv2.IMREAD_GRAYSCALE)
    if src is None:
        raise FileNotFoundError(f"Could not load image: {args.input_image}")

    # 2) Optional EXPAND (dilate) BEFORE resizing (helps thin features survive)
    pre = src.copy()
    if args.expand_k > 0:
        if args.expand_thresh < 0:
            # Otsu auto threshold; invert if defects are darker
            tflag = cv2.THRESH_BINARY_INV if args.expand_dark else cv2.THRESH_BINARY
            _, mask = cv2.threshold(src, 0, 255, tflag + cv2.THRESH_OTSU)
        else:
            tflag = cv2.THRESH_BINARY_INV if args.expand_dark else cv2.THRESH_BINARY
            _, mask = cv2.threshold(src, args.expand_thresh, 255, tflag)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (args.expand_k, args.expand_k))
        dil = cv2.dilate(mask, kernel, iterations=args.expand_iter)

        # Keep original grayscale, just thicken the "on" regions (safer than replacing everything)
        pre = cv2.max(src, dil)

        save_gray(os.path.join(args.output_dir, f"expand_mask.png"), mask)
        save_gray(os.path.join(args.output_dir, f"expand_dilated.png"), dil)
        save_gray(os.path.join(args.output_dir, f"pre_after_expand.png"), pre)

    # 3) Optional pre-blur (after expand, before resize)
    if args.blur > 0:
        k = max(3, int(2 * round(3 * args.blur) + 1))  # odd kernel ≈ 6σ
        pre_blur = cv2.GaussianBlur(pre, (k, k), sigmaX=args.blur, sigmaY=args.blur)
    else:
        pre_blur = pre

    # 4) Downsample (bilinear + nearest) to visualize resampling effects
    size = (args.img_size, args.img_size)
    down_bilinear = cv2.resize(pre_blur, size, interpolation=cv2.INTER_LINEAR)
    down_nearest  = cv2.resize(pre_blur, size, interpolation=cv2.INTER_NEAREST)

    save_gray(os.path.join(args.output_dir, "downsized_bilinear.png"), down_bilinear)
    save_gray(os.path.join(args.output_dir, "downsized_nearest.png"),  down_nearest)

    # 5) Use bilinear version for edge comparisons (matches your v2)
    work = down_bilinear

    # 6) Edge methods
    canny = cv2.Canny(work, args.canny_low, args.canny_high)
    sobel = sobel_mag(work, ksize=3)
    lap   = laplacian_edges(work, ksize=3)
    scharr= scharr_mag(work)

    save_gray(os.path.join(args.output_dir, "canny.png"), canny)
    save_gray(os.path.join(args.output_dir, "sobel.png"), sobel)
    save_gray(os.path.join(args.output_dir, "laplacian.png"), lap)
    save_gray(os.path.join(args.output_dir, "scharr.png"), scharr)

    # 7) Optional binary-based methods (better for strict 0/1 masks)
    if args.binary_thresh > 0:
        _, bin_img = cv2.threshold(work, args.binary_thresh, 255, cv2.THRESH_BINARY)
        save_gray(os.path.join(args.output_dir, "binary.png"), bin_img)

        grad = morph_gradient(bin_img, ksize=args.morph_ksize)
        save_gray(os.path.join(args.output_dir, "morph_gradient.png"), grad)

        cont = draw_contours_from_binary(bin_img, thickness=1)
        save_gray(os.path.join(args.output_dir, "contours.png"), cont)

    print(f"Saved results in: {os.path.abspath(args.output_dir)}")
    print("Files:")
    for f in sorted(os.listdir(args.output_dir)):
        print(" -", f)


if __name__ == "__main__":
    main()
