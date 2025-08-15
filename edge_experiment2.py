#!/usr/bin/env python3
"""
edge_experiment.py

Quick visual experiment for wafer images:
- Downsample to a target size (both bilinear and nearest-neighbor).
- Compute several edge maps (Canny, Sobel magnitude, Laplacian, Scharr).
- Optional: morphological gradient and contour outline from the (thresholded) image.

Outputs are saved as PNGs in the chosen output directory.

Example:
  python edge_experiment.py data/Op3176_DefectMap/image.png \
      --output_dir experiment/edge --img_size 128 \
      --canny_low 50 --canny_high 150 --morph_ksize 3
"""

import os
import cv2
import argparse
import numpy as np


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_gray(path: str, img: np.ndarray):
    # Clamp to 0..255 and save as 8-bit
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
    # Binary morphological gradient = dilation - erosion
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    dil = cv2.dilate(bin_img, kernel)
    ero = cv2.erode(bin_img, kernel)
    grad = cv2.absdiff(dil, ero)
    return grad


def draw_contours_from_binary(bin_img: np.ndarray, thickness: int = 1) -> np.ndarray:
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    edge = np.zeros_like(bin_img)
    cv2.drawContours(edge, contours, -1, color=255, thickness=thickness)
    return edge


def main():
    parser = argparse.ArgumentParser(description="Edge detection experiment on a wafer image")
    parser.add_argument("input_image", help="Path to input wafer image (grayscale or RGB)")
    parser.add_argument("--output_dir", default="experiment/edge2", help="Directory to save outputs")
    parser.add_argument("--img_size", type=int, default=128, help="Downsample target size (H=W=img_size)")
    parser.add_argument("--canny_low", type=int, default=20, help="Canny low threshold")
    parser.add_argument("--canny_high", type=int, default=60, help="Canny high threshold")
    parser.add_argument("--blur", type=float, default=0.0,
                        help="Optional Gaussian blur sigma before edges (0 = off).")
    parser.add_argument("--morph_ksize", type=int, default=3, help="Kernel size for morphological gradient")
    parser.add_argument("--binary_thresh", type=int, default=0,
                        help="If >0, threshold value (0-255) to make a binary mask for morph/contours.")
    args = parser.parse_args()

    ensure_dir(args.output_dir)

    # Load image as grayscale
    src = cv2.imread(args.input_image, cv2.IMREAD_GRAYSCALE)
    if src is None:
        raise FileNotFoundError(f"Could not load image: {args.input_image}")

    # Optional pre-blur to tame salt-and-pepper 0/1 maps before edge detection
    if args.blur > 0:
        k = max(3, int(2 * round(3 * args.blur) + 1))  # odd kernel approx 6σ
        src_blur = cv2.GaussianBlur(src, (k, k), sigmaX=args.blur, sigmaY=args.blur)
    else:
        src_blur = src

    # Downsample (bilinear + nearest) to visualize resampling effects
    size = (args.img_size, args.img_size)
    down_bilinear = cv2.resize(src_blur, size, interpolation=cv2.INTER_LINEAR)
    down_nearest  = cv2.resize(src_blur, size, interpolation=cv2.INTER_NEAREST)

    save_gray(os.path.join(args.output_dir, "downsized_bilinear.png"), down_bilinear)
    save_gray(os.path.join(args.output_dir, "downsized_nearest.png"),  down_nearest)

    # Use the bilinear-downsampled image for the edge comparisons by default
    work = down_bilinear

    # Edge methods
    canny = cv2.Canny(work, args.canny_low, args.canny_high)
    sobel = sobel_mag(work, ksize=3)
    lap = laplacian_edges(work, ksize=3)
    scharr = scharr_mag(work)

    save_gray(os.path.join(args.output_dir, "canny.png"), canny)
    save_gray(os.path.join(args.output_dir, "sobel.png"), sobel)
    save_gray(os.path.join(args.output_dir, "laplacian.png"), lap)
    save_gray(os.path.join(args.output_dir, "scharr.png"), scharr)

    # Optional binary-based methods (better for strict 0/1 masks)
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
