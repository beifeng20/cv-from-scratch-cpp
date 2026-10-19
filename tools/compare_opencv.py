"""Compare the from-scratch Canny edges against OpenCV's Canny on the same input and parameters.

cvfs_demo masks out the top of the image by default, so run it with --roi 0 for this comparison.

Usage:
    cvfs_demo data/road.ppm out --roi 0
    python tools/compare_opencv.py data/road.ppm out/4_edges.pgm

Reports precision / recall / F1 of our edge pixels against OpenCV's, allowing 1 pixel of
position tolerance (two correct implementations can differ by a pixel at ties).
"""
import sys
import time

import cv2
import numpy as np


def read_pnm(path: str) -> np.ndarray:
    raw = open(path, "rb").read()
    magic, dims, _maxval, pixels = raw.split(b"\n", 3)
    w, h = map(int, dims.split())
    data = np.frombuffer(pixels, np.uint8)
    return data.reshape(h, w, 3) if magic == b"P6" else data.reshape(h, w)


def main(img_path: str, ours_path: str, sigma: float = 1.4, low: float = 50, high: float = 150) -> None:
    rgb = read_pnm(img_path)
    gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)
    radius = int(np.ceil(3 * sigma))
    blur = cv2.GaussianBlur(gray, (2 * radius + 1, 2 * radius + 1), sigma, borderType=cv2.BORDER_REPLICATE)

    t0 = time.perf_counter()
    # OpenCV needs 8-bit input; L2gradient=True uses sqrt(gx^2+gy^2) like our implementation
    ref = cv2.Canny(np.clip(blur.round(), 0, 255).astype(np.uint8), low, high, apertureSize=3, L2gradient=True)
    ms = (time.perf_counter() - t0) * 1000

    ours = read_pnm(ours_path) > 0
    ref = ref > 0
    kernel = np.ones((3, 3), np.uint8)
    ref_near = cv2.dilate(ref.astype(np.uint8), kernel) > 0
    ours_near = cv2.dilate(ours.astype(np.uint8), kernel) > 0
    precision = (ours & ref_near).sum() / max(ours.sum(), 1)
    recall = (ref & ours_near).sum() / max(ref.sum(), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    print(f"edge pixels: ours {ours.sum()}, OpenCV {ref.sum()}")
    print(f"precision {precision:.3f}  recall {recall:.3f}  F1 {f1:.3f}  (1-pixel tolerance)")
    print(f"OpenCV Canny alone: {ms:.2f} ms")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
