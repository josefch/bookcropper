#!/usr/bin/env python3
"""Cut out and align each detected book: warp the (possibly tilted) detection
rect to an upright rectangle and save it.

Usage:
    python cutout_all.py <source_folder> [output_folder]

    source_folder   folder with scanner images (jpg/jpeg/png/tif)
    output_folder   default: <source_folder>/cutouts

Optionally writes a side-by-side preview per image to <output>/previews/
(disable with --no-previews).
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from unified_detect import detect

EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def order_corners(box: np.ndarray) -> np.ndarray:
    """Return corners as [top-left, top-right, bottom-right, bottom-left]."""
    pts = box.astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def cutout(img: np.ndarray, box: np.ndarray) -> np.ndarray:
    src = order_corners(box)
    w = int(round(max(np.linalg.norm(src[1] - src[0]), np.linalg.norm(src[2] - src[3]))))
    h = int(round(max(np.linalg.norm(src[3] - src[0]), np.linalg.norm(src[2] - src[1]))))
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_CUBIC)


def make_preview(img: np.ndarray, box: np.ndarray, note: str) -> np.ndarray:
    h, w = img.shape[:2]
    vis = img.copy()
    cv2.polylines(vis, [box.astype(np.int32)], True, (0, 0, 255), max(4, h // 400))
    th = 1200
    if vis.shape[0] > th:
        s = th / vis.shape[0]
        vis = cv2.resize(vis, None, fx=s, fy=s)
    cv2.putText(vis, note[:60], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return vis


def main():
    ap = argparse.ArgumentParser(description="Detect, cut out and align book covers/spines.")
    ap.add_argument("source", type=Path, help="folder with scanner images")
    ap.add_argument("output", type=Path, nargs="?", default=None,
                    help="output folder (default: <source>/cutouts)")
    ap.add_argument("--no-previews", action="store_true", help="skip preview overlays")
    args = ap.parse_args()

    src_dir = args.source.expanduser()
    if not src_dir.is_dir():
        sys.exit(f"not a folder: {src_dir}")
    out_dir = (args.output or src_dir / "cutouts").expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    prev_dir = out_dir / "previews"
    if not args.no_previews:
        prev_dir.mkdir(exist_ok=True)

    files = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in EXTS)
    if not files:
        sys.exit(f"no images found in {src_dir}")

    ok = 0
    for f in files:
        name = f.stem
        try:
            img, box, note = detect(f)
        except Exception as e:
            print(f"FAIL {name}: {e}")
            continue
        if box is None:
            print(f"{name}: NO BOX ({note})")
            continue
        out = cutout(img, box)
        cv2.imwrite(str(out_dir / f"{name}.jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not args.no_previews:
            cv2.imwrite(str(prev_dir / f"{name}_preview.jpg"),
                        make_preview(img, box, note), [cv2.IMWRITE_JPEG_QUALITY, 85])
        ok += 1
        print(f"{name}: {out.shape[1]}x{out.shape[0]}  [{note[:50]}]")
    print(f"DONE — {ok}/{len(files)} cut out to {out_dir}")


if __name__ == "__main__":
    main()
