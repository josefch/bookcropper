#!/usr/bin/env python3
"""Run unified_detect on a source folder and save preview overlays."""
import argparse
from pathlib import Path
import cv2
import numpy as np

from unified_detect import detect

ap = argparse.ArgumentParser()
ap.add_argument("source", type=Path)
ap.add_argument("output", type=Path, nargs="?", default=None)
args = ap.parse_args()
ARCHIVE = args.source.expanduser()
OUT_DIR = (args.output or ARCHIVE / "previews_unified").expanduser()
OUT_DIR.mkdir(parents=True, exist_ok=True)

files = sorted(p for p in ARCHIVE.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
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

    h, w = img.shape[:2]
    preview = img.copy()
    cv2.polylines(preview, [box.astype(np.int32)], True, (0, 0, 255), max(4, h // 400))
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [box.astype(np.int32)], 255)
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    side = np.hstack([preview, mask_rgb])
    th = 1200
    if side.shape[0] > th:
        s = th / side.shape[0]
        side = cv2.resize(side, None, fx=s, fy=s)
    short_note = note[:55]
    cv2.putText(side, short_note, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    out_path = OUT_DIR / f"{name}_preview.jpg"
    cv2.imwrite(str(out_path), side, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"{name}: {short_note}")

print("DONE")
print(f"Saved {len(list(OUT_DIR.iterdir()))} previews")
