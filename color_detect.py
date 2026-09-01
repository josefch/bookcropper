#!/usr/bin/env python3
"""Generalized color-sample book detector.

Strategy:
  1. Auto-detect background color from image corners.
  2. Auto-detect book color from image center (largest patch most distinct from bg).
  3. Build mask of pixels close to book color in LAB.
  4. Union with high-saturation regions (red straps, etc.) and very-bright regions
     (white photos / labels) — these are also book content even if their color differs.
  5. Morphology: open to kill bg noise, close to bridge book + sub-features, erode
     to pull boundary back to true book edge.
  6. Largest connected component → minAreaRect.

Spine fallback: if detected rect is too thin or too small, defer to ml_extend.
"""
import sys
from pathlib import Path

import cv2
import numpy as np


def bg_lab(img_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    corners = np.concatenate([
        lab[:60, :60].reshape(-1, 3),
        lab[:60, -60:].reshape(-1, 3),
        lab[-60:, :60].reshape(-1, 3),
        lab[-60:, -60:].reshape(-1, 3),
    ])
    return np.median(corners, axis=0)


def sample_book_color(img_bgr: np.ndarray, bg: np.ndarray, n_clusters: int = 4) -> np.ndarray:
    """Find the most book-like color in the image center via k-means on LAB."""
    h, w = img_bgr.shape[:2]
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    # central 40% area — likely book
    y0, y1 = int(h * 0.30), int(h * 0.70)
    x0, x1 = int(w * 0.30), int(w * 0.70)
    sample = lab[y0:y1, x0:x1].reshape(-1, 3)
    # k-means clusters
    if len(sample) > 5000:
        idx = np.random.RandomState(0).choice(len(sample), 5000, replace=False)
        sample = sample[idx]
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, centers = cv2.kmeans(sample, n_clusters, None, crit, 5, cv2.KMEANS_PP_CENTERS)
    centers = centers.astype(np.float32)
    counts = np.bincount(labels.ravel(), minlength=n_clusters)

    # pick cluster that is (a) far from bg and (b) reasonably populated
    bg_dists = np.linalg.norm(centers - bg, axis=1)
    # weight: prefer large clusters far from bg
    # exclude clusters too close to bg (< 8) — those are bg leak
    scores = []
    for i in range(n_clusters):
        if bg_dists[i] < 8.0:
            scores.append(-1.0)
            continue
        scores.append(counts[i] * (bg_dists[i] / 30.0))
    best = int(np.argmax(scores))
    return centers[best]


def color_dist_mask(img_bgr: np.ndarray, color_lab: np.ndarray, thresh: float = 12.0) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    diff = lab - color_lab
    dist = np.sqrt((diff * diff).sum(axis=2))
    return (dist < thresh).astype(np.uint8) * 255


def saturated_or_bright_mask(img_bgr: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """Pixels with strong color saturation or extreme brightness — usually part
    of the book (red straps, white photos, accent prints)."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    # saturated red/blue/green: a or b channel far from neutral (128)
    a_far = np.abs(lab[:, :, 1] - 128) > 16
    b_far = np.abs(lab[:, :, 2] - 128) > 16
    # very bright (white photo) or very dark (black accent)
    L_high = lab[:, :, 0] > 200
    bg_L = float(bg[0])
    L_low_far_from_bg = (lab[:, :, 0] < 30) & (np.abs(lab[:, :, 0] - bg_L) > 25)
    combined = a_far | b_far | L_high | L_low_far_from_bg
    return (combined.astype(np.uint8)) * 255


def detect(img_path: Path):
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    bg = bg_lab(img)

    # quick already-cropped check: corners disagree wildly
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    corner_meds = [
        np.median(lab[:60, :60].reshape(-1, 3), axis=0),
        np.median(lab[:60, -60:].reshape(-1, 3), axis=0),
        np.median(lab[-60:, :60].reshape(-1, 3), axis=0),
        np.median(lab[-60:, -60:].reshape(-1, 3), axis=0),
    ]
    max_d = max(
        float(np.linalg.norm(corner_meds[i] - corner_meds[j]))
        for i in range(4) for j in range(i + 1, 4)
    )
    if max_d > 25.0:
        # already cropped
        return img, np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.int32), "cropped"

    image_area = h * w
    book_color = sample_book_color(img, bg)

    # build masks
    book_mask = color_dist_mask(img, book_color, thresh=14.0)
    sub_mask = saturated_or_bright_mask(img, bg)
    full_mask = cv2.bitwise_or(book_mask, sub_mask)

    # morphology — kernels scale gently with image size (clamp to reasonable max)
    md = min(h, w)
    k_open = min(max(7, md // 200), 15)
    if k_open % 2 == 0:
        k_open += 1
    k_close = min(max(15, md // 100), 25)
    if k_close % 2 == 0:
        k_close += 1
    k_erode = min(max(11, md // 120), 23)
    if k_erode % 2 == 0:
        k_erode += 1

    # Only OPEN to kill bg noise speckles. Do NOT close globally (would bridge
    # bg noise across the image). Use connected-component aggregation instead.
    full_mask = cv2.morphologyEx(
        full_mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open)),
    )

    # find all connected components, take those with area > 0.5% of image
    # AND inside or near the largest one — this aggregates book + strap + photo
    # without bridging across bg.
    nc, labels, stats, _ = cv2.connectedComponentsWithStats(full_mask, connectivity=8)
    if nc <= 1:
        return img, None, "no-blob"
    areas = stats[1:, cv2.CC_STAT_AREA]
    big_idx = 1 + int(np.argmax(areas))
    big_area = stats[big_idx, cv2.CC_STAT_AREA]
    bx, by, bw, bh = stats[big_idx, :4]
    big_cx = bx + bw / 2
    big_cy = by + bh / 2
    big_radius = max(bw, bh) * 0.6

    # accept components within `big_radius` of the largest center AND with area
    # at least 0.5% of image
    keep_mask = np.zeros_like(full_mask)
    for i in range(1, nc):
        x, y, cw, ch, area = stats[i]
        if area < 0.005 * image_area:
            continue
        cx = x + cw / 2
        cy = y + ch / 2
        d = np.sqrt((cx - big_cx) ** 2 + (cy - big_cy) ** 2)
        if d <= big_radius:
            keep_mask[labels == i] = 255

    # close gently to bridge keep components into one
    full_mask = cv2.morphologyEx(
        keep_mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close)),
        iterations=1,
    )
    # erode to undo close's outward expansion
    full_mask = cv2.erode(
        full_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_erode, k_erode)),
        iterations=1,
    )

    cnts, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return img, None, "no-contour"
    filled = np.zeros_like(full_mask)
    cv2.drawContours(filled, cnts, -1, 255, thickness=cv2.FILLED)
    big = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(big)
    box = cv2.boxPoints(rect)

    # sanity check
    area = cv2.contourArea(box)
    rect_pct = area / image_area
    sides = [np.linalg.norm(box[i] - box[(i + 1) % 4]) for i in range(4)]
    aspect = max(sides) / max(min(sides), 1)
    note = f"book_L={book_color[0]:.0f} rect_pct={rect_pct:.3f} aspect={aspect:.1f}"

    # too small or too thin: defer to ml_extend
    if rect_pct < 0.05 or (rect_pct < 0.10 and aspect > 4):
        from ml_extend import detect as ml_detect
        img2, ml_box = ml_detect(img_path)
        if ml_box is not None:
            return img, ml_box, f"deferred to ml_extend ({note})"

    return img, box.astype(np.int32), note


def main():
    img_path = Path(sys.argv[1])
    name = img_path.stem
    img, box, note = detect(img_path)
    print(f"{name}: {note}")
    if box is None:
        return
    PROJECT = Path("/Users/murat/git/private/bookcoverfixer")
    v26_path = PROJECT / "previews_v26" / f"{name}_preview.jpg"
    OUT = PROJECT / "overlay"
    OUT.mkdir(exist_ok=True)
    out_path = OUT / f"{name}_color.jpg"
    if v26_path.exists():
        v26 = cv2.imread(str(v26_path))
        pw = v26.shape[1] // 2
        v26_left = v26[:, :pw]
        oh, ow = img.shape[:2]
        ph = v26_left.shape[0]
        sx = pw / ow
        sy = ph / oh
        scaled = box.astype(np.float32).copy()
        scaled[:, 0] *= sx
        scaled[:, 1] *= sy
        out = v26_left.copy()
        cv2.polylines(out, [scaled.astype(np.int32)], True, (255, 200, 0), 2)
        cv2.putText(out, "color_detect (CYAN)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imwrite(str(out_path), out, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"-> {out_path}")


if __name__ == "__main__":
    main()
