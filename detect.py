#!/usr/bin/env python3
"""Detect book cover/spine rectangle in scan. Outputs preview with red quad."""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


# ---------- background ----------

def _corner_medians(img_lab: np.ndarray, patch: int = 60) -> np.ndarray:
    h, w = img_lab.shape[:2]
    p = patch
    corners = [
        img_lab[0:p, 0:p],
        img_lab[0:p, w - p : w],
        img_lab[h - p : h, 0:p],
        img_lab[h - p : h, w - p : w],
    ]
    return np.array([np.median(c.reshape(-1, 3), axis=0) for c in corners])


def is_already_cropped(img_lab: np.ndarray) -> bool:
    medians = _corner_medians(img_lab)
    max_d = max(
        float(np.linalg.norm(medians[i] - medians[j]))
        for i in range(4)
        for j in range(i + 1, 4)
    )
    if max_d > 25.0:
        return True
    bg_L = float(np.median(medians[:, 0]))
    img_L = float(np.median(img_lab[:, :, 0]))
    if bg_L > 150 and abs(bg_L - img_L) < 20:
        return True
    return False


def sample_bg_lab(img_lab: np.ndarray) -> np.ndarray:
    return np.median(_corner_medians(img_lab), axis=0)


# ---------- masks ----------

def color_distance_mask(img_lab: np.ndarray, bg: np.ndarray, thresh: float) -> np.ndarray:
    diff = img_lab.astype(np.float32) - bg.astype(np.float32)
    dist = np.sqrt((diff * diff).sum(axis=2))
    return (dist > thresh).astype(np.uint8) * 255


def edge_floodfill_mask(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
    eq = clahe.apply(gray)
    blur = cv2.GaussianBlur(eq, (5, 5), 0)
    med = float(np.median(blur))
    lo = max(10, int(0.66 * med))
    hi = max(30, int(1.33 * med))
    raw_edges = cv2.Canny(blur, lo, hi)
    h, w = raw_edges.shape
    image_area = h * w

    best = None
    for dilate_iter, erode_iter in [(2, 0), (2, 1), (1, 2), (0, 3)]:
        edges = raw_edges.copy()
        if dilate_iter:
            edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=dilate_iter)
        if erode_iter:
            edges = cv2.erode(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=erode_iter)
        border = max(10, min(h, w) // 100)
        edges[:border, :] = 0
        edges[-border:, :] = 0
        edges[:, :border] = 0
        edges[:, -border:] = 0
        ff = cv2.bitwise_not(edges)
        flood = ff.copy()
        mask = np.zeros((h + 2, w + 2), np.uint8)
        for sy, sx in [(2, 2), (2, w - 3), (h - 3, 2), (h - 3, w - 3)]:
            if flood[sy, sx] == 255:
                cv2.floodFill(flood, mask, (sx, sy), 128)
        interior = (flood == 255).astype(np.uint8) * 255
        best = interior
        if cv2.countNonZero(interior) < 0.6 * image_area:
            return interior
    return best


def clean_mask(mask: np.ndarray, image_min_dim: int) -> np.ndarray:
    k = max(3, image_min_dim // 200)
    if k % 2 == 0:
        k += 1
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, se, iterations=3)
    return mask


# ---------- blob ----------

def largest_interior_component(mask: np.ndarray, dilate_first: bool = True):
    """Dilate to merge attached protrusions, take largest non-edge component, fill holes."""
    h, w = mask.shape
    image_area = h * w
    work = mask
    if dilate_first:
        # bridge thin gaps so attached objects (straps, spiral binding) merge with cover
        k = max(3, min(h, w) // 200)
        if k % 2 == 0:
            k += 1
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        work = cv2.dilate(mask, se, iterations=1)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(work, connectivity=8)
    if n <= 1:
        return None
    best, best_area = None, 0
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        touches = x <= 2 or y <= 2 or x + cw >= w - 2 or y + ch >= h - 2
        if touches and area < 0.25 * image_area:
            continue
        if area > best_area:
            best_area = area
            best = i
    if best is None:
        idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        best = idx
        best_area = int(stats[idx, cv2.CC_STAT_AREA])
    blob = (labels == best).astype(np.uint8) * 255
    # restrict back to original mask area + nearby (intersect with dilated original)
    if dilate_first:
        blob = cv2.bitwise_and(blob, work)
    # fill holes
    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        filled = np.zeros_like(blob)
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
        blob = filled
        best_area = int(cv2.countNonZero(blob))
    return blob, best_area


def strip_thin_appendages(blob: np.ndarray, image_min_dim: int) -> np.ndarray:
    """Remove only thin appendages that protrude beyond the main body.
    Uses opening, but only if the open-then-close result is similar in area to
    the original (i.e., opening didn't eat into the body). Spines bypass."""
    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return blob
    rect = cv2.minAreaRect(max(contours, key=cv2.contourArea))
    (_, _), (rw, rh), _ = rect
    short_side = min(rw, rh)
    if short_side < image_min_dim / 25:  # spine - skip
        return blob
    k = max(5, image_min_dim // 130)
    if k % 2 == 0:
        k += 1
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    opened = cv2.morphologyEx(blob, cv2.MORPH_OPEN, se)
    # bring back the body shape (close after open kills small protrusions but preserves main rectangle)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, se, iterations=3)
    # re-fill via outer contour to ensure solid rectangle interior
    contours2, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours2:
        return blob
    out = np.zeros_like(closed)
    biggest = max(contours2, key=cv2.contourArea)
    cv2.drawContours(out, [biggest], -1, 255, thickness=cv2.FILLED)
    # safety: if we lost > 20% of area, revert to original
    if cv2.countNonZero(out) < 0.8 * cv2.countNonZero(blob):
        return blob
    return out


# ---------- density-band rectangle fitter ----------

def _principal_angle_deg(blob: np.ndarray) -> float:
    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    rect = cv2.minAreaRect(max(contours, key=cv2.contourArea))
    (_, _), (rw, rh), ang = rect
    if rw < rh:
        ang = ang + 90.0
    while ang > 45:
        ang -= 90
    while ang <= -45:
        ang += 90
    return float(ang)


def _density_rect_at_angle(blob: np.ndarray, angle: float, threshold: float = 0.5):
    """Rotate by angle, compute density bands, return (rect_corners, sharpness_score)."""
    h, w = blob.shape
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    cos_a, sin_a = abs(M[0, 0]), abs(M[0, 1])
    new_w = int(round(h * sin_a + w * cos_a))
    new_h = int(round(h * cos_a + w * sin_a))
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    rotated = cv2.warpAffine(blob, M, (new_w, new_h), flags=cv2.INTER_NEAREST, borderValue=0)

    col_sum = (rotated > 0).sum(axis=0).astype(np.float32)
    row_sum = (rotated > 0).sum(axis=1).astype(np.float32)
    if col_sum.max() < 2 or row_sum.max() < 2:
        return None, 0.0

    k = max(3, min(new_w, new_h) // 100)
    if k % 2 == 0:
        k += 1
    col_smooth = cv2.blur(col_sum.reshape(1, -1), (1, k)).ravel()
    row_smooth = cv2.blur(row_sum.reshape(-1, 1), (k, 1)).ravel()

    col_thr = threshold * float(col_smooth.max())
    row_thr = threshold * float(row_smooth.max())
    col_above = np.where(col_smooth >= col_thr)[0]
    row_above = np.where(row_smooth >= row_thr)[0]
    if len(col_above) < 2 or len(row_above) < 2:
        return None, 0.0

    # take longest contiguous above-threshold run, not just min/max
    # — this rejects outlier columns from protrusions far from the main body
    def longest_run(above):
        if len(above) == 0:
            return 0, 0
        gaps = np.where(np.diff(above) > 1)[0]
        starts = np.concatenate([[0], gaps + 1])
        ends = np.concatenate([gaps + 1, [len(above)]])
        runs = list(zip(starts, ends))
        if not runs:
            return int(above.min()), int(above.max())
        best_run = max(runs, key=lambda r: r[1] - r[0])
        s, e = best_run
        return int(above[s]), int(above[e - 1])

    x1, x2 = longest_run(col_above)
    y1, y2 = longest_run(row_above)

    # sharpness: avg gradient at the rect edges (steeper = better angle)
    edges_score = 0.0
    for idx in (x1, x2):
        if 1 < idx < len(col_smooth) - 1:
            edges_score += abs(col_smooth[idx + 1] - col_smooth[idx - 1])
    for idx in (y1, y2):
        if 1 < idx < len(row_smooth) - 1:
            edges_score += abs(row_smooth[idx + 1] - row_smooth[idx - 1])

    corners_rot = np.array(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32
    )
    M_inv = cv2.invertAffineTransform(M)
    corners = cv2.transform(corners_rot.reshape(-1, 1, 2), M_inv).reshape(-1, 2)
    return corners, float(edges_score)


def density_band_rect(blob: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Best-angle search around minAreaRect angle, then density-band fit.
    Wider search catches cases where minAreaRect angle is far off due to noisy mask."""
    if cv2.countNonZero(blob) < 50:
        return None
    base_angle = _principal_angle_deg(blob)
    best_box, best_score = None, -1.0
    # coarse pass (wider)
    for delta in (-8, -5, -3, -1.5, 0, 1.5, 3, 5, 8):
        box, score = _density_rect_at_angle(blob, base_angle + delta, threshold)
        if box is None:
            continue
        if score > best_score:
            best_score = score
            best_box = box
            best_delta = delta
    # fine pass around best
    fine_center = best_delta if best_box is not None else 0
    for delta in (fine_center - 1.0, fine_center - 0.5, fine_center + 0.5, fine_center + 1.0):
        box, score = _density_rect_at_angle(blob, base_angle + delta, threshold)
        if box is None:
            continue
        if score > best_score:
            best_score = score
            best_box = box
    return best_box


def pad_box(box: np.ndarray, pad: float) -> np.ndarray:
    """Expand a 4-corner rect outward by `pad` pixels."""
    c = box.mean(axis=0)
    out = box.copy().astype(np.float32)
    for i in range(4):
        v = out[i] - c
        n = np.linalg.norm(v)
        if n < 1e-6:
            continue
        out[i] = c + v * (n + pad) / n
    return out


# ---------- edge-snap ----------

def snap_rect_to_gradient(box4: np.ndarray, gray: np.ndarray, bg_gray: float, search: int = 50) -> np.ndarray:
    """Snap each rect edge to the physical book/bg boundary.

    For each edge: sample brightness along the outward normal at N points along
    the edge. Average those profiles for a stable 1D signal. Smooth. Find the
    position where smoothed brightness crosses the midpoint between bg and the
    book interior brightness — that is the physical boundary.
    """
    h, w = gray.shape
    pts = box4.astype(np.float32)
    center = pts.mean(axis=0)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    moved = pts.copy()

    for (a, b) in edges:
        ea, eb = moved[a], moved[b]
        edge_vec = eb - ea
        L = float(np.linalg.norm(edge_vec))
        if L < 1e-6:
            continue
        normal = np.array([-edge_vec[1], edge_vec[0]], dtype=np.float32) / L
        midpoint = (ea + eb) / 2
        if np.dot(normal, midpoint - center) < 0:
            normal = -normal

        N = max(11, int(L / 25))
        ts = np.linspace(0.1, 0.9, N)
        D = 2 * search + 1
        profile = np.zeros(D, dtype=np.float32)
        cnt = np.zeros(D, dtype=np.float32)
        for t in ts:
            p = ea + t * edge_vec
            for i, d in enumerate(range(-search, search + 1)):
                px = int(round(p[0] + normal[0] * d))
                py = int(round(p[1] + normal[1] * d))
                if 0 <= px < w and 0 <= py < h:
                    profile[i] += float(gray[py, px])
                    cnt[i] += 1.0
        if cnt.max() < 1:
            continue
        avg = np.where(cnt > 0, profile / np.maximum(cnt, 1), bg_gray)
        # wide median filter to flatten highlight/dark bands at the edge,
        # leaving only the overall book→bg transition.
        med_window = 7
        padded = np.pad(avg, med_window // 2, mode="edge")
        avg = np.array([
            float(np.median(padded[i : i + med_window]))
            for i in range(len(avg))
        ], dtype=np.float32)
        # estimate book interior brightness from the innermost few samples
        inner_n = min(8, search // 4)
        book_brightness = float(np.median(avg[:inner_n]))
        # if book is effectively same as bg, can't snap
        if abs(book_brightness - bg_gray) < 4:
            continue
        # is the pixel bg-like? tolerance scaled by bg/book gap
        bg_tol = max(2.0, abs(bg_gray - book_brightness) * 0.50)

        def is_bg(v):
            return abs(v - bg_gray) < bg_tol

        # walk from outside (d=+search) inward; find first non-bg pixel.
        # since we wide-median-filtered the profile, highlight/band spikes are
        # flattened and only the actual book→bg transition remains.
        center_i = search
        crossing = None
        for i in range(D - 1, -1, -1):
            if not is_bg(avg[i]):
                crossing = i + 1  # the bg pixel just outside the book
                break
        if crossing is None or crossing >= D:
            # fall back: walk outward from current edge to first bg
            for i in range(center_i, D):
                if is_bg(avg[i]):
                    crossing = i
                    break
        if crossing is None:
            continue
        shift = crossing - center_i
        moved[a] = ea + normal * shift
        moved[b] = eb + normal * shift
    return moved


def rect_coverage(blob: np.ndarray, box: np.ndarray) -> float:
    h, w = blob.shape
    rect_mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(rect_mask, [box.astype(np.int32)], 255)
    inter = cv2.bitwise_and(blob, rect_mask)
    rect_area = float(cv2.countNonZero(rect_mask))
    if rect_area == 0:
        return 0.0
    return cv2.countNonZero(inter) / rect_area


# ---------- hough spine ----------

def hough_spine_box(img_bgr: np.ndarray):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # try regular gray first, then aggressive CLAHE for nearly-invisible spines
    lines = None
    for enhance in (False, True):
        if enhance:
            clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8))
            g = clahe.apply(gray)
            edges = cv2.Canny(g, 20, 60)
        else:
            edges = cv2.Canny(gray, 30, 90)
        candidate = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, minLineLength=max(60, h // 20), maxLineGap=int(h * 0.05))
        if candidate is not None and len(candidate) >= 3:
            lines = candidate
            break
    if lines is None:
        return None

    def orient(line):
        x1, y1, x2, y2 = line
        dx, dy = x2 - x1, y2 - y1
        if abs(dx) < 1:
            return "v"
        slope = abs(dy / dx)
        if slope > 4:
            return "v"
        if slope < 0.25:
            return "h"
        return None

    verts = [l[0] for l in lines if orient(l[0]) == "v"]
    horzs = [l[0] for l in lines if orient(l[0]) == "h"]
    pool = verts if len(verts) > len(horzs) else horzs
    if len(pool) < 3:
        return None
    is_vert = pool is verts

    pts = []
    for x1, y1, x2, y2 in pool:
        pts.append((x1, y1))
        pts.append((x2, y2))
    pts = np.array(pts, dtype=np.float32)
    sec = pts[:, 0] if is_vert else pts[:, 1]
    med = float(np.median(sec))
    mad = float(np.median(np.abs(sec - med))) or 1.0
    pts = pts[np.abs(sec - med) < 3 * mad]
    if len(pts) < 6:
        return None

    if is_vert:
        cx = float(np.median(pts[:, 0]))
        y_min = float(pts[:, 1].min())
        y_max = float(pts[:, 1].max())
        # spine width: spread of detected line endpoints around center (robust)
        x_mad = float(np.median(np.abs(pts[:, 0] - cx)))
        spread_width = 2.5 * x_mad + 8
        measured_width = _measure_spine_width(gray, cx, y_min, y_max, vertical=True)
        width = max(spread_width, measured_width)
        width = max(10.0, min(50.0, width))
        y_min, y_max = _extend_spine_endpoints(gray, cx, y_min, y_max, width, vertical=True)
        return np.array(
            [[cx - width / 2, y_min], [cx + width / 2, y_min],
             [cx + width / 2, y_max], [cx - width / 2, y_max]],
            dtype=np.float32,
        )
    else:
        cy = float(np.median(pts[:, 1]))
        x_min = float(pts[:, 0].min())
        x_max = float(pts[:, 0].max())
        y_mad = float(np.median(np.abs(pts[:, 1] - cy)))
        spread_height = 2.5 * y_mad + 8
        measured_height = _measure_spine_width(gray, cy, x_min, x_max, vertical=False)
        height = max(spread_height, measured_height)
        height = max(10.0, min(50.0, height))
        x_min, x_max = _extend_spine_endpoints(gray, cy, x_min, x_max, height, vertical=False)
        return np.array(
            [[x_min, cy - height / 2], [x_max, cy - height / 2],
             [x_max, cy + height / 2], [x_min, cy + height / 2]],
            dtype=np.float32,
        )


def _extend_spine_endpoints(gray: np.ndarray, center: float, lo: float, hi: float, width: float, vertical: bool):
    """Walk past lo and hi along the spine axis to capture endcap pixels.
    Stop when a stretch of bg-matching pixels is encountered."""
    h, w = gray.shape
    bg_L = float(np.median(gray[:30, :30]))
    half_w = max(3.0, width / 2)
    c = int(round(center))

    def axis_strip(idx: int):
        if vertical:
            if not (0 <= idx < h):
                return None
            return gray[idx, max(0, c - int(half_w)) : min(w, c + int(half_w) + 1)]
        else:
            if not (0 <= idx < w):
                return None
            return gray[max(0, c - int(half_w)) : min(h, c + int(half_w) + 1), idx]

    def is_book_row(strip):
        if strip is None or strip.size == 0:
            return False
        # any pixel deviating from bg by > 12 → still book
        return float(np.max(np.abs(strip.astype(np.float32) - bg_L))) > 12

    new_lo, new_hi = lo, hi
    # tolerate longer gaps in low-contrast spines
    consecutive = 0
    for d in range(1, 350, 2):
        idx = int(lo) - d
        if is_book_row(axis_strip(idx)):
            new_lo = idx
            consecutive = 0
        else:
            consecutive += 2
            if consecutive > 80:
                break
    consecutive = 0
    for d in range(1, 350, 2):
        idx = int(hi) + d
        if is_book_row(axis_strip(idx)):
            new_hi = idx
            consecutive = 0
        else:
            consecutive += 2
            if consecutive > 80:
                break
    return new_lo, new_hi


def _measure_spine_width(gray: np.ndarray, center_coord: float, low: float, high: float, vertical: bool, fallback: float = 22.0) -> float:
    """Sample perpendicular gradient along the spine, return spine width via Sobel peaks.
    Falls back to `fallback` if measurement fails. Clamped to [10, 55]."""
    h, w = gray.shape
    samples = np.linspace(low + (high - low) * 0.1, low + (high - low) * 0.9, 9)
    widths = []
    for s in samples:
        s = int(round(s))
        cidx = int(round(center_coord))
        if vertical:
            if not (0 <= s < h):
                continue
            profile = gray[s, :].astype(np.float32)
        else:
            if not (0 <= s < w):
                continue
            profile = gray[:, s].astype(np.float32)
        if not (5 < cidx < len(profile) - 5):
            continue
        # gradient magnitude perpendicular to spine
        smooth = cv2.blur(profile.reshape(-1, 1), (5, 1)).ravel()
        grad = np.abs(np.diff(smooth))
        # search outward from center; book edge = local gradient peak
        win = min(40, cidx, len(grad) - cidx - 1)
        if win < 5:
            continue
        left_seg = grad[max(0, cidx - win):cidx]
        right_seg = grad[cidx:min(len(grad), cidx + win)]
        if len(left_seg) < 3 or len(right_seg) < 3:
            continue
        thresh = max(3.5, float(grad.std()) * 2.0)
        left_hits = np.where(left_seg > thresh)[0]
        right_hits = np.where(right_seg > thresh)[0]
        if len(left_hits) == 0 or len(right_hits) == 0:
            continue
        left_edge = cidx - win + int(left_hits[-1])
        right_edge = cidx + int(right_hits[0])
        if right_edge - left_edge > 3:
            widths.append(right_edge - left_edge)
    if not widths:
        return fallback
    width = float(np.median(widths))
    return max(8.0, min(35.0, width))


# ---------- detect ----------

def detect(img_path: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"cannot read {img_path}")
    h, w = img.shape[:2]
    scale = 1500 / max(h, w)
    small = cv2.resize(img, None, fx=scale, fy=scale) if scale < 1 else img.copy()
    sh, sw = small.shape[:2]
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    bg = sample_bg_lab(lab)

    if is_already_cropped(lab):
        box = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.int32)
        return img, box, np.full((sh, sw), 255, np.uint8), bg, "cropped"

    bg_L = float(bg[0])
    diff = lab.astype(np.float32) - bg.astype(np.float32)
    dist = np.sqrt((diff * diff).sum(axis=2))
    corner_dist = dist[:60, :60]
    bg_noise_95 = float(np.percentile(corner_dist, 95))
    # color threshold permissive enough to catch dark-on-dark cases
    thresh = max(bg_noise_95 - 1.0, 6.0)
    if bg_L > 30:
        thresh = max(thresh, 14.0)

    m_color = color_distance_mask(lab, bg, thresh)
    m_edge = edge_floodfill_mask(small)
    mask = cv2.bitwise_or(m_color, m_edge)
    mask = clean_mask(mask, min(sh, sw))

    blob_res = largest_interior_component(mask, dilate_first=True)
    image_area = sh * sw
    use_hough = blob_res is None or blob_res[1] > 0.85 * image_area
    if not use_hough:
        # if blob is small AND not thin-spine-shaped, treat as fragment -> Hough
        contours, _ = cv2.findContours(blob_res[0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            rect = cv2.minAreaRect(max(contours, key=cv2.contourArea))
            (_, _), (rw, rh), _ = rect
            short, long_ = (min(rw, rh), max(rw, rh))
            aspect = long_ / max(short, 1)
            if blob_res[1] < 0.03 * image_area and aspect < 6:
                use_hough = True

    gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    bg_gray = float(np.median(gray_small[:60, :60]))

    if use_hough:
        box_s = hough_spine_box(small)
        if box_s is None:
            return img, None, mask, bg, "no-contour"
        box_s = snap_rect_to_gradient(box_s, gray_small, bg_gray, search=15)
        if scale < 1:
            box_s = box_s / scale
        return img, box_s.astype(np.int32), mask, bg, "hough-spine"

    blob, _ = blob_res
    blob = strip_thin_appendages(blob, min(sh, sw))

    box_s = density_band_rect(blob, threshold=0.5)
    if box_s is None:
        return img, None, blob, bg, "density-fail"
    box_s = snap_rect_to_gradient(box_s, gray_small, bg_gray, search=40)

    cov = rect_coverage(blob, box_s)
    note = f"cov={cov:.2f}"

    if scale < 1:
        box_s = box_s / scale
    return img, box_s.astype(np.int32), blob, bg, note


def render_preview(img: np.ndarray, box, mask, bg, note: str, out_path: Path) -> None:
    h, w = img.shape[:2]
    preview = img.copy()
    if box is not None:
        cv2.polylines(preview, [box], True, (0, 0, 255), max(4, h // 400))
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_rgb = cv2.resize(mask_rgb, (w, h))
    side = np.hstack([preview, mask_rgb])
    th = 1200
    if side.shape[0] > th:
        s = th / side.shape[0]
        side = cv2.resize(side, None, fx=s, fy=s)
    cv2.putText(side, note, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.imwrite(str(out_path), side, [cv2.IMWRITE_JPEG_QUALITY, 85])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("previews"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    files = [args.input] if args.input.is_file() else sorted(
        p for p in args.input.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    )

    for f in files:
        try:
            img, box, mask, bg, note = detect(f)
            out = args.out / f"{f.stem}_preview.jpg"
            render_preview(img, box, mask, bg, note, out)
            status = "ok" if box is not None else "NO BOX"
            print(f"{status:7s} {note:18s}  bg=L{bg[0]:.0f}a{bg[1]:.0f}b{bg[2]:.0f}  {f.name}")
        except Exception as e:
            print(f"FAIL    {f.name}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
