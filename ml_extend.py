#!/usr/bin/env python3
"""ML + spine-axis extension.

1. Use rembg to get foreground mask + initial rect.
2. If rect is thin (aspect>4) AND small (area < 8% of image), assume it's a
   partial spine detection (ML locked onto brightest fragment).
3. Extend along the rect's long axis: scan perpendicular slices and include any
   slice that contains at least one non-bg pixel (within a perpendicular window).
4. Output overlay vs v26 (with user's green target).
"""
import sys
import threading
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from rembg import remove, new_session

PROJECT = Path("/Users/murat/git/private/bookcoverfixer")
PREVIEWS_V26 = PROJECT / "previews_v26"
OUT = PROJECT / "overlay"
_session = None
_session_lock = threading.Lock()


def _get_session():
    global _session
    if _session is None:
        with _session_lock:
            if _session is None:
                _session = new_session("u2net")
    return _session


def warm_model() -> None:
    _get_session()


def ml_mask(img_path: Path) -> np.ndarray:
    with Image.open(img_path) as image:
        mask = remove(image, session=_get_session(), only_mask=True)
        alpha = np.asarray(mask.convert("L"))
    _, mask = cv2.threshold(alpha, 64, 255, cv2.THRESH_BINARY)
    return mask


def bg_gray_estimate(img_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(np.median(gray[:60, :60]))


def extend_along_axis(img_bgr: np.ndarray, rect_box: np.ndarray, bg_gray: float) -> np.ndarray:
    """Project ALL non-bg pixels in the image onto the rect's long axis.
    Keep only those within (width/2 + margin) perpendicular distance from axis.
    New rect length = min..max projection of kept pixels."""
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    pts = rect_box.astype(np.float32)
    edges = [(pts[i], pts[(i + 1) % 4]) for i in range(4)]
    edge_lens = [np.linalg.norm(b - a) for a, b in edges]
    long_idx = int(np.argmax(edge_lens))
    short_idx = (long_idx + 1) % 4
    a, b = edges[long_idx]
    long_vec = b - a
    L_long = float(np.linalg.norm(long_vec))
    long_unit = long_vec / L_long
    short_unit = np.array([-long_unit[1], long_unit[0]], dtype=np.float32)
    sa, sb = edges[short_idx]
    width = float(np.linalg.norm(sb - sa))

    center = pts.mean(axis=0)

    # find all non-bg pixels — use stricter threshold to ignore scanner noise
    bg_tol = 18.0
    diff = np.abs(gray - bg_gray)
    # zero out image-border margins (scanner edge artifacts)
    border_mask = np.zeros_like(diff, dtype=bool)
    bm = max(40, min(h, w) // 30)
    border_mask[bm:-bm, bm:-bm] = True
    mask = (diff > bg_tol) & border_mask
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return rect_box

    pts_xy = np.column_stack([xs, ys]).astype(np.float32) - center
    proj_long = pts_xy @ long_unit
    proj_short = pts_xy @ short_unit

    # keep only pixels TIGHT to the axis (true spine fragments, not bg noise)
    band_half = max(width / 2 + 4, 8.0)
    keep = np.abs(proj_short) < band_half
    pl = proj_long[keep]
    if len(pl) < 5:
        return rect_box

    # extents — density-filtered: bin projections into 5px bins, keep only
    # bins with ≥5 pixels (kills isolated noise that raw percentiles let
    # through), then take 0.5/99.5 percentiles of the kept mass. Plain 2/98
    # trimmed ~5px off the tulsa_1_3 spine tip; plain 0.5/99.5 over-extended
    # into noise.
    bins = np.floor(pl / 5.0).astype(np.int64)
    uniq, counts = np.unique(bins, return_counts=True)
    good = set(uniq[counts >= 5].tolist())
    pl_f = pl[np.isin(bins, list(good))] if good else pl
    if len(pl_f) < 5:
        pl_f = pl
    p_min = float(np.percentile(pl_f, 0.5))
    p_max = float(np.percentile(pl_f, 99.5))

    # ensure we don't shrink relative to current rect
    p_min = min(p_min, -L_long / 2)
    p_max = max(p_max, L_long / 2)

    new_center = center + long_unit * (p_min + p_max) / 2
    new_long_half = (p_max - p_min) / 2
    new_short_half = width / 2
    p1 = new_center - long_unit * new_long_half - short_unit * new_short_half
    p2 = new_center + long_unit * new_long_half - short_unit * new_short_half
    p3 = new_center + long_unit * new_long_half + short_unit * new_short_half
    p4 = new_center - long_unit * new_long_half + short_unit * new_short_half
    return np.array([p1, p2, p3, p4], dtype=np.float32)


def ml_rect(img_bgr: np.ndarray, mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    big = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(big)
    return cv2.boxPoints(rect)


def bg_lab(img_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    sh, sw = lab.shape[:2]
    return np.median(
        np.concatenate([
            lab[:60, :60].reshape(-1, 3),
            lab[:60, -60:].reshape(-1, 3),
            lab[-60:, :60].reshape(-1, 3),
            lab[-60:, -60:].reshape(-1, 3),
        ]),
        axis=0,
    )


def bg_distance_mask(img_bgr: np.ndarray, thresh: float = 8.0) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg = bg_lab(img_bgr).astype(np.float32)
    diff = lab - bg
    dist = np.sqrt((diff * diff).sum(axis=2))
    return (dist > thresh).astype(np.uint8) * 255


def snap_via_distance(box: np.ndarray, img_bgr: np.ndarray, max_outward: int = 40, max_inward: int = 40) -> np.ndarray:
    """For each rect edge, walk along the normal and find the position where
    mean LAB distance from bg crosses a clear book threshold. This is the
    precise physical book/bg boundary."""
    h, w = img_bgr.shape[:2]
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg = bg_lab(img_bgr).astype(np.float32)

    pts = box.astype(np.float32).copy()
    center = pts.mean(axis=0)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]

    for a, b in edges:
        ea, eb = pts[a], pts[b]
        edge_vec = eb - ea
        L = float(np.linalg.norm(edge_vec))
        if L < 1e-6:
            continue
        normal = np.array([-edge_vec[1], edge_vec[0]], dtype=np.float32) / L
        midpoint = (ea + eb) / 2
        if np.dot(normal, midpoint - center) < 0:
            normal = -normal

        # build distance profile along normal; d>0 = outward, d<0 = inward
        N = 31
        ts = np.linspace(0.1, 0.9, N)
        D_total = max_outward + max_inward + 1
        prof = np.zeros(D_total, dtype=np.float32)
        cnt = np.zeros(D_total, dtype=np.float32)
        for t in ts:
            p_base = ea + t * edge_vec
            for i, d in enumerate(range(-max_outward, max_inward + 1)):
                # d>0 = outward (along normal); d<0 = inward (against normal)
                # but here we want d positive to mean OUTWARD relative to box
                # we sample at p_base + normal*d
                p = p_base + normal * d
                ix, iy = int(round(p[0])), int(round(p[1]))
                if 0 <= ix < w and 0 <= iy < h:
                    diff = lab[iy, ix] - bg
                    dist = float(np.sqrt((diff * diff).sum()))
                    prof[i] += dist
                    cnt[i] += 1
        avg = np.where(cnt > 0, prof / np.maximum(cnt, 1), 0)
        # smooth (5-tap median to kill spikes)
        med = max(5, 7)
        padded = np.pad(avg, med // 2, mode="edge")
        avg = np.array([float(np.median(padded[k:k + med])) for k in range(len(avg))], dtype=np.float32)

        # the book/bg edge is where the distance profile JUMPS sharpest.
        # find the position of maximum gradient (first derivative) along the profile.
        grad = np.abs(np.diff(avg))
        # peak gradient = boundary
        if len(grad) < 3 or float(grad.max()) < 3:
            continue
        crossing = int(np.argmax(grad)) + 1  # +1 because diff shifts by 1
        if crossing is None:
            continue
        # convert i back to d offset: d = -max_outward + i; but our d here means
        # OUTWARD relative to rect (i.e., away from center). So d>0 means edge
        # should move outward, d<0 means inward.
        d_off = -max_outward + crossing
        # move edge in normal direction by d_off (positive = outward, negative = inward)
        pts[a] = ea + normal * d_off
        pts[b] = eb + normal * d_off
    return pts


def snap_inward_via_bgmask(box: np.ndarray, img_bgr: np.ndarray, max_inward: int = 25) -> np.ndarray:
    """Per-edge tighten: for each rect edge, count how many sample points along
    it lie OUTSIDE the bg-distance mask (i.e., in bg). If many do, pull that
    edge inward until most samples are inside the bg-distance mask."""
    h, w = img_bgr.shape[:2]
    bgd = bg_distance_mask(img_bgr, thresh=15.0)
    bgd = cv2.morphologyEx(
        bgd, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )
    bgd = cv2.morphologyEx(
        bgd, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)),
        iterations=2,
    )
    # keep ONLY the largest blob (the book), kill scattered remnants
    nc, labels, stats, _ = cv2.connectedComponentsWithStats(bgd, connectivity=8)
    if nc > 1:
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        bgd = ((labels == biggest).astype(np.uint8)) * 255
    # tiny erode ~1 px to align mask boundary close to true book edge
    bgd = cv2.erode(
        bgd, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1
    )

    pts = box.astype(np.float32).copy()
    center = pts.mean(axis=0)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]

    for a, b in edges:
        ea, eb = pts[a], pts[b]
        edge_vec = eb - ea
        L = float(np.linalg.norm(edge_vec))
        if L < 1e-6:
            continue
        normal = np.array([-edge_vec[1], edge_vec[0]], dtype=np.float32) / L
        midpoint = (ea + eb) / 2
        if np.dot(normal, midpoint - center) < 0:
            normal = -normal

        N = 21
        ts = np.linspace(0.1, 0.9, N)

        def inside_frac(d_off):
            count = 0
            valid = 0
            for t in ts:
                p = ea + t * edge_vec - normal * d_off
                ix, iy = int(round(p[0])), int(round(p[1]))
                if 0 <= ix < w and 0 <= iy < h:
                    valid += 1
                    if bgd[iy, ix] > 0:
                        count += 1
            return count / max(valid, 1)

        # only tighten if current edge has bg samples
        cur_frac = inside_frac(0)
        if cur_frac >= 0.95:
            continue  # already on book

        # walk inward until ≥95% of samples are on book
        best_d = None
        for d in range(1, max_inward + 1):
            if inside_frac(d) >= 0.95:
                best_d = d
                break
        if best_d is None:
            continue
        pts[a] = ea - normal * best_d
        pts[b] = eb - normal * best_d
    return pts


def snap_inward_to_book(box: np.ndarray, img_bgr: np.ndarray, bg_lab_v: np.ndarray, max_inward: int = 20) -> np.ndarray:
    """For each rect edge, walk INWARD and find the position where mean color
    distance from bg is HIGHEST (= deepest into book). Then snap edge to a
    position slightly outside that maximum — at the boundary."""
    h, w = img_bgr.shape[:2]
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    pts = box.astype(np.float32).copy()
    center = pts.mean(axis=0)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]

    for a, b in edges:
        ea, eb = pts[a], pts[b]
        edge_vec = eb - ea
        L = float(np.linalg.norm(edge_vec))
        if L < 1e-6:
            continue
        normal = np.array([-edge_vec[1], edge_vec[0]], dtype=np.float32) / L
        midpoint = (ea + eb) / 2
        if np.dot(normal, midpoint - center) < 0:
            normal = -normal

        N = 21
        ts = np.linspace(0.1, 0.9, N)
        # measure average dist-from-bg at each step from d=-5 (outside edge) to d=+max_inward
        avgs = []
        for d in range(-5, max_inward + 1):
            total = 0.0
            cnt = 0
            for t in ts:
                p = ea + t * edge_vec - normal * d
                ix, iy = int(round(p[0])), int(round(p[1]))
                if 0 <= ix < w and 0 <= iy < h:
                    diff = lab[iy, ix] - bg_lab_v
                    total += float(np.sqrt((diff * diff).sum()))
                    cnt += 1
            avgs.append(total / max(cnt, 1))
        avgs = np.array(avgs, dtype=np.float32)
        # smooth profile
        avgs = cv2.blur(avgs.reshape(1, -1), (1, 5)).ravel()
        # find first big jump going inward from outside
        # the boundary is where avg crosses midpoint between bg-side (low) and book-side (high)
        outside_avg = float(np.median(avgs[:3]))  # d=-5,-4,-3
        inside_avg = float(np.median(avgs[-5:]))  # last 5 = deepest inside
        if inside_avg - outside_avg < 3:
            continue  # not enough contrast to tighten reliably
        threshold = outside_avg + (inside_avg - outside_avg) * 0.55
        # find first index where avg >= threshold
        crossing = None
        for i in range(len(avgs)):
            if avgs[i] >= threshold:
                crossing = i
                break
        if crossing is None:
            continue
        # convert i to d offset (d = i - 5 because we started at d=-5)
        d_off = crossing - 5
        # snap inward by d_off (if positive, move inward; if negative, move outward slightly)
        if d_off != 0:
            pts[a] = ea - normal * d_off
            pts[b] = eb - normal * d_off
    return pts


def snap_via_gradient(box: np.ndarray, img_bgr: np.ndarray, max_inward: int = 80, max_outward: int = 15, only_long_edges: bool = False) -> np.ndarray:
    """Per-edge snap to peak image gradient.
    Walks each rect edge along its outward normal across [-max_inward, +max_outward],
    averages Sobel gradient magnitude over the edge length, snaps to the peak.
    Works on low-contrast books (e.g. black-on-near-black) where LAB-distance
    methods fail — the visible boundary still produces a gradient peak.
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)
    gmag = np.sqrt(gx * gx + gy * gy)
    gmag = cv2.GaussianBlur(gmag, (5, 5), 0)

    pts = box.astype(np.float32).copy()
    center = pts.mean(axis=0)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    if only_long_edges:
        edge_lens = [float(np.linalg.norm(pts[bi] - pts[ai])) for ai, bi in edges]
        long_max = max(edge_lens)
        edges = [edges[i] for i in range(4) if edge_lens[i] > 0.6 * long_max]
    for ai, bi in edges:
        ea, eb = pts[ai], pts[bi]
        edge_vec = eb - ea
        L = float(np.linalg.norm(edge_vec))
        if L < 1e-6:
            continue
        normal = np.array([-edge_vec[1], edge_vec[0]], dtype=np.float32) / L
        midpoint = (ea + eb) / 2
        if np.dot(normal, midpoint - center) < 0:
            normal = -normal  # OUTWARD

        N = 31
        ts = np.linspace(0.1, 0.9, N)
        ds = list(range(-max_inward, max_outward + 1))
        prof = np.zeros(len(ds), dtype=np.float32)
        gprof = np.zeros(len(ds), dtype=np.float32)
        cnt = np.zeros(len(ds), dtype=np.float32)
        for t in ts:
            p_base = ea + t * edge_vec
            for i, d in enumerate(ds):
                p = p_base + normal * d
                ix, iy = int(round(p[0])), int(round(p[1]))
                if 0 <= ix < w and 0 <= iy < h:
                    prof[i] += gmag[iy, ix]
                    gprof[i] += gray[iy, ix]
                    cnt[i] += 1
        avg = np.where(cnt > 0, prof / np.maximum(cnt, 1), 0)
        gavg = np.where(cnt > 0, gprof / np.maximum(cnt, 1), 0)
        # require a CLEAR DOMINANT peak — at least 5x median.
        # Muddled profiles (max/median < 5) signal the ml rect is already on the
        # boundary AND internal book content noise dominates — don't snap.
        # Clean cases (tulsa_1_2: max/median ~30) snap correctly.
        med = float(np.median(avg))
        if avg.max() < max(5.0, 5.0 * med):
            continue
        thresh = max(5.0, 2.0 * med)
        # Find the OUTERMOST gradient peak (walking from outside inward).
        # The book/bg boundary is the FIRST significant gradient we hit from outside;
        # any inner peaks (title text, strap, spiral binding) are internal book content
        # whose gradient may be STRONGER than the book/bg boundary itself.
        # The peak must also reach ≥25% of the global max: weak secondary ridges
        # (the soft shadow line below tulsa_1_2's bottom edge) otherwise win and
        # drag the edge into bg. 0.5×max was too strict — it skipped the true
        # boundary ridge when an inner feature had a stronger gradient.
        peak_floor = max(thresh, 0.25 * float(avg.max()))
        best = None
        for i in range(len(ds) - 2, 0, -1):  # iterate from outer side, skip endpoints
            if avg[i] >= peak_floor and avg[i] >= avg[i - 1] and avg[i] >= avg[i + 1]:
                # bright→dark gate: a real book edge on the dark mat falls in
                # luminance going OUTWARD (even black covers end in a bright
                # page sliver). The scanner-shadow/bg boundary below the book
                # RISES outward (shadow 13 → bg 25). Probe ±3 — the shadow
                # strip is only ~7px tall, so a ±6 probe jumped OVER it back
                # into the bright sliver and the gate passed the false edge.
                i_in = max(0, i - 3)
                i_out = min(len(ds) - 1, i + 3)
                if gavg[i_in] <= gavg[i_out]:
                    continue
                best = i
                break
        if best is not None:
            # nudge OUTWARD from the peak center by at most 2px: peak center
            # sits mid-transition (tulsa top edges landed ~5px inset vs the
            # green outer-boundary), but anything more rides gradual shadow
            # ramps below the book (a 4px cap pushed five other edges 6-10px
            # into shadow/bg).
            jj = best
            half = 0.5 * avg[best]
            while jj + 1 < len(ds) and avg[jj + 1] >= half and (jj - best) < 2:
                jj += 1
            best = jj
        if best is None:
            continue
        d_off = ds[best]
        pts[ai] = ea + normal * d_off
        pts[bi] = eb + normal * d_off
    return pts


def detect(img_path: Path):
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    mask = ml_mask(img_path)
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    # erode ML mask slightly — rembg masks tend to overshoot the true book edge
    # by a few px due to soft-alpha boundaries.
    erode_k = max(3, min(h, w) // 300)
    if erode_k % 2 == 0:
        erode_k += 1
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_k, erode_k))
    mask_tight = cv2.erode(mask, se, iterations=2)
    if cv2.countNonZero(mask_tight) > 0.5 * cv2.countNonZero(mask):
        mask = mask_tight
    box = ml_rect(img, mask)
    if box is None:
        return img, None
    rect_area = cv2.contourArea(box)
    image_area = h * w
    sides = [np.linalg.norm(box[i] - box[(i + 1) % 4]) for i in range(4)]
    long_side = max(sides)
    short_side = min(sides)
    aspect = long_side / max(short_side, 1)
    rect_pct = rect_area / image_area
    note = f"rect_pct={rect_pct:.3f} aspect={aspect:.1f}"
    print(note)

    # detect "ML missed surrounding book" — when bg-distance mask has many pixels
    # outside the ML rect, ML locked onto an inner subject. Expand to outer extent.
    bgd = bg_distance_mask(img, thresh=18.0)
    # only consider pixels not at image borders
    bm = max(40, min(h, w) // 30)
    bgd_clean = np.zeros_like(bgd)
    bgd_clean[bm:-bm, bm:-bm] = bgd[bm:-bm, bm:-bm]
    # how much of bg-distance mask is inside the ML rect?
    rect_mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(rect_mask, [box.astype(np.int32)], 255)
    inside = cv2.bitwise_and(bgd_clean, rect_mask)
    total_bgd = max(cv2.countNonZero(bgd_clean), 1)
    inside_frac = cv2.countNonZero(inside) / total_bgd
    print(f"bg-distance pixels inside ML rect: {inside_frac:.2f}")

    # if ML is a thin spine (high aspect, small area) the axis extension below
    # will handle it — DON'T fall back to heuristic (which loses the tilt).
    is_thin_spine = aspect > 3 and rect_pct < 0.08
    if inside_frac < 0.75 and not is_thin_spine:
        # ML missed a lot of book content; fall back to detect.py heuristic
        # which uses edge-floodfill + bg-distance and handles low-contrast books.
        from detect import detect as heuristic_detect
        _, h_box, _, _, _ = heuristic_detect(img_path)
        if h_box is not None:
            box = h_box.astype(np.float32)
            print("fell back to detect.py heuristic")

    # extension along long axis for thin spines
    sides = [np.linalg.norm(box[i] - box[(i + 1) % 4]) for i in range(4)]
    aspect = max(sides) / max(min(sides), 1)
    rect_pct = cv2.contourArea(box) / image_area
    is_spine = (aspect > 3 and rect_pct < 0.08) or rect_pct < 0.01
    if is_spine:
        bg_g = bg_gray_estimate(img)
        box = extend_along_axis(img, box, bg_g)
        print("extended along axis")

    # gradient-snap (only for non-spine rects): pulls overshoot edges tight to
    # visible book boundary. Critical for low-contrast books (tulsa_1_2) where
    # LAB-distance methods can't detect book/bg boundary.
    if is_spine:
        # only tighten the long sides — leave short ends alone (axis-extension already set them).
        # max_inward sized to spine width: each side can pull up to ~1/3 of current width.
        sides = [np.linalg.norm(box[i] - box[(i + 1) % 4]) for i in range(4)]
        spine_width = float(min(sides))
        mi = max(8, int(spine_width // 3))
        box = snap_via_gradient(box, img, max_inward=mi, max_outward=8, only_long_edges=True)
    else:
        box = snap_via_gradient(box, img)
    return img, box


def main():
    img_path = Path(sys.argv[1])
    name = img_path.stem
    img, box = detect(img_path)
    if box is None:
        print(f"FAIL {name}")
        return
    v26 = cv2.imread(str(PREVIEWS_V26 / f"{name}_preview.jpg"))
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
    cv2.putText(out, "ML+extend (CYAN)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    OUT.mkdir(exist_ok=True)
    p = OUT / f"{name}_mlext.jpg"
    cv2.imwrite(str(p), out, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"-> {p}")


if __name__ == "__main__":
    main()
