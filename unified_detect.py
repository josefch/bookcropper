#!/usr/bin/env python3
"""Unified book detection: score-based ensemble.

Run multiple detectors (ml, ml+snap, color, heuristic), score each candidate
on physical edge fit, pick the highest-scoring one. No hand-tuned rules.

Score = mean Sobel gradient along rect perimeter * sqrt(bg-distance density inside).
Rationale:
- High perimeter gradient = rect edges land on real image edges (book boundary)
- High interior bg-distance density = rect contains real book content, not bg
- Product rewards rects that are BOTH well-located AND tight on actual book.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

import ml_extend
import color_detect
import detect as heuristic


def rect_area(box: np.ndarray) -> float:
    return float(cv2.contourArea(box.astype(np.float32)))


def perimeter_gradient(box: np.ndarray, gmag: np.ndarray) -> float:
    h, w = gmag.shape[:2]
    vals = []
    for ai, bi in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        ea, eb = box[ai], box[bi]
        L = float(np.linalg.norm(eb - ea))
        n = max(10, int(L / 8))
        ts = np.linspace(0.0, 1.0, n)
        for t in ts:
            p = ea + t * (eb - ea)
            ix, iy = int(round(p[0])), int(round(p[1]))
            if 0 <= ix < w and 0 <= iy < h:
                vals.append(float(gmag[iy, ix]))
    return float(np.mean(vals)) if vals else 0.0


def perimeter_ridge_fraction(box: np.ndarray, gmag: np.ndarray, probe: int = 6, floor: float = 3.0) -> float:
    """Fraction of perimeter samples sitting ON a gradient ridge (local max across
    the edge normal). Contrast-INVARIANT: a weak-but-real book edge on a dark book
    counts the same as a strong edge. Raw gradient strength rewards rects that snag
    high-contrast features (straps, shadows) instead of the true low-contrast
    boundary — this metric doesn't."""
    h, w = gmag.shape[:2]
    pts = box.astype(np.float32)
    center = pts.mean(axis=0)
    on_ridge = 0
    total = 0
    for ai, bi in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        ea, eb = pts[ai], pts[bi]
        edge_vec = eb - ea
        L = float(np.linalg.norm(edge_vec))
        if L < 1e-6:
            continue
        normal = np.array([-edge_vec[1], edge_vec[0]], dtype=np.float32) / L
        midpoint = (ea + eb) / 2
        if np.dot(normal, midpoint - center) < 0:
            normal = -normal
        n = max(10, int(L / 12))
        for t in np.linspace(0.05, 0.95, n):
            p = ea + t * edge_vec
            p_in = p - normal * probe
            p_out = p + normal * probe
            coords = []
            ok = True
            for q in (p, p_in, p_out):
                ix, iy = int(round(q[0])), int(round(q[1]))
                if not (0 <= ix < w and 0 <= iy < h):
                    ok = False
                    break
                coords.append(gmag[iy, ix])
            if not ok:
                continue
            total += 1
            g0, gi, go = coords
            if g0 > floor and g0 >= gi and g0 >= go:
                on_ridge += 1
    return on_ridge / max(total, 1)


def interior_bg_density(box: np.ndarray, bgd_mask: np.ndarray) -> float:
    h, w = bgd_mask.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [box.astype(np.int32)], 255)
    area = max(cv2.countNonZero(mask), 1)
    inside = cv2.countNonZero(cv2.bitwise_and(mask, bgd_mask))
    return inside / area


def captured_fraction(box: np.ndarray, ref_mask: np.ndarray, ref_total: int) -> float:
    """Fraction of reference book area (union of all candidate rects) inside this rect.
    Penalizes rects that miss book content other detectors found."""
    if ref_total == 0:
        return 1.0
    h, w = ref_mask.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [box.astype(np.int32)], 255)
    inside = cv2.countNonZero(cv2.bitwise_and(mask, ref_mask))
    return inside / ref_total


def perimeter_bands(box: np.ndarray, bgd_mask: np.ndarray, band: int = 12):
    """(inner_book_fraction, outer_bg_fraction) for strips just inside/outside
    the rect perimeter. A correctly-placed rect has book pixels in its inner
    band and bg pixels in its outer band. A rect snapped to a cast SHADOW edge
    (e.g. moreno set: ml box 55px out on the shadow ridge) has bg-like shadow
    pixels in its inner band — punished here, invisible to whole-area density."""
    h, w = bgd_mask.shape[:2]
    rect = np.zeros((h, w), np.uint8)
    cv2.fillPoly(rect, [box.astype(np.int32)], 255)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1, 2 * band + 1))
    inner_ring = cv2.subtract(rect, cv2.erode(rect, k))
    outer_ring = cv2.subtract(cv2.dilate(rect, k), rect)
    inner_n = max(cv2.countNonZero(inner_ring), 1)
    outer_n = max(cv2.countNonZero(outer_ring), 1)
    inner_book = cv2.countNonZero(cv2.bitwise_and(inner_ring, bgd_mask)) / inner_n
    outer_bg = 1.0 - cv2.countNonZero(cv2.bitwise_and(outer_ring, bgd_mask)) / outer_n
    return inner_book, outer_bg


def per_edge_penalties(box: np.ndarray, bgd_mask: np.ndarray, band: int = 12) -> float:
    """Multiplicative penalty for per-EDGE defects that perimeter averaging hides.
    For each of the 4 edges, sample a strip just outside and just inside:
      - outer strip mostly book (>50%)  → this edge CUTS the book (kanko: the
        dark top band fills ~90% of the top edge's outer strip; a dangling
        strap fills only ~10% of its edge → tolerated).
      - inner strip mostly bg (>50%)    → this edge floats in bg/shadow
        (moreno ml box: left edge 55px out, its inner strip is all shadow).
    Each defective edge multiplies the score by 0.3."""
    h, w = bgd_mask.shape[:2]
    pts = box.astype(np.float32)
    center = pts.mean(axis=0)
    penalty = 1.0
    for ai, bi in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        ea, eb = pts[ai], pts[bi]
        edge_vec = eb - ea
        L = float(np.linalg.norm(edge_vec))
        if L < 1e-6:
            continue
        normal = np.array([-edge_vec[1], edge_vec[0]], dtype=np.float32) / L
        midpoint = (ea + eb) / 2
        if np.dot(normal, midpoint - center) < 0:
            normal = -normal
        n = max(10, int(L / 15))
        out_book = in_bg = total = 0
        for t in np.linspace(0.05, 0.95, n):
            p = ea + t * edge_vec
            # probe at TWO depths (12 and 24px): a defect must persist at both
            # to count. A ~15px overshoot (kanko ml box) only registers at the
            # shallow probe and is tolerated; a 55px shadow-float (moreno ml)
            # or a 40px band cut (kanko heur) registers at both.
            probes_o = [p + normal * band, p + normal * 2 * band]
            probes_i = [p - normal * band, p - normal * 2 * band]
            coords = []
            ok = True
            for q in probes_o + probes_i:
                qx, qy = int(round(q[0])), int(round(q[1]))
                if not (0 <= qx < w and 0 <= qy < h):
                    ok = False
                    break
                coords.append(bgd_mask[qy, qx])
            if not ok:
                continue
            total += 1
            o1, o2, i1, i2 = coords
            if o1 > 0 and o2 > 0:
                out_book += 1
            if i1 == 0 and i2 == 0:
                in_bg += 1
        if total < 5:
            continue
        if out_book / total > 0.5:
            penalty *= 0.3
        if in_bg / total > 0.5:
            penalty *= 0.3
    return penalty


def score_box(box: np.ndarray, gmag: np.ndarray, bgd_mask: np.ndarray, ref_mask: np.ndarray, ref_total: int, low_contrast: bool = False) -> float:
    cf = captured_fraction(box, ref_mask, ref_total)
    # On low-contrast images (tulsa set: book LAB dist from bg ≈ 4.5 < thresh)
    # bgd is ~empty, band fractions are meaningless — use ridge fraction ONLY.
    # No captured_fraction here: the candidate union is untrustworthy when bgd
    # can't validate it (tulsa_1_3: heur's box was 94% background, ml's correct
    # tilted spine box covered 21% of that union and lost despite 4x better
    # ridge alignment).
    if low_contrast:
        return perimeter_ridge_fraction(box, gmag)
    inner_book, outer_bg = perimeter_bands(box, bgd_mask)
    # geometric, contrast-invariant:
    # - inner_book: strip inside perimeter is book content. WEIGHTED HIGH (²):
    #   nothing foreign intrudes inside the true book hull, so a low inner_book
    #   reliably means the rect overshoots (shadow/bg swallowed).
    # - outer_bg: strip outside perimeter is background. WEIGHTED LOW (√):
    #   legitimate accessories (strap tails, ribbons) dangle OUTSIDE the ideal
    #   rect and would otherwise punish the correct box (moreno set).
    # - captured: rect doesn't miss book area found by other detectors
    # - per_edge_penalties: catch single-edge defects that averaging hides
    pep = per_edge_penalties(box, bgd_mask)
    return (inner_book ** 2) * (outer_bg ** 0.5) * cf * pep


def refine_edges(box: np.ndarray, bgd_mask: np.ndarray, gmag: np.ndarray,
                 low_contrast: bool, max_shift: int = 40, gray_img: np.ndarray = None,
                 br_map: np.ndarray = None) -> np.ndarray:
    """Final per-edge snap of the WINNING box to the actual book boundary.
    Selection picks the best candidate, but candidates are routinely 5-190px
    off (audits measure preview px ~4-6x smaller than real px).

    Per edge, build two profiles along the outward normal:
      f(d)    = fraction of edge samples on book (bgd mask)
      g(d)    = mean gradient magnitude
    The boundary is the significant f-DESCENT that coincides with the
    STRONGEST GRADIENT RIDGE. f alone is ambiguous: scanner-lid shadow darker
    than corner-bg reads as "book" (keuken top: false descent 18px out,
    gmag 224, while the true junction at d=0 has gmag 2325); the shadow-filter
    eats cover slivers (moreno_2: true edge ridge gmag 364 at d=-14).

    Ring/plateau extension: spiral binding rings beyond the solid edge are a
    sparse tail (0.05≤f≤0.5) ENDING in empty bg, with SPIKY gradient (wire
    edges, CoV≥0.5) — extend to the tail end. Smooth shadow slivers have the
    same f-signature but low gradient variance — not extended.

    Edges where bgd is blind (f flat — dark cover region on dark bg): snap to
    the strongest gradient ridge searching INWARD up to 200px (nues_1: the
    heur box right edge floats ~190 real px out in flat background).

    Low-contrast images are NOT skipped: their edges have flat/inverted f and
    route automatically into the per-sample voting path (the old blanket skip
    silently left tulsa/keuken bottom edges 9-10px deep in the shadow strip —
    no downstream stage could ever correct them). Voting re-snaps to the same
    dominant falling ridge ml's internal snap found, so double-snap drift is
    bounded to ~1px.
    """
    h, w = bgd_mask.shape[:2]
    pts = box.astype(np.float32).copy()
    center = pts.mean(axis=0)
    side01 = float(np.linalg.norm(pts[1] - pts[0]))
    side12 = float(np.linalg.norm(pts[2] - pts[1]))
    short_side = max(min(side01, side12), 1.0)
    # inward moves may not exceed 15% of the box's short side: on ultra-thin
    # spines (keuken_17_3: ~17px wide) each long edge's vote window sees the
    # OPPOSITE edge's ridge and the box collapses to a line.
    max_inward = -max(3.0, 0.08 * short_side)
    for ai, bi in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        ea, eb = pts[ai], pts[bi]
        edge_vec = eb - ea
        L = float(np.linalg.norm(edge_vec))
        if L < 1e-6:
            continue
        normal = np.array([-edge_vec[1], edge_vec[0]], dtype=np.float32) / L
        midpoint = (ea + eb) / 2
        if np.dot(normal, midpoint - center) < 0:
            normal = -normal  # outward
        n = max(15, int(L / 10))
        ts = np.linspace(0.05, 0.95, n)
        ds = list(range(-max_shift, max_shift + 1))
        f = np.zeros(len(ds), np.float32)
        g = np.zeros(len(ds), np.float32)
        cnt = np.zeros(len(ds), np.float32)
        for t in ts:
            p0 = ea + t * edge_vec
            for i, d in enumerate(ds):
                p = p0 + normal * d
                x, y = int(round(p[0])), int(round(p[1]))
                if 0 <= x < w and 0 <= y < h:
                    cnt[i] += 1
                    g[i] += gmag[y, x]
                    if bgd_mask[y, x] > 0:
                        f[i] += 1
        f = np.where(cnt > 0, f / np.maximum(cnt, 1), 0)
        g = np.where(cnt > 0, g / np.maximum(cnt, 1), 0)
        # edge-padded smoothing (convolve 'same' zero-pads -> fake end-drop)
        f = np.convolve(np.pad(f, 1, mode="edge"), [0.25, 0.5, 0.25], mode="valid")

        # bgd unreliable on this edge when: no transition at all (flat f), OR
        # the inside isn't even book (inverted mask: vignetting makes far
        # fabric read "book" at f≈0.5 while the black book itself reads ~0 —
        # the dark-cover tulsa/keuken family). Resolution per sample:
        #   STAY  if the inside is continuously dark with NO bright sliver
        #         within 12px (keuken_17_2 right: black cover IS the book,
        #         the edge already sits on the cover→fabric boundary), or a
        #         falling-luminance ridge is already within ±4px.
        #   VOTE  otherwise: the edge floats in shadow beyond the book's
        #         bright page sliver (tulsa/keuken bottoms+tops) — snap to
        #         the per-sample strongest falling-luminance ridge (median).
        if f.max() - f.min() < 0.3 or float(np.mean(f[:10])) < 0.5:
            if gray_img is None:
                continue
            stay = 0
            checked = 0
            for t in ts[:: max(1, len(ts) // 12)]:
                p0 = ea + t * edge_vec
                # local fabric reference beyond the edge
                ref_vals = []
                for d in range(20, 51):
                    p = p0 + normal * d
                    x, y = int(round(p[0])), int(round(p[1]))
                    if 0 <= x < w and 0 <= y < h:
                        ref_vals.append(float(gray_img[y, x]))
                if len(ref_vals) < 15:
                    continue
                fab = float(np.median(ref_vals))
                checked += 1
                # (a) falling ridge within ±4 → on boundary
                found = False
                for d in range(-4, 5):
                    p = p0 + normal * d
                    x, y = int(round(p[0])), int(round(p[1]))
                    xi, yi = int(round(p[0] - normal[0] * 3)), int(round(p[1] - normal[1] * 3))
                    xo, yo = int(round(p[0] + normal[0] * 3)), int(round(p[1] + normal[1] * 3))
                    if not (0 <= x < w and 0 <= y < h and 0 <= xi < w and 0 <= yi < h and 0 <= xo < w and 0 <= yo < h):
                        continue
                    if float(gmag[y, x]) >= 30.0 and float(gray_img[yi, xi]) > float(gray_img[yo, xo]) + 2:
                        found = True
                        break
                # (b) inside continuously dark, no sliver within 12px →
                #     the dark zone is the book itself, edge is correct
                if not found:
                    dark_run = True
                    has_bright = False
                    # window 18 > max observed float distance (edges sit up to
                    # ~13px out in shadow; a 12px window missed the sliver by
                    # 0.4px and held the wrong bottoms)
                    for d in range(-18, 0):
                        p = p0 + normal * d
                        x, y = int(round(p[0])), int(round(p[1]))
                        if not (0 <= x < w and 0 <= y < h):
                            continue
                        gv = float(gray_img[y, x])
                        if gv > fab + 5:
                            has_bright = True
                        if gv >= fab - 3:
                            dark_run = False
                    if dark_run and not has_bright:
                        found = True
                if found:
                    stay += 1
            # vote: per-sample strongest falling-luminance ridge
            g_med = float(np.median(g[g > 0])) if np.any(g > 0) else 0.0
            vote_floor = max(15.0, 1.5 * g_med)
            # two-phase range: normally -60..+15 (a -200 range let interior
            # text lines capture the vote). If almost NO sample finds a ridge
            # (edge floats in flat background — nues_1 right is 140 real px
            # out), there is no text risk either: extend inward to -200.
            def collect_votes(d_lo):
                vv = []
                for t in ts:
                    p0 = ea + t * edge_vec
                    best_d = None
                    # adaptive floor: a fixed 15 sat below the bg grain noise
                    # (gmag 17-35 on dark fabric) — noise votes with median ~0
                    # froze nues_1's right edge 140px out in flat background
                    best_g = vote_floor
                    for d in range(d_lo, 16):
                        p = p0 + normal * d
                        x, y = int(round(p[0])), int(round(p[1]))
                        xi, yi = int(round(p[0] - normal[0] * 3)), int(round(p[1] - normal[1] * 3))
                        xo, yo = int(round(p[0] + normal[0] * 3)), int(round(p[1] + normal[1] * 3))
                        if not (0 <= x < w and 0 <= y < h and 0 <= xi < w and 0 <= yi < h and 0 <= xo < w and 0 <= yo < h):
                            continue
                        gv = float(gmag[y, x])
                        if gv <= best_g:
                            continue
                        if float(gray_img[yi, xi]) <= float(gray_img[yo, xo]) + 2:
                            continue  # luminance rises outward — shadow boundary
                        best_g = gv
                        best_d = d
                    if best_d is not None:
                        vv.append(best_d)
                return vv
            votes = collect_votes(-60)
            if len(votes) < 0.3 * len(ts):
                votes = collect_votes(-200)
            consensus = (
                len(votes) >= 0.8 * len(ts)
                and float(np.percentile(votes, 75) - np.percentile(votes, 25)) <= 8.0
            )
            d_off = int(np.median(votes)) if votes else 0
            # CHROMA GATE resolves the stay/vote ambiguity for inward moves:
            # the zone between the voted edge and the current edge is either
            #   shadowed FABRIC (bluish, B-R≈+5: tulsa/keuken bottoms — the
            #   book truly ends at the sliver; shadow falls BELOW the book)
            # or NEUTRAL black book material (B-R≈0: keuken_17_2's right edge
            # black cover, tulsa tops' dark cover band the user's green marks
            # include). Apply the votes only over bluish (shadow) zones.
            apply_votes = False
            if consensus and d_off < -3 and br_map is not None:
                zone_br = []
                for t in ts[:: max(1, len(ts) // 12)]:
                    p0 = ea + t * edge_vec
                    for d in range(d_off + 2, -1):
                        p = p0 + normal * d
                        x, y = int(round(p[0])), int(round(p[1]))
                        if 0 <= x < w and 0 <= y < h:
                            zone_br.append(float(br_map[y, x]))
                if zone_br and float(np.mean(zone_br)) >= 1.5:
                    apply_votes = True
            # (no outward override: an outward vote consensus once chased a
            # faint lum-13 shadow band in lum-11 background 33px out on
            # nues_1 — outward moves stay gated by the stay/vote fallback)
            if not apply_votes:
                # fall back to the stay heuristic for ambiguous cases
                if checked > 0 and stay / checked >= 0.4:
                    continue
                if len(votes) < 0.5 * len(ts) or abs(d_off) <= 2:
                    continue
            if abs(d_off) <= 2:
                continue
            d_off = max(float(d_off), max_inward)
            pts[ai] = pts[ai] + normal * d_off
            pts[bi] = pts[bi] + normal * d_off
            continue

        slope = np.diff(f)
        lo, hi = 2, len(slope) - 2
        seg = slope[lo:hi]
        if len(seg) == 0:
            continue
        global_min = float(seg.min())
        if global_min > -0.05:
            continue
        sig = min(-0.05, 0.5 * global_min)
        # candidate descents, gated by VALIDITY:
        # 1. f stays <0.5 everywhere beyond (a true book→bg boundary; false
        #    inner descents like moiver_6's white-spine→dark-cap junction have
        #    f returning to 1.0 beyond).
        # 2. luminance FALLS outward across it (real edges on dark mats are
        #    bright→dark; the scanner-shadow→bg boundary RISES outward —
        #    keuken_17_2's bottom snapped 8px into the shadow strip there).
        # Pick the OUTERMOST valid descent.
        i_steep = None
        for i in range(hi - 1, lo - 1, -1):
            if slope[i] > sig:
                continue
            beyond = f[i + 2:]
            if len(beyond) > 0 and float(beyond.max()) >= 0.5:
                continue
            if gray_img is not None:
                # ±3 probe: the shadow strip below bottoms is only ~7px tall;
                # a ±6 probe jumps over it into the bright sliver and the
                # falling-luminance test passes the false shadow→fabric edge.
                gi_in = max(0, i - 3)
                gi_out = min(len(ds) - 1, i + 3)
                gin = gout = 0.0
                cin = cout = 0
                for t in ts[:: max(1, len(ts) // 12)]:
                    p0 = ea + t * edge_vec
                    for gi, acc in ((gi_in, "in"), (gi_out, "out")):
                        p = p0 + normal * ds[gi]
                        x, y = int(round(p[0])), int(round(p[1]))
                        if 0 <= x < w and 0 <= y < h:
                            if acc == "in":
                                gin += float(gray_img[y, x]); cin += 1
                            else:
                                gout += float(gray_img[y, x]); cout += 1
                if cin and cout and gin / cin <= gout / cout:
                    continue
            i_steep = i
            break
        if i_steep is None:
            continue
        d_off = max(float(ds[i_steep]), max_inward)
        if d_off != 0:
            pts[ai] = pts[ai] + normal * d_off
            pts[bi] = pts[bi] + normal * d_off
    return pts


def expand_for_straddlers(box: np.ndarray, bgd_mask: np.ndarray, gray_img: np.ndarray,
                          bg_gray: float, max_reach: int = 100) -> np.ndarray:
    """Post-refinement 2-D expansion for BRIGHT book structures that STRADDLE
    an edge — spiral binding rings (formes_nues set: wire loops enter the
    cover, tips protrude 20-60px outside any 1-D-profile edge; their f-tail is
    too smeared by rotation/gaps for profile logic). Per edge sample: if book
    content continues from just inside the edge into a run of bright
    (gray > bg+15) bgd pixels outside, record that run's extent. If ≥15% of
    the edge has such straddlers, push the edge to their 95th-percentile tip.
    Dark shadow bands fail the brightness test; dangling straps cover <15% of
    their edge and fail support."""
    h, w = bgd_mask.shape[:2]
    pts = box.astype(np.float32).copy()
    center = pts.mean(axis=0)
    for ai, bi in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        ea, eb = pts[ai], pts[bi]
        edge_vec = eb - ea
        L = float(np.linalg.norm(edge_vec))
        if L < 1e-6:
            continue
        normal = np.array([-edge_vec[1], edge_vec[0]], dtype=np.float32) / L
        midpoint = (ea + eb) / 2
        if np.dot(normal, midpoint - center) < 0:
            normal = -normal
        n = max(30, int(L / 8))
        t_list = []
        e_list = []
        supported = 0
        total = 0
        for t in np.linspace(0.02, 0.98, n):
            p0 = ea + t * edge_vec
            # inner anchor: book content just inside this edge point
            ix, iy = int(round(p0[0] - normal[0] * 2)), int(round(p0[1] - normal[1] * 2))
            if not (0 <= ix < w and 0 <= iy < h):
                continue
            total += 1
            # outer run: bright book pixels with ≤4px gap tolerance
            last_hit = 0
            gap = 0
            if bgd_mask[iy, ix] > 0:
                for d in range(1, max_reach + 1):
                    x, y = int(round(p0[0] + normal[0] * d)), int(round(p0[1] + normal[1] * d))
                    if not (0 <= x < w and 0 <= y < h):
                        break
                    if bgd_mask[y, x] > 0 and gray_img[y, x] > bg_gray + 15:
                        last_hit = d
                        gap = 0
                    else:
                        gap += 1
                        if gap > 8:
                            break
            if last_hit >= 4:
                supported += 1
            t_list.append(t)
            e_list.append(float(last_hit))
        if total >= 10 and supported / total >= 0.15 and t_list:
            # LINEAR FIT of extent over t, not a parallel shift: rings that
            # span only part of the edge (nues_3: top third) would otherwise
            # push the whole edge out and float the ring-less remainder ~9px
            # off the book. Zero-extents pull the fit down where nothing
            # straddles. Each vertex moves by the fitted extent at its end.
            tt = np.array(t_list)
            ee = np.array(e_list)
            A = np.vstack([np.ones_like(tt), tt]).T
            coef, *_ = np.linalg.lstsq(A, ee, rcond=None)
            a0, b1 = float(coef[0]), float(coef[1])
            # margin = p80 of positive residuals (≥6): ring-tip apexes scatter
            # far above the least-squares line; a fixed +6 cut tips by 8-15px
            resid = ee - (a0 + b1 * tt)
            pos = resid[resid > 0]
            margin = max(6.0, float(np.percentile(pos, 80)) if len(pos) else 6.0)
            e_at = lambda t: max(0.0, min(max_reach, a0 + b1 * t + margin))
            ea_shift = e_at(0.0)
            eb_shift = e_at(1.0)
            if max(ea_shift, eb_shift) >= 4:
                pts[ai] = pts[ai] + normal * ea_shift
                pts[bi] = pts[bi] + normal * eb_shift
    return pts


def detect(img_path: Path):
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    image_area = h * w

    # pre-cropped check: if the image corners disagree wildly, this scan is
    # already cropped to the book (e.g. moiver spine strips) — the answer is
    # the full image. Candidate boxes at image borders break perimeter metrics
    # (ridge probes leave the image → score 0) and got beaten by truncated
    # interior boxes (moiver_3: color box covering only "CI-C" of "CI-CONTRE").
    lab_full = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    corner_meds = [
        np.median(lab_full[:60, :60].reshape(-1, 3), axis=0),
        np.median(lab_full[:60, -60:].reshape(-1, 3), axis=0),
        np.median(lab_full[-60:, :60].reshape(-1, 3), axis=0),
        np.median(lab_full[-60:, -60:].reshape(-1, 3), axis=0),
    ]
    max_corner_d = max(
        float(np.linalg.norm(corner_meds[i] - corner_meds[j]))
        for i in range(4) for j in range(i + 1, 4)
    )
    # extreme strip aspect (>6) only occurs when the scan is already cropped
    # to a spine (moiver_3: 167x3511, corners uniformly white so the corner
    # check alone misses it).
    strip = max(h, w) / max(min(h, w), 1) > 6.0
    if max_corner_d > 25.0 or strip:
        full = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.int32)
        return img, full, "pre-cropped (full image)"

    # precompute gradient magnitude and bg-distance mask (reused for all candidates)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)
    gmag = np.sqrt(gx * gx + gy * gy)
    gmag = cv2.GaussianBlur(gmag, (5, 5), 0)
    bgd_mask = ml_extend.bg_distance_mask(img, thresh=8.0)
    # Shadow removal: a cast shadow is achromatic darkening — L drops slightly,
    # chroma (a/b) stays at bg values. It registers as dist>8 ("book") and fools
    # every perimeter metric (moreno set: ml box snapped to the shadow's outer
    # ridge 55px off the real edge and still scored best). Treat shadow as bg.
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg = ml_extend.bg_lab(img).astype(np.float32)
    dL = bg[0] - lab[:, :, 0]          # positive = darker than bg
    da = np.abs(lab[:, :, 1] - bg[1])
    db = np.abs(lab[:, :, 2] - bg[2])
    # window dL<12: true cast shadows are a SLIGHT darkening (moreno: ~5).
    # Dark book bands (kanko hardcover: ~20 below bg) must stay classified
    # as book or the score rewards rects that cut them off.
    shadow = (dL > 3) & (dL < 12) & (da < 6) & (db < 6)
    bgd_mask[shadow] = 0
    # ignore image-border artifacts when counting book content
    bm = max(40, min(h, w) // 30)
    bgd_clean = np.zeros_like(bgd_mask)
    bgd_clean[bm:-bm, bm:-bm] = bgd_mask[bm:-bm, bm:-bm]
    bgd_total = int(cv2.countNonZero(bgd_clean))

    candidates = []

    _, ml_box = ml_extend.detect(img_path)
    if ml_box is not None:
        candidates.append(("ml", ml_box.astype(np.float32)))

    _, cd_box, _ = color_detect.detect(img_path)
    if cd_box is not None:
        candidates.append(("color", cd_box.astype(np.float32)))

    _, he_box, _, _, _ = heuristic.detect(img_path)
    if he_box is not None:
        candidates.append(("heur", he_box.astype(np.float32)))

    if not candidates:
        return img, None, "all failed"

    # filter out malformed. Upper bound is loose (0.995) because pre-cropped
    # strip images (e.g. spine-only scans) legitimately fill the whole image.
    valid = []
    for name, box in candidates:
        area = rect_area(box)
        pct = area / image_area
        if pct < 0.005 or pct > 0.995:
            continue
        valid.append((name, box, area))
    if not valid:
        # fallback to first available
        name, box = candidates[0]
        return img, box.astype(np.int32), f"{name} (fallback)"

    # build reference book mask = union of all valid candidate rects.
    # Any pixel that ANY detector thinks is book counts as book content
    # that the chosen rect should ideally cover.
    ref_mask = np.zeros((h, w), np.uint8)
    for _, box, _ in valid:
        cv2.fillPoly(ref_mask, [box.astype(np.int32)], 255)
    ref_total = int(cv2.countNonZero(ref_mask))

    # global low-contrast check: does the bg-distance mask see the book at all
    # inside the candidate-union region? If not (dark book on dark bg), band
    # scoring is blind — use ridge-based scoring and PURE argmax (no heur
    # preference: low-contrast images are exactly where v26's heuristic was
    # weak, e.g. the tulsa set).
    # Threshold 0.55: tulsa_1_1's pasted photo plate alone is ~35% of the
    # union — at 0.30 the image passed as "visible" and band scoring crowned
    # the photo-plate box. A book must be MOSTLY visible for band scoring.
    visible_global = cv2.countNonZero(cv2.bitwise_and(bgd_clean, ref_mask)) / max(ref_total, 1)
    low_contrast = visible_global < 0.55

    # In low-contrast images, drop the color candidate entirely: color
    # separation is by definition absent there, so color_detect locks onto
    # inner high-contrast features (tulsa_1_1: boxed the pasted photo plate,
    # missing the whole black cover).
    if low_contrast:
        valid = [v for v in valid if v[0] != "color"] or valid

    # REFINE EACH CANDIDATE FIRST, THEN SCORE THE REFINED BOX. Scoring raw
    # boxes punished ml on nues_1 for excluding the spiral rings (two per-edge
    # penalties) that the post-selection straddler expansion would have
    # absorbed anyway — the floating heur box won. Selecting on refined boxes
    # judges what each candidate actually BECOMES; if refinement fixes several
    # candidates to the same place, any choice is right.
    br_map = img[:, :, 0].astype(np.float32) - img[:, :, 2].astype(np.float32)
    bg_gray_v = float(np.median(gray[:60, :60]))
    scored = []
    for name, box, area in valid:
        rbox = refine_edges(box, bgd_clean, gmag, low_contrast, gray_img=gray, br_map=br_map)
        if not low_contrast:
            rbox = expand_for_straddlers(rbox, bgd_clean, gray, bg_gray_v)
        s = score_box(rbox, gmag, bgd_clean, ref_mask, ref_total, low_contrast=low_contrast)
        scored.append((s, name, rbox, area))

    # Selection: argmax score, with a SIZE tie-break. After per-candidate
    # refinement, candidates converge almost everywhere — the residual
    # difference between near-tied boxes is exactly the dark cap/band material
    # that bgd is blind to (kanko tops/spine ends, moiver_6's cover cap). When
    # the top scores are within 8%, prefer the LARGER refined box: visible
    # overshoot is already punished by the band score, so a tie means the
    # extra area is invisible-to-bgd book material. (The old heur-preference
    # rule broke these ties wrong — it even kept heur over a higher-scoring
    # ml on kanko_3.)
    scored.sort(key=lambda x: -x[0])
    best = scored[0]
    if len(scored) >= 2:
        top_s = scored[0][0]
        contenders = [c for c in scored if top_s > 0 and c[0] >= 0.92 * top_s]
        if len(contenders) >= 2:
            best = max(contenders, key=lambda c: rect_area(c[2]))
    best_score, best_name, best_box, best_area = best
    others = ", ".join(f"{n}={s:.3f}" for s, n, _, _ in scored if n != best_name)
    note = f"{best_name} (score={best_score:.3f}; {others})" if others else f"{best_name} (score={best_score:.3f})"
    # best_box is already refined (candidates were refined before scoring)
    return img, best_box.astype(np.int32), note


def main():
    img_path = Path(sys.argv[1])
    name = img_path.stem
    img, box, note = detect(img_path)
    print(f"{name}: {note}")
    if box is None:
        return
    PROJECT = Path("/Users/murat/git/private/bookcoverfixer")
    v26 = cv2.imread(str(PROJECT / "previews_v26" / f"{name}_preview.jpg"))
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
    cv2.putText(out, "unified (CYAN)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    out_path = PROJECT / "overlay" / f"{name}_unified.jpg"
    out_path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(out_path), out, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
