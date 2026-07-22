#!/usr/bin/env python3
"""Colour-free spatial structure of the area-based nectar-guide mask.

A functional guide should not be scattered at random over the corolla. Two patterns
are tested against matched random pixels drawn from the same reviewed ROI:

1. Basal concentration: the fraction of guide pixels in the proximal third and the
   mean position along the corolla length (0 = base, 1 = lobe tips).
2. Petal-midline concentration: distance to the corolla medial skeleton, compared
   with random pixels within the same along-length bands so the transverse pattern
   is separated from the basal pattern.

Only corollas with at least 200 guide pixels enter these spatial tests. The metrics
are based on mask geometry, not dried-specimen colour values. Writes
``results_shimask_all/guide_spatial.csv`` and ``guide_spatial.npz``.
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage, stats
from skimage.morphology import skeletonize

import measure_guides as base
import shimask_input
import remeasure_medial as rm
from guide_traits import guide_mask
from pollination_traits import upright
from run_all_shimask_confirmed import find_raw
from plot_island_traits import ORDER, island_of

MIN_GUIDE_PX = 200
CANON = (160, 120)
N_RANDOM = 5000


def norm_dist_to_skeleton(roi: np.ndarray):
    """Return 0 on the medial skeleton and values approaching 1 at the ROI edge."""
    skel = skeletonize(roi)
    dt_skel = ndimage.distance_transform_edt(~skel)
    dt_boundary = ndimage.distance_transform_edt(roi)
    return dt_skel / (dt_skel + dt_boundary + 1e-6)


def corolla_frames(sheet: str):
    """Yield (corolla id, upright ROI, upright guide mask)."""
    _, raw_path = find_raw(sheet, Path("shimahotarubukuro"))
    raw = base.load_bgr(str(raw_path))
    ann = base.load_bgr(str(Path("shimask") / (sheet + ".jpg")))
    strokes = shimask_input.stroke_masks(raw, ann)
    comps = shimask_input.red_corolla_components(raw, ann, strokes=strokes)
    for cid0, comp in enumerate(comps):
        pieces = rm.split_merged_pair(comp["mask"].astype(np.uint8))
        suffixes = [""] if len(pieces) == 1 else ["a", "b"]
        for suffix, piece in zip(suffixes, pieces):
            corolla_id = f"{cid0 + 1}{suffix}"
            ys, xs = np.where(piece)
            x0, y0 = int(xs.min()), int(ys.min())
            trimmed = (sheet, corolla_id) in rm.TRIM_TO_PETAL
            mask_local = rm.crop_to_petal(raw, piece) if trimmed else rm.crop_to_mask(piece)
            solid, _ = rm._solid_roi((mask_local > 0).astype(np.uint8))
            measured = rm.medial_axis(mask_local)
            rot, transform = upright(solid, float(measured["angle_deg"]))

            guide, (bx0, by0), _roi_px = guide_mask(raw, piece)
            local_guide = np.zeros_like(mask_local)
            if int(guide.sum()) > 0:
                gy, gx = np.where(guide)
                yy, xx = gy + (by0 - y0), gx + (bx0 - x0)
                ok = (
                    (yy >= 0) & (yy < local_guide.shape[0]) &
                    (xx >= 0) & (xx < local_guide.shape[1])
                )
                local_guide[yy[ok], xx[ok]] = 1

            h, w = mask_local.shape
            pad = int(max(h, w))
            big = cv2.copyMakeBorder(local_guide, pad, pad, pad, pad, cv2.BORDER_CONSTANT, 0)
            guide_rot = cv2.warpAffine(
                big, transform, (big.shape[1], big.shape[0]), flags=cv2.INTER_NEAREST
            )
            rys, rxs = np.where(rot > 0)
            ya, yb, xa, xb = rys.min(), rys.max() + 1, rxs.min(), rxs.max() + 1
            roi = rot[ya:yb, xa:xb] > 0
            guide_roi = (guide_rot[ya:yb, xa:xb] > 0) & roi
            yield corolla_id, roi, guide_roi


def stratified_midline_ratio(
    distances, guide_pos, random_pos, guide_sel, random_sel,
    lo: float = 0.0, hi: float = 1.0, bins: int = 8,
):
    """Random/guide midline-distance ratio averaged within length bands."""
    ratios, weights = [], []
    for index in range(bins):
        band_lo = lo + (hi - lo) * index / bins
        band_hi = lo + (hi - lo) * (index + 1) / bins
        gm = guide_sel & (guide_pos >= band_lo) & (guide_pos < band_hi)
        rm_ = random_sel & (random_pos >= band_lo) & (random_pos < band_hi)
        if gm.sum() < 20 or rm_.sum() < 20:
            continue
        ratios.append(distances["random"][rm_].mean() / (distances["guide"][gm].mean() + 1e-6))
        weights.append(gm.sum())
    return float(np.average(ratios, weights=weights)) if ratios else float("nan")


def main() -> None:
    rng = np.random.RandomState(0)
    sheets = sorted(
        p.stem for p in Path("shimask").iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    rows = []
    pooled = {
        "g_pos": [], "r_pos": [], "g_mid": [], "r_mid": [],
        "g_mid_distal": [], "r_mid_distal": [],
    }
    canon = {island: np.zeros(CANON) for island in ORDER}
    canon_cov = {island: np.zeros(CANON) for island in ORDER}
    canon_all = np.zeros(CANON)
    canon_all_cov = np.zeros(CANON)

    for sheet in sheets:
        island = island_of(sheet)
        n_ok = 0
        for corolla_id, roi, guide in corolla_frames(sheet):
            if int(guide.sum()) < MIN_GUIDE_PX:
                continue
            n_ok += 1
            norm_dist = norm_dist_to_skeleton(roi)
            height = roi.shape[0]
            guide_pixels = np.argwhere(guide)
            roi_pixels = np.argwhere(roi)
            random_pixels = roi_pixels[
                rng.choice(len(roi_pixels), min(N_RANDOM, len(roi_pixels)), replace=False)
            ]
            guide_pos = guide_pixels[:, 0] / height
            random_pos = random_pixels[:, 0] / height
            distances = {
                "guide": norm_dist[guide_pixels[:, 0], guide_pixels[:, 1]],
                "random": norm_dist[random_pixels[:, 0], random_pixels[:, 1]],
            }
            guide_all = np.ones(len(guide_pos), bool)
            random_all = np.ones(len(random_pos), bool)

            basal_frac = float((guide_pos < 0.33).mean())
            mean_pos = float(guide_pos.mean())
            midline_all = stratified_midline_ratio(
                distances, guide_pos, random_pos, guide_all, random_all
            )
            guide_distal = guide_pos >= 0.5
            random_distal = random_pos >= 0.5
            midline_distal = stratified_midline_ratio(
                distances, guide_pos, random_pos, guide_distal, random_distal,
                lo=0.5, hi=1.0, bins=4,
            )
            if guide_distal.sum() > 20 and random_distal.sum() > 20:
                _, midline_p = stats.mannwhitneyu(
                    distances["guide"][guide_distal],
                    distances["random"][random_distal],
                    alternative="less",
                )
            else:
                midline_p = float("nan")

            rows.append({
                "sheet": sheet,
                "corolla_id": corolla_id,
                "island": island,
                "n_guide_px": int(guide.sum()),
                "basal_frac_prox_third": round(basal_frac, 3),
                "rand_basal_frac": round(float((random_pos < 0.33).mean()), 3),
                "mean_along_length_pos": round(mean_pos, 3),
                "midline_ratio_all": round(midline_all, 3) if midline_all == midline_all else "",
                "midline_ratio_distal": (
                    round(midline_distal, 3) if midline_distal == midline_distal else ""
                ),
                "distal_midline_p": f"{midline_p:.2e}" if midline_p == midline_p else "",
            })

            pooled["g_pos"] += list(guide_pos)
            pooled["r_pos"] += list(random_pos)
            pooled["g_mid"] += list(distances["guide"])
            pooled["r_mid"] += list(distances["random"])
            pooled["g_mid_distal"] += list(distances["guide"][guide_distal])
            pooled["r_mid_distal"] += list(distances["random"][random_distal])

            guide_resized = cv2.resize(
                guide.astype(np.float32), (CANON[1], CANON[0]), interpolation=cv2.INTER_AREA
            )
            roi_resized = cv2.resize(
                roi.astype(np.float32), (CANON[1], CANON[0]), interpolation=cv2.INTER_AREA
            )
            canon[island] += guide_resized
            canon_cov[island] += roi_resized
            canon_all += guide_resized
            canon_all_cov += roi_resized
        print(f"[{sheet}] {n_ok} guided corollas analysed", flush=True)

    out = Path("results_shimask_all/guide_spatial.csv")
    with out.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}  ({len(rows)} guided corollas)")

    guide_pos = np.array(pooled["g_pos"])
    random_pos = np.array(pooled["r_pos"])
    _, basal_p = stats.mannwhitneyu(guide_pos, random_pos, alternative="less")
    guide_mid = np.array(pooled["g_mid_distal"])
    random_mid = np.array(pooled["r_mid_distal"])
    _, midline_p = stats.mannwhitneyu(guide_mid, random_mid, alternative="less")
    print("\n=== pooled guide-vs-random tests ===")
    print(
        f"along-length position: guide {guide_pos.mean():.3f} vs random {random_pos.mean():.3f} "
        f"(basal, MW p={basal_p:.2e})"
    )
    print(
        f"distal midline distance: guide {guide_mid.mean():.3f} vs random {random_mid.mean():.3f} "
        f"(midline, MW p={midline_p:.2e})"
    )

    density_all = np.where(
        canon_all_cov > 0.05 * canon_all_cov.max(),
        canon_all / (canon_all_cov + 1e-6),
        np.nan,
    )
    density_island = {
        island: np.where(
            canon_cov[island] > 0.05 * (canon_cov[island].max() + 1e-9),
            canon[island] / (canon_cov[island] + 1e-6),
            np.nan,
        )
        for island in ORDER
    }

    def thin(values, maximum: int = 150_000):
        values = np.asarray(values)
        return values if values.size <= maximum else values[rng.choice(values.size, maximum, replace=False)]

    np.savez_compressed(
        Path("results_shimask_all/guide_spatial.npz"),
        g_pos=thin(guide_pos),
        r_pos=thin(random_pos),
        g_mid=thin(np.array(pooled["g_mid"])),
        r_mid=thin(np.array(pooled["r_mid"])),
        g_mid_distal=thin(guide_mid),
        r_mid_distal=thin(random_mid),
        dens_all=density_all,
        **{f"dens_{island}": density_island[island] for island in ORDER},
    )
    print("wrote results_shimask_all/guide_spatial.npz")


if __name__ == "__main__":
    main()
