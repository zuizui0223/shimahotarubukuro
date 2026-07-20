#!/usr/bin/env python3
"""Spatial structure of the nectar-guide spots: are they really a guide?

A nectar guide should not be scattered at random over the corolla; it should point
a pollinator to the reward. Two directional patterns are expected and tested here,
each against a matched random (complete-spatial-randomness) null drawn from the same
corolla ROI:

1. Basal concentration (longitudinal). Guides pack toward the tube base, where the
   nectar sits. Measured as the fraction of guide pixels in the proximal third of the
   corolla length and the mean along-length position (0 = base, 1 = lobe tips),
   compared with random ROI pixels.

2. Petal-midline concentration (transverse). Within each along-length band the guide
   sits nearer the lobe midline than the lobe margins - the midribs of the five
   petals, which would align a bumblebee's approach along each petal axis. The
   corolla's medial-axis skeleton branches toward each lobe tip, so distance to that
   skeleton (normalised to 0 on the midline, ->1 at the edge) is the midline score.
   Comparing guide vs random WITHIN along-length bands removes the basal effect, so
   this isolates the transverse pattern; it is reported for the whole corolla and for
   the distal (lobed) half, where the petals actually fan out.

Writes per-corolla metrics to results_shimask_all/guide_spatial.csv, pooled
guide-vs-null distributions and canonical density maps to
results_shimask_all/guide_spatial.npz, and prints pooled significance tests.
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
from pollination_traits import upright, guide_and_color
from run_all_shimask_confirmed import find_raw
from plot_island_traits import ORDER, island_of

MIN_GUIDE_PX = 200      # spatial stats need a real spot cloud, not a fleck
CANON = (160, 120)      # canonical corolla frame (H, W) for the density map
N_RANDOM = 5000


def norm_dist_to_skeleton(R: np.ndarray):
    """0 on the midline skeleton, ->1 at the petal edge, per ROI pixel."""
    skel = skeletonize(R)
    dt_skel = ndimage.distance_transform_edt(~skel)
    dt_bnd = ndimage.distance_transform_edt(R)
    return dt_skel / (dt_skel + dt_bnd + 1e-6)


def corolla_frames(sheet: str):
    """Yield (corolla_id, R, G) upright ROI and guide masks cropped to the ROI bbox."""
    _, rp = find_raw(sheet, Path("shimahotarubukuro"))
    raw = base.load_bgr(str(rp))
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
            m = rm.medial_axis(mask_local)
            rot, M = upright(solid, float(m["angle_deg"]))
            spot, (bx0, by0), _c, _s = guide_and_color(raw, piece)
            gloc = np.zeros_like(mask_local)
            if int(spot.sum()) > 0:
                gy, gx = np.where(spot)
                yy, xx = gy + (by0 - y0), gx + (bx0 - x0)
                ok = (yy >= 0) & (yy < gloc.shape[0]) & (xx >= 0) & (xx < gloc.shape[1])
                gloc[yy[ok], xx[ok]] = 1
            h, w = mask_local.shape
            pad = int(max(h, w))
            big = cv2.copyMakeBorder(gloc, pad, pad, pad, pad, cv2.BORDER_CONSTANT, 0)
            grot = cv2.warpAffine(big, M, (big.shape[1], big.shape[0]), flags=cv2.INTER_NEAREST)
            rys, rxs = np.where(rot > 0)
            ya, yb, xa, xb = rys.min(), rys.max() + 1, rxs.min(), rxs.max() + 1
            R = rot[ya:yb, xa:xb] > 0
            G = (grot[ya:yb, xa:xb] > 0) & R
            yield corolla_id, R, G


def strat_ratio(nd, gpos, rpos, gmask, rmask, lo=0.0, hi=1.0, nb=8):
    """Random/guide ratio of midline score, averaged within along-length bands."""
    ratios, weights = [], []
    for b in range(nb):
        blo, bhi = lo + (hi - lo) * b / nb, lo + (hi - lo) * (b + 1) / nb
        gm = gmask & (gpos >= blo) & (gpos < bhi)
        rmk = rmask & (rpos >= blo) & (rpos < bhi)
        if gm.sum() < 20 or rmk.sum() < 20:
            continue
        ratios.append(nd["r"][rmk].mean() / (nd["g"][gm].mean() + 1e-6))
        weights.append(gm.sum())
    return float(np.average(ratios, weights=weights)) if ratios else float("nan")


def main() -> None:
    rng = np.random.RandomState(0)
    sheets = sorted(p.stem for p in Path("shimask").iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    rows = []
    pooled = {"g_pos": [], "r_pos": [], "g_mid": [], "r_mid": [],
              "g_pos_distal": [], "r_pos_distal": [], "g_mid_distal": [], "r_mid_distal": []}
    canon = {i: np.zeros(CANON) for i in ORDER}
    canon_cov = {i: np.zeros(CANON) for i in ORDER}
    canon_all = np.zeros(CANON)
    canon_all_cov = np.zeros(CANON)

    for sheet in sheets:
        island = island_of(sheet)
        n_ok = 0
        for corolla_id, R, G in corolla_frames(sheet):
            if int(G.sum()) < MIN_GUIDE_PX:
                continue
            n_ok += 1
            nd_full = norm_dist_to_skeleton(R)
            H = R.shape[0]
            gpix = np.argwhere(G)
            rpts = np.argwhere(R)
            sel = rpts[rng.choice(len(rpts), min(N_RANDOM, len(rpts)), replace=False)]
            gpos = gpix[:, 0] / H
            rpos = sel[:, 0] / H
            nd = {"g": nd_full[gpix[:, 0], gpix[:, 1]], "r": nd_full[sel[:, 0], sel[:, 1]]}
            gmask = np.ones(len(gpos), bool)
            rmask = np.ones(len(rpos), bool)

            basal_frac = float((gpos < 0.33).mean())
            mean_pos = float(gpos.mean())
            strat_all = strat_ratio(nd, gpos, rpos, gmask, rmask)
            strat_distal = strat_ratio(nd, gpos, rpos, gpos >= 0.5, rpos >= 0.5, lo=0.5, hi=1.0, nb=4)
            # per-corolla midline significance in the distal half (rank test)
            gd, rd = gpos >= 0.5, rpos >= 0.5
            if gd.sum() > 20 and rd.sum() > 20:
                _, pmid = stats.mannwhitneyu(nd["g"][gd], nd["r"][rd], alternative="less")
            else:
                pmid = float("nan")

            rows.append({
                "sheet": sheet, "corolla_id": corolla_id, "island": island,
                "n_guide_px": int(G.sum()),
                "basal_frac_prox_third": round(basal_frac, 3),
                "rand_basal_frac": round(float((rpos < 0.33).mean()), 3),
                "mean_along_length_pos": round(mean_pos, 3),
                "midline_ratio_all": round(strat_all, 3) if strat_all == strat_all else "",
                "midline_ratio_distal": round(strat_distal, 3) if strat_distal == strat_distal else "",
                "distal_midline_p": f"{pmid:.2e}" if pmid == pmid else "",
            })

            pooled["g_pos"] += list(gpos)
            pooled["r_pos"] += list(rpos)
            pooled["g_mid"] += list(nd["g"])
            pooled["r_mid"] += list(nd["r"])
            pooled["g_pos_distal"] += list(gpos[gd])
            pooled["r_pos_distal"] += list(rpos[rd])
            pooled["g_mid_distal"] += list(nd["g"][gd])
            pooled["r_mid_distal"] += list(nd["r"][rd])

            gr = cv2.resize(G.astype(np.float32), (CANON[1], CANON[0]), interpolation=cv2.INTER_AREA)
            rr = cv2.resize(R.astype(np.float32), (CANON[1], CANON[0]), interpolation=cv2.INTER_AREA)
            canon[island] += gr
            canon_cov[island] += rr
            canon_all += gr
            canon_all_cov += rr
        print(f"[{sheet}] {n_ok} guided corollas analysed", flush=True)

    out = Path("results_shimask_all/guide_spatial.csv")
    with out.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}  ({len(rows)} guided corollas)")

    # Pooled significance.
    print("\n=== pooled tests (guide vs random ROI pixels) ===")
    gp, rp_ = np.array(pooled["g_pos"]), np.array(pooled["r_pos"])
    _, p1 = stats.mannwhitneyu(gp, rp_, alternative="less")
    print(f"along-length pos: guide {gp.mean():.3f} vs random {rp_.mean():.3f}  "
          f"(basal, MW p={p1:.2e})")
    gm, rmv = np.array(pooled["g_mid_distal"]), np.array(pooled["r_mid_distal"])
    _, p2 = stats.mannwhitneyu(gm, rmv, alternative="less")
    print(f"distal midline score: guide {gm.mean():.3f} vs random {rmv.mean():.3f}  "
          f"(midline, MW p={p2:.2e})")

    dens_all = np.where(canon_all_cov > 0.05 * canon_all_cov.max(),
                        canon_all / (canon_all_cov + 1e-6), np.nan)
    dens_isl = {i: np.where(canon_cov[i] > 0.05 * (canon_cov[i].max() + 1e-9),
                            canon[i] / (canon_cov[i] + 1e-6), np.nan) for i in ORDER}

    def thin(a, n=150_000):  # subsample pooled pixels so the npz stays small
        a = np.asarray(a)
        return a if a.size <= n else a[rng.choice(a.size, n, replace=False)]

    np.savez_compressed(
        Path("results_shimask_all/guide_spatial.npz"),
        g_pos=thin(gp), r_pos=thin(rp_),
        g_mid=thin(np.array(pooled["g_mid"])), r_mid=thin(np.array(pooled["r_mid"])),
        g_mid_distal=thin(gm), r_mid_distal=thin(rmv),
        dens_all=dens_all, **{f"dens_{i}": dens_isl[i] for i in ORDER})
    print("wrote results_shimask_all/guide_spatial.npz")


if __name__ == "__main__":
    main()
