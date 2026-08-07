#!/usr/bin/env python3
"""Compare the fixed nectar-guide threshold with an unsupervised colour model.

This experiment changes *only* nectar-guide pixel extraction. Corolla ROIs,
numbering, fold state, size measurements and reproductive-organ measurements are
read exactly as the publication pipeline already defines them.

Why this model
--------------
The guide consists of many small purple marks, so hand-painting every spot boundary
would itself introduce substantial area error. Instead, all reviewed corolla pixels
from all 218 flowers are pooled in CIELAB chromatic space (a*, b*) and a three-
component Gaussian mixture is fitted without manual pixel labels. The three
components are interpreted as neutral/petal tissue, oxidised/brown tissue and
purple/magenta guide tissue from their component centres. The guide component is the
one with the largest magenta opponent score (a* - b*); oxidation is the remaining
component with the largest yellow-red score (a* + b*).

For area, the preferred estimate is *soft*: each pixel contributes its posterior
probability of belonging to the guide component. This avoids an arbitrary hard
boundary around tiny dots. A posterior >= 0.5 mask is produced only for visual/spatial
inspection.

Outputs are written below ``results_shimask_all/guide_segmentation_experiment`` and
do not overwrite the publication CSVs.
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from scipy import stats
from sklearn.mixture import GaussianMixture

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import measure_guides as base
import remeasure_medial as rm
import shimask_input
from plot_island_traits import COLOUR, ORDER, island_of
from run_all_shimask_confirmed import find_raw

OUT = Path("results_shimask_all/guide_segmentation_experiment")
SAMPLE_PER_COROLLA = 900
RANDOM_SEED = 20260807
N_COMPONENTS = 3
HARD_POSTERIOR = 0.50  # visualization/spatial mask only; coverage uses soft probabilities


def old_fixed_mask(raw: np.ndarray, piece: np.ndarray) -> tuple[np.ndarray, tuple[int, int], np.ndarray]:
    """Current publication threshold, retained here only as a comparison baseline."""
    ys, xs = np.where(piece)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sub = raw[y0:y1, x0:x1]
    roi = piece[y0:y1, x0:x1] > 0
    b, g, r = cv2.split(sub.astype(int))
    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    guide = (((r - g) > 18) & ((b - g) > -10) &
             (hsv[:, :, 1] > 60) & (hsv[:, :, 2] < 205) & roi)
    guide = cv2.morphologyEx(
        guide.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)
    ) > 0
    return guide, (int(x0), int(y0)), roi


def lab_features(sub_bgr: np.ndarray, roi: np.ndarray) -> np.ndarray:
    """Return CIELAB chromatic coordinates (a*, b*) for ROI pixels.

    OpenCV stores a and b with +128 offsets. The offset is removed; absolute L* is
    deliberately excluded so scanner brightness/shadow does not define a guide.
    """
    lab = cv2.cvtColor(sub_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    a = lab[:, :, 1][roi] - 128.0
    b = lab[:, :, 2][roi] - 128.0
    return np.column_stack([a, b]).astype(np.float64)


def sheet_records(sheet: str):
    """Yield (corolla_id, raw, piece) using the unchanged reviewed ROI pipeline."""
    _, raw_path = find_raw(sheet, Path("shimahotarubukuro"))
    raw = base.load_bgr(str(raw_path))
    ann = base.load_bgr(str(Path("shimask") / f"{sheet}.jpg"))
    strokes = shimask_input.stroke_masks(raw, ann)
    comps = shimask_input.red_corolla_components(raw, ann, strokes=strokes)
    for cid0, comp in enumerate(comps):
        pieces = rm.split_merged_pair(comp["mask"].astype(np.uint8))
        suffixes = [""] if len(pieces) == 1 else ["a", "b"]
        for suffix, piece in zip(suffixes, pieces):
            yield f"{cid0 + 1}{suffix}", raw, piece


def all_sheets() -> list[str]:
    return sorted(
        p.stem for p in Path("shimask").iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )


def crop_data(raw: np.ndarray, piece: np.ndarray):
    ys, xs = np.where(piece)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sub = raw[y0:y1, x0:x1]
    roi = piece[y0:y1, x0:x1] > 0
    return sub, roi, (int(x0), int(y0))


def fit_global_model(sheets: list[str]) -> tuple[GaussianMixture, int, int, int, np.ndarray]:
    """Fit an equal-flower-weighted three-component GMM in (a*, b*) space."""
    rng = np.random.RandomState(RANDOM_SEED)
    samples = []
    n_corollas = 0
    for sheet in sheets:
        for _cid, raw, piece in sheet_records(sheet):
            sub, roi, _origin = crop_data(raw, piece)
            x = lab_features(sub, roi)
            if len(x) > SAMPLE_PER_COROLLA:
                x = x[rng.choice(len(x), SAMPLE_PER_COROLLA, replace=False)]
            samples.append(x)
            n_corollas += 1
        print(f"[sample] {sheet}", flush=True)
    training = np.vstack(samples)
    model = GaussianMixture(
        n_components=N_COMPONENTS,
        covariance_type="full",
        n_init=6,
        max_iter=400,
        reg_covar=0.35,
        random_state=RANDOM_SEED,
    ).fit(training)

    means = model.means_
    magenta_score = means[:, 0] - means[:, 1]
    guide_idx = int(np.argmax(magenta_score))
    remaining = [i for i in range(N_COMPONENTS) if i != guide_idx]
    yellow_red_score = means[:, 0] + means[:, 1]
    oxidation_idx = int(max(remaining, key=lambda i: yellow_red_score[i]))
    petal_idx = int(next(i for i in remaining if i != oxidation_idx))
    print(
        f"fitted GMM on {len(training):,} pixels from {n_corollas} corollas; "
        f"guide={guide_idx}, oxidation={oxidation_idx}, petal={petal_idx}",
        flush=True,
    )
    return model, guide_idx, oxidation_idx, petal_idx, training


def write_model(model, guide_idx, oxidation_idx, petal_idx, training):
    OUT.mkdir(parents=True, exist_ok=True)
    role = {guide_idx: "guide", oxidation_idx: "oxidation", petal_idx: "petal"}
    rows = []
    for i, (mean, weight) in enumerate(zip(model.means_, model.weights_)):
        a, b = map(float, mean)
        rows.append({
            "component": i,
            "role": role[i],
            "weight": round(float(weight), 5),
            "mean_a_star": round(a, 3),
            "mean_b_star": round(b, 3),
            "chroma": round(math.hypot(a, b), 3),
            "hue_deg": round(math.degrees(math.atan2(b, a)), 2),
            "magenta_score_a_minus_b": round(a - b, 3),
            "yellow_red_score_a_plus_b": round(a + b, 3),
        })
    with (OUT / "gmm_components.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    # Diagnostic chromatic-space sample with component centres.
    rng = np.random.RandomState(RANDOM_SEED)
    sample = training if len(training) <= 30000 else training[rng.choice(len(training), 30000, replace=False)]
    labels = model.predict(sample)
    fig, ax = plt.subplots(figsize=(7.2, 6.3))
    for i in range(N_COMPONENTS):
        q = labels == i
        ax.scatter(sample[q, 0], sample[q, 1], s=2, alpha=0.12, label=f"{i}: {role[i]}")
        ax.scatter(model.means_[i, 0], model.means_[i, 1], s=130, marker="x", linewidth=3)
    ax.set_xlabel("CIELAB a*  (green <- -> red)")
    ax.set_ylabel("CIELAB b*  (blue <- -> yellow)")
    ax.set_title("Unsupervised corolla-pixel colour components")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "gmm_chromatic_space.png", dpi=160)
    plt.close(fig)


def segment(model: GaussianMixture, guide_idx: int, oxidation_idx: int,
            raw: np.ndarray, piece: np.ndarray):
    sub, roi, origin = crop_data(raw, piece)
    features = lab_features(sub, roi)
    posterior = model.predict_proba(features)
    p_guide = posterior[:, guide_idx]
    p_oxid = posterior[:, oxidation_idx]
    guide_prob = np.zeros(roi.shape, np.float32)
    oxid_prob = np.zeros(roi.shape, np.float32)
    guide_prob[roi] = p_guide.astype(np.float32)
    oxid_prob[roi] = p_oxid.astype(np.float32)
    hard = (guide_prob >= HARD_POSTERIOR) & roi
    return sub, roi, origin, guide_prob, oxid_prob, hard


def make_overlay(sub: np.ndarray, mask: np.ndarray, colour_bgr: tuple[int, int, int]) -> np.ndarray:
    image = sub.copy()
    layer = image.copy()
    layer[mask] = colour_bgr
    return cv2.addWeighted(layer, 0.55, image, 0.45, 0)


def main() -> None:
    sheets = all_sheets()
    OUT.mkdir(parents=True, exist_ok=True)
    model, guide_idx, oxidation_idx, petal_idx, training = fit_global_model(sheets)
    write_model(model, guide_idx, oxidation_idx, petal_idx, training)

    rows = []
    island_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for sheet in sheets:
        island = island_of(sheet)
        for cid, raw, piece in sheet_records(sheet):
            old, _old_origin, roi_old = old_fixed_mask(raw, piece)
            sub, roi, _origin, guide_prob, oxid_prob, hard = segment(
                model, guide_idx, oxidation_idx, raw, piece
            )
            assert roi.shape == roi_old.shape
            n_roi = int(roi.sum())
            old_pct = 100.0 * int(old.sum()) / n_roi
            soft_pct = 100.0 * float(guide_prob[roi].sum()) / n_roi
            hard_pct = 100.0 * int(hard.sum()) / n_roi
            oxidation_pct = 100.0 * float(oxid_prob[roi].sum()) / n_roi
            rows.append({
                "island": island,
                "sheet": sheet,
                "corolla_id": cid,
                "old_fixed_pct": round(old_pct, 3),
                "gmm_soft_pct": round(soft_pct, 3),
                "gmm_hard_pct": round(hard_pct, 3),
                "gmm_oxidation_pct": round(oxidation_pct, 3),
                "delta_soft_minus_old": round(soft_pct - old_pct, 3),
                "roi_pixels": n_roi,
            })
            island_values[island]["old"].append(old_pct)
            island_values[island]["soft"].append(soft_pct)
        print(f"[segment] {sheet}", flush=True)

    compare_path = OUT / "coverage_comparison.csv"
    with compare_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    old_all = np.array([r["old_fixed_pct"] for r in rows], float)
    new_all = np.array([r["gmm_soft_pct"] for r in rows], float)
    delta = new_all - old_all
    pearson = stats.pearsonr(old_all, new_all)
    spearman = stats.spearmanr(old_all, new_all)
    summary = [
        ("n_corollas", len(rows)),
        ("old_mean_pct", old_all.mean()),
        ("gmm_soft_mean_pct", new_all.mean()),
        ("mean_delta_pct_points", delta.mean()),
        ("median_abs_delta_pct_points", np.median(np.abs(delta))),
        ("pearson_r", pearson.statistic),
        ("pearson_p", pearson.pvalue),
        ("spearman_rho", spearman.statistic),
        ("spearman_p", spearman.pvalue),
    ]
    for island in ORDER:
        if island not in island_values:
            continue
        summary.append((f"{island}_old_mean_pct", np.mean(island_values[island]["old"])))
        summary.append((f"{island}_gmm_mean_pct", np.mean(island_values[island]["soft"])))
    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh); writer.writerow(["metric", "value"])
        for key, value in summary:
            writer.writerow([key, f"{value:.8g}" if isinstance(value, (float, np.floating)) else value])

    # Overall old-vs-new comparison.
    fig, ax = plt.subplots(figsize=(6.5, 6.2))
    for island in ORDER:
        rr = [r for r in rows if r["island"] == island]
        if not rr:
            continue
        ax.scatter([r["old_fixed_pct"] for r in rr], [r["gmm_soft_pct"] for r in rr],
                   s=24, alpha=0.75, label=island, color=COLOUR[island])
    maximum = max(float(old_all.max()), float(new_all.max())) * 1.04
    ax.plot([0, maximum], [0, maximum], "--", color="#555", lw=1)
    ax.set_xlim(0, maximum); ax.set_ylim(0, maximum)
    ax.set_xlabel("Old fixed-threshold coverage (%)")
    ax.set_ylabel("GMM soft guide coverage (%)")
    ax.set_title(f"Nectar-guide extraction: fixed threshold vs unsupervised GMM\n"
                 f"Spearman rho={spearman.statistic:.3f}, n={len(rows)}")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "old_vs_gmm_scatter.png", dpi=160)
    plt.close(fig)

    # Select the most informative specimens: largest changes plus broad coverage range.
    by_abs_delta = sorted(rows, key=lambda r: abs(float(r["delta_soft_minus_old"])), reverse=True)
    selected = {(r["sheet"], r["corolla_id"]) for r in by_abs_delta[:12]}
    for island in ORDER:
        rr = sorted([r for r in rows if r["island"] == island], key=lambda r: float(r["old_fixed_pct"]))
        if rr:
            for q in (0.2, 0.5, 0.8):
                selected.add((rr[min(int(q * (len(rr) - 1)), len(rr) - 1)]["sheet"],
                              rr[min(int(q * (len(rr) - 1)), len(rr) - 1)]["corolla_id"]))
    selected = list(selected)[:27]

    panels = []
    row_lookup = {(r["sheet"], r["corolla_id"]): r for r in rows}
    for sheet in sheets:
        for cid, raw, piece in sheet_records(sheet):
            if (sheet, cid) not in selected:
                continue
            old, _origin, _roi_old = old_fixed_mask(raw, piece)
            sub, roi, _origin, guide_prob, _oxid_prob, hard = segment(
                model, guide_idx, oxidation_idx, raw, piece
            )
            old_img = make_overlay(sub, old, (50, 40, 230))
            gmm_img = make_overlay(sub, hard, (230, 80, 40))
            panels.append(((sheet, cid), sub, old_img, gmm_img, row_lookup[(sheet, cid)]))

    per_page = 9
    for page0 in range(0, len(panels), per_page):
        page = panels[page0:page0 + per_page]
        fig, axes = plt.subplots(len(page), 3, figsize=(10.2, 2.75 * len(page)))
        axes = np.atleast_2d(axes)
        for row_i, ((sheet, cid), raw_crop, old_img, gmm_img, record) in enumerate(page):
            images = [raw_crop, old_img, gmm_img]
            titles = [
                f"{sheet} C{cid}  raw",
                f"old {record['old_fixed_pct']:.1f}%",
                f"GMM soft {record['gmm_soft_pct']:.1f}% / hard {record['gmm_hard_pct']:.1f}%",
            ]
            for col_i, (image, title) in enumerate(zip(images, titles)):
                axes[row_i, col_i].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                axes[row_i, col_i].set_title(title, fontsize=8.5)
                axes[row_i, col_i].axis("off")
        fig.suptitle("Guide segmentation comparison: raw / fixed threshold / GMM posterior>=0.5",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        fig.savefig(OUT / f"comparison_page_{page0 // per_page + 1}.png", dpi=140)
        plt.close(fig)

    print(f"wrote {compare_path} and diagnostics in {OUT}")


if __name__ == "__main__":
    main()
