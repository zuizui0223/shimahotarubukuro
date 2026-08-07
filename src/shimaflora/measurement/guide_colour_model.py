"""Unsupervised colour model for nectar-guide pixels.

The reviewed corolla ROI is unchanged.  This module only classifies pixels *inside*
that ROI.  A global three-component Gaussian mixture is fitted in CIELAB chromatic
coordinates (a*, b*) using an equal-sized random sample from every corolla.  This
lets the data separate pale petal tissue, yellow/brown oxidation and purple/magenta
guide tissue without hand-painting hundreds of tiny spot boundaries.

Guide pixels use a conservative posterior-majority rule: P(guide) >= 0.5.  Coverage
therefore has no hand-tuned RGB/HSV cut-off, while pixels that are ambiguous between
guide and non-guide remain excluded.  The fit is deterministic for the committed
inputs and fixed random seed.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture

import measure_guides as base
import remeasure_medial as rm
import shimask_input
from run_all_shimask_confirmed import find_raw

RESULTS = Path("results_shimask_all")
MODEL_PATH = RESULTS / "guide_gmm_model.npz"
COMPONENT_PATH = RESULTS / "guide_gmm_components.csv"

N_COMPONENTS = 3
SAMPLE_PER_COROLLA = 900
RANDOM_SEED = 20260807
POSTERIOR_CUTOFF = 0.50
N_INIT = 6
REG_COVAR = 0.35

_MODEL_CACHE: dict[str, np.ndarray | int] | None = None


def all_sheets() -> list[str]:
    return sorted(
        p.stem for p in Path("shimask").iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )


def iter_corollas(sheet: str):
    """Yield (corolla_id, raw scan, reviewed corolla piece) for one sheet."""
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


def crop_piece(raw: np.ndarray, piece: np.ndarray):
    ys, xs = np.where(piece)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sub = raw[y0:y1, x0:x1]
    roi = piece[y0:y1, x0:x1] > 0
    return sub, roi, (int(x0), int(y0))


def lab_features(sub_bgr: np.ndarray, roi: np.ndarray) -> np.ndarray:
    """CIELAB chromatic coordinates (a*, b*) for pixels inside ``roi``.

    L* is deliberately omitted: scanner brightness, folds and shading should not
    define the guide class.  OpenCV stores a and b with a +128 offset, removed here.
    """
    lab = cv2.cvtColor(sub_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    a_star = lab[:, :, 1][roi] - 128.0
    b_star = lab[:, :, 2][roi] - 128.0
    return np.column_stack([a_star, b_star])


def _roles(means: np.ndarray) -> tuple[int, int, int]:
    """Identify guide, oxidation and petal components from their Lab centres."""
    magenta_score = means[:, 0] - means[:, 1]
    guide = int(np.argmax(magenta_score))
    remaining = [i for i in range(len(means)) if i != guide]
    yellow_red_score = means[:, 0] + means[:, 1]
    oxidation = int(max(remaining, key=lambda i: yellow_red_score[i]))
    petal = int(next(i for i in remaining if i != oxidation))

    # Fail loudly if a future dataset no longer has the colour geometry observed in
    # this specimen series rather than silently relabelling unrelated components.
    if magenta_score[guide] <= 0:
        raise RuntimeError("GMM has no distinct magenta/purple component")
    if means[oxidation, 1] <= means[guide, 1]:
        raise RuntimeError("GMM oxidation component is not more yellow than guide")
    if means[guide, 0] <= means[petal, 0]:
        raise RuntimeError("GMM guide component is not redder than petal tissue")
    return guide, oxidation, petal


def _write_component_table(params: dict[str, np.ndarray | int]) -> None:
    means = np.asarray(params["means"], float)
    weights = np.asarray(params["weights"], float)
    guide = int(params["guide_idx"])
    oxidation = int(params["oxidation_idx"])
    petal = int(params["petal_idx"])
    role = {guide: "guide", oxidation: "oxidation", petal: "petal"}
    rows = []
    for i, (mean, weight) in enumerate(zip(means, weights)):
        a, b = map(float, mean)
        rows.append({
            "component": i,
            "role": role[i],
            "weight": round(float(weight), 6),
            "mean_a_star": round(a, 4),
            "mean_b_star": round(b, 4),
            "chroma": round(math.hypot(a, b), 4),
            "hue_deg": round(math.degrees(math.atan2(b, a)), 3),
            "magenta_score_a_minus_b": round(a - b, 4),
            "yellow_red_score_a_plus_b": round(a + b, 4),
        })
    RESULTS.mkdir(parents=True, exist_ok=True)
    with COMPONENT_PATH.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fit_global_model() -> dict[str, np.ndarray | int]:
    """Fit the deterministic global model with equal sampling weight per corolla."""
    rng = np.random.RandomState(RANDOM_SEED)
    samples: list[np.ndarray] = []
    n_corollas = 0
    for sheet in all_sheets():
        for _cid, raw, piece in iter_corollas(sheet):
            sub, roi, _origin = crop_piece(raw, piece)
            features = lab_features(sub, roi)
            if len(features) > SAMPLE_PER_COROLLA:
                features = features[
                    rng.choice(len(features), SAMPLE_PER_COROLLA, replace=False)
                ]
            samples.append(features)
            n_corollas += 1
    training = np.vstack(samples)
    model = GaussianMixture(
        n_components=N_COMPONENTS,
        covariance_type="full",
        n_init=N_INIT,
        max_iter=400,
        reg_covar=REG_COVAR,
        random_state=RANDOM_SEED,
    ).fit(training)
    guide, oxidation, petal = _roles(model.means_)
    params: dict[str, np.ndarray | int] = {
        "weights": model.weights_.astype(np.float64),
        "means": model.means_.astype(np.float64),
        "covariances": model.covariances_.astype(np.float64),
        "guide_idx": guide,
        "oxidation_idx": oxidation,
        "petal_idx": petal,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        MODEL_PATH,
        weights=params["weights"],
        means=params["means"],
        covariances=params["covariances"],
        guide_idx=np.array(guide, dtype=np.int16),
        oxidation_idx=np.array(oxidation, dtype=np.int16),
        petal_idx=np.array(petal, dtype=np.int16),
    )
    _write_component_table(params)
    print(
        f"fitted guide GMM on {len(training):,} sampled pixels from "
        f"{n_corollas} corollas; guide={guide}, oxidation={oxidation}, petal={petal}",
        flush=True,
    )
    global _MODEL_CACHE
    _MODEL_CACHE = params
    return params


def load_model() -> dict[str, np.ndarray | int]:
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE
    if not MODEL_PATH.exists():
        return fit_global_model()
    data = np.load(MODEL_PATH)
    params: dict[str, np.ndarray | int] = {
        "weights": data["weights"].astype(np.float64),
        "means": data["means"].astype(np.float64),
        "covariances": data["covariances"].astype(np.float64),
        "guide_idx": int(data["guide_idx"]),
        "oxidation_idx": int(data["oxidation_idx"]),
        "petal_idx": int(data["petal_idx"]),
    }
    _MODEL_CACHE = params
    return params


def posterior(features: np.ndarray, params: dict[str, np.ndarray | int]) -> np.ndarray:
    """Posterior component probabilities from saved full-covariance GMM parameters."""
    weights = np.asarray(params["weights"], float)
    means = np.asarray(params["means"], float)
    covariances = np.asarray(params["covariances"], float)
    d = features.shape[1]
    log_prob = np.empty((len(features), len(weights)), dtype=np.float64)
    for i, (weight, mean, covariance) in enumerate(zip(weights, means, covariances)):
        sign, logdet = np.linalg.slogdet(covariance)
        if sign <= 0:
            raise RuntimeError(f"non-positive GMM covariance determinant for component {i}")
        inverse = np.linalg.inv(covariance)
        diff = features - mean
        mahalanobis = np.einsum("ij,jk,ik->i", diff, inverse, diff)
        log_prob[:, i] = (
            math.log(max(float(weight), 1e-300))
            - 0.5 * (d * math.log(2.0 * math.pi) + logdet + mahalanobis)
        )
    normalizer = logsumexp(log_prob, axis=1, keepdims=True)
    return np.exp(log_prob - normalizer)


def segment_piece(
    raw: np.ndarray,
    piece: np.ndarray,
    params: dict[str, np.ndarray | int] | None = None,
) -> tuple[np.ndarray, tuple[int, int], int]:
    """Return hard guide mask, bbox origin and reviewed ROI size.

    A pixel is guide only when the guide component alone has posterior probability
    >= 0.5, i.e. it is more likely to be guide than all non-guide components combined.
    No morphological opening is applied because genuine guide marks are often tiny.
    """
    if params is None:
        params = load_model()
    sub, roi, origin = crop_piece(raw, piece)
    features = lab_features(sub, roi)
    probabilities = posterior(features, params)
    p_guide = probabilities[:, int(params["guide_idx"])]
    guide = np.zeros(roi.shape, dtype=np.uint8)
    guide[roi] = (p_guide >= POSTERIOR_CUTOFF).astype(np.uint8)
    return guide, origin, int(roi.sum())
