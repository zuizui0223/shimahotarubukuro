"""Colour-pattern segmentation and spatial measurements within floral ROIs."""
from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np
from sklearn.mixture import GaussianMixture


def _features(image_bgr: np.ndarray, roi: np.ndarray, channels: tuple[str, ...]) -> np.ndarray:
    lab = cv2.cvtColor(np.asarray(image_bgr), cv2.COLOR_BGR2LAB).astype(np.float64)
    roi = np.asarray(roi) > 0
    mapping = {
        "L": lab[:, :, 0] * (100.0 / 255.0),
        "a": lab[:, :, 1] - 128.0,
        "b": lab[:, :, 2] - 128.0,
    }
    try:
        return np.column_stack([mapping[c][roi] for c in channels])
    except KeyError as exc:
        raise ValueError("channels must contain only 'L', 'a', or 'b'") from exc


@dataclass
class ColorPatternModel:
    """Unsupervised colour-component model for pixels inside floral masks.

    Components are intentionally unlabeled. Biological interpretation (nectar guide,
    bullseye, oxidation, etc.) belongs in a preset or downstream analysis.
    """

    n_components: int = 3
    channels: tuple[str, ...] = ("a", "b")
    sample_per_roi: int = 900
    random_state: int = 0
    reg_covar: float = 0.35

    def __post_init__(self) -> None:
        self.model_: GaussianMixture | None = None

    def fit(self, images: list[np.ndarray], masks: list[np.ndarray]) -> "ColorPatternModel":
        if len(images) != len(masks) or not images:
            raise ValueError("images and masks must have the same non-zero length")
        rng = np.random.RandomState(self.random_state)
        chunks = []
        for image, mask in zip(images, masks):
            x = _features(image, mask, self.channels)
            if len(x) > self.sample_per_roi:
                x = x[rng.choice(len(x), self.sample_per_roi, replace=False)]
            chunks.append(x)
        training = np.vstack(chunks)
        self.model_ = GaussianMixture(
            n_components=self.n_components,
            covariance_type="full",
            n_init=6,
            max_iter=400,
            reg_covar=self.reg_covar,
            random_state=self.random_state,
        ).fit(training)
        return self

    @property
    def component_centres_(self) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("fit the model first")
        return self.model_.means_.copy()

    def posterior(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.model_ is None:
            raise RuntimeError("fit the model first")
        roi = np.asarray(mask) > 0
        x = _features(image, roi, self.channels)
        probs = self.model_.predict_proba(x)
        return probs, roi

    def segment(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        component: int,
        min_posterior: float = 0.5,
    ) -> np.ndarray:
        if not 0 <= component < self.n_components:
            raise ValueError("component index out of range")
        probs, roi = self.posterior(image, mask)
        out = np.zeros(roi.shape, dtype=np.uint8)
        out[roi] = (probs[:, component] >= min_posterior).astype(np.uint8)
        return out


def measure_pattern(pattern_mask: np.ndarray, roi_mask: np.ndarray) -> dict[str, float]:
    """Quantify coverage and geometry of any binary intrafloral pattern."""
    pattern = np.asarray(pattern_mask) > 0
    roi = np.asarray(roi_mask) > 0
    if pattern.shape != roi.shape:
        raise ValueError("pattern_mask and roi_mask must have identical shapes")
    pattern &= roi
    roi_n = int(roi.sum())
    if roi_n == 0:
        raise ValueError("roi_mask is empty")
    n = int(pattern.sum())
    result = {
        "coverage": n / roi_n,
        "coverage_pct": 100.0 * n / roi_n,
        "component_count": 0.0,
        "centroid_x_rel": float("nan"),
        "centroid_y_rel": float("nan"),
        "spatial_dispersion": float("nan"),
    }
    if n == 0:
        return result
    count, _labels = cv2.connectedComponents(pattern.astype(np.uint8), connectivity=8)
    result["component_count"] = float(max(count - 1, 0))
    ry, rx = np.where(roi)
    py, px = np.where(pattern)
    x0, x1 = float(rx.min()), float(rx.max())
    y0, y1 = float(ry.min()), float(ry.max())
    cx, cy = float(px.mean()), float(py.mean())
    result["centroid_x_rel"] = (cx - x0) / max(x1 - x0, 1.0)
    result["centroid_y_rel"] = (cy - y0) / max(y1 - y0, 1.0)
    diag = max(float(np.hypot(x1 - x0, y1 - y0)), 1.0)
    result["spatial_dispersion"] = float(np.mean(np.hypot(px - cx, py - cy)) / diag)
    return result
