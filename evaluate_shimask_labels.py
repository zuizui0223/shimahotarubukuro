# -*- coding: utf-8 -*-
"""Inspect hand-labelled shimask images without using them as runtime inputs.

Red pixels are treated as reviewed corolla mask labels and green pixels as
reviewed reproductive-organ labels.  This tool only creates evaluation data and
previews; the extraction pipeline continues to operate on the raw scans.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def load_bgr(path: Path) -> np.ndarray:
    data = np.fromfile(path, np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def label_masks(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract saturated red and green annotations from a labelled preview."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    vivid = (sat >= 105) & (val >= 80)
    red = vivid & ((hue <= 12) | (hue >= 168))
    green = vivid & (hue >= 35) & (hue <= 95)

    kernel = np.ones((3, 3), np.uint8)
    red = cv2.morphologyEx(red.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    green = cv2.morphologyEx(green.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    return red, green


def components(mask: np.ndarray, minimum: int = 8) -> list[dict]:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    rows: list[dict] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < minimum:
            continue
        rows.append(
            {
                "area_px": area,
                "left": int(stats[label, cv2.CC_STAT_LEFT]),
                "top": int(stats[label, cv2.CC_STAT_TOP]),
                "width": int(stats[label, cv2.CC_STAT_WIDTH]),
                "height": int(stats[label, cv2.CC_STAT_HEIGHT]),
                "cx": round(float(centroids[label][0]), 2),
                "cy": round(float(centroids[label][1]), 2),
            }
        )
    return rows


def make_preview(image: np.ndarray, red: np.ndarray, green: np.ndarray) -> np.ndarray:
    preview = image.copy()
    red_contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    green_contours, _ = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(preview, red_contours, -1, (0, 0, 255), 4)
    cv2.drawContours(preview, green_contours, -1, (0, 255, 0), 4)
    return preview


def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", default="shimask")
    parser.add_argument("--out-dir", default="results_v3/shimask_ground_truth")
    args = parser.parse_args()

    label_root = Path(args.labels)
    out = Path(args.out_dir)
    previews = out / "previews"
    masks = out / "masks"
    previews.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)

    files = sorted(
        path for path in label_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not files:
        raise SystemExit(f"No labelled images found under {label_root}")

    summary_rows: list[dict] = []
    component_rows: list[dict] = []
    for path in files:
        image = load_bgr(path)
        red, green = label_masks(image)
        red_parts = components(red)
        green_parts = components(green)
        relative = path.relative_to(label_root)
        safe_stem = "__".join(relative.with_suffix("").parts)

        cv2.imencode(".png", red * 255)[1].tofile(str(masks / f"{safe_stem}__corolla.png"))
        cv2.imencode(".png", green * 255)[1].tofile(str(masks / f"{safe_stem}__organs.png"))
        preview = make_preview(image, red, green)
        scale = min(1.0, 1800.0 / max(preview.shape[:2]))
        preview = cv2.resize(preview, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(
            str(previews / f"{safe_stem}.jpg")
        )

        summary_rows.append(
            {
                "label_file": relative.as_posix(),
                "width_px": image.shape[1],
                "height_px": image.shape[0],
                "red_pixels": int(red.sum()),
                "green_pixels": int(green.sum()),
                "red_components": len(red_parts),
                "green_components": len(green_parts),
                "red_fraction": round(float(red.mean()), 6),
                "green_fraction": round(float(green.mean()), 6),
            }
        )
        for label_name, records in (("corolla", red_parts), ("organ", green_parts)):
            for index, record in enumerate(records, 1):
                component_rows.append(
                    {
                        "label_file": relative.as_posix(),
                        "label_type": label_name,
                        "component_id": index,
                        **record,
                    }
                )

    write_csv(out / "summary.csv", summary_rows)
    write_csv(out / "components.csv", component_rows)
    (out / "FILES.txt").write_text(
        "\n".join(row["label_file"] for row in summary_rows) + "\n",
        encoding="utf-8",
    )
    print(f"label_images={len(summary_rows)} components={len(component_rows)} -> {out}")


if __name__ == "__main__":
    main()
