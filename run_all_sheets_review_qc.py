#!/usr/bin/env python3
"""Run the current QC pipeline on every raw scan, then emit human-review artifacts.

This is deliberately a PRE-QC pass. It automates measurement and produces per-sheet
raw/spot/trait/organ outputs plus mask-first symmetry-axis overlays. A human still
accepts, corrects, or excludes each sheet before results are treated as reviewed.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

import measure_guides as base
import measure_guides_symmetry_axis as symmetry

ISLAND_FOLDERS = ("oshima", "toshima", "niijima", "shikinejima", "kozushima")


def _write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Could not encode {path}")
    encoded.tofile(str(path))


def _extract_corolla_contours(overlay: np.ndarray) -> list[np.ndarray]:
    """Extract accepted green corolla outlines from the current QC overlay."""
    b, g, r = cv2.split(overlay)
    green = ((g > 165) & (r < 135) & (b < 150)).astype(np.uint8) * 255
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return [c for c in contours if cv2.contourArea(c) > 1000]


def _sort_contours(contours: list[np.ndarray]) -> list[np.ndarray]:
    rows = []
    for contour in contours:
        m = cv2.moments(contour)
        if m["m00"] <= 0:
            continue
        rows.append((m["m10"] / m["m00"], m["m01"] / m["m00"], contour))
    # Reading order after canonical ruler-at-top normalization.
    rows.sort(key=lambda item: (round(item[1] / 220.0), item[0]))
    return [item[2] for item in rows]


def write_symmetry_qc(raw_path: Path, overlay_path: Path, out_dir: Path) -> tuple[int, float]:
    raw = base.load_bgr(str(raw_path))
    overlay_data = np.fromfile(str(overlay_path), dtype=np.uint8)
    overlay = cv2.imdecode(overlay_data, cv2.IMREAD_COLOR)
    if raw is None or overlay is None:
        raise RuntimeError(f"Could not load symmetry inputs for {raw_path}")

    contours = _sort_contours(_extract_corolla_contours(overlay))
    if not contours:
        raise RuntimeError(f"No reviewed corolla contours found in {overlay_path}")

    rh, rw = raw.shape[:2]
    oh, ow = overlay.shape[:2]
    sx, sy = rw / ow, rh / oh

    raw_qc = raw.copy()
    mask_qc = np.full_like(raw, 255)
    axis_rows: list[dict[str, object]] = []

    for corolla_id, contour in enumerate(contours, start=1):
        scaled = contour.astype(np.float32).copy()
        scaled[:, 0, 0] *= sx
        scaled[:, 0, 1] *= sy
        scaled = np.rint(scaled).astype(np.int32)

        mask = np.zeros((rh, rw), dtype=np.uint8)
        cv2.drawContours(mask, [scaled], -1, 1, -1)
        axis = symmetry.estimate_symmetry_axis(mask)

        bx, by = map(int, map(round, axis.base_xy))
        tx, ty = map(int, map(round, axis.tip_xy))
        cv2.drawContours(mask_qc, [scaled], -1, (0, 180, 0), 3)
        mask_qc[mask > 0] = (238, 238, 228)
        cv2.drawContours(mask_qc, [scaled], -1, (0, 180, 0), 3)

        for canvas in (raw_qc, mask_qc):
            cv2.circle(canvas, (bx, by), 10, (0, 0, 255), -1)
            cv2.arrowedLine(canvas, (bx, by), (tx, ty), (0, 0, 220), 4, cv2.LINE_AA, tipLength=0.035)
            cv2.putText(
                canvas,
                f"C{corolla_id} sym={axis.score_iou:.3f}",
                (bx + 8, max(24, by - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (25, 25, 25),
                2,
                cv2.LINE_AA,
            )

        axis_rows.append(
            {
                "corolla_id": corolla_id,
                "symmetry_iou": axis.score_iou,
                "axis_angle_deg_from_x": axis.angle_deg_from_x,
                "axis_offset_px": axis.offset_px,
                "base_x": axis.base_xy[0],
                "base_y": axis.base_xy[1],
                "tip_x": axis.tip_xy[0],
                "tip_y": axis.tip_xy[1],
                "human_axis_review": "PENDING",
                "human_fold_state": "PENDING",
                "human_corolla_mask_review": "PENDING",
            }
        )

    sym_dir = out_dir / "symmetry_qc"
    _write_png(sym_dir / "raw_axis_qc.png", raw_qc)
    _write_png(sym_dir / "mask_axis_qc.png", mask_qc)
    with (sym_dir / "symmetry_axes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(axis_rows[0]))
        writer.writeheader()
        writer.writerows(axis_rows)

    return len(axis_rows), float(np.mean([float(row["symmetry_iou"]) for row in axis_rows]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("shimahotarubukuro"))
    parser.add_argument("--out-dir", type=Path, default=Path("results_all_review"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    failures: list[str] = []

    raw_paths: list[tuple[str, Path]] = []
    for folder in ISLAND_FOLDERS:
        island_dir = args.data_root / folder
        for path in sorted(island_dir.glob("*.jpg")):
            raw_paths.append((folder, path))
        for path in sorted(island_dir.glob("*.jpeg")):
            raw_paths.append((folder, path))

    if not raw_paths:
        raise SystemExit(f"No raw scans found under {args.data_root}")

    for folder, raw_path in raw_paths:
        sheet_out = args.out_dir / "sheets" / folder / raw_path.stem
        sheet_out.mkdir(parents=True, exist_ok=True)
        raw_copy = sheet_out / "raw" / raw_path.name
        raw_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(raw_path, raw_copy)

        command = [
            sys.executable,
            "qc_single_sheet.py",
            "--image",
            str(raw_path),
            "--folder",
            folder,
            "--out-dir",
            str(sheet_out),
        ]
        completed = subprocess.run(command, text=True, capture_output=True)
        (sheet_out / "pipeline_stdout.txt").write_text(completed.stdout, encoding="utf-8")
        (sheet_out / "pipeline_stderr.txt").write_text(completed.stderr, encoding="utf-8")

        island_name, _ = base.ISLANDS.get(folder, (folder, ""))
        overlay_path = sheet_out / "overlays" / f"{island_name}_{raw_path.stem}.png"
        status = "success" if completed.returncode == 0 else "failed"
        n_corollas = ""
        mean_symmetry = ""
        if completed.returncode == 0:
            try:
                n_corollas, mean_symmetry = write_symmetry_qc(raw_path, overlay_path, sheet_out)
            except Exception as exc:  # keep the all-sheet run going, but surface the failure
                status = "symmetry_failed"
                failures.append(f"{folder}/{raw_path.name}: symmetry: {exc}")
        else:
            failures.append(f"{folder}/{raw_path.name}: qc_single_sheet exit {completed.returncode}")

        manifest.append(
            {
                "island_folder": folder,
                "sheet": raw_path.stem,
                "raw_path": str(raw_path),
                "status": status,
                "n_corollas_for_symmetry_qc": n_corollas,
                "mean_symmetry_iou": mean_symmetry,
                "human_sheet_review": "PENDING",
            }
        )
        print(f"[{status}] {folder}/{raw_path.name}")

    with (args.out_dir / "review_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)

    (args.out_dir / "HUMAN_REVIEW_REQUIRED.txt").write_text(
        "AUTO PRE-QC ONLY. Review every sheet against raw_axis_qc.png, mask_axis_qc.png, "
        "spot overlay, trait overlay, organ overlay/CSV, and raw scan before accepting results.\n",
        encoding="utf-8",
    )
    if failures:
        (args.out_dir / "FAILURES.txt").write_text("\n".join(failures) + "\n", encoding="utf-8")
        raise SystemExit(f"{len(failures)} sheet(s) failed; see FAILURES.txt")

    print(f"Completed {len(manifest)} raw sheets -> {args.out_dir}")


if __name__ == "__main__":
    main()
