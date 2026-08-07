#!/usr/bin/env python3
"""Publication overlays: show every measured trait on the raw scan.

For each sheet this draws the measured corolla ROI, length/width constructs, the same
GMM nectar-guide mask used for coverage and spatial analysis, and the reviewed
reproductive-organ stroke.  Numbers come from ``corolla_traits_final.csv``.
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

import guide_traits
import measure_guides as base
import organ_traits
import remeasure_medial as rm
import shimask_input
from run_all_shimask_confirmed import find_raw

MM = float(base.MM_PX)
OUT = Path("results_shimask_all/paper_overlays")

# BGR colours (bright, distinct on the pale scan).
C_ROI = (60, 220, 60)
C_LEN = (40, 200, 255)
C_WID = (255, 150, 40)
C_GUIDE = (200, 60, 220)
C_ORGAN = (60, 60, 235)
C_TXT = (20, 20, 20)


def final_table() -> dict[tuple[str, str], dict]:
    path = Path("results_shimask_all/corolla_traits_final.csv")
    return {
        (row["sheet"], row["corolla_id"]): row
        for row in csv.DictReader(path.open(encoding="utf-8-sig"))
    }


def guide_spot_mask(raw: np.ndarray, piece: np.ndarray) -> np.ndarray:
    """Full-frame mask using the exact classifier behind guide_coverage_pct."""
    local, (x0, y0), _roi_pixels = guide_traits.guide_mask(raw, piece)
    full = np.zeros(raw.shape[:2], np.uint8)
    height, width = local.shape
    full[y0:y0 + height, x0:x0 + width] = local
    return full


def organ_for_corolla(raw, ann, strokes, comps, sheet: str) -> dict[str, dict]:
    """Map corolla_id ("3", "8a", ...) to its reviewed green organ stroke."""
    pieces = organ_traits.build_pieces(comps)
    greens = shimask_input.green_organ_rows(raw, ann, strokes=strokes)
    greens += organ_traits.manual_green_rows(sheet, raw, strokes)
    return organ_traits.associate_organs(sheet, pieces, greens)


def draw_label(img, x, y, lines, colour=C_TXT):
    fs, thickness, pad, line_height = 1.1, 2, 8, 34
    width = max(
        cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, thickness)[0][0]
        for text in lines
    ) + 2 * pad
    height = line_height * len(lines) + pad
    x = int(np.clip(x, 0, img.shape[1] - width - 1))
    y = int(np.clip(y, 0, img.shape[0] - height - 1))
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.78, img, 0.22, 0, img)
    cv2.rectangle(img, (x, y), (x + width, y + height), colour, 1)
    for i, text in enumerate(lines):
        cv2.putText(
            img,
            text,
            (x + pad, y + line_height * (i + 1) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            fs,
            colour,
            thickness,
            cv2.LINE_AA,
        )


def process_sheet(sheet: str, final: dict) -> Path | None:
    _, raw_path = find_raw(sheet, Path("shimahotarubukuro"))
    raw = base.load_bgr(str(raw_path))
    ann = base.load_bgr(str(Path("shimask") / (sheet + ".jpg")))
    strokes = shimask_input.stroke_masks(raw, ann)
    comps = shimask_input.red_corolla_components(raw, ann, strokes=strokes)
    organ = organ_for_corolla(raw, ann, strokes, comps, sheet)
    image = raw.copy()

    for cid0, comp in enumerate(comps):
        full_mask = comp["mask"].astype(np.uint8)
        pieces = rm.split_merged_pair(full_mask)
        suffixes = [""] if len(pieces) == 1 else ["a", "b"]
        for suffix, piece in zip(suffixes, pieces):
            corolla_id = f"{cid0 + 1}{suffix}"
            ys, xs = np.where(piece)
            x0, y0 = int(xs.min()), int(ys.min())
            trimmed = (sheet, corolla_id) in rm.TRIM_TO_PETAL
            mask_local = rm.crop_to_petal(raw, piece) if trimmed else rm.crop_to_mask(piece)
            solid, _ = rm._solid_roi((mask_local > 0).astype(np.uint8))
            measured = rm.medial_axis(mask_local)

            contours, _ = cv2.findContours(
                solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for contour in contours:
                cv2.polylines(
                    image, [contour + [x0, y0]], True, C_ROI, 2, cv2.LINE_AA
                )

            def off(point):
                return (
                    int(round(point[0] + x0)),
                    int(round(point[1] + y0)),
                )

            cv2.arrowedLine(
                image,
                off(measured["base_xy"]),
                off(measured["tip_xy"]),
                C_LEN,
                2,
                cv2.LINE_AA,
                tipLength=0.03,
            )
            cv2.arrowedLine(
                image,
                off(measured["tip_xy"]),
                off(measured["base_xy"]),
                C_LEN,
                2,
                cv2.LINE_AA,
                tipLength=0.03,
            )
            cv2.line(
                image,
                off(measured["w0_xy"]),
                off(measured["w1_xy"]),
                C_WID,
                2,
                cv2.LINE_AA,
            )

            spots = guide_spot_mask(raw, piece)
            guide_contours, _ = cv2.findContours(
                spots, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(image, guide_contours, -1, C_GUIDE, 2, cv2.LINE_AA)

            organ_row = organ.get(corolla_id)
            if organ_row is not None:
                cv2.line(
                    image,
                    (int(organ_row["x1"]), int(organ_row["y1"])),
                    (int(organ_row["x2"]), int(organ_row["y2"])),
                    C_ORGAN,
                    3,
                    cv2.LINE_AA,
                )

            final_row = final.get((sheet, corolla_id))
            lines = [f"C{corolla_id}"]
            if final_row:
                lines.append(
                    f"L {float(final_row['corolla_length_mm']):.1f}  "
                    f"W {float(final_row['corolla_width_obs_mm']):.1f} mm"
                )
                coverage = final_row.get("guide_coverage_pct", "")
                lines.append(
                    f"guide {float(coverage):.1f}%"
                    if coverage not in ("", None) else "guide -"
                )
                if final_row.get("organ_length_mm"):
                    lines[-1] += f"   organ {float(final_row['organ_length_mm']):.1f}mm"
            else:
                lines.append(
                    f"L {measured['length_mm']:.1f}  W {measured['width_mm']:.1f} mm"
                )
            draw_label(image, x0, y0 - 120, lines)

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{sheet}.png"
    cv2.imwrite(str(out), image)
    return out


def main() -> None:
    final = final_table()
    sheets = sorted(
        p.stem for p in Path("shimask").iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    for sheet in sheets:
        out = process_sheet(sheet, final)
        print(f"[{sheet}] wrote {out}", flush=True)

    key = np.full((260, 620, 3), 255, np.uint8)
    items = [
        ("Corolla ROI outline", C_ROI),
        ("Length axis (base->tip)", C_LEN),
        ("Width bar", C_WID),
        ("GMM nectar-guide pixels", C_GUIDE),
        ("Reproductive-organ stroke", C_ORGAN),
    ]
    cv2.putText(
        key, "Overlay legend", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, C_TXT, 2, cv2.LINE_AA
    )
    for i, (text, colour) in enumerate(items):
        y = 85 + i * 34
        cv2.line(key, (24, y - 6), (74, y - 6), colour, 4, cv2.LINE_AA)
        cv2.putText(
            key, text, (90, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_TXT, 2, cv2.LINE_AA
        )
    cv2.imwrite(str(OUT / "_legend.png"), key)
    print(f"wrote {OUT / '_legend.png'}")


if __name__ == "__main__":
    main()
