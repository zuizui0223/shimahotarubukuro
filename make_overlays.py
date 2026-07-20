#!/usr/bin/env python3
"""Publication overlays: show every measured trait on the raw scan.

For each sheet this draws, on a copy of the raw scan, for every corolla:
- the corolla ROI outline (the measured silhouette),
- the length axis (base->tip) and the width bar at the widest cross-section,
- the detected purple nectar-guide spots,
- the reviewer's green reproductive-organ stroke, and
- a label carrying the final numbers (length / width / guide coverage / organ),

so a reader can see exactly what each reported value corresponds to. Numbers are
read from results_shimask_all/corolla_traits_final.csv (iPhone-registered ROI where
available, hand ROI otherwise); the drawn geometry uses the hand ROI, which lives in
the scan's own coordinates. Writes results_shimask_all/paper_overlays/<sheet>.png.
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

import measure_guides as base
import shimask_input
import remeasure_medial as rm
import organ_traits
from run_all_shimask_confirmed import find_raw

MM = float(base.MM_PX)
OUT = Path("results_shimask_all/paper_overlays")

# BGR colours (bright, distinct on the pale scan).
C_ROI = (60, 220, 60)      # green  - corolla outline
C_LEN = (40, 200, 255)     # amber  - length axis
C_WID = (255, 150, 40)     # blue   - width bar
C_GUIDE = (200, 60, 220)   # magenta- nectar guide spots
C_ORGAN = (60, 60, 235)    # red    - organ stroke
C_TXT = (20, 20, 20)


def final_table() -> dict[tuple[str, str], dict]:
    path = Path("results_shimask_all/corolla_traits_final.csv")
    return {(r["sheet"], r["corolla_id"]): r
            for r in csv.DictReader(path.open(encoding="utf-8-sig"))}


def guide_spot_mask(raw: np.ndarray, piece: np.ndarray) -> np.ndarray:
    """Full-frame binary mask of nectar-guide spots inside the corolla piece."""
    ys, xs = np.where(piece)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sub = raw[y0:y1, x0:x1]
    inside = piece[y0:y1, x0:x1] > 0
    b, g, r = cv2.split(sub.astype(int))
    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    spot = (((r - g) > 18) & ((b - g) > -10) & (hsv[:, :, 1] > 60) & (hsv[:, :, 2] < 205)) & inside
    spot = cv2.morphologyEx(spot.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    full = np.zeros(raw.shape[:2], np.uint8)
    full[y0:y1, x0:x1] = spot
    return full


def organ_for_corolla(raw, ann, strokes, comps, sheet: str) -> dict[str, dict]:
    """Map corolla_id ("3", "8a", ...) -> its green organ stroke.

    Delegates to organ_traits so the drawn organs match organ_traits.csv exactly,
    including the split-pair handling and the manual ORGAN_ASSIGN pins.
    """
    pieces = organ_traits.build_pieces(comps)
    greens = shimask_input.green_organ_rows(raw, ann, strokes=strokes)
    greens += organ_traits.manual_green_rows(sheet, raw, strokes)
    return organ_traits.associate_organs(sheet, pieces, greens)


def draw_label(img, x, y, lines, colour=C_TXT):
    """Draw a small white-boxed multi-line label anchored at (x, y)."""
    fs, th, pad, lh = 1.1, 2, 8, 34
    w = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, fs, th)[0][0] for t in lines) + 2 * pad
    h = lh * len(lines) + pad
    x = int(np.clip(x, 0, img.shape[1] - w - 1))
    y = int(np.clip(y, 0, img.shape[0] - h - 1))
    ov = img.copy()
    cv2.rectangle(ov, (x, y), (x + w, y + h), (255, 255, 255), -1)
    cv2.addWeighted(ov, 0.78, img, 0.22, 0, img)
    cv2.rectangle(img, (x, y), (x + w, y + h), colour, 1)
    for i, t in enumerate(lines):
        cv2.putText(img, t, (x + pad, y + lh * (i + 1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, colour, th, cv2.LINE_AA)


def process_sheet(sheet: str, final: dict) -> Path | None:
    _, raw_path = find_raw(sheet, Path("shimahotarubukuro"))
    raw = base.load_bgr(str(raw_path))
    ann = base.load_bgr(str(Path("shimask") / (sheet + ".jpg")))
    strokes = shimask_input.stroke_masks(raw, ann)
    comps = shimask_input.red_corolla_components(raw, ann, strokes=strokes)
    organ = organ_for_corolla(raw, ann, strokes, comps, sheet)
    img = raw.copy()

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
            m = rm.medial_axis(mask_local)

            # ROI outline (repaired solid region) in scan coordinates.
            cnts, _ = cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                cv2.polylines(img, [c + [x0, y0]], True, C_ROI, 2, cv2.LINE_AA)

            def off(p):
                return (int(round(p[0] + x0)), int(round(p[1] + y0)))

            cv2.arrowedLine(img, off(m["base_xy"]), off(m["tip_xy"]), C_LEN, 2, cv2.LINE_AA, tipLength=0.03)
            cv2.arrowedLine(img, off(m["tip_xy"]), off(m["base_xy"]), C_LEN, 2, cv2.LINE_AA, tipLength=0.03)
            cv2.line(img, off(m["w0_xy"]), off(m["w1_xy"]), C_WID, 2, cv2.LINE_AA)

            # Nectar-guide spots.
            spots = guide_spot_mask(raw, piece)
            gc, _ = cv2.findContours(spots, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, gc, -1, C_GUIDE, 2, cv2.LINE_AA)

            # Organ stroke for this corolla (each split half has its own).
            gr = organ.get(corolla_id)
            if gr is not None:
                cv2.line(img, (int(gr["x1"]), int(gr["y1"])), (int(gr["x2"]), int(gr["y2"])),
                         C_ORGAN, 3, cv2.LINE_AA)

            fr = final.get((sheet, corolla_id))
            lines = [f"C{corolla_id}"]
            if fr:
                lines.append(f"L {float(fr['corolla_length_mm']):.1f}  W {float(fr['corolla_width_obs_mm']):.1f} mm")
                cov = fr.get("guide_coverage_pct", "")
                lines.append(f"guide {float(cov):.1f}%" if cov not in ("", None)
                             else "guide -")
                if fr.get("organ_length_mm"):
                    lines[-1] += f"   organ {float(fr['organ_length_mm']):.1f}mm"
            else:
                lines.append(f"L {m['length_mm']:.1f}  W {m['width_mm']:.1f} mm")
            draw_label(img, x0, y0 - 120, lines)

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{sheet}.png"
    cv2.imwrite(str(out), img)
    return out


def main() -> None:
    final = final_table()
    sheets = sorted(p.stem for p in Path("shimask").iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    for sheet in sheets:
        out = process_sheet(sheet, final)
        print(f"[{sheet}] wrote {out}", flush=True)

    # Legend key figure so the colour code is documented once.
    key = np.full((260, 620, 3), 255, np.uint8)
    items = [("Corolla ROI outline", C_ROI), ("Length axis (base->tip)", C_LEN),
             ("Width bar", C_WID), ("Nectar-guide spots", C_GUIDE),
             ("Reproductive-organ stroke", C_ORGAN)]
    cv2.putText(key, "Overlay legend", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, C_TXT, 2, cv2.LINE_AA)
    for i, (t, c) in enumerate(items):
        y = 85 + i * 34
        cv2.line(key, (24, y - 6), (74, y - 6), c, 4, cv2.LINE_AA)
        cv2.putText(key, t, (90, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_TXT, 2, cv2.LINE_AA)
    cv2.imwrite(str(OUT / "_legend.png"), key)
    print(f"wrote {OUT / '_legend.png'}")


if __name__ == "__main__":
    main()
