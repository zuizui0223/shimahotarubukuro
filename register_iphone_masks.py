#!/usr/bin/env python3
"""Register iPhone-extracted corolla masks onto the reviewed scan corollas.

The masks in ``mask/<sheet>/*.{heic,jpg}`` are clean per-corolla silhouettes lifted
on an iPhone, but they carry no scan position, scale, or corolla id. This registers
each one to a reviewed hand-mask corolla by an IoU-optimal similarity transform
(rotation + scale + translation, with flips), transferring the ruler-calibrated
scan scale to the clean iPhone shape. The iPhone silhouette then supplies the ROI
for measurement - filling hollow hand masks and refining boundaries - while the
scale stays anchored to the scan.

Outputs results_shimask_all/iphone_traits.csv and a validation montage.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pillow_heif

import measure_guides as base
import shimask_input
import remeasure_medial as rm
from run_all_shimask_confirmed import find_raw

pillow_heif.register_heif_opener()
MM = float(base.MM_PX)
MASK_ROOT = Path("mask")


def zip_dir(sheet: str) -> str:
    return sheet.replace("niijiama", "niijima")  # zip folders drop the 'a'


def load_silhouette(path: Path) -> np.ndarray:
    im = np.array(Image.open(path))
    if im.ndim == 3 and im.shape[2] == 4:
        m = (im[:, :, 3] > 40).astype(np.uint8)           # alpha = subject
    else:
        g = cv2.cvtColor(im[:, :, :3], cv2.COLOR_RGB2GRAY)
        m = (g < 235).astype(np.uint8)                    # non-white = subject
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = rm._fill_holes(m)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if count > 1:
        m = (labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    ys, xs = np.where(m)
    return m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def upright(mask: np.ndarray) -> np.ndarray:
    """Rotate so the minAreaRect long axis is vertical; return the cropped mask."""
    cnt = max(cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)
    (cx, cy), (w, h), ang = cv2.minAreaRect(cnt)
    if w > h:
        ang += 90.0
    pad = int(max(mask.shape))
    big = cv2.copyMakeBorder(mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, 0)
    M = cv2.getRotationMatrix2D((cx + pad, cy + pad), ang, 1.0)
    r = cv2.warpAffine(big, M, (big.shape[1], big.shape[0]), flags=cv2.INTER_NEAREST)
    ys, xs = np.where(r)
    return r[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int((a & b).sum())
    union = int((a | b).sum())
    return inter / union if union else 0.0


def register(iph: np.ndarray, hand: np.ndarray) -> tuple[float, np.ndarray]:
    """Best IoU of the iPhone silhouette scaled to the (scan-scale) hand mask.

    Both are upright. Search a scale range and the four axis flips, centre-aligned;
    return (best_iou, iphone_registered_at_scan_scale) where the registered mask is
    on the hand mask's canvas (hand pixels == scan pixels).
    """
    H, W = hand.shape
    hy, hx = np.where(hand)
    hcen = np.array([hx.mean(), hy.mean()])
    best_iou, best_reg = 0.0, None
    base_scale = math.sqrt(hand.sum() / max(iph.sum(), 1))
    for s in np.linspace(base_scale * 0.8, base_scale * 1.2, 13):
        rs = cv2.resize(iph, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
        for flip in (rs, rs[::-1], rs[:, ::-1], rs[::-1, ::-1]):
            fy, fx = np.where(flip)
            if fy.size == 0:
                continue
            fcen = np.array([fx.mean(), fy.mean()])
            dx, dy = int(round(hcen[0] - fcen[0])), int(round(hcen[1] - fcen[1]))
            canvas = np.zeros((H, W), np.uint8)
            ys = fy + dy
            xs = fx + dx
            ok = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
            canvas[ys[ok], xs[ok]] = 1
            iou = _iou(canvas, hand)
            if iou > best_iou:
                best_iou, best_reg = iou, canvas
    return best_iou, best_reg


def side_lengths(mask: np.ndarray) -> tuple[float, float]:
    """Long and short minAreaRect sides of the mask, in mm."""
    cnt = max(cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)
    (_c, (w, h), _a) = cv2.minAreaRect(cnt)
    return max(w, h) * MM, min(w, h) * MM


def load_hand_traits() -> dict[tuple[str, str], tuple[float, float]]:
    """Reviewed hand-mask length/width per corolla (defines the length/width roles)."""
    out = {}
    path = Path("results_shimask_all/medial_traits.csv")
    for r in csv.DictReader(path.open(encoding="utf-8-sig")):
        out[(r["sheet"], r["corolla_id"])] = (
            float(r["corolla_length_mm"]), float(r["corolla_width_obs_mm"]))
    return out


def process_sheet(sheet: str, hand_traits: dict) -> list[dict]:
    zdir = MASK_ROOT / zip_dir(sheet)
    if not zdir.is_dir():
        return []
    files = sorted(f for f in zdir.iterdir() if f.suffix.lower() in (".heic", ".jpg", ".jpeg"))
    iph = [(f, upright(load_silhouette(f))) for f in files]
    _, raw_path = find_raw(sheet, Path("shimahotarubukuro"))
    raw = base.load_bgr(str(raw_path))
    ann = base.load_bgr(str(Path("shimask") / (sheet + ".jpg")))
    comps = shimask_input.red_corolla_components(raw, ann, strokes=shimask_input.stroke_masks(raw, ann))
    hands = []
    for cid, comp in enumerate(comps, 1):
        m = rm.crop_to_mask(comp["mask"].astype(np.uint8))
        solid, _ = rm._solid_roi(m)
        hands.append((str(cid), upright(solid)))
    # all pairs -> greedy 1:1 by best IoU
    pairs = []
    regs = {}
    for i, (f, im) in enumerate(iph):
        for j, (cid, hm) in enumerate(hands):
            iou, reg = register(im, hm)
            pairs.append((iou, i, j))
            regs[(i, j)] = reg
    pairs.sort(reverse=True)
    used_i, used_j, rows = set(), set(), []
    for iou, i, j in pairs:
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        cid = hands[j][0]
        reg = regs[(i, j)]
        long_mm, short_mm = side_lengths(reg)
        # Assign length/width to the same physical roles the hand mask used: if the
        # reviewed length is the longer side, the iPhone length is its long side too
        # (folded corollas), otherwise the short side (opened corollas are wider).
        hand_L, hand_W = hand_traits.get((sheet, cid), (long_mm, short_mm))
        L, W = (long_mm, short_mm) if hand_L >= hand_W else (short_mm, long_mm)
        A = float(reg.sum()) * MM * MM
        opened = rm.is_full_open(sheet, int(cid))
        rows.append({
            "sheet": sheet, "corolla_id": cid, "iphone_file": iph[i][0].name,
            "match_iou": round(iou, 3),
            "corolla_length_mm": round(L, 2), "corolla_width_obs_mm": round(W, 2),
            "corolla_area_obs_mm2": round(A, 1),
            "corolla_width_fulleq_mm": round(W * (1.0 if opened else 2.0), 2),
            "corolla_area_fulleq_mm2": round(A * (1.0 if opened else 2.0), 1),
            "fold_state": "opened_full" if opened else "folded_half",
        })
    rows.sort(key=lambda r: int(r["corolla_id"]))
    return rows


def main() -> None:
    sheets = sorted(p.stem for p in Path("shimask").iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    hand_traits = load_hand_traits()
    all_rows = []
    for sheet in sheets:
        rows = process_sheet(sheet, hand_traits)
        all_rows.extend(rows)
        if rows:
            miou = np.mean([r["match_iou"] for r in rows])
            print(f"[{sheet}] {len(rows)} matched  mean IoU={miou:.2f}", flush=True)
    out = Path("results_shimask_all/iphone_traits.csv")
    with out.open("w", newline="", encoding="utf-8-sig") as h:
        w = csv.DictWriter(h, fieldnames=list(all_rows[0]))
        w.writeheader()
        w.writerows(all_rows)
    print(f"wrote {out}  ({len(all_rows)} corollas)")


if __name__ == "__main__":
    main()
