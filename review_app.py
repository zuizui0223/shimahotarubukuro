# -*- coding: utf-8 -*-
"""Streamlit review app for image-grounded floral trait correction.

Everything is in the canonical ruler-at-top orientation. Per-corolla masks come
from the detector foreground (so paper-shadow that leaked into a mask is visible
and can be erased). Corrections export to the same reviewed formats the pipeline
uses (REVIEWED_AXIS_OVERRIDES snippet + reviewed CSVs).

Run:
    streamlit run review_app.py
"""
from __future__ import annotations
import os, json, csv, glob
import numpy as np, cv2
import streamlit as st

import measure_guides as base
import measure_guides_symmetry_axis as sym
import trait_review
from mask_editor_component import (
    bgr_to_jpeg_data_url,
    component_value,
    display_line_to_raw,
    display_polygons_to_raw,
    image_editor,
    mask_to_display_polygons,
    raw_line_to_display,
    stroke_to_raw_polygons,
)

MM_PX = base.MM_PX
MM2_PX = base.MM2_PX
DATA_ROOT = "shimahotarubukuro"
REVIEW_DIR = "results/reviewed"
STATE_DIR = "results/review_state"
ISLAND_FOLDERS = ("oshima", "toshima", "niijima", "shikinejima", "kozushima")
DISPLAY_W = 720
SHEET_DISPLAY_W = 900
FOLD_STATE_LABELS = {
    "全展開（5裂片）": "open",
    "半折り（約2.5裂片）": "folded_half",
    "不明": "unknown",
}
ORGAN_STATE_KEY = "_organ_reviews"
os.makedirs(STATE_DIR, exist_ok=True)


# ----------------------------- data loading -----------------------------
def sheet_dash(stem: str) -> str:
    return stem.replace("~", "-")


def list_sheets():
    out = []
    for folder in ISLAND_FOLDERS:
        for p in sorted(glob.glob(os.path.join(DATA_ROOT, folder, "*.jpg"))):
            out.append((folder, os.path.splitext(os.path.basename(p))[0], p))
    return out


def committed_traits(stem: str):
    _, rows = committed_trait_table(stem)
    return rows or None


@st.cache_data(show_spinner=False)
def committed_trait_table(stem: str):
    path = os.path.join(REVIEW_DIR, sheet_dash(stem), "traits.csv")
    if not os.path.exists(path):
        return [], []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames or []), list(reader)


def trait_rows_by_corolla(rows):
    out = {}
    for row in rows:
        try:
            out[int(row["corolla_id"])] = row
        except (KeyError, TypeError, ValueError):
            continue
    return out


def find_overlay(folder: str, stem: str):
    """Locate a CANONICAL reviewed overlay (ruler-at-top) for this sheet."""
    island, _ = base.ISLANDS.get(folder, (folder, ""))
    name = f"{island}_{stem}.png"
    for d in (os.path.join("results", "review_overlays", sheet_dash(stem)),  # committed (Cloud)
              os.path.join("results", "review_cache", sheet_dash(stem), "overlays"),
              os.path.join("results_single", sheet_dash(stem), "overlays"),
              os.path.join("results_all_review", "sheets", folder, stem, "overlays")):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def _load_raw_matching(path, ov, cen):
    """Load the raw scan in the SAME frame as the accepted overlay. `load_bgr`'s
    per-sheet SHEET_ROTATION can disagree with the committed frame, so orientation
    is derived from the overlay aspect and the committed centroids being in-bounds."""
    from PIL import Image, ImageOps
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    raw0 = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    oh, ow = ov.shape[:2]
    target = ow / oh
    cand = []
    for rot in (None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180):
        raw = raw0 if rot is None else cv2.rotate(raw0, rot)
        rh, rw = raw.shape[:2]
        if abs((rw / rh) - target) > 0.02:
            continue
        inb = bool(cen) and all(0 <= cx < rw and 0 <= cy < rh for cx, cy in cen.values())
        cand.append((0 if inb else 1, raw))
    cand.sort(key=lambda t: t[0])
    return cand[0][1] if cand else raw0


@st.cache_data(show_spinner="Loading sheet + PRE-QC masks/axes…")
def load_sheet(folder: str, stem: str, path: str, overlay_path: str):
    """Masks come from the accepted reviewed overlay outlines (canonical), assigned
    to committed corolla ids by point-sampling each id's centroid inside the filled
    outlines. Split pairs that share one outline are divided at their mid-x."""
    ov = cv2.imdecode(np.fromfile(overlay_path, np.uint8), cv2.IMREAD_COLOR)
    _rows = committed_traits(stem)
    _cen = {int(t["corolla_id"]): (float(t["cx"]), float(t["cy"])) for t in _rows} if _rows else {}
    raw = _load_raw_matching(path, ov, _cen)
    rh, rw = raw.shape[:2]
    oh, ow = ov.shape[:2]
    sx, sy = rw / ow, rh / oh
    b, g, r = cv2.split(ov)
    green = ((g > 165) & (r < 135) & (b < 150)).astype(np.uint8) * 255
    mag = ((r > 150) & (b > 150) & (g < 120)).astype(np.uint8) * 255
    outl = cv2.morphologyEx(cv2.bitwise_or(green, mag), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(outl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = [c for c in cnts if cv2.contourArea(c) > 1000]
    labels = np.zeros((rh, rw), np.int32)
    for k, c in enumerate(cnts, start=1):
        cc = c.astype(np.float32).copy(); cc[:, 0, 0] *= sx; cc[:, 0, 1] *= sy
        cv2.drawContours(labels, [np.rint(cc).astype(np.int32)], -1, k, -1)

    rows = committed_traits(stem)
    if rows:
        cen = {int(t["corolla_id"]): (float(t["cx"]), float(t["cy"])) for t in rows}
    else:
        # fall back: number filled outlines in reading order
        cen = {}
        for k in range(1, len(cnts) + 1):
            ys, xs = np.where(labels == k)
            if len(xs):
                cen[k] = (float(xs.mean()), float(ys.mean()))
        cen = {i + 1: cen[k] for i, k in enumerate(sorted(cen, key=lambda k: (cen[k][1] // 200, cen[k][0])))}

    # point-sample the outline region under each committed centroid; group shared ones
    from collections import defaultdict
    region_of = {}
    for cid, (cx, cy) in cen.items():
        region_of[cid] = int(labels[int(round(cy)), int(round(cx))])
    groups = defaultdict(list)
    for cid, reg in region_of.items():
        groups[reg].append(cid)

    masks = {}
    for reg, cids in groups.items():
        union = (labels == reg).astype(np.uint8) if reg > 0 else np.zeros((rh, rw), np.uint8)
        if reg == 0:
            for cid in cids:
                masks[cid] = union.copy()
            continue
        if len(cids) == 1:
            masks[cids[0]] = union
        else:  # split pair sharing one outline -> divide at successive mid-x
            cids = sorted(cids, key=lambda c: cen[c][0])
            bounds = [0] + [int(round((cen[cids[i]][0] + cen[cids[i + 1]][0]) / 2))
                            for i in range(len(cids) - 1)] + [rw]
            for i, cid in enumerate(cids):
                m = union.copy(); m[:, :bounds[i]] = 0; m[:, bounds[i + 1]:] = 0
                masks[cid] = m

    return raw, masks, cen


@st.cache_data(show_spinner=False)
def preqc_axis(stem: str, cid: int, _mask):
    """PRE-QC symmetry axis for one mask (lazy + cached per corolla)."""
    if int(_mask.sum()) < 50:
        return None
    try:
        ax = sym.estimate_symmetry_axis(_mask)
        return (list(ax.base_xy), list(ax.tip_xy))
    except Exception:
        return None


# ----------------------------- state -----------------------------
def state_path(stem):
    return os.path.join(STATE_DIR, f"{sheet_dash(stem)}.json")


def load_state(stem):
    p = state_path(stem)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}


def save_state(stem, state):
    json.dump(state, open(state_path(stem), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def organ_id_sort_key(value):
    text = str(value).strip().removeprefix("O")
    return (0, int(text)) if text.isdigit() else (1, text)


def next_organ_id(reviews):
    numbers = []
    for value in (review.get("organ_id", "") for review in reviews):
        text = str(value).strip().removeprefix("O")
        if text.isdigit():
            numbers.append(int(text))
    return str(max(numbers, default=0) + 1)


def sheet_organ_reviews(state):
    """Return sheet-level O records, migrating old C-linked measurements."""
    reviews = state.setdefault(ORGAN_STATE_KEY, [])
    if isinstance(reviews, dict):
        reviews = list(reviews.values())
        state[ORGAN_STATE_KEY] = reviews

    used_ids = {
        str(review.get("organ_id", "")).strip()
        for review in reviews
        if str(review.get("organ_id", "")).strip()
    }
    for corolla_id, corolla in list(state.items()):
        if not str(corolla_id).isdigit() or not isinstance(corolla, dict):
            continue
        for old_record in corolla.pop("detached_organs", []):
            requested_id = str(old_record.get("candidate_id", "")).strip()
            if not requested_id.isdigit() or requested_id in used_ids:
                requested_id = str(
                    max(
                        [int(value) for value in used_ids if value.isdigit()],
                        default=0,
                    )
                    + 1
                )
            record = dict(old_record)
            record.update(
                organ_id=requested_id,
                source="legacy_corolla_review",
                nearest_corolla_hint=str(corolla_id),
                association_status="unconfirmed",
            )
            reviews.append(record)
            used_ids.add(requested_id)

    for review in reviews:
        review["organ_id"] = (
            str(review.get("organ_id", "")).strip().removeprefix("O")
        )
        if review.get("organ_type") in (
            "unclassified",
            "unclassified_reproductive_organ",
        ):
            review["organ_type"] = "unknown"
        review.setdefault("organ_type", "unknown")
        review.setdefault("identity_status", "uncertain")
        review.setdefault("source", "manual")
        review.setdefault("nearest_corolla_hint", "")
        review["association_status"] = "unconfirmed"
        review.setdefault("note", "")
    return reviews


def corolla_state(state, cid, preqc_axis):
    key = str(cid)
    if key not in state:
        base_axis = preqc_axis if preqc_axis else [[0, 0], [0, 0]]
        state[key] = {
            "axis_base": list(map(float, base_axis[0])),
            "axis_tip": list(map(float, base_axis[1])),
            "axis_changed": False,
            "exclude": False, "reason": "",
            "fold_state": "unknown", "pistil": False,
            "subtract": [], "add": [],
            "edge_mask_polys": [], "edge_mask_changed": False,
            "measurement_lines": {}, "measurement_lines_changed": [],
            "measurement_protocol": "half_corolla_widths_v1",
            "flat_n_lobes": None, "fold_changed": False,
            "organ_exclusions": [],
            "region_edits": {},
            "guide_presence_reviewed": "unreviewed",
            "review_complete": False,
        }
    cs = state[key]
    cs.setdefault("subtract", [])
    cs.setdefault("add", [])
    cs.setdefault("edge_mask_polys", [])
    cs.setdefault("edge_mask_changed", bool(cs.get("edge_mask_poly")))
    if cs.get("edge_mask_poly") and not cs["edge_mask_polys"]:
        cs["edge_mask_polys"] = [cs.pop("edge_mask_poly")]
    cs.setdefault("measurement_lines", {})
    cs.setdefault("measurement_lines_changed", [])
    if cs.get("measurement_protocol") != "half_corolla_widths_v1":
        for guide_name in trait_review.CORE_SHAPE_GUIDES:
            cs["measurement_lines"].pop(guide_name, None)
        cs["measurement_lines_changed"] = [
            guide_name
            for guide_name in cs["measurement_lines_changed"]
            if guide_name not in trait_review.CORE_SHAPE_GUIDES
        ]
        cs["measurement_protocol"] = "half_corolla_widths_v1"
    cs.setdefault("flat_n_lobes", None)
    cs.setdefault("fold_changed", False)
    cs.setdefault("fold_state", "unknown")
    cs.setdefault("organ_exclusions", [])
    for organ_list in (cs["organ_exclusions"],):
        for organ in organ_list:
            if organ.get("organ_type") in (
                "unclassified",
                "unclassified_reproductive_organ",
            ):
                organ["organ_type"] = "unknown"
            organ.setdefault("identity_status", "uncertain")
            organ.setdefault("note", "")
    cs.setdefault("region_edits", {})
    for target in trait_review.REGION_TARGETS:
        edits = cs["region_edits"].setdefault(target, {})
        edits.setdefault("add", [])
        edits.setdefault("subtract", [])
    cs.setdefault("guide_presence_reviewed", "unreviewed")
    cs.setdefault("review_complete", False)
    # Older app states silently defaulted unreviewed flowers to open.
    if (
        cs["fold_state"] == "open"
        and not cs["fold_changed"]
        and not cs["review_complete"]
    ):
        cs["fold_state"] = "unknown"
    return cs


# ----------------------------- geometry / rendering -----------------------------
def mask_was_edited(cs):
    return bool(
        cs.get("edge_mask_polys") or cs.get("subtract") or cs.get("add")
    )


def apply_mask_edits(mask, cs):
    m = np.zeros_like(mask, dtype=np.uint8) if cs.get("edge_mask_polys") else mask.copy()
    for poly in cs.get("edge_mask_polys", []):
        cv2.fillPoly(m, [np.array(poly, np.int32)], 1)
    for poly in cs.get("add", []):
        cv2.fillPoly(m, [np.array(poly, np.int32)], 1)
    for poly in cs.get("subtract", []):
        cv2.fillPoly(m, [np.array(poly, np.int32)], 0)
    return m


def analysis_mask(mask, cs):
    return apply_mask_edits(mask, cs)


def corolla_traits(mask, base_xy, tip_xy):
    area_mm2 = float(mask.sum()) * MM2_PX
    b = np.array(base_xy, float); t = np.array(tip_xy, float)
    length_mm = float(np.linalg.norm(t - b)) * MM_PX
    return {
        "area_mm2": round(area_mm2, 1),
        "length_mm": round(length_mm, 2),
    }


def region_was_edited(cs):
    return any(
        edits.get("add") or edits.get("subtract")
        for edits in cs.get("region_edits", {}).values()
    )


def pollination_trait_values(raw, mask, cs, row):
    edited = apply_mask_edits(mask, cs)
    mm_per_px = float(row.get("mm_per_px") or MM_PX)
    trait_review.ensure_measurement_lines(cs, edited, row)
    values = trait_review.shape_trait_values(edited, cs, mm_per_px)
    colour_box = bbox_of(edited, 30, raw.shape)
    colour_values, _ = trait_review.colour_trait_values(
        raw, analysis_mask(mask, cs), colour_box, cs, mm_per_px
    )
    values.update(colour_values)
    return values


def stringify_trait_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def bbox_of(mask, mgn, shape):
    ys, xs = np.where(mask > 0)
    if not len(xs):
        return 0, 0, shape[1], shape[0]
    return (max(0, xs.min() - mgn), max(0, ys.min() - mgn),
            min(shape[1], xs.max() + mgn), min(shape[0], ys.max() + mgn))


def render_organ_sheet(raw, masks, reviews, selected_organ_id):
    canvas = raw.copy()
    font_scale = max(0.55, min(canvas.shape[:2]) / 2400.0)
    for candidate_cid, mask in masks.items():
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        colour = (150, 150, 150)
        thickness = 1
        cv2.drawContours(canvas, contours, -1, colour, thickness)
        ys, xs = np.where(mask > 0)
        if len(xs):
            cv2.putText(
                canvas, f"C{candidate_cid}", (int(xs.mean()) - 18, int(ys.mean())),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, colour, 2, cv2.LINE_AA,
            )
    for organ in reviews:
        line = organ.get("line", [])
        if len(line) != 2:
            continue
        organ_id = str(organ.get("organ_id", ""))
        first, second = (tuple(np.rint(point).astype(int)) for point in line)
        colour = (
            (20, 145, 25)
            if organ_id == str(selected_organ_id)
            else (195, 105, 20)
        )
        cv2.line(canvas, first, second, colour, 3, cv2.LINE_AA)
        midpoint = tuple(np.rint(trait_review.line_midpoint(line)).astype(int))
        cv2.putText(
            canvas, f"O{organ_id}", midpoint,
            cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, colour, 2, cv2.LINE_AA,
        )
    return canvas


# ----------------------------- export -----------------------------
def export_all(folder, stem, state, raw, masks, cen, trait_fields=None, trait_rows=None):
    outdir = os.path.join(REVIEW_DIR, sheet_dash(stem), "app_review")
    os.makedirs(outdir, exist_ok=True)
    island, _ = base.ISLANDS.get(folder, (folder, ""))
    axes_rows, hr_rows, excl_rows, corr_rows = [], [], [], []
    measurement_rows, region_rows = [], []
    pollination_rows, reproductive_rows = [], []
    trait_by_cid = trait_rows_by_corolla(trait_rows or [])
    organ_reviews = sheet_organ_reviews(state)
    sheet_mm_per_px = next(
        (
            float(row["mm_per_px"])
            for row in (trait_rows or [])
            if row.get("mm_per_px")
        ),
        MM_PX,
    )
    ovr_lines = []
    for cid in sorted(cen):
        cs = state.get(str(cid))
        if cs is None:
            continue
        edited = apply_mask_edits(masks[cid], cs)
        tr = corolla_traits(edited, cs["axis_base"], cs["axis_tip"])
        original_trait = trait_by_cid.get(cid, {})
        mm_per_px = float(original_trait.get("mm_per_px") or MM_PX)
        trait_review.ensure_measurement_lines(cs, edited, original_trait)
        core_values = pollination_trait_values(raw, masks[cid], cs, original_trait)
        presence_reviewed = cs.get("guide_presence_reviewed", "unreviewed")
        analysis_values = trait_review.reviewed_guide_trait_values(
            core_values, presence_reviewed
        )
        presence_analysis = analysis_values.get(
            "guide_present_incl_oxidized", ""
        )
        pollination_rows.append({
            "island": island,
            "sheet": stem,
            "corolla_id": cid,
            "site_no": original_trait.get("site_no", ""),
            "individual_id": original_trait.get("individual_id", ""),
            **{
                field: stringify_trait_value(analysis_values.get(field, ""))
                for field, _, _, _ in trait_review.CORE_POLLINATION_TRAITS
            },
            "guide_presence_reviewed": presence_reviewed,
            "guide_present_analysis": presence_analysis,
            "n_spots_incl_oxidized": analysis_values.get(
                "n_spots_incl_oxidized", ""
            ),
            "brown_frac": core_values.get("brown_frac", ""),
            "degraded_flag": core_values.get("degraded_flag", ""),
            "fold_state_reviewed": cs.get("fold_state", ""),
            "area_correction_factor": core_values.get(
                "area_correction_factor", ""
            ),
            "area_scope": core_values.get(
                "area_scope", ""
            ),
            "excluded": "yes" if cs.get("exclude") else "",
            "review_complete": "yes" if cs.get("review_complete") else "",
            "scale_qc": original_trait.get("scale_qc", ""),
            "note": cs.get("reason", ""),
        })
        axes_rows.append({
            "island": island, "sheet": stem, "corolla_id": cid,
            "base_x": round(cs["axis_base"][0], 3), "base_y": round(cs["axis_base"][1], 3),
            "tip_x": round(cs["axis_tip"][0], 3), "tip_y": round(cs["axis_tip"][1], 3),
            "axis_changed": "yes" if cs["axis_changed"] else "no",
            "length_mm": tr["length_mm"], "area_mm2": tr["area_mm2"],
        })
        hr_rows.append({
            "island": island, "sheet": stem, "corolla_id": cid,
            "review_status": "excluded" if cs["exclude"] else "retained",
            "visible_pistil_attached": "",
            "fold_state": cs["fold_state"],
            "axis_reviewed": "yes" if cs["axis_changed"] else "",
            "mask_edited": "yes" if mask_was_edited(cs) else "",
            "measurement_guides_reviewed": "yes" if cs.get("measurement_lines_changed") else "",
            "pigment_regions_reviewed": "yes" if region_was_edited(cs) else "",
            "organ_exclusions": 0,
            "note": cs["reason"],
        })
        if cs["exclude"]:
            excl_rows.append({"island": island, "sheet": stem, "corolla_id": cid,
                              "excluded": "yes", "reason": cs["reason"] or "excluded_by_review"})
        if mask_was_edited(cs):
            corr_rows.append({"island": island, "sheet": stem, "corolla_id": cid,
                              "edge_mask_changed": "yes" if cs.get("edge_mask_polys") else "",
                              "n_edge_polys": len(cs.get("edge_mask_polys", [])),
                              "n_subtract_polys": len(cs["subtract"]), "n_add_polys": len(cs["add"]),
                              "n_outline_organ_exclusions": 0,
                              "kept_area_mm2": tr["area_mm2"]})
        for guide_name, line in cs.get("measurement_lines", {}).items():
            if guide_name not in trait_review.CORE_SHAPE_GUIDES or len(line) != 2:
                continue
            measurement_rows.append({
                "island": island, "sheet": stem, "corolla_id": cid,
                "guide": guide_name,
                "x1": round(line[0][0], 3), "y1": round(line[0][1], 3),
                "x2": round(line[1][0], 3), "y2": round(line[1][1], 3),
                "length_mm": round(trait_review.line_length(line) * mm_per_px, 3),
                "reviewed": "yes" if guide_name in cs.get("measurement_lines_changed", []) else "",
            })
        for target, edits in cs.get("region_edits", {}).items():
            region_rows.append({
                "island": island, "sheet": stem, "corolla_id": cid, "region": target,
                "n_add_polys": len(edits.get("add", [])),
                "n_subtract_polys": len(edits.get("subtract", [])),
            })
        if cs["axis_changed"]:
            ovr_lines.append(
                f'    ("{folder}", "{stem}", {cid}): {{\n'
                f'        "base_x": {cs["axis_base"][0]:.3f}, "base_y": {cs["axis_base"][1]:.3f},\n'
                f'        "tip_x": {cs["axis_tip"][0]:.3f}, "tip_y": {cs["axis_tip"][1]:.3f},\n'
                f'        "review_status": "ACCEPT",\n'
                f'        "review_note": "Reviewed in app (manual axis on central petal).",\n'
                f'    }},'
            )

    for organ in sorted(
        organ_reviews, key=lambda item: organ_id_sort_key(item.get("organ_id", ""))
    ):
        line = organ.get("line", [])
        if len(line) != 2:
            continue
        reproductive_rows.append({
            "island": island,
            "sheet": stem,
            "organ_id": organ.get("organ_id", ""),
            "source": organ.get("source", "manual"),
            "organ_type": organ.get("organ_type", "unknown"),
            "identity_status": organ.get("identity_status", "uncertain"),
            "length_mm": round(
                trait_review.line_length(line) * sheet_mm_per_px, 3
            ),
            "x1": round(line[0][0], 3),
            "y1": round(line[0][1], 3),
            "x2": round(line[1][0], 3),
            "y2": round(line[1][1], 3),
            "nearest_corolla_hint": organ.get("nearest_corolla_hint", ""),
            "corolla_id": "",
            "association_status": "unconfirmed",
            "note": organ.get("note", ""),
        })

    def _w(name, rows, fields=None):
        p = os.path.join(outdir, name)
        if rows or fields:
            with open(p, "w", newline="", encoding="utf-8-sig") as fh:
                fieldnames = fields or list(rows[0])
                w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)
        else:
            open(p, "w").close()
        return p

    _w("reviewed_axes.csv", axes_rows)
    _w("human_review.csv", hr_rows)
    _w("reviewed_exclusions.csv", excl_rows)
    _w("reviewed_mask_corrections.csv", corr_rows)
    _w("reviewed_measurement_guides.csv", measurement_rows)
    _w("reviewed_reproductive_organs.csv", reproductive_rows, [
        "island", "sheet", "organ_id", "source", "organ_type",
        "identity_status", "length_mm", "x1", "y1", "x2", "y2",
        "nearest_corolla_hint", "corolla_id", "association_status", "note",
    ])
    _w("reviewed_region_corrections.csv", region_rows)
    _w("reviewed_pollination_traits.csv", pollination_rows, [
        "island", "sheet", "corolla_id", "site_no", "individual_id",
        *[field for field, _, _, _ in trait_review.CORE_POLLINATION_TRAITS],
        "guide_presence_reviewed", "guide_present_analysis",
        "n_spots_incl_oxidized", "brown_frac", "degraded_flag",
        "fold_state_reviewed", "area_correction_factor", "area_scope",
        "excluded", "review_complete", "scale_qc", "note",
    ])
    for obsolete_name in (
        "reviewed_organ_exclusions.csv",
        "reviewed_traits.csv",
        "reviewed_trait_overrides.csv",
    ):
        obsolete_path = os.path.join(outdir, obsolete_name)
        if os.path.exists(obsolete_path):
            os.remove(obsolete_path)
    with open(os.path.join(outdir, "axis_overrides_snippet.py"), "w", encoding="utf-8") as fh:
        fh.write("# paste into REVIEWED_AXIS_OVERRIDES in measure_guides_review_axis_overrides.py\n")
        fh.write("\n".join(ovr_lines) + ("\n" if ovr_lines else ""))
    return outdir


# ----------------------------- UI -----------------------------
st.set_page_config(page_title="花形質レビュー", layout="wide")
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.1rem; padding-bottom: 2.5rem; max-width: 1440px;}
      [data-testid="stSidebar"] {min-width: 276px; max-width: 320px;}
      [data-testid="stMetric"] {border-left: 3px solid #2e7d32; padding-left: .7rem;}
      [data-testid="stMetricValue"] {font-size: 1.65rem;}
      .stDataFrame {border-top: 1px solid #d6d9dc;}
      .block-container h2 {font-size: 1.65rem; letter-spacing: 0;}
      .block-container h3 {font-size: 1.2rem; letter-spacing: 0;}
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown("## シマホタルブクロ花形質レビュー")

sheets = list_sheets()
if not sheets:
    st.error(f"{DATA_ROOT}/ に原画像がありません。")
    st.stop()

sheet_labels = [f"{folder}/{stem}" for folder, stem, _ in sheets]
default_sheet = "oshima/oshima10~13"
selected_sheet = st.sidebar.selectbox(
    "シート",
    sheet_labels,
    index=sheet_labels.index(default_sheet) if default_sheet in sheet_labels else 0,
)
folder, stem, path = sheets[sheet_labels.index(selected_sheet)]

overlay_path = find_overlay(folder, stem)
if overlay_path is None:
    st.error(f"{folder}/{stem} のレビュー用オーバーレイがありません。")
    st.code(
        f'python qc_single_sheet.py --image "{path}" '
        f'--folder {folder} --out-dir results/review_cache/{sheet_dash(stem)}'
    )
    st.stop()

raw, masks, cen = load_sheet(folder, stem, path, overlay_path)
trait_fields, trait_rows = committed_trait_table(stem)
trait_by_cid = trait_rows_by_corolla(trait_rows)


def get_preqc(corolla_id):
    return preqc_axis(stem, corolla_id, masks[corolla_id])


if st.session_state.get("stem") != stem:
    st.session_state.stem = stem
    st.session_state.state = load_state(stem)
    st.session_state.editor_generation = 0
state = st.session_state.state
organ_reviews = sheet_organ_reviews(state)

ids = sorted(cen)
cid = st.sidebar.selectbox(
    "花冠",
    ids,
    format_func=lambda value: f"C{value}",
    key=f"corolla_{stem}",
)
cs = corolla_state(state, cid, get_preqc(cid))
trait_row = trait_by_cid.get(cid, {})
mm_per_px = float(trait_row.get("mm_per_px") or MM_PX)

reviewed_count = sum(
    bool(state.get(str(corolla_id), {}).get("review_complete"))
    for corolla_id in ids
)
st.sidebar.progress(reviewed_count / max(len(ids), 1))
st.sidebar.caption(f"確認済み {reviewed_count} / {len(ids)}")

edited = apply_mask_edits(masks[cid], cs)
trait_review.ensure_measurement_lines(cs, edited, trait_row)
box = bbox_of(edited, 80, raw.shape)
scale = DISPLAY_W / max(box[2] - box[0], 1)
display_height = max(180, int(round((box[3] - box[1]) * scale)))
crop = raw[box[1]:box[3], box[0]:box[2]].copy()
image_url = bgr_to_jpeg_data_url(crop, DISPLAY_W, display_height)
display_mask = mask_to_display_polygons(
    edited, box, DISPLAY_W, display_height
)
editor_generation = int(st.session_state.get("editor_generation", 0))


def commit_change():
    save_state(stem, state)
    st.session_state.editor_generation = editor_generation + 1
    st.rerun()


stage = st.segmented_control(
    "レビュー工程",
    ["マスク", "形", "斑点", "雄しべ・雌しべ", "確認"],
    default="マスク",
    key=f"review_stage_{stem}_{cid}",
    width="stretch",
) or "マスク"

status_col, progress_col = st.columns([4, 1])
with status_col:
    st.subheader(
        f"{folder}/{stem}  器官O番号"
        if stage == "雄しべ・雌しべ"
        else f"{folder}/{stem}  C{cid}"
    )
with progress_col:
    st.metric(
        "状態",
        f"測定 {len(organ_reviews)}"
        if stage == "雄しべ・雌しべ"
        else "確認済み"
        if cs.get("review_complete")
        else "作業中",
    )

if stage == "マスク":
    left, right = st.columns([3, 1])
    with left:
        mask_mode = st.segmented_control(
            "編集",
            ["境界", "ブラシ"],
            default="境界",
            key=f"mask_mode_{stem}_{cid}",
        ) or "境界"
        if mask_mode == "境界":
            editor_key = f"mask_boundary_{stem}_{cid}_{editor_generation}"
            draft_polygons = component_value(
                st.session_state.get(editor_key), "polygons", display_mask
            )
            result = image_editor(
                key=editor_key,
                data={
                    "image_url": image_url,
                    "width": DISPLAY_W,
                    "height": display_height,
                    "mode": "polygon",
                    "polygons": draft_polygons,
                },
                default={"polygons": draft_polygons},
                on_polygons_change=lambda: None,
                width="stretch",
                height="content",
            )
            draft_polygons = component_value(
                result, "polygons", draft_polygons
            )
        else:
            brush_size = st.slider(
                "ブラシ幅",
                5,
                70,
                24,
                key=f"mask_brush_{stem}_{cid}",
            )
            paint_label = st.segmented_control(
                "処理",
                ["除く", "足す"],
                default="除く",
                key=f"mask_effect_{stem}_{cid}",
            ) or "除く"
            paint_effect = {"除く": "subtract", "足す": "add"}[paint_label]
            editor_key = f"mask_paint_{stem}_{cid}_{editor_generation}"
            draft_stroke = component_value(
                st.session_state.get(editor_key), "stroke", []
            )
            result = image_editor(
                key=editor_key,
                data={
                    "image_url": image_url,
                    "width": DISPLAY_W,
                    "height": display_height,
                    "mode": "paint",
                    "polygons": display_mask,
                    "stroke": draft_stroke,
                    "brush": brush_size,
                    "effect": paint_effect,
                },
                default={"stroke": draft_stroke},
                on_stroke_change=lambda: None,
                width="stretch",
                height="content",
            )
            draft_stroke = component_value(result, "stroke", draft_stroke)

    with right:
        current_shape = trait_review.shape_trait_values(
            edited, cs, mm_per_px
        )
        standardized_area = current_shape["corolla_area_standardized_mm2"]
        st.metric(
            "花冠面積（全花冠換算）",
            (
                f"{standardized_area:.1f} mm2"
                if standardized_area != ""
                else "状態を選択"
            ),
        )
        st.metric("マスク修正", "あり" if mask_was_edited(cs) else "なし")
        if mask_mode == "境界":
            if st.button(
                "境界を適用",
                type="primary",
                width="stretch",
                key=f"apply_boundary_{stem}_{cid}",
            ):
                raw_polygons = display_polygons_to_raw(
                    draft_polygons,
                    box,
                    DISPLAY_W,
                    display_height,
                    raw.shape,
                )
                if raw_polygons:
                    cs["edge_mask_polys"] = raw_polygons
                    cs["edge_mask_changed"] = True
                    cs["add"] = []
                    cs["subtract"] = []
                    commit_change()
                st.warning("有効な境界がありません。")
            if st.button(
                "境界を自動値へ戻す",
                width="stretch",
                key=f"reset_boundary_{stem}_{cid}",
            ):
                cs["edge_mask_polys"] = []
                cs["edge_mask_changed"] = False
                cs["add"] = []
                cs["subtract"] = []
                commit_change()
        else:
            if st.button(
                "ブラシ修正を適用",
                type="primary",
                width="stretch",
                key=f"apply_mask_paint_{stem}_{cid}",
            ):
                polygons = stroke_to_raw_polygons(
                    draft_stroke,
                    brush_size,
                    box,
                    DISPLAY_W,
                    display_height,
                )
                if polygons:
                    cs[paint_effect].extend(polygons)
                    commit_change()
                st.warning("ブラシ軌跡がありません。")
            if st.button(
                "直前のブラシ修正を戻す",
                width="stretch",
                key=f"undo_mask_paint_{stem}_{cid}",
            ):
                fallback = "add" if paint_effect == "subtract" else "subtract"
                for operation in (paint_effect, fallback):
                    if cs[operation]:
                        cs[operation].pop()
                        commit_change()

elif stage == "形":
    left, right = st.columns([3, 1])
    lines = trait_review.ensure_measurement_lines(cs, edited, trait_row)
    shape_targets = {
        "花冠長（基部－先端）": "axis",
        "最大幅": "max_span",
        "開口幅（谷の高さで左右端－端）": "throat_span",
        "筒の基部幅": "basal_tube_span",
    }
    with left:
        selected_shape_label = st.selectbox(
            "測定線",
            list(shape_targets),
            key=f"shape_target_{stem}_{cid}",
        )
        selected_shape = shape_targets[selected_shape_label]
        if selected_shape == "axis":
            active_raw_line = [cs["axis_base"], cs["axis_tip"]]
            active_colour = "#d32f2f"
            endpoint_labels = ["BASE", "TIP"]
        else:
            active_raw_line = lines[selected_shape]
            active_colour = trait_review.MEASUREMENT_GUIDES[selected_shape]["colour"]
            endpoint_labels = (
                ["AXIS", "EDGE"]
                if cs.get("fold_state") in {"open", "opened_full"}
                else ["EDGE 1", "EDGE 2"]
            )

        context_lines = []
        if selected_shape != "axis":
            context_lines.append({
                "points": raw_line_to_display(
                    [cs["axis_base"], cs["axis_tip"]],
                    box,
                    DISPLAY_W,
                    display_height,
                ),
                "colour": "rgba(211,47,47,.65)",
            })
        for guide_name in trait_review.CORE_SHAPE_GUIDES:
            if guide_name == selected_shape:
                continue
            context_lines.append({
                "points": raw_line_to_display(
                    lines[guide_name], box, DISPLAY_W, display_height
                ),
                "colour": trait_review.MEASUREMENT_GUIDES[guide_name]["colour"],
            })

        editor_key = (
            f"shape_{selected_shape}_{stem}_{cid}_{editor_generation}"
        )
        initial_line = raw_line_to_display(
            active_raw_line, box, DISPLAY_W, display_height
        )
        draft_line = component_value(
            st.session_state.get(editor_key), "line", initial_line
        )
        result = image_editor(
            key=editor_key,
            data={
                "image_url": image_url,
                "width": DISPLAY_W,
                "height": display_height,
                "mode": "line",
                "polygons": display_mask,
                "line": draft_line,
                "line_colour": active_colour,
                "line_labels": endpoint_labels,
                "context_lines": context_lines,
            },
            default={"line": draft_line},
            on_line_change=lambda: None,
            width="stretch",
            height="content",
        )
        draft_line = component_value(result, "line", draft_line)

    with right:
        fold_values = list(FOLD_STATE_LABELS.values())
        current_fold = cs.get("fold_state", "unknown")
        selected_fold_label = st.radio(
            "花冠の展開状態",
            list(FOLD_STATE_LABELS),
            index=(
                fold_values.index(current_fold)
                if current_fold in fold_values
                else fold_values.index("unknown")
            ),
            key=f"shape_fold_{stem}_{cid}",
        )
        selected_fold = FOLD_STATE_LABELS[selected_fold_label]
        if selected_fold != current_fold:
            cs["fold_state"] = selected_fold
            cs["fold_changed"] = True
            for guide_name in trait_review.CORE_SHAPE_GUIDES:
                if guide_name not in cs["measurement_lines_changed"]:
                    cs["measurement_lines"].pop(guide_name, None)
            commit_change()
        if selected_fold == "open":
            st.caption(
                "全展開：幅3本は中軸から片側外縁まで。面積は全体を実測。"
            )
        elif selected_fold == "folded_half":
            st.caption(
                "半折り：幅3本は見えている半花冠を実測。面積だけ2倍。"
            )
        else:
            st.caption("状態不明：実測線は保存し、比較値は保留。")

        reviewed_line = display_line_to_raw(
            draft_line, box, DISPLAY_W, display_height, raw.shape
        )
        st.metric(
            "選択線",
            f"{trait_review.line_length(reviewed_line) * mm_per_px:.2f} mm",
        )
        shape_values = trait_review.shape_trait_values(edited, cs, mm_per_px)
        standardization_pending = shape_values["area_scope"] == "unresolved"
        st.metric("花冠長", f'{shape_values["corolla_length_ruler_mm"]:.2f} mm')
        st.metric(
            "最大幅（半花冠）",
            (
                "状態を選択"
                if standardization_pending
                else f'{shape_values["corolla_max_span_standardized_mm"]:.2f} mm'
            ),
        )
        st.metric(
            "開口幅（半花冠）",
            (
                "状態を選択"
                if standardization_pending
                else f'{shape_values["throat_span_standardized_mm"]:.2f} mm'
            ),
        )
        st.metric("相対開口率", f'{shape_values["flat_throat_openness"]:.3f}')
        st.metric(
            "筒基部幅（半花冠）",
            (
                "状態を選択"
                if standardization_pending
                else f'{shape_values["basal_tube_width_standardized_mm"]:.2f} mm'
            ),
        )
        st.metric(
            "筒の広がり比",
            f'{shape_values["flat_tube_taper_ratio"]:.3f}',
        )

        if st.button(
            "測定線を適用",
            type="primary",
            width="stretch",
            key=f"apply_shape_{selected_shape}_{stem}_{cid}",
        ):
            if trait_review.line_length(reviewed_line) <= 1:
                st.warning("測定線が短すぎます。")
            elif selected_shape == "axis":
                cs["axis_base"] = reviewed_line[0]
                cs["axis_tip"] = reviewed_line[1]
                cs["axis_changed"] = True
                for guide_name in list(cs["measurement_lines"]):
                    if guide_name not in cs["measurement_lines_changed"]:
                        cs["measurement_lines"].pop(guide_name, None)
                trait_review.ensure_measurement_lines(cs, edited, trait_row)
                commit_change()
            else:
                cs["measurement_lines"][selected_shape] = reviewed_line
                if selected_shape not in cs["measurement_lines_changed"]:
                    cs["measurement_lines_changed"].append(selected_shape)
                commit_change()

        if st.button(
            "選択線を自動値へ戻す",
            width="stretch",
            key=f"reset_shape_{selected_shape}_{stem}_{cid}",
        ):
            if selected_shape == "axis":
                original_axis = get_preqc(cid)
                if original_axis:
                    cs["axis_base"] = list(map(float, original_axis[0]))
                    cs["axis_tip"] = list(map(float, original_axis[1]))
                    cs["axis_changed"] = False
            else:
                automatic = trait_review.automatic_measurement_lines(
                    edited,
                    cs["axis_base"],
                    cs["axis_tip"],
                    trait_row,
                    half_widths=cs.get("fold_state") in {
                        "open",
                        "opened_full",
                    },
                )
                cs["measurement_lines"][selected_shape] = automatic[selected_shape]
                if selected_shape in cs["measurement_lines_changed"]:
                    cs["measurement_lines_changed"].remove(selected_shape)
            commit_change()

elif stage == "斑点":
    analysis = analysis_mask(masks[cid], cs)
    colour_values, colour_masks = trait_review.colour_trait_values(
        raw, analysis, box, cs, mm_per_px
    )
    left, right = st.columns([3, 1])
    target_labels = {
        "紫色のネクターガイド": "guide",
        "酸化したガイド": "oxidized",
        "褐変・劣化": "brown",
    }
    presence_labels = {
        "未判定": "unreviewed",
        "あり": "present",
        "なし": "absent",
        "不明": "uncertain",
    }
    current_presence = cs.get("guide_presence_reviewed", "unreviewed")
    with left:
        selected_target_label = st.selectbox(
            "領域",
            list(target_labels),
            key=f"region_target_{stem}_{cid}",
        )
        target = target_labels[selected_target_label]
        operation_label = st.segmented_control(
            "処理",
            ["除く", "足す"],
            default="除く",
            key=f"region_operation_{stem}_{cid}",
        ) or "除く"
        operation = {"除く": "subtract", "足す": "add"}[operation_label]
        brush_size = st.slider(
            "ブラシ幅",
            3,
            50,
            12,
            key=f"region_brush_{stem}_{cid}",
        )
        overlay_crop = trait_review.render_region_overlay(
            raw, analysis, box, colour_masks, target
        )
        overlay_url = bgr_to_jpeg_data_url(
            overlay_crop, DISPLAY_W, display_height
        )
        editor_key = (
            f"region_{target}_{stem}_{cid}_{editor_generation}"
        )
        draft_stroke = component_value(
            st.session_state.get(editor_key), "stroke", []
        )
        result = image_editor(
            key=editor_key,
            data={
                "image_url": overlay_url,
                "width": DISPLAY_W,
                "height": display_height,
                "mode": "paint",
                "polygons": [],
                "stroke": draft_stroke,
                "brush": brush_size,
                "effect": operation,
            },
            default={"stroke": draft_stroke},
            on_stroke_change=lambda: None,
            width="stretch",
            height="content",
        )
        draft_stroke = component_value(result, "stroke", draft_stroke)

    with right:
        presence_label = st.selectbox(
            "ガイド判定",
            list(presence_labels),
            index=list(presence_labels.values()).index(current_presence)
            if current_presence in presence_labels.values()
            else 0,
            key=f"guide_presence_{stem}_{cid}",
        )
        standardized_guide_area = colour_values[
            "guide_area_incl_oxidized_standardized_mm2"
        ]
        st.metric(
            "ガイド面積（紫＋酸化、全花冠換算）",
            (
                f"{standardized_guide_area:.2f} mm2"
                if standardized_guide_area != ""
                else "状態を選択"
            ),
        )
        st.metric(
            "面積率（紫＋酸化）",
            f'{colour_values["guide_cov_incl_oxidized_pct"]:.2f}%',
        )
        st.metric(
            "領域数（紫＋酸化）",
            colour_values["n_spots_incl_oxidized"],
        )
        if st.button(
            "斑点レビューを保存",
            type="primary",
            width="stretch",
            key=f"apply_region_{stem}_{cid}",
        ):
            polygons = stroke_to_raw_polygons(
                draft_stroke,
                brush_size,
                box,
                DISPLAY_W,
                display_height,
            )
            if polygons:
                cs["region_edits"][target][operation].extend(polygons)
            cs["guide_presence_reviewed"] = presence_labels[presence_label]
            commit_change()
        if st.button(
            "直前の領域修正を戻す",
            width="stretch",
            key=f"undo_region_{stem}_{cid}",
        ):
            fallback = "add" if operation == "subtract" else "subtract"
            for candidate_operation in (operation, fallback):
                if cs["region_edits"][target][candidate_operation]:
                    cs["region_edits"][target][candidate_operation].pop()
                    commit_change()
        if st.button(
            "この領域を自動値へ戻す",
            width="stretch",
            key=f"reset_region_{stem}_{cid}",
        ):
            cs["region_edits"][target] = {"add": [], "subtract": []}
            commit_change()

elif stage == "雄しべ・雌しべ":
    organ_type_labels = {
        "不明": "unknown",
        "雌しべ": "pistil",
        "雄しべ": "stamen",
        "その他": "artifact",
    }
    identity_labels = {
        "不確実": "uncertain",
        "確実": "confirmed",
    }

    organ_status = st.columns([1, 3])
    organ_status[0].metric("測定済み", len(organ_reviews))
    if organ_status[1].button(
        "新しい器官を手動測定",
        type="primary",
        width="stretch",
        key=f"manual_organ_{stem}_{editor_generation}",
    ):
        st.session_state[f"focus_manual_organ_{stem}"] = "new"
        st.session_state.editor_generation = editor_generation + 1
        st.rerun()

    review_by_id = {
        str(item.get("organ_id", "")): item
        for item in organ_reviews
        if str(item.get("organ_id", "")).strip()
    }
    available_ids = sorted(review_by_id, key=organ_id_sort_key)
    new_organ_id = next_organ_id(organ_reviews)
    organ_options = available_ids + ["new"]
    requested_organ = st.session_state.pop(
        f"focus_manual_organ_{stem}", None
    )
    default_organ = (
        requested_organ
        if requested_organ in organ_options
        else available_ids[0]
        if available_ids
        else "new"
    )

    left, right = st.columns([3, 1])
    with left:
        selected_organ = st.selectbox(
            "器官番号",
            organ_options,
            index=organ_options.index(default_organ),
            format_func=lambda value: (
                f"新しいO番号（O{new_organ_id}）"
                if value == "new"
                else f'O{value}  {"測定済み" if value in review_by_id else "未測定"}'
            ),
            key=f"organ_number_{stem}_{editor_generation}",
        )
        working_organ_id = (
            new_organ_id if selected_organ == "new" else selected_organ
        )
        existing_review = review_by_id.get(working_organ_id)
        initial_raw_line = (
            existing_review.get("line")
            if existing_review
            else [
                [raw.shape[1] * 0.5, raw.shape[0] * 0.45],
                [raw.shape[1] * 0.5, raw.shape[0] * 0.55],
            ]
        )

        sheet_canvas = render_organ_sheet(
            raw,
            masks,
            organ_reviews,
            working_organ_id,
        )
        sheet_height = max(
            240,
            int(round(raw.shape[0] * SHEET_DISPLAY_W / raw.shape[1])),
        )
        sheet_url = bgr_to_jpeg_data_url(
            sheet_canvas, SHEET_DISPLAY_W, sheet_height
        )
        sheet_box = (0, 0, raw.shape[1], raw.shape[0])
        editor_key = (
            f"organ_{working_organ_id}_{stem}_"
            f"{editor_generation}"
        )
        initial_line = raw_line_to_display(
            initial_raw_line,
            sheet_box,
            SHEET_DISPLAY_W,
            sheet_height,
        )
        draft_line = component_value(
            st.session_state.get(editor_key), "line", initial_line
        )
        result = image_editor(
            key=editor_key,
            data={
                "image_url": sheet_url,
                "width": SHEET_DISPLAY_W,
                "height": sheet_height,
                "mode": "line",
                "polygons": [],
                "line": draft_line,
                "line_colour": "#1565c0",
                "line_labels": ["BASE", "TIP"],
                "context_lines": [],
            },
            default={"line": draft_line},
            on_line_change=lambda: None,
            width="stretch",
            height="content",
        )
        draft_line = component_value(result, "line", draft_line)

    with right:
        st.metric("器官番号", f"O{working_organ_id}")
        st.metric("花冠との対応", "未確認")
        type_values = list(organ_type_labels.values())
        current_type = (
            existing_review.get("organ_type", "unknown")
            if existing_review
            else "unknown"
        )
        selected_type_label = st.selectbox(
            "器官",
            list(organ_type_labels),
            index=(
                type_values.index(current_type)
                if current_type in type_values
                else 0
            ),
            key=f"organ_type_{working_organ_id}_{stem}_{editor_generation}",
        )
        identity_values = list(identity_labels.values())
        current_identity = (
            existing_review.get("identity_status", "uncertain")
            if existing_review
            else "uncertain"
        )
        selected_identity_label = st.selectbox(
            "同定",
            list(identity_labels),
            index=(
                identity_values.index(current_identity)
                if current_identity in identity_values
                else 0
            ),
            key=f"organ_identity_{working_organ_id}_{stem}_{editor_generation}",
        )
        organ_note = st.text_input(
            "メモ",
            value=existing_review.get("note", "") if existing_review else "",
            key=f"organ_note_{working_organ_id}_{stem}_{editor_generation}",
        )
        raw_organ_line = display_line_to_raw(
            draft_line,
            sheet_box,
            SHEET_DISPLAY_W,
            sheet_height,
            raw.shape,
        )
        st.metric(
            "器官長",
            f"{trait_review.line_length(raw_organ_line) * mm_per_px:.2f} mm",
        )
        if st.button(
            f"O{working_organ_id}の測定を保存",
            type="primary",
            width="stretch",
            key=f"save_organ_{working_organ_id}_{stem}_{editor_generation}",
        ):
            if trait_review.line_length(raw_organ_line) <= 1:
                st.warning("器官線が短すぎます。")
            else:
                record = {
                    "organ_id": working_organ_id,
                    "line": raw_organ_line,
                    "organ_type": organ_type_labels[selected_type_label],
                    "identity_status": identity_labels[
                        selected_identity_label
                    ],
                    "source": "manual",
                    "nearest_corolla_hint": "",
                    "association_status": "unconfirmed",
                    "note": organ_note,
                }
                if existing_review:
                    existing_review.update(record)
                else:
                    organ_reviews.append(record)
                commit_change()

    if organ_reviews:
        sorted_reviews = sorted(
            organ_reviews,
            key=lambda item: organ_id_sort_key(item.get("organ_id", "")),
        )
        st.dataframe(
            [
                {
                    "番号": f'O{item.get("organ_id", "")}',
                    "器官": item.get("organ_type", "unknown"),
                    "長さ mm": round(
                        trait_review.line_length(item.get("line", []))
                        * mm_per_px,
                        3,
                    ),
                    "同定": item.get("identity_status", "uncertain"),
                    "花冠対応": "未確認",
                }
                for item in sorted_reviews
            ],
            hide_index=True,
            width="stretch",
        )
        remove_organ_id = st.selectbox(
            "削除する器官番号",
            [str(item.get("organ_id", "")) for item in sorted_reviews],
            format_func=lambda value: f"O{value}",
            key=f"remove_organ_{stem}_{editor_generation}",
        )
        if st.button(
            "選択した器官測定を削除",
            key=f"delete_organ_{stem}_{editor_generation}",
        ):
            organ_reviews[:] = [
                item
                for item in organ_reviews
                if str(item.get("organ_id", "")) != remove_organ_id
            ]
            commit_change()


elif stage == "確認":
    live_values = pollination_trait_values(
        raw, masks[cid], cs, trait_row
    )
    presence_review = cs.get("guide_presence_reviewed", "unreviewed")
    analysis_values = trait_review.reviewed_guide_trait_values(
        live_values, presence_review
    )
    presence_display = {
        "unreviewed": "未判定",
        "present": "あり",
        "absent": "なし",
        "uncertain": "不明",
    }.get(presence_review, presence_review)
    category_labels = {
        "Size": "花サイズ",
        "Access": "開口・形",
        "Guide": "ネクターガイド",
    }
    core_rows = []
    for field, label, unit, category in trait_review.CORE_POLLINATION_TRAITS:
        value = analysis_values.get(field, "")
        core_rows.append({
            "区分": category_labels[category],
            "形質": label,
            "値": value,
            "単位": unit,
        })

    top_metrics = st.columns(4)
    top_metrics[0].metric(
        "花冠長",
        f'{live_values.get("corolla_length_ruler_mm", 0):.2f} mm',
    )
    standardized_throat = live_values.get("throat_span_standardized_mm", "")
    top_metrics[1].metric(
        "開口幅（半花冠）",
        (
            f"{standardized_throat:.2f} mm"
            if standardized_throat != ""
            else "状態を選択"
        ),
    )
    top_metrics[2].metric(
        "ガイド面積率",
        (
            f'{analysis_values.get("guide_cov_incl_oxidized_pct", 0):.2f}%'
            if analysis_values.get("guide_cov_incl_oxidized_pct", "") != ""
            else "判定保留"
        ),
    )
    top_metrics[3].metric("ガイド判定", presence_display)

    left, right = st.columns([3, 1])
    with left:
        st.dataframe(
            core_rows,
            hide_index=True,
            width="stretch",
            column_config={
                "区分": st.column_config.TextColumn("区分"),
                "形質": st.column_config.TextColumn("送粉関連形質"),
                "値": st.column_config.TextColumn("レビュー値"),
                "単位": st.column_config.TextColumn("単位"),
            },
        )
    with right:
        fold_values = list(FOLD_STATE_LABELS.values())
        current_fold = cs.get("fold_state", "unknown")
        selected_fold_label = st.radio(
            "花冠の状態",
            list(FOLD_STATE_LABELS),
            index=(
                fold_values.index(current_fold)
                if current_fold in fold_values
                else fold_values.index("unknown")
            ),
            key=f"qc_fold_{stem}_{cid}",
        )
        exclude_corolla = st.checkbox(
            "この花冠を除外",
            value=bool(cs.get("exclude")),
            key=f"qc_exclude_{stem}_{cid}",
        )
        review_note = st.text_area(
            "レビュー記録",
            value=cs.get("reason", ""),
            key=f"qc_note_{stem}_{cid}",
        )
        review_complete = st.checkbox(
            "この花冠のレビュー完了",
            value=bool(cs.get("review_complete")),
            key=f"qc_complete_{stem}_{cid}",
        )
        if st.button(
            "確認内容を保存",
            type="primary",
            width="stretch",
            key=f"save_qc_{stem}_{cid}",
        ):
            new_fold = FOLD_STATE_LABELS[selected_fold_label]
            if new_fold != cs.get("fold_state"):
                cs["fold_changed"] = True
            cs["fold_state"] = new_fold
            cs["exclude"] = exclude_corolla
            cs["reason"] = review_note
            cs["review_complete"] = review_complete
            commit_change()

st.sidebar.divider()
if st.sidebar.button(
    "作業状態を保存",
    width="stretch",
    key=f"save_state_{stem}",
):
    save_state(stem, state)
    st.sidebar.success("保存しました")

if st.sidebar.button(
    "レビューCSVを書き出す",
    type="primary",
    width="stretch",
    key=f"export_state_{stem}",
):
    save_state(stem, state)
    export_dir = export_all(
        folder,
        stem,
        state,
        raw,
        masks,
        cen,
        trait_fields,
        trait_rows,
    )
    st.sidebar.success(f"{export_dir} に書き出しました")

with st.expander("シート全体"):
    canvas = raw.copy()
    for preview_cid in ids:
        preview_state = state.get(str(preview_cid))
        preview_mask = (
            apply_mask_edits(masks[preview_cid], preview_state)
            if preview_state
            else masks[preview_cid]
        )
        contours, _ = cv2.findContours(
            preview_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        excluded = bool(preview_state and preview_state.get("exclude"))
        reviewed = bool(
            preview_state and preview_state.get("review_complete")
        )
        outline_colour = (
            (0, 0, 210)
            if excluded
            else (30, 125, 220)
            if reviewed
            else (0, 175, 0)
        )
        cv2.drawContours(canvas, contours, -1, outline_colour, 3)
        ys, xs = np.where(preview_mask > 0)
        if len(xs):
            label = f"C{preview_cid}"
            if reviewed:
                label += " OK"
            if excluded:
                label += " EXCL"
            cv2.putText(
                canvas,
                label,
                (int(xs.mean()) - 30, int(ys.mean())),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                outline_colour,
                2,
                cv2.LINE_AA,
            )
    preview_scale = 1100 / max(canvas.shape[:2])
    preview = cv2.resize(
        canvas,
        None,
        fx=preview_scale,
        fy=preview_scale,
        interpolation=cv2.INTER_AREA,
    )
    st.image(
        cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
        caption=f"{folder}/{stem}",
        width="stretch",
    )
