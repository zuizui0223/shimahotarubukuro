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
from streamlit_image_coordinates import streamlit_image_coordinates as click_image

import measure_guides as base
import measure_guides_symmetry_axis as sym
import trait_review
from mask_editor_component import (
    bgr_to_jpeg_data_url,
    buffered_line_polygon,
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
LOCKED_TRAIT_FIELDS = {
    "island", "region_order", "sheet", "corolla_id", "cx", "cy",
    "source_component_id", "split_piece", "split_status",
    "site_no", "site_candidates", "site_lat", "site_lon", "individual_id",
    "scale_source", "scale_qc", "mm_per_px", "px_per_cm",
    "ruler_spacing_cv", "ruler_tick_count", "ruler_regular_interval_count",
    "ruler_baseline_y", "ruler_tick_x",
}
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


@st.cache_data(show_spinner=False)
def committed_organs(stem: str):
    path = os.path.join(REVIEW_DIR, sheet_dash(stem), "organs.csv")
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


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


def corolla_state(state, cid, preqc_axis):
    key = str(cid)
    if key not in state:
        base_axis = preqc_axis if preqc_axis else [[0, 0], [0, 0]]
        state[key] = {
            "axis_base": list(map(float, base_axis[0])),
            "axis_tip": list(map(float, base_axis[1])),
            "axis_changed": False,
            "exclude": False, "reason": "",
            "fold_state": "open", "pistil": False,
            "subtract": [], "add": [],
            "edge_mask_polys": [], "edge_mask_changed": False,
            "measurement_lines": {}, "measurement_lines_changed": [],
            "flat_n_lobes": None, "fold_changed": False,
            "organ_exclusions": [],
            "region_edits": {},
            "trait_overrides": {},
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
    cs.setdefault("flat_n_lobes", None)
    cs.setdefault("fold_changed", False)
    cs.setdefault("organ_exclusions", [])
    cs.setdefault("region_edits", {})
    for target in trait_review.REGION_TARGETS:
        edits = cs["region_edits"].setdefault(target, {})
        edits.setdefault("add", [])
        edits.setdefault("subtract", [])
    cs.setdefault("trait_overrides", {})
    return cs


# ----------------------------- geometry / rendering -----------------------------
def mask_was_edited(cs):
    organ_outline = any(item.get("scope") == "outline" for item in cs.get("organ_exclusions", []))
    return bool(cs.get("edge_mask_polys") or cs.get("subtract") or cs.get("add") or organ_outline)


def apply_mask_edits(mask, cs):
    m = np.zeros_like(mask, dtype=np.uint8) if cs.get("edge_mask_polys") else mask.copy()
    for poly in cs.get("edge_mask_polys", []):
        cv2.fillPoly(m, [np.array(poly, np.int32)], 1)
    for poly in cs.get("add", []):
        cv2.fillPoly(m, [np.array(poly, np.int32)], 1)
    for poly in cs.get("subtract", []):
        cv2.fillPoly(m, [np.array(poly, np.int32)], 0)
    for organ in cs.get("organ_exclusions", []):
        if organ.get("scope") != "outline":
            continue
        for poly in organ.get("polygons", []):
            cv2.fillPoly(m, [np.array(poly, np.int32)], 0)
    return m


def analysis_mask(mask, cs):
    output = apply_mask_edits(mask, cs)
    for organ in cs.get("organ_exclusions", []):
        for poly in organ.get("polygons", []):
            cv2.fillPoly(output, [np.asarray(poly, np.int32)], 0)
    return output


def corolla_traits(mask, base_xy, tip_xy):
    area_mm2 = float(mask.sum()) * MM2_PX
    b = np.array(base_xy, float); t = np.array(tip_xy, float)
    length_mm = float(np.linalg.norm(t - b)) * MM_PX
    ys, xs = np.where(mask > 0)
    width_mm = 0.0
    if len(xs) and length_mm > 0:
        axis = (t - b) / (np.linalg.norm(t - b) + 1e-9)
        normal = np.array([-axis[1], axis[0]])
        pts = np.stack([xs, ys], 1) - b
        perp = pts @ normal
        width_mm = float(perp.max() - perp.min()) * MM_PX
    return dict(area_mm2=round(area_mm2, 1), length_mm=round(length_mm, 2), width_mm=round(width_mm, 2))


def region_was_edited(cs):
    return any(
        edits.get("add") or edits.get("subtract")
        for edits in cs.get("region_edits", {}).values()
    )


def shape_review_active(cs):
    return bool(
        cs.get("axis_changed")
        or mask_was_edited(cs)
        or cs.get("measurement_lines_changed")
        or cs.get("flat_n_lobes") is not None
        or cs.get("fold_changed")
    )


def colour_review_active(cs):
    return bool(
        shape_review_active(cs)
        or region_was_edited(cs)
        or cs.get("organ_exclusions")
    )


def current_trait_updates(raw, mask, cs, row, *, preview=False):
    edited = apply_mask_edits(mask, cs)
    mm_per_px = float(row.get("mm_per_px") or MM_PX)
    trait_review.ensure_measurement_lines(cs, edited, row)
    updates = {}
    if preview or shape_review_active(cs):
        updates.update(trait_review.shape_trait_values(edited, cs, mm_per_px))
    if preview or colour_review_active(cs):
        colour_box = bbox_of(edited, 30, raw.shape)
        colour_values, _ = trait_review.colour_trait_values(
            raw, analysis_mask(mask, cs), colour_box, cs, mm_per_px
        )
        updates.update(colour_values)
    if "exclude" in row:
        updates["exclude"] = "yes" if cs.get("exclude") else row.get("exclude", "")
    return {field: value for field, value in updates.items() if field in row}


def editable_trait_fields(fields):
    return [field for field in fields if field not in LOCKED_TRAIT_FIELDS]


def stringify_trait_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def values_differ(left, right):
    return stringify_trait_value(left).strip() != stringify_trait_value(right).strip()


def build_trait_review_rows(fields, row, live, cs):
    overrides = cs.setdefault("trait_overrides", {})
    rows = []
    for field in editable_trait_fields(fields):
        original = row.get(field, "")
        baseline = live.get(field, original)
        reviewed = overrides.get(field, baseline)
        if field in overrides:
            source = "manual"
        elif field in live:
            source = "image/recalculated"
        else:
            source = "original"
        rows.append({
            "field": field,
            "original": stringify_trait_value(original),
            "reviewed_value": stringify_trait_value(reviewed),
            "source": source,
        })
    return rows


def edited_records(table):
    return table.to_dict("records") if hasattr(table, "to_dict") else list(table)


def apply_trait_review_table(cs, records, fields, row, live):
    next_overrides = {}
    allowed = set(editable_trait_fields(fields))
    for item in records:
        field = item.get("field")
        if field not in allowed:
            continue
        reviewed = stringify_trait_value(item.get("reviewed_value", ""))
        baseline = live.get(field, row.get(field, ""))
        if values_differ(reviewed, baseline):
            next_overrides[field] = reviewed
    cs["trait_overrides"] = next_overrides


def reviewed_trait_outputs(folder, stem, state, raw, masks, cen, fields, rows):
    if not fields or not rows:
        return [], [], []
    island, _ = base.ISLANDS.get(folder, (folder, ""))
    extra_fields = [
        "app_review_status", "app_review_note", "visible_pistil_attached",
        "fold_state_reviewed", "axis_reviewed", "mask_edited",
    ]
    out_fields = list(fields)
    for field in extra_fields:
        if field not in out_fields:
            out_fields.append(field)
    reviewed_rows = []
    override_rows = []
    for original in rows:
        try:
            cid = int(original["corolla_id"])
        except (KeyError, TypeError, ValueError):
            reviewed_rows.append(dict(original))
            continue
        reviewed = {field: original.get(field, "") for field in fields}
        cs = state.get(str(cid))
        if cs and cid in masks:
            for field, value in current_trait_updates(raw, masks[cid], cs, reviewed).items():
                if values_differ(value, original.get(field, "")):
                    reviewed[field] = stringify_trait_value(value)
                    override_rows.append({
                        "island": island, "sheet": stem, "corolla_id": cid,
                        "field": field, "original_value": original.get(field, ""),
                        "reviewed_value": stringify_trait_value(value),
                        "source": "image_recalculation",
                    })
            for field, value in cs.get("trait_overrides", {}).items():
                if field not in reviewed:
                    out_fields.append(field)
                if values_differ(value, original.get(field, "")):
                    reviewed[field] = stringify_trait_value(value)
                    override_rows.append({
                        "island": island, "sheet": stem, "corolla_id": cid,
                        "field": field, "original_value": original.get(field, ""),
                        "reviewed_value": stringify_trait_value(value),
                        "source": "manual_trait_review",
                    })
            reviewed["app_review_status"] = "excluded" if cs.get("exclude") else "retained"
            reviewed["app_review_note"] = cs.get("reason", "")
            reviewed["visible_pistil_attached"] = "yes" if cs.get("pistil") else ""
            reviewed["fold_state_reviewed"] = cs.get("fold_state", "")
            reviewed["axis_reviewed"] = "yes" if cs.get("axis_changed") else ""
            reviewed["mask_edited"] = "yes" if mask_was_edited(cs) else ""
        reviewed_rows.append(reviewed)
    # Keep dynamically added manual fields only once and in insertion order.
    deduped_fields = []
    for field in out_fields:
        if field not in deduped_fields:
            deduped_fields.append(field)
    return deduped_fields, reviewed_rows, override_rows


def bbox_of(mask, mgn, shape):
    ys, xs = np.where(mask > 0)
    if not len(xs):
        return 0, 0, shape[1], shape[0]
    return (max(0, xs.min() - mgn), max(0, ys.min() - mgn),
            min(shape[1], xs.max() + mgn), min(shape[0], ys.max() + mgn))


def render_crop(raw, mask, cs, cid, box, scale, pending_poly):
    x0, y0, x1, y1 = box
    crop = raw[y0:y1, x0:x1].copy()
    edited = apply_mask_edits(mask, cs)
    sub = crop[y0:y1, x0:x1] if False else crop  # noqa
    over = crop.copy()
    over[edited[y0:y1, x0:x1] > 0] = (0, 190, 0)
    vis = cv2.addWeighted(crop, 0.6, over, 0.4, 0)
    # removed (shadow) region in red = original minus edited
    removed = (mask[y0:y1, x0:x1] > 0) & (edited[y0:y1, x0:x1] == 0)
    vis[removed] = (0, 0, 235)
    cnts, _ = cv2.findContours(edited[y0:y1, x0:x1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, (0, 120, 0), 2)
    # axis
    bx, by = cs["axis_base"]; tx, ty = cs["axis_tip"]
    cv2.arrowedLine(vis, (int(bx - x0), int(by - y0)), (int(tx - x0), int(ty - y0)),
                    (0, 0, 220), 2, cv2.LINE_AA, tipLength=0.05)
    cv2.circle(vis, (int(bx - x0), int(by - y0)), 7, (0, 0, 255), -1)
    cv2.circle(vis, (int(tx - x0), int(ty - y0)), 6, (255, 0, 0), -1)
    # pending polygon vertices
    for (px, py) in pending_poly:
        cv2.circle(vis, (int(px - x0), int(py - y0)), 5, (255, 160, 0), -1)
    if len(pending_poly) >= 2:
        pts = np.array([[int(px - x0), int(py - y0)] for px, py in pending_poly], np.int32)
        cv2.polylines(vis, [pts], False, (255, 160, 0), 2)
    big = cv2.resize(vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(big, cv2.COLOR_BGR2RGB), edited


# ----------------------------- export -----------------------------
def export_all(folder, stem, state, raw, masks, cen, trait_fields=None, trait_rows=None):
    outdir = os.path.join(REVIEW_DIR, sheet_dash(stem), "app_review")
    os.makedirs(outdir, exist_ok=True)
    island, _ = base.ISLANDS.get(folder, (folder, ""))
    axes_rows, hr_rows, excl_rows, corr_rows = [], [], [], []
    measurement_rows, organ_rows, region_rows = [], [], []
    trait_by_cid = trait_rows_by_corolla(trait_rows or [])
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
        axes_rows.append({
            "island": island, "sheet": stem, "corolla_id": cid,
            "base_x": round(cs["axis_base"][0], 3), "base_y": round(cs["axis_base"][1], 3),
            "tip_x": round(cs["axis_tip"][0], 3), "tip_y": round(cs["axis_tip"][1], 3),
            "axis_changed": "yes" if cs["axis_changed"] else "no",
            "length_mm": tr["length_mm"], "width_mm": tr["width_mm"], "area_mm2": tr["area_mm2"],
        })
        hr_rows.append({
            "island": island, "sheet": stem, "corolla_id": cid,
            "review_status": "excluded" if cs["exclude"] else "retained",
            "visible_pistil_attached": "yes" if cs["pistil"] else "",
            "fold_state": cs["fold_state"],
            "axis_reviewed": "yes" if cs["axis_changed"] else "",
            "mask_edited": "yes" if mask_was_edited(cs) else "",
            "measurement_guides_reviewed": "yes" if cs.get("measurement_lines_changed") else "",
            "pigment_regions_reviewed": "yes" if region_was_edited(cs) else "",
            "organ_exclusions": len(cs.get("organ_exclusions", [])),
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
                              "n_outline_organ_exclusions": sum(
                                  item.get("scope") == "outline" for item in cs.get("organ_exclusions", [])
                              ),
                              "kept_area_mm2": tr["area_mm2"]})
        for guide_name, line in cs.get("measurement_lines", {}).items():
            if len(line) != 2:
                continue
            measurement_rows.append({
                "island": island, "sheet": stem, "corolla_id": cid,
                "guide": guide_name,
                "x1": round(line[0][0], 3), "y1": round(line[0][1], 3),
                "x2": round(line[1][0], 3), "y2": round(line[1][1], 3),
                "length_mm": round(trait_review.line_length(line) * mm_per_px, 3),
                "reviewed": "yes" if guide_name in cs.get("measurement_lines_changed", []) else "",
            })
        for index, organ in enumerate(cs.get("organ_exclusions", []), start=1):
            line = organ.get("line", [])
            if len(line) != 2:
                continue
            organ_rows.append({
                "island": island, "sheet": stem, "corolla_id": cid, "exclusion_id": index,
                "organ_type": organ.get("organ_type", "unclassified"),
                "seed_candidate_id": organ.get("seed_candidate_id", ""),
                "scope": organ.get("scope", ""), "width_mm": organ.get("width_mm", ""),
                "x1": round(line[0][0], 3), "y1": round(line[0][1], 3),
                "x2": round(line[1][0], 3), "y2": round(line[1][1], 3),
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
    _w("reviewed_organ_exclusions.csv", organ_rows)
    _w("reviewed_region_corrections.csv", region_rows)
    trait_fields_out, reviewed_traits, trait_overrides = reviewed_trait_outputs(
        folder, stem, state, raw, masks, cen, trait_fields or [], trait_rows or []
    )
    _w("reviewed_traits.csv", reviewed_traits, trait_fields_out)
    _w("reviewed_trait_overrides.csv", trait_overrides, [
        "island", "sheet", "corolla_id", "field",
        "original_value", "reviewed_value", "source",
    ])
    with open(os.path.join(outdir, "axis_overrides_snippet.py"), "w", encoding="utf-8") as fh:
        fh.write("# paste into REVIEWED_AXIS_OVERRIDES in measure_guides_review_axis_overrides.py\n")
        fh.write("\n".join(ovr_lines) + ("\n" if ovr_lines else ""))
    return outdir


# ----------------------------- UI -----------------------------
st.set_page_config(page_title="Corolla review", layout="wide")
st.title("シマホタルブクロ — sheet review / manual correction")

sheets = list_sheets()
if not sheets:
    st.error(f"No raw scans under {DATA_ROOT}/. Check the data path.")
    st.stop()

labels = [f"{f}/{s}" for f, s, _ in sheets]
sel = st.sidebar.selectbox("Sheet (ruler rotated to top)", labels, index=labels.index("oshima/oshima10~13") if "oshima/oshima10~13" in labels else 0)
folder, stem, path = sheets[labels.index(sel)]

overlay_path = find_overlay(folder, stem)
if overlay_path is None:
    st.error(
        f"No canonical PRE-QC overlay for **{folder}/{stem}**. Extract it first:\n\n"
        f"```\npython qc_single_sheet.py --image \"{path}\" "
        f"--folder {folder} --out-dir results/review_cache/{sheet_dash(stem)}\n```"
    )
    st.stop()
raw, masks, cen = load_sheet(folder, stem, path, overlay_path)
trait_fields, trait_rows = committed_trait_table(stem)
trait_by_cid = trait_rows_by_corolla(trait_rows)
organ_candidates = committed_organs(stem)
st.sidebar.caption(f"overlay: {overlay_path}")


def get_preqc(cid):
    return preqc_axis(stem, cid, masks[cid])

if "stem" not in st.session_state or st.session_state.stem != stem:
    st.session_state.stem = stem
    st.session_state.state = load_state(stem)
    st.session_state.pending = []
    st.session_state.last_click = None
state = st.session_state.state

ids = sorted(cen)
cid = st.sidebar.selectbox("Corolla", ids, format_func=lambda c: f"C{c}")
cs = corolla_state(state, cid, get_preqc(cid))

# per-corolla flags live in the sidebar (apply to the selected corolla)
st.sidebar.divider()
st.sidebar.subheader(f"C{cid} flags")
cs["exclude"] = st.sidebar.checkbox("Exclude corolla", value=cs["exclude"])
cs["reason"] = st.sidebar.text_input("Reason / note", value=cs["reason"])
fold_state = st.sidebar.radio(
    "Fold state", ["open", "folded_half"],
    index=0 if cs["fold_state"] == "open" else 1, horizontal=True,
)
if fold_state != cs["fold_state"]:
    cs["fold_state"] = fold_state
    cs["fold_changed"] = True
cs["pistil"] = st.sidebar.checkbox("Visible attached pistil", value=cs["pistil"])

trait_row = trait_by_cid.get(cid, {})
mm_per_px = float(trait_row.get("mm_per_px") or MM_PX)
edited = apply_mask_edits(masks[cid], cs)
trait_review.ensure_measurement_lines(cs, edited, trait_row)
box = bbox_of(edited, 80, raw.shape)
scale = DISPLAY_W / (box[2] - box[0])
disp_h = int((box[3] - box[1]) * scale)
mgen = st.session_state.get("mask_gen", 0)
crop = raw[box[1]:box[3], box[0]:box[2]].copy()
image_url = bgr_to_jpeg_data_url(crop, DISPLAY_W, disp_h)
display_mask = mask_to_display_polygons(edited, box, DISPLAY_W, disp_h)

review_view = st.segmented_control(
    "Review workspace",
    ["Axis", "Mask", "Measurements", "Pigment", "Organs", "All traits"],
    default="Axis",
    key=f"review_view_{stem}_{cid}",
    width="stretch",
)

if review_view == "Axis":
    c1, c2 = st.columns([3, 1])
    with c1:
        atool = st.radio("Set point", ["BASE (throat / top-centre)", "TIP (central lobe)"], horizontal=True)
        disp, edited = render_crop(raw, masks[cid], cs, cid, box, scale, [])
        click = click_image(disp, key=f"axis_{stem}_{cid}")
        if click is not None and click != st.session_state.last_click:
            st.session_state.last_click = click
            rx = box[0] + click["x"] / scale
            ry = box[1] + click["y"] / scale
            if atool.startswith("BASE"):
                cs["axis_base"] = [rx, ry]
            else:
                cs["axis_tip"] = [rx, ry]
            cs["axis_changed"] = True
            for name in list(cs["measurement_lines"]):
                if name not in cs["measurement_lines_changed"]:
                    cs["measurement_lines"].pop(name, None)
            st.rerun()
    with c2:
        shape_values = trait_review.shape_trait_values(edited, cs, mm_per_px)
        st.metric("length mm", shape_values["corolla_length_ruler_mm"])
        st.metric("width mm", shape_values["corolla_max_span_ruler_mm"])
        st.metric("area mm²", shape_values["corolla_area_ruler_mm2"])
        if st.button("Reset axis to PRE-QC", width="stretch"):
            pa = get_preqc(cid)
            if pa:
                cs["axis_base"] = list(map(float, pa[0])); cs["axis_tip"] = list(map(float, pa[1]))
                cs["axis_changed"] = False
                for name in list(cs["measurement_lines"]):
                    if name not in cs["measurement_lines_changed"]:
                        cs["measurement_lines"].pop(name, None)
            st.rerun()

elif review_view == "Mask":
    c1, c2 = st.columns([3, 1])
    with c1:
        edit_mode = st.segmented_control(
            "Mask edit mode", ["Vertices / edges", "Brush"], default="Vertices / edges"
        )
        if edit_mode == "Vertices / edges":
            editor_key = f"edge_{stem}_{cid}_{mgen}"
            draft_polygons = component_value(st.session_state.get(editor_key), "polygons", display_mask)
            result = image_editor(
                key=editor_key,
                data={
                    "image_url": image_url,
                    "width": DISPLAY_W,
                    "height": disp_h,
                    "mode": "polygon",
                    "polygons": draft_polygons,
                },
                default={"polygons": draft_polygons},
                on_polygons_change=lambda: None,
                width="stretch",
                height="content",
            )
            draft_polygons = component_value(result, "polygons", draft_polygons)
        else:
            brush = st.slider("Brush size (px)", 5, 70, 24)
            paint_effect = st.segmented_control("Paint effect", ["Subtract", "Add"], default="Subtract")
            editor_key = f"mask_paint_{stem}_{cid}_{mgen}"
            draft_stroke = component_value(st.session_state.get(editor_key), "stroke", [])
            result = image_editor(
                key=editor_key,
                data={
                    "image_url": image_url,
                    "width": DISPLAY_W,
                    "height": disp_h,
                    "mode": "paint",
                    "polygons": display_mask,
                    "stroke": draft_stroke,
                    "brush": brush,
                    "effect": paint_effect.lower(),
                },
                default={"stroke": draft_stroke},
                on_stroke_change=lambda: None,
                width="stretch",
                height="content",
            )
            draft_stroke = component_value(result, "stroke", draft_stroke)
    with c2:
        st.metric("area mm²", round(float(edited.sum()) * mm_per_px * mm_per_px, 2))
        if edit_mode == "Vertices / edges":
            if st.button("Apply edge mask", width="stretch"):
                polygons = display_polygons_to_raw(
                    draft_polygons, box, DISPLAY_W, disp_h, raw.shape
                )
                if polygons:
                    cs["edge_mask_polys"] = polygons
                    cs["edge_mask_changed"] = True
                    cs["add"] = []
                    cs["subtract"] = []
                    st.session_state.mask_gen = mgen + 1
                    st.rerun()
                else:
                    st.warning("No valid mask polygon was returned.")
            if st.button("Reset edge mask", width="stretch"):
                cs["edge_mask_polys"] = []
                cs["edge_mask_changed"] = False
                st.session_state.mask_gen = mgen + 1
                st.rerun()
        else:
            if st.button("Apply paint", width="stretch"):
                polygons = stroke_to_raw_polygons(
                    draft_stroke, brush, box, DISPLAY_W, disp_h
                )
                target = paint_effect.lower()
                if polygons:
                    cs[target].extend(polygons)
                    st.session_state.mask_gen = mgen + 1
                    st.rerun()
            if st.button("Undo last paint edit", width="stretch"):
                fallback = "add" if paint_effect == "Subtract" else "subtract"
                for k in (paint_effect.lower(), fallback):
                    if cs[k]:
                        cs[k].pop(); break
                st.session_state.mask_gen = mgen + 1
                st.rerun()

elif review_view == "Measurements":
    c1, c2 = st.columns([3, 1])
    lines = trait_review.ensure_measurement_lines(cs, edited, trait_row)
    guide_names = list(trait_review.MEASUREMENT_GUIDES)
    with c1:
        selected_guide = st.selectbox(
            "Measurement guide",
            guide_names,
            format_func=lambda name: trait_review.MEASUREMENT_GUIDES[name]["label"],
        )
        line_key = f"measure_{selected_guide}_{stem}_{cid}_{mgen}"
        initial_line = raw_line_to_display(lines[selected_guide], box, DISPLAY_W, disp_h)
        draft_line = component_value(st.session_state.get(line_key), "line", initial_line)
        context_lines = [{
            "points": raw_line_to_display(
                [cs["axis_base"], cs["axis_tip"]], box, DISPLAY_W, disp_h
            ),
            "colour": "rgba(210,35,35,.65)",
        }]
        for name, line in lines.items():
            if name == selected_guide:
                continue
            context_lines.append({
                "points": raw_line_to_display(line, box, DISPLAY_W, disp_h),
                "colour": trait_review.MEASUREMENT_GUIDES[name]["colour"],
            })
        result = image_editor(
            key=line_key,
            data={
                "image_url": image_url,
                "width": DISPLAY_W,
                "height": disp_h,
                "mode": "line",
                "polygons": display_mask,
                "line": draft_line,
                "line_colour": trait_review.MEASUREMENT_GUIDES[selected_guide]["colour"],
                "context_lines": context_lines,
            },
            default={"line": draft_line},
            on_line_change=lambda: None,
            width="stretch",
            height="content",
        )
        draft_line = component_value(result, "line", draft_line)
    with c2:
        reviewed_line = display_line_to_raw(
            draft_line, box, DISPLAY_W, disp_h, raw.shape
        )
        st.metric(
            "selected span mm",
            round(trait_review.line_length(reviewed_line) * mm_per_px, 3),
        )
        lobe_default = cs.get("flat_n_lobes")
        if lobe_default is None:
            lobe_default = int(float(
                trait_row.get("flat_n_lobes") or trait_row.get("prov_n_lobes") or 5
            ))
        lobe_count = st.number_input(
            "Visible lobe count", 1, 12, int(lobe_default), step=1
        )
        if st.button("Apply measurement guide", width="stretch"):
            if reviewed_line:
                cs["measurement_lines"][selected_guide] = reviewed_line
                if selected_guide not in cs["measurement_lines_changed"]:
                    cs["measurement_lines_changed"].append(selected_guide)
                cs["flat_n_lobes"] = int(lobe_count)
                st.session_state.mask_gen = mgen + 1
                st.rerun()
        if st.button("Reset selected guide", width="stretch"):
            automatic = trait_review.automatic_measurement_lines(
                edited, cs["axis_base"], cs["axis_tip"], trait_row
            )
            cs["measurement_lines"][selected_guide] = automatic[selected_guide]
            if selected_guide in cs["measurement_lines_changed"]:
                cs["measurement_lines_changed"].remove(selected_guide)
            st.session_state.mask_gen = mgen + 1
            st.rerun()
        shape_values = trait_review.shape_trait_values(edited, cs, mm_per_px)
        st.metric("tube length mm", shape_values["flat_tube_length_mm"])
        st.metric("lobe length mm", shape_values["flat_lobe_length_mm"])

elif review_view == "Pigment":
    analysis = analysis_mask(masks[cid], cs)
    colour_values, colour_masks = trait_review.colour_trait_values(
        raw, analysis, box, cs, mm_per_px
    )
    c1, c2 = st.columns([3, 1])
    with c1:
        target = st.selectbox(
            "Region",
            list(trait_review.REGION_TARGETS),
            format_func=lambda name: trait_review.REGION_TARGETS[name]["label"],
        )
        operation = st.segmented_control(
            "Region edit", ["Subtract", "Add"], default="Subtract"
        )
        brush = st.slider("Region brush size (px)", 3, 50, 12)
        overlay_crop = trait_review.render_region_overlay(
            raw, analysis, box, colour_masks, target
        )
        overlay_url = bgr_to_jpeg_data_url(overlay_crop, DISPLAY_W, disp_h)
        region_key = f"region_{target}_{stem}_{cid}_{mgen}"
        draft_stroke = component_value(st.session_state.get(region_key), "stroke", [])
        result = image_editor(
            key=region_key,
            data={
                "image_url": overlay_url,
                "width": DISPLAY_W,
                "height": disp_h,
                "mode": "paint",
                "polygons": [],
                "stroke": draft_stroke,
                "brush": brush,
                "effect": operation.lower(),
            },
            default={"stroke": draft_stroke},
            on_stroke_change=lambda: None,
            width="stretch",
            height="content",
        )
        draft_stroke = component_value(result, "stroke", draft_stroke)
    with c2:
        if target == "guide":
            st.metric("guide coverage %", colour_values["guide_cov_pct"])
            st.metric("spots", colour_values["n_spots"])
        elif target == "oxidized":
            st.metric(
                "inclusive coverage %", colour_values["guide_cov_incl_oxidized_pct"]
            )
            st.metric("oxidized spots", colour_values["n_oxidized_recovered_spots"])
        else:
            st.metric("brown fraction", colour_values["brown_frac"])
            st.metric("degraded", colour_values["degraded_flag"])
        if st.button("Apply region edit", width="stretch"):
            polygons = stroke_to_raw_polygons(
                draft_stroke, brush, box, DISPLAY_W, disp_h
            )
            if polygons:
                cs["region_edits"][target][operation.lower()].extend(polygons)
                st.session_state.mask_gen = mgen + 1
                st.rerun()
        if st.button("Undo region edit", width="stretch"):
            fallback = "add" if operation == "Subtract" else "subtract"
            for name in (operation.lower(), fallback):
                if cs["region_edits"][target][name]:
                    cs["region_edits"][target][name].pop()
                    break
            st.session_state.mask_gen = mgen + 1
            st.rerun()

elif review_view == "Organs":
    c1, c2 = st.columns([3, 1])
    organ_seed = trait_review.best_organ_candidate(
        organ_candidates, masks[cid], mm_per_px
    )
    with c1:
        scope_label = st.segmented_control(
            "Organ exclusion scope",
            ["Outline + shape", "Pigment only"],
            default="Outline + shape",
            key=f"organ_scope_{stem}_{cid}",
        )
        organ_type = st.selectbox(
            "Organ type",
            ["Unclassified", "Pistil", "Stamen"],
            key=f"organ_type_{stem}_{cid}",
        )
        default_organ_width = round(
            organ_seed["width_mm"] if organ_seed else 1.5, 1
        )
        organ_width_mm = st.slider(
            "Organ width (mm)",
            0.5,
            5.0,
            float(default_organ_width),
            0.1,
            key=f"organ_width_{stem}_{cid}",
        )
        organ_key = f"organ_{stem}_{cid}_{mgen}"
        initial_organ_line = raw_line_to_display(
            organ_seed["line"] if organ_seed else [cs["axis_base"], cs["axis_tip"]],
            box,
            DISPLAY_W,
            disp_h,
        )
        draft_line = component_value(
            st.session_state.get(organ_key), "line", initial_organ_line
        )
        existing_lines = [{
            "points": raw_line_to_display(item["line"], box, DISPLAY_W, disp_h),
            "colour": "rgba(210,0,120,.55)",
        } for item in cs.get("organ_exclusions", []) if len(item.get("line", [])) == 2]
        result = image_editor(
            key=organ_key,
            data={
                "image_url": image_url,
                "width": DISPLAY_W,
                "height": disp_h,
                "mode": "line",
                "polygons": display_mask,
                "line": draft_line,
                "line_colour": "#d00078",
                "context_lines": existing_lines,
            },
            default={"line": draft_line},
            on_line_change=lambda: None,
            width="stretch",
            height="content",
        )
        draft_line = component_value(result, "line", draft_line)
    with c2:
        raw_organ_line = display_line_to_raw(
            draft_line, box, DISPLAY_W, disp_h, raw.shape
        )
        st.metric(
            "organ length mm",
            round(trait_review.line_length(raw_organ_line) * mm_per_px, 3),
        )
        st.metric("saved exclusions", len(cs.get("organ_exclusions", [])))
        if st.button("Apply organ exclusion", width="stretch"):
            if organ_seed and organ_seed.get("polygons"):
                polygons = trait_review.transform_seed_polygons(
                    organ_seed["polygons"],
                    organ_seed["line"],
                    raw_organ_line,
                    organ_width_mm / max(organ_seed["width_mm"], 1e-9),
                    raw.shape,
                )
            else:
                polygons = buffered_line_polygon(
                    raw_organ_line,
                    organ_width_mm / max(mm_per_px, 1e-9),
                    raw.shape,
                )
            if polygons:
                cs["organ_exclusions"].append({
                    "line": raw_organ_line,
                    "width_mm": float(organ_width_mm),
                    "organ_type": organ_type.lower(),
                    "seed_candidate_id": organ_seed["candidate_id"] if organ_seed else "",
                    "scope": (
                        "outline" if scope_label == "Outline + shape" else "pigment"
                    ),
                    "polygons": polygons,
                })
                st.session_state.mask_gen = mgen + 1
                st.rerun()
        if st.button("Undo organ exclusion", width="stretch"):
            if cs["organ_exclusions"]:
                cs["organ_exclusions"].pop()
                st.session_state.mask_gen = mgen + 1
                st.rerun()

elif review_view == "All traits":
    if not trait_fields or not trait_row:
        st.info("No committed traits.csv row was found for this corolla.")
    else:
        live_updates = current_trait_updates(
            raw, masks[cid], cs, trait_row, preview=True
        )
        c1, c2 = st.columns([3, 1])
        with c1:
            review_rows = build_trait_review_rows(
                trait_fields, trait_row, live_updates, cs
            )
            edited_table = st.data_editor(
                review_rows,
                hide_index=True,
                width="stretch",
                height=600,
                disabled=["field", "original", "source"],
                column_config={
                    "field": st.column_config.TextColumn("trait"),
                    "original": st.column_config.TextColumn("original"),
                    "reviewed_value": st.column_config.TextColumn("reviewed"),
                    "source": st.column_config.TextColumn("source"),
                },
                key=f"traits_{stem}_{cid}_all",
            )
        with c2:
            shape_values = trait_review.shape_trait_values(edited, cs, mm_per_px)
            st.metric(
                "reviewed length mm",
                live_updates.get(
                    "corolla_length_ruler_mm", shape_values["corolla_length_ruler_mm"]
                ),
            )
            st.metric(
                "reviewed width mm",
                live_updates.get(
                    "corolla_max_span_ruler_mm", shape_values["corolla_max_span_ruler_mm"]
                ),
            )
            st.metric(
                "reviewed area mm²",
                live_updates.get(
                    "corolla_area_ruler_mm2", shape_values["corolla_area_ruler_mm2"]
                ),
            )
            if st.button("Apply trait edits", width="stretch"):
                apply_trait_review_table(
                    cs,
                    edited_records(edited_table),
                    trait_fields,
                    trait_row,
                    live_updates,
                )
                st.rerun()
            if st.button("Reset trait edits", width="stretch"):
                cs["trait_overrides"] = {}
                st.rerun()
            st.write(f"manual overrides: {len(cs.get('trait_overrides', {}))}")

st.sidebar.divider()
if st.sidebar.button("💾 Save state", width="stretch"):
    save_state(stem, state); st.sidebar.success("Saved state JSON")
if st.sidebar.button("⤓ Export reviewed CSVs + axis snippet", width="stretch"):
    save_state(stem, state)
    outdir = export_all(folder, stem, state, raw, masks, cen, trait_fields, trait_rows)
    st.sidebar.success(f"Exported → {outdir}")

# full-sheet preview (uses only already-reviewed state; no bulk axis compute)
with st.expander("Full-sheet preview (reviewed so far)"):
    canvas = raw.copy()
    for c in ids:
        s = state.get(str(c))
        m = apply_mask_edits(masks[c], s) if s else masks[c]
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        excl = bool(s and s["exclude"])
        cv2.drawContours(canvas, cnts, -1, (0, 0, 230) if excl else (0, 190, 0), 3)
        if s and not excl and (s["axis_changed"] or c == cid):
            b, t = s["axis_base"], s["axis_tip"]
            cv2.arrowedLine(canvas, (int(b[0]), int(b[1])), (int(t[0]), int(t[1])),
                            (0, 0, 220) if c == cid else (70, 70, 70), 3, cv2.LINE_AA, tipLength=0.03)
        ys, xs = np.where(m > 0)
        if len(xs):
            lab = f"C{c}" + (" EXCL" if excl else "") + (" +p" if (s and s["pistil"]) else "")
            cv2.putText(canvas, lab, (int(xs.mean()) - 30, int(ys.mean())),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    sc = 1100 / max(canvas.shape[:2])
    st.image(cv2.cvtColor(cv2.resize(canvas, None, fx=sc, fy=sc), cv2.COLOR_BGR2RGB),
             caption=f"{folder}/{stem}", width="stretch")
