# -*- coding: utf-8 -*-
"""Streamlit review app: one sheet at a time, manual correction of corolla MASK
and central AXIS (plus exclude / fold-state / pistil flags).

Everything is in the canonical ruler-at-top orientation. Per-corolla masks come
from the detector foreground (so paper-shadow that leaked into a mask is visible
and can be erased). Corrections export to the same reviewed formats the pipeline
uses (REVIEWED_AXIS_OVERRIDES snippet + reviewed CSVs).

Run:
    streamlit run review_app.py
"""
from __future__ import annotations
import os, json, math, csv, glob
import numpy as np, cv2
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates as click_image

import measure_guides as base
import measure_guides_symmetry_axis as sym
from PIL import Image

# streamlit-drawable-canvas needs streamlit.elements.image.image_to_url, which moved
# (and changed signature) in Streamlit >=1.3x. Restore an old-signature shim.
import streamlit.elements.image as _st_image_mod
if not hasattr(_st_image_mod, "image_to_url"):
    from streamlit.elements.lib.image_utils import image_to_url as _new_i2u
    from streamlit.elements.lib.layout_utils import create_layout_config as _clc

    def _image_to_url_compat(image, width, clamp, channels, output_format, image_id, *a, **k):
        lc = _clc(width=int(width)) if isinstance(width, (int, float)) else _clc()
        return _new_i2u(image, lc, clamp, channels, output_format, image_id)

    _st_image_mod.image_to_url = _image_to_url_compat
from streamlit_drawable_canvas import st_canvas

MM_PX = base.MM_PX
MM2_PX = base.MM2_PX
DATA_ROOT = "shimahotarubukuro"
REVIEW_DIR = "results/reviewed"
STATE_DIR = "results/review_state"
ISLAND_FOLDERS = ("oshima", "toshima", "niijima", "shikinejima", "kozushima")
DISPLAY_W = 720
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
    path = os.path.join(REVIEW_DIR, sheet_dash(stem), "traits.csv")
    if not os.path.exists(path):
        return None
    return list(csv.DictReader(open(path, encoding="utf-8-sig")))


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
        }
    return state[key]


# ----------------------------- geometry / rendering -----------------------------
def apply_mask_edits(mask, cs):
    m = mask.copy()
    for poly in cs.get("add", []):
        cv2.fillPoly(m, [np.array(poly, np.int32)], 1)
    for poly in cs.get("subtract", []):
        cv2.fillPoly(m, [np.array(poly, np.int32)], 0)
    return m


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
def export_all(folder, stem, state, masks, cen):
    outdir = os.path.join(REVIEW_DIR, sheet_dash(stem), "app_review")
    os.makedirs(outdir, exist_ok=True)
    island, _ = base.ISLANDS.get(folder, (folder, ""))
    axes_rows, hr_rows, excl_rows, corr_rows = [], [], [], []
    ovr_lines = []
    for cid in sorted(cen):
        cs = state.get(str(cid))
        if cs is None:
            continue
        edited = apply_mask_edits(masks[cid], cs)
        tr = corolla_traits(edited, cs["axis_base"], cs["axis_tip"])
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
            "mask_edited": "yes" if (cs["subtract"] or cs["add"]) else "",
            "note": cs["reason"],
        })
        if cs["exclude"]:
            excl_rows.append({"island": island, "sheet": stem, "corolla_id": cid,
                              "excluded": "yes", "reason": cs["reason"] or "excluded_by_review"})
        if cs["subtract"] or cs["add"]:
            corr_rows.append({"island": island, "sheet": stem, "corolla_id": cid,
                              "n_subtract_polys": len(cs["subtract"]), "n_add_polys": len(cs["add"]),
                              "kept_area_mm2": tr["area_mm2"]})
        if cs["axis_changed"]:
            ovr_lines.append(
                f'    ("{folder}", "{stem}", {cid}): {{\n'
                f'        "base_x": {cs["axis_base"][0]:.3f}, "base_y": {cs["axis_base"][1]:.3f},\n'
                f'        "tip_x": {cs["axis_tip"][0]:.3f}, "tip_y": {cs["axis_tip"][1]:.3f},\n'
                f'        "review_status": "ACCEPT",\n'
                f'        "review_note": "Reviewed in app (manual axis on central petal).",\n'
                f'    }},'
            )

    def _w(name, rows):
        p = os.path.join(outdir, name)
        if rows:
            with open(p, "w", newline="", encoding="utf-8-sig") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        else:
            open(p, "w").close()
        return p

    _w("reviewed_axes.csv", axes_rows)
    _w("human_review.csv", hr_rows)
    _w("reviewed_exclusions.csv", excl_rows)
    _w("reviewed_mask_corrections.csv", corr_rows)
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
cs["fold_state"] = st.sidebar.radio("Fold state", ["open", "folded_half"],
                                    index=0 if cs["fold_state"] == "open" else 1, horizontal=True)
cs["pistil"] = st.sidebar.checkbox("Visible attached pistil", value=cs["pistil"])

box = bbox_of(masks[cid], 80, raw.shape)
scale = DISPLAY_W / (box[2] - box[0])
disp_h = int((box[3] - box[1]) * scale)
mgen = st.session_state.get("mask_gen", 0)

tab_axis, tab_mask = st.tabs(["✎ Axis — click base→tip", "🖌 Mask — drag to paint"])

with tab_axis:
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
            st.rerun()
    with c2:
        tr = corolla_traits(edited, cs["axis_base"], cs["axis_tip"])
        st.metric("length mm", tr["length_mm"])
        st.metric("width mm", tr["width_mm"])
        st.metric("area mm²", tr["area_mm2"])
        if st.button("Reset axis to PRE-QC", width="stretch"):
            pa = get_preqc(cid)
            if pa:
                cs["axis_base"] = list(map(float, pa[0])); cs["axis_tip"] = list(map(float, pa[1]))
                cs["axis_changed"] = False
            st.rerun()

with tab_mask:
    c1, c2 = st.columns([3, 1])
    with c1:
        brush = st.slider("Brush size (px)", 5, 70, 24)
        mode = st.radio("Paint effect", ["SUBTRACT — erase shadow/noise", "ADD — recover tissue"], horizontal=True)
        edited = apply_mask_edits(masks[cid], cs)
        crop = raw[box[1]:box[3], box[0]:box[2]].copy()
        ecnts, _ = cv2.findContours(edited[box[1]:box[3], box[0]:box[2]], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(crop, ecnts, -1, (0, 190, 0), 2)
        bg = Image.fromarray(cv2.cvtColor(cv2.resize(crop, (DISPLAY_W, disp_h)), cv2.COLOR_BGR2RGB))
        stroke = "rgba(255,0,0,0.5)" if mode.startswith("SUBTRACT") else "rgba(0,200,255,0.5)"
        res = st_canvas(background_image=bg, drawing_mode="freedraw", stroke_width=brush,
                        stroke_color=stroke, height=disp_h, width=DISPLAY_W, display_toolbar=True,
                        key=f"mask_{stem}_{cid}_{mgen}")
    with c2:
        st.metric("area mm²", corolla_traits(edited, cs["axis_base"], cs["axis_tip"])["area_mm2"])
        st.caption("Drag over the region, then **Apply paint**.")
        if st.button("Apply paint", width="stretch"):
            if res is not None and res.image_data is not None:
                drawn = (res.image_data[..., 3] > 10).astype(np.uint8)
                if drawn.sum() > 20:
                    dm = cv2.resize(drawn, (box[2] - box[0], box[3] - box[1]), interpolation=cv2.INTER_NEAREST)
                    pcnts, _ = cv2.findContours(dm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    target = "subtract" if mode.startswith("SUBTRACT") else "add"
                    for pc in pcnts:
                        if cv2.contourArea(pc) < 15:
                            continue
                        cs[target].append([[float(px + box[0]), float(py + box[1])] for [[px, py]] in pc])
                    st.session_state.mask_gen = mgen + 1
                    st.rerun()
        if st.button("Undo last mask edit", width="stretch"):
            for k in ("subtract", "add"):
                if cs[k]:
                    cs[k].pop(); break
            st.session_state.mask_gen = mgen + 1
            st.rerun()

st.sidebar.divider()
if st.sidebar.button("💾 Save state", width="stretch"):
    save_state(stem, state); st.sidebar.success("Saved state JSON")
if st.sidebar.button("⤓ Export reviewed CSVs + axis snippet", width="stretch"):
    save_state(stem, state)
    outdir = export_all(folder, stem, state, masks, cen)
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
