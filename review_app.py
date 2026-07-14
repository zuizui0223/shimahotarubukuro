# -*- coding: utf-8 -*-
"""Reliable Streamlit entrypoint: lazy masks, atomic autosave, and crash resume."""
from __future__ import annotations

from pathlib import Path


def _replace_section(source: str, start: str, end: str, replacement: str, label: str) -> str:
    start_at = source.find(start)
    if start_at < 0:
        raise RuntimeError(f"review app patch failed: missing {label} start marker")
    end_at = source.find(end, start_at)
    if end_at < 0:
        raise RuntimeError(f"review app patch failed: missing {label} end marker")
    return source[:start_at] + replacement.rstrip() + "\n\n" + source[end_at:]


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"review app patch failed: expected one {label}, found {count}")
    return source.replace(old, new, 1)


impl_path = Path(__file__).with_name("review_app_impl.py")
source = impl_path.read_text(encoding="utf-8")

lazy_loader = r'''
class LazyMaskStore:
    """Generate only the requested full-resolution mask and retain at most two."""

    def __init__(self, shape, contour_of, bounds_of, max_cached=2):
        from collections import OrderedDict
        self.shape = tuple(shape)
        self._contour_of = dict(contour_of)
        self._bounds_of = dict(bounds_of)
        self._max_cached = max(1, int(max_cached))
        self._cache = OrderedDict()

    def __getitem__(self, cid):
        cid = int(cid)
        cached = self._cache.pop(cid, None)
        if cached is not None:
            self._cache[cid] = cached
            return cached
        mask = np.zeros(self.shape, np.uint8)
        contour = self._contour_of.get(cid)
        if contour is not None:
            cv2.drawContours(mask, [contour], -1, 1, -1)
        left, right = self._bounds_of.get(cid, (0, self.shape[1]))
        if left > 0:
            mask[:, :left] = 0
        if right < self.shape[1]:
            mask[:, right:] = 0
        self._cache[cid] = mask
        while len(self._cache) > self._max_cached:
            self._cache.popitem(last=False)
        return mask

    def __iter__(self):
        return iter(self._contour_of)

    def __len__(self):
        return len(self._contour_of)

    def keys(self):
        return self._contour_of.keys()

    def items(self):
        for cid in self._contour_of:
            yield cid, self[cid]


@st.cache_resource(
    show_spinner="Loading sheet + PRE-QC masks/axes…",
    max_entries=1,
)
def load_sheet(
    folder: str,
    stem: str,
    path: str,
    overlay_path: str,
    overlay_version: float,
):
    """Load one scan and keep contours; materialize masks only when requested."""
    del folder, overlay_version
    ov = cv2.imdecode(np.fromfile(overlay_path, np.uint8), cv2.IMREAD_COLOR)
    if ov is None:
        raise RuntimeError(f"レビュー用オーバーレイを読めません: {overlay_path}")
    raw = _load_canonical_raw(path)
    rh, rw = raw.shape[:2]
    oh, ow = ov.shape[:2]
    sx, sy = rw / ow, rh / oh
    b, g, r = cv2.split(ov)
    green = ((g > 165) & (r < 135) & (b < 150)).astype(np.uint8) * 255
    mag = ((r > 150) & (b > 150) & (g < 120)).astype(np.uint8) * 255
    outl = cv2.morphologyEx(
        cv2.bitwise_or(green, mag),
        cv2.MORPH_CLOSE,
        np.ones((3, 3), np.uint8),
    )
    cnts, _ = cv2.findContours(outl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = [c for c in cnts if cv2.contourArea(c) > 1000]
    raw_cnts = []
    for contour in cnts:
        scaled = contour.astype(np.float32).copy()
        scaled[:, 0, 0] *= sx
        scaled[:, 0, 1] *= sy
        raw_cnts.append(np.rint(scaled).astype(np.int32))
    del ov, b, g, r, green, mag, outl, cnts

    rows = committed_traits(stem)
    if rows:
        cen = {
            int(item["corolla_id"]): (float(item["cx"]), float(item["cy"]))
            for item in rows
        }
    else:
        provisional = {}
        for index, contour in enumerate(raw_cnts, start=1):
            moments = cv2.moments(contour)
            if moments["m00"]:
                provisional[index] = (
                    float(moments["m10"] / moments["m00"]),
                    float(moments["m01"] / moments["m00"]),
                )
        ordered = sorted(
            provisional,
            key=lambda key: (provisional[key][1] // 200, provisional[key][0]),
        )
        cen = {index + 1: provisional[key] for index, key in enumerate(ordered)}

    from collections import defaultdict
    region_of = {}
    for cid, (cx, cy) in cen.items():
        region = 0
        for index, contour in enumerate(raw_cnts, start=1):
            if cv2.pointPolygonTest(contour, (float(cx), float(cy)), False) >= 0:
                region = index
                break
        region_of[cid] = region

    groups = defaultdict(list)
    for cid, region in region_of.items():
        groups[region].append(cid)

    contour_of = {}
    bounds_of = {cid: (0, rw) for cid in cen}
    for region, cids in groups.items():
        contour = raw_cnts[region - 1] if region > 0 else None
        for cid in cids:
            contour_of[cid] = contour
        if region > 0 and len(cids) > 1:
            ordered = sorted(cids, key=lambda value: cen[value][0])
            bounds = [0] + [
                int(round((cen[ordered[i]][0] + cen[ordered[i + 1]][0]) / 2))
                for i in range(len(ordered) - 1)
            ] + [rw]
            for index, cid in enumerate(ordered):
                bounds_of[cid] = (bounds[index], bounds[index + 1])

    masks = LazyMaskStore((rh, rw), contour_of, bounds_of, max_cached=2)
    return raw, masks, cen
'''
source = _replace_section(
    source,
    '@st.cache_data(show_spinner="Loading sheet + PRE-QC masks/axes…")\ndef load_sheet(',
    '@st.cache_data(show_spinner=False)\ndef preqc_axis',
    lazy_loader,
    "sheet loader",
)

state_helpers = r'''
def state_path(stem):
    return os.path.join(STATE_DIR, f"{sheet_dash(stem)}.json")


def state_snapshot_dir(stem):
    return os.path.join(STATE_DIR, sheet_dash(stem))


def resume_path():
    return os.path.join(STATE_DIR, "_resume.json")


def _read_json_with_backup(path, default):
    for candidate in (path, f"{path}.bak"):
        if not os.path.exists(candidate):
            continue
        try:
            with open(candidate, encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, ValueError, TypeError):
            continue
    return default


def _atomic_json_write(path, payload):
    import shutil
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp.{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1)
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:
            pass
    if os.path.exists(path):
        try:
            shutil.copy2(path, f"{path}.bak")
        except OSError:
            pass
    os.replace(temporary, path)


_STATE_DIGESTS = {}


def load_state(stem):
    state = _read_json_with_backup(state_path(stem), {})
    if not isinstance(state, dict):
        state = {}
    snapshot_dir = state_snapshot_dir(stem)
    if os.path.isdir(snapshot_dir):
        for filename in os.listdir(snapshot_dir):
            if filename.startswith("C") and filename.endswith(".json"):
                cid = filename[1:-5]
                snapshot = _read_json_with_backup(
                    os.path.join(snapshot_dir, filename), None
                )
                if cid.isdigit() and isinstance(snapshot, dict):
                    state[cid] = snapshot
        organs = _read_json_with_backup(
            os.path.join(snapshot_dir, "_organs.json"), None
        )
        if isinstance(organs, list):
            state[ORGAN_STATE_KEY] = organs
    return state


def save_state(stem, state):
    import hashlib
    _atomic_json_write(state_path(stem), state)
    snapshot_dir = state_snapshot_dir(stem)
    for key, value in state.items():
        if str(key).isdigit():
            filename = f"C{key}.json"
        elif key == ORGAN_STATE_KEY:
            filename = "_organs.json"
        else:
            continue
        serialized = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        digest_key = (stem, filename)
        if _STATE_DIGESTS.get(digest_key) == digest:
            continue
        _atomic_json_write(os.path.join(snapshot_dir, filename), value)
        _STATE_DIGESTS[digest_key] = digest


def load_resume():
    value = _read_json_with_backup(resume_path(), {})
    return value if isinstance(value, dict) else {}


def save_resume(**updates):
    resume = load_resume()
    for key, value in updates.items():
        if value is None:
            resume.pop(key, None)
        else:
            resume[key] = value
    _atomic_json_write(resume_path(), resume)
    return resume
'''
source = _replace_section(
    source,
    'def state_path(stem):',
    'def organ_id_sort_key(value):',
    state_helpers,
    "state helpers",
)

source = _replace_once(
    source,
    '''sheet_labels = [f"{folder}/{stem}" for folder, stem, _ in sheets]\ndefault_sheet = "oshima/oshima10~13"\nselected_sheet = st.sidebar.selectbox(\n    "シート",\n    sheet_labels,\n    index=sheet_labels.index(default_sheet) if default_sheet in sheet_labels else 0,\n)\nfolder, stem, path = sheets[sheet_labels.index(selected_sheet)]''',
    '''sheet_labels = [f"{folder}/{stem}" for folder, stem, _ in sheets]\nresume = load_resume()\ndefault_sheet = resume.get("sheet", "oshima/oshima10~13")\nif default_sheet not in sheet_labels:\n    default_sheet = sheet_labels[0]\nselected_sheet = st.sidebar.selectbox(\n    "シート",\n    sheet_labels,\n    index=sheet_labels.index(default_sheet),\n)\nsave_resume(pending_sheet=selected_sheet)\nfolder, stem, path = sheets[sheet_labels.index(selected_sheet)]''',
    "sheet resume selector",
)

source = _replace_once(
    source,
    '''raw, masks, cen = load_sheet(\n    folder,\n    stem,\n    path,\n    overlay_path,\n    os.path.getmtime(overlay_path),\n)\ntrait_fields, trait_rows = committed_trait_table(stem)''',
    '''raw, masks, cen = load_sheet(\n    folder,\n    stem,\n    path,\n    overlay_path,\n    os.path.getmtime(overlay_path),\n)\nsave_resume(sheet=selected_sheet, pending_sheet=None)\ntrait_fields, trait_rows = committed_trait_table(stem)''',
    "successful sheet load marker",
)

source = _replace_once(
    source,
    '''ids = sorted(cen)\ncid = st.sidebar.selectbox(\n    "花冠",\n    ids,\n    format_func=lambda value: f"C{value}",\n    key=f"corolla_{stem}",\n)''',
    '''ids = sorted(cen)\nresume_cid = resume.get("cid")\nresume_cid = int(resume_cid) if str(resume_cid).isdigit() else None\ncid = st.sidebar.selectbox(\n    "花冠",\n    ids,\n    index=ids.index(resume_cid) if resume_cid in ids else 0,\n    format_func=lambda value: f"C{value}",\n    key=f"corolla_{stem}",\n)\nsave_resume(sheet=selected_sheet, cid=int(cid))''',
    "corolla resume selector",
)

source = _replace_once(
    source,
    '''def commit_change():\n    save_state(stem, state)\n    st.session_state.editor_generation = editor_generation + 1\n    st.rerun()''',
    '''def commit_change():\n    save_state(stem, state)\n    save_resume(sheet=selected_sheet, cid=int(cid), stage=stage)\n    st.session_state.editor_generation = editor_generation + 1\n    st.rerun()''',
    "commit autosave",
)

source = _replace_once(
    source,
    '''stage = st.segmented_control(\n    "レビュー工程",\n    ["マスク", "形", "斑点", "雄しべ・雌しべ", "確認"],\n    default="マスク",\n    key=f"review_stage_{stem}_{cid}",\n    width="stretch",\n) or "マスク"''',
    '''stage_options = ["マスク", "形", "斑点", "雄しべ・雌しべ", "確認"]\nresume_stage = resume.get("stage", "マスク")\nif resume_stage not in stage_options:\n    resume_stage = "マスク"\nstage = st.segmented_control(\n    "レビュー工程",\n    stage_options,\n    default=resume_stage,\n    key=f"review_stage_{stem}_{cid}",\n    width="stretch",\n) or resume_stage\nsave_resume(sheet=selected_sheet, cid=int(cid), stage=stage)''',
    "stage resume selector",
)

source = _replace_once(
    source,
    '''        standardized_guide_area = colour_values[\n            "guide_area_incl_oxidized_standardized_mm2"\n        ]''',
    '''        new_presence = presence_labels[presence_label]\n        if new_presence != cs.get("guide_presence_reviewed", "unreviewed"):\n            cs["guide_presence_reviewed"] = new_presence\n            save_state(stem, state)\n        standardized_guide_area = colour_values[\n            "guide_area_incl_oxidized_standardized_mm2"\n        ]''',
    "guide judgement autosave",
)

source = _replace_once(
    source,
    '''        if st.button(\n            "確認内容を保存",\n            type="primary",\n            width="stretch",\n            key=f"save_qc_{stem}_{cid}",\n        ):\n            new_fold = FOLD_STATE_LABELS[selected_fold_label]\n            if new_fold != cs.get("fold_state"):\n                cs["fold_changed"] = True\n            cs["fold_state"] = new_fold\n            cs["exclude"] = exclude_corolla\n            cs["reason"] = review_note\n            cs["review_complete"] = review_complete\n            commit_change()''',
    '''        new_fold = FOLD_STATE_LABELS[selected_fold_label]\n        if new_fold != cs.get("fold_state"):\n            cs["fold_changed"] = True\n        cs["fold_state"] = new_fold\n        cs["exclude"] = exclude_corolla\n        cs["reason"] = review_note\n        cs["review_complete"] = review_complete\n        save_state(stem, state)\n        save_resume(sheet=selected_sheet, cid=int(cid), stage=stage)\n        st.caption("この花冠は自動保存済みです")''',
    "confirmation autosave",
)

preview_start = 'with st.expander("シート全体"):'
preview_at = source.find(preview_start)
if preview_at < 0:
    raise RuntimeError("review app patch failed: missing sheet preview")
source = source[:preview_at] + r'''with st.expander("シート全体"):
    st.caption("メモリ節約のため、全体図は必要なときだけ生成します。")
    if st.button("シート全体図を生成", key=f"sheet_preview_{stem}"):
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
'''

namespace = {
    "__name__": "__main__",
    "__file__": str(impl_path),
    "__package__": None,
}
exec(compile(source, str(impl_path), "exec"), namespace, namespace)
