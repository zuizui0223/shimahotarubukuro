# -*- coding: utf-8 -*-
"""Crash-guarded launcher for the floral-trait review app."""
from __future__ import annotations

import faulthandler
import html
import json
import os
import runpy
import sys
from pathlib import Path

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_name, "1")
os.environ.setdefault("OPENCV_OPENCL_RUNTIME", "disabled")

faulthandler.enable(all_threads=True)
print(
    f"[review_app bootstrap] python={sys.version.split()[0]} platform={sys.platform}",
    flush=True,
)

import streamlit as st

st.set_page_config(page_title="花形質レビュー", layout="wide")

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "shimahotarubukuro"
STATE_DIR = ROOT / "results" / "review_state"
RESUME_PATH = STATE_DIR / "_resume.json"
RUNTIME_PATH = ROOT / "review_app_runtime.py"
ISLAND_FOLDERS = ("oshima", "toshima", "niijima", "shikinejima", "kozushima")


def sheet_labels() -> list[str]:
    labels: list[str] = []
    for folder in ISLAND_FOLDERS:
        for image_path in sorted((DATA_ROOT / folder).glob("*.jpg")):
            labels.append(f"{folder}/{image_path.stem}")
    return labels


def read_resume() -> dict:
    for path in (RESUME_PATH, Path(f"{RESUME_PATH}.bak")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                return value
        except (OSError, ValueError, TypeError):
            continue
    return {}


def save_resume_sheet(sheet: str) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    value = read_resume()
    value["sheet"] = sheet
    temporary = Path(f"{RESUME_PATH}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    if RESUME_PATH.exists():
        try:
            Path(f"{RESUME_PATH}.bak").write_bytes(RESUME_PATH.read_bytes())
        except OSError:
            pass
    os.replace(temporary, RESUME_PATH)


def _safe_dataframe(data=None, *args, **kwargs):
    """Render small review tables without pandas/pyarrow conversion.

    Streamlit's dataframe conversion can enter pandas' Arrow-backed string
    constructor and crash the worker in native code. The review app only passes
    short lists of dictionaries here, so a plain HTML table is safer.
    """
    del args, kwargs
    if data is None:
        st.caption("データなし")
        return None

    if isinstance(data, dict):
        rows = [data]
    elif isinstance(data, (list, tuple)):
        rows = list(data)
    else:
        st.code(str(data))
        return None

    if not rows:
        st.caption("データなし")
        return None

    if not all(isinstance(row, dict) for row in rows):
        st.code("\n".join(str(row) for row in rows))
        return None

    headers: list[str] = []
    for row in rows:
        for key in row:
            text = str(key)
            if text not in headers:
                headers.append(text)

    head = "".join(
        f"<th style='padding:.45rem .6rem;text-align:left;"
        f"border-bottom:1px solid #d9d9d9'>{html.escape(header)}</th>"
        for header in headers
    )
    body_rows = []
    for row in rows:
        cells = "".join(
            f"<td style='padding:.42rem .6rem;border-bottom:1px solid #eeeeee'>"
            f"{html.escape(str(row.get(header, '')))}</td>"
            for header in headers
        )
        body_rows.append(f"<tr>{cells}</tr>")

    st.markdown(
        "<div style='overflow-x:auto'>"
        "<table style='width:100%;border-collapse:collapse;font-size:.92rem'>"
        f"<thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table></div>",
        unsafe_allow_html=True,
    )
    return None


labels = sheet_labels()
if not labels:
    st.error("shimahotarubukuro/ に原画像がありません。")
    st.stop()

resume = read_resume()
default_sheet = resume.get("sheet", "oshima/oshima10~13")
if default_sheet not in labels:
    default_sheet = labels[0]

if not st.session_state.get("_review_runtime_enabled", False):
    st.markdown("## シマホタルブクロ花形質レビュー")
    st.success("花冠ごとの編集内容は自動保存されます。")
    st.warning(
        "画像は起動時には読み込みません。シートを選び、下のボタンを押した後に"
        "選択した1枚だけを読み込みます。"
    )
    selected_sheet = st.selectbox(
        "レビューするシート",
        labels,
        index=labels.index(default_sheet),
        key="_bootstrap_sheet_selector",
    )
    if st.button("レビューを開始・再開", type="primary", width="stretch"):
        save_resume_sheet(selected_sheet)
        st.session_state["_review_runtime_enabled"] = True
        st.rerun()
    st.stop()

if st.sidebar.button("安全なシート選択画面へ戻る", width="stretch"):
    st.session_state["_review_runtime_enabled"] = False
    st.rerun()

print("[review_app bootstrap] importing numpy", flush=True)
import numpy as np
print(f"[review_app bootstrap] numpy={np.__version__}", flush=True)

print("[review_app bootstrap] importing cv2", flush=True)
import cv2
cv2.setNumThreads(1)
try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass
print(f"[review_app bootstrap] cv2={cv2.__version__}", flush=True)

_original_set_page_config = st.set_page_config
_original_dataframe = st.dataframe
st.set_page_config = lambda *args, **kwargs: None
st.dataframe = _safe_dataframe
try:
    print("[review_app bootstrap] starting review runtime", flush=True)
    runpy.run_path(str(RUNTIME_PATH), run_name="__main__")
finally:
    st.dataframe = _original_dataframe
    st.set_page_config = _original_set_page_config
