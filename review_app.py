# -*- coding: utf-8 -*-
"""Crash-guarded launcher for the floral-trait review app."""
from __future__ import annotations

import faulthandler
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
st.set_page_config = lambda *args, **kwargs: None
try:
    print("[review_app bootstrap] starting review runtime", flush=True)
    runpy.run_path(str(RUNTIME_PATH), run_name="__main__")
finally:
    st.set_page_config = _original_set_page_config
