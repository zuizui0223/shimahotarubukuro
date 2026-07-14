# -*- coding: utf-8 -*-
"""Read-only emergency recovery page for Streamlit review state."""
from __future__ import annotations

import io
import json
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="レビュー状態の緊急復旧", layout="wide")

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
STATE_DIR = RESULTS_DIR / "review_state"


@st.cache_data(show_spinner=False)
def build_recovery_archive(signature: tuple[tuple[str, int, int], ...]) -> bytes:
    """Create an in-memory ZIP without modifying any source file."""
    del signature
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(STATE_DIR.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(RESULTS_DIR))
        manifest = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source": str(STATE_DIR),
            "warning": "Recovered from Streamlit local runtime storage; preserve this ZIP.",
        }
        archive.writestr(
            "review_state/RECOVERY_MANIFEST.json",
            json.dumps(manifest, ensure_ascii=False, indent=2),
        )
    return payload.getvalue()


def state_files() -> list[Path]:
    if not STATE_DIR.is_dir():
        return []
    return sorted(path for path in STATE_DIR.rglob("*") if path.is_file())


st.title("レビュー状態の緊急復旧")
st.error(
    "レビュー作業は一時停止しています。この画面は残っている保存データを"
    "書き換えずに回収するためだけのものです。"
)
st.warning("回収が終わるまで、Manage app の Reboot・Delete・再デプロイはしないでください。")

files = state_files()
if not files:
    st.error(
        "現在のコンテナ内に results/review_state の保存ファイルが見つかりませんでした。"
    )
    st.caption(
        "この表示の場合、過去の再デプロイ時にローカル保存領域が交換された可能性が高いです。"
    )
    st.stop()

signature = tuple(
    (
        str(path.relative_to(STATE_DIR)),
        int(path.stat().st_size),
        int(path.stat().st_mtime_ns),
    )
    for path in files
)
total_bytes = sum(size for _, size, _ in signature)
json_files = [path for path in files if path.suffix == ".json"]
backup_files = [path for path in files if path.name.endswith(".bak")]
corolla_files = [
    path
    for path in json_files
    if path.parent != STATE_DIR and path.stem.startswith("C") and path.stem[1:].isdigit()
]

left, middle, right = st.columns(3)
left.metric("残存ファイル", len(files))
middle.metric("花冠スナップショット", len(corolla_files))
right.metric("バックアップ", len(backup_files))

st.success(
    f"{len(files)}個、合計 {total_bytes / 1024:.1f} KiB の保存データが残っています。"
)
archive = build_recovery_archive(signature)
st.download_button(
    "残っているレビュー状態をZIPで保存",
    data=archive,
    file_name="shimahotarubukuro_review_state_recovery.zip",
    mime="application/zip",
    type="primary",
    width="stretch",
)

with st.expander("残存ファイル一覧"):
    lines = []
    for relative, size, mtime_ns in signature:
        modified = datetime.fromtimestamp(mtime_ns / 1_000_000_000).astimezone()
        lines.append(f"{relative}\t{size} bytes\t{modified.isoformat(timespec='seconds')}")
    st.code("\n".join(lines), language=None)

st.info(
    "ZIPを手元に保存できるまでレビューは再開しません。回収後は、"
    "Streamlitの一時ディスクではなく外部の永続保存先へ毎回同期する構成に変更します。"
)
