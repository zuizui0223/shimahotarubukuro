"""Targeted startup patch for the Streamlit review app.

The review app previously cached a full-resolution image plus every full-sheet
mask with ``st.cache_data``. That cache is pickled, so restoring it can require
another large contiguous allocation and fail with ``MemoryError``. Only the
``load_sheet`` decorator is redirected to ``st.cache_resource``; all other
``st.cache_data`` uses keep their normal behavior.
"""
from __future__ import annotations

import os
import sys


def _running_review_app() -> bool:
    return any(
        os.path.basename(str(argument)).lower() == "review_app.py"
        for argument in sys.argv
    )


if _running_review_app():
    import streamlit as _st

    _original_cache_data = _st.cache_data
    _original_cache_resource = _st.cache_resource

    class _TargetedCacheDataProxy:
        def __call__(self, func=None, **kwargs):
            def decorate(target):
                if getattr(target, "__name__", "") == "load_sheet":
                    return _original_cache_resource(
                        show_spinner=kwargs.get("show_spinner", True),
                        max_entries=1,
                    )(target)
                if kwargs:
                    return _original_cache_data(**kwargs)(target)
                return _original_cache_data(target)

            return decorate if func is None else decorate(func)

        def clear(self):
            return _original_cache_data.clear()

        def __getattr__(self, name):
            return getattr(_original_cache_data, name)

    _st.cache_data = _TargetedCacheDataProxy()
