# -*- coding: utf-8 -*-
"""Streamlit entrypoint with a targeted, non-pickling sheet cache.

The implementation returns a full-resolution image and all masks from
``load_sheet``. Decorating that function with ``st.cache_data`` serializes the
whole result and can fail while unpickling with ``MemoryError``. This entrypoint
changes only that decorator to ``st.cache_resource(max_entries=1)`` while the
implementation is executed, then restores Streamlit's normal cache API.
"""
from __future__ import annotations

import os
import runpy

import streamlit as st


class _TargetedCacheDataProxy:
    def __init__(self, cache_data, cache_resource):
        self._cache_data = cache_data
        self._cache_resource = cache_resource

    def __call__(self, func=None, **kwargs):
        def decorate(target):
            if getattr(target, "__name__", "") == "load_sheet":
                return self._cache_resource(
                    show_spinner=kwargs.get("show_spinner", True),
                    max_entries=1,
                )(target)
            if kwargs:
                return self._cache_data(**kwargs)(target)
            return self._cache_data(target)

        return decorate if func is None else decorate(func)

    def clear(self):
        return self._cache_data.clear()

    def __getattr__(self, name):
        return getattr(self._cache_data, name)


_original_cache_data = st.cache_data
st.cache_data = _TargetedCacheDataProxy(st.cache_data, st.cache_resource)
try:
    runpy.run_path(
        os.path.join(os.path.dirname(__file__), "review_app_impl.py"),
        run_name="__main__",
    )
finally:
    st.cache_data = _original_cache_data
