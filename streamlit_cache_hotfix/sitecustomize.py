"""Targeted cache patch for the Streamlit review app.

``load_sheet`` returns a full-resolution scan plus every full-sheet mask. Keeping
that value in ``st.cache_data`` pickles and unpickles hundreds of megabytes and
can raise ``MemoryError``. This module changes only that decorator to
``st.cache_resource``.

The patch is installed lazily, after Streamlit's normal import has completed.
Importing Streamlit directly from ``sitecustomize`` can collide with the
Streamlit CLI while Python is still starting and make the app process exit
before the health endpoint is available.
"""
from __future__ import annotations

import builtins
import os
import sys


def _running_review_app() -> bool:
    return any(
        os.path.basename(str(argument)).lower() == "review_app.py"
        for argument in sys.argv
    )


if _running_review_app():
    _original_import = builtins.__import__
    _patch_installed = False

    def _install_patch_if_ready() -> bool:
        global _patch_installed
        if _patch_installed:
            return True

        streamlit_module = sys.modules.get("streamlit")
        if streamlit_module is None:
            return False
        if not hasattr(streamlit_module, "cache_data") or not hasattr(
            streamlit_module, "cache_resource"
        ):
            return False

        original_cache_data = streamlit_module.cache_data
        original_cache_resource = streamlit_module.cache_resource

        class _TargetedCacheDataProxy:
            def __call__(self, func=None, **kwargs):
                def decorate(target):
                    if getattr(target, "__name__", "") == "load_sheet":
                        return original_cache_resource(
                            show_spinner=kwargs.get("show_spinner", True),
                            max_entries=1,
                        )(target)
                    if kwargs:
                        return original_cache_data(**kwargs)(target)
                    return original_cache_data(target)

                return decorate if func is None else decorate(func)

            def clear(self):
                return original_cache_data.clear()

            def __getattr__(self, name):
                return getattr(original_cache_data, name)

        streamlit_module.cache_data = _TargetedCacheDataProxy()
        _patch_installed = True
        builtins.__import__ = _original_import
        return True

    def _deferred_import(name, globals=None, locals=None, fromlist=(), level=0):
        module = _original_import(name, globals, locals, fromlist, level)
        if name == "streamlit" or name.startswith("streamlit."):
            try:
                _install_patch_if_ready()
            except Exception:
                # Fail open: a cache optimization must never prevent app startup.
                builtins.__import__ = _original_import
        return module

    builtins.__import__ = _deferred_import
