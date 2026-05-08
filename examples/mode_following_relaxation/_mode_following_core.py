"""Backward-compatible shim — implementation moved to the package.

The actual implementation lives in:
    autoflow_srxn/vibrational/mode_following.py

This file exists only so that the subdir wrappers (precursor/, slab/) can
continue to do ``from _mode_following_core import run_mode_following`` while
the real code is maintained inside the installable package.
"""
from autoflow_srxn.vibrational.mode_following import run_mode_following  # noqa: F401

__all__ = ["run_mode_following"]
