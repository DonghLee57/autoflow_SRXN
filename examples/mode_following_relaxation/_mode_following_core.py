"""Backward-compatible shim — implementation moved to the package.

The actual implementation lives in:
    autoflow_srxn/vibrational/mode_following.py

This file exists only so that legacy callers can still do:
    from _mode_following_core import run_mode_following
"""
import os, sys
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from autoflow_srxn.vibrational.mode_following import run_mode_following  # noqa: F401

__all__ = ["run_mode_following"]
