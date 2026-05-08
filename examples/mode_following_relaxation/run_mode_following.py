"""Mode-following relaxation — legacy entry point (parent directory).

Prefer running from within the sub-example directories:

    cd precursor && python run_mode_following.py
    cd slab     && python run_mode_following.py

This script is kept for backward compatibility.  It delegates directly to
:func:`autoflow_srxn.vibrational.mode_following.run_mode_following`.

Usage
-----
    python run_mode_following.py <config.yaml>
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from autoflow_srxn.vibrational.mode_following import run_mode_following

if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    run_mode_following(os.path.abspath(cfg))
