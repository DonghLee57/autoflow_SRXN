"""Mode-following relaxation — run from this directory.

    python run_mode_following.py

Reads config.yaml in the same directory.  All options (structure type,
PHVA/FHVA, vacuum centering, frozen zone) are set there.
"""
import os, sys
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, "..", "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)
from autoflow_srxn.vibrational.mode_following import run_mode_following

if __name__ == "__main__":
    run_mode_following(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"))
