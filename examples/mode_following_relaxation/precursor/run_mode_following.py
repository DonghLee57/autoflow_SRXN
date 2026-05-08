"""Mode-following relaxation — run from this directory.

    python run_mode_following.py

Reads config.yaml in the same directory.  All options (structure type,
PHVA/FHVA, vacuum centering, frozen zone) are set there.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _mode_following_core import run_mode_following

if __name__ == "__main__":
    run_mode_following(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"))
