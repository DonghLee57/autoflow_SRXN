"""Mode-following relaxation — run from this directory.

    python run_mode_following.py

Reads config.yaml in the same directory.
"""
import os
from autoflow_srxn import run_mode_following

if __name__ == "__main__":
    # Get absolute path to config.yaml in the same directory as this script
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    run_mode_following(cfg_path)
