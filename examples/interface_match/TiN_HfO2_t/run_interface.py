"""
Interface Lattice-Match Search — minimal runner
==============================================
This script uses the core :func:`~autoflow_srxn.interface.run_interface_screening` 
utility to perform 2D ZSL screening and slab construction.

Usage
-----
    python run_interface.py              # reads config.yaml in this directory
    python run_interface.py config.yaml  # explicit config path
"""

import sys
import os
from autoflow_srxn.utils import load_yaml_config
from autoflow_srxn.interface import run_interface_screening

if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    if not os.path.exists(config_file):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(script_dir, "config.yaml")
    
    if not os.path.exists(config_file):
        print(f"Error: Config file not found: {config_file}")
        sys.exit(1)
        
    config = load_yaml_config(config_file)
    run_interface_screening(config)
