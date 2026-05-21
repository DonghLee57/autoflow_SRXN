"""
t-ZrO2 / B-Nb2O5 interface screening
Usage: python run_interface.py [config.yaml]
"""
import sys, os
from autoflow_srxn.utils import load_yaml_config
from autoflow_srxn.interface import run_interface_screening

if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    run_interface_screening(load_yaml_config(cfg_path))
