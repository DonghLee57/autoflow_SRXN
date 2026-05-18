import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from autoflow_srxn.surface.main_workflow import run_generic_adsorption_study

if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    run_generic_adsorption_study(config_path)
