"""TiN(111) + TiCl4 adsorption study using SevenNet-0 (7net-0)."""
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from autoflow_srxn.surface.main_workflow import run_generic_adsorption_study

if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    if not os.path.exists(config_path):
        print(f"Error: config file '{config_path}' not found.")
        sys.exit(1)

    run_generic_adsorption_study(config_path)
