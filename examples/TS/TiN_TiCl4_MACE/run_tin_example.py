"""TiN(111) + TiCl4 adsorption study using MACE-MP."""
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from autoflow_srxn.surface import run_generic_adsorption_study

if __name__ == "__main__":
    print("--- Starting TiN(111) + TiCl4 (MACE-MP) Study ---")
    
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        sys.exit(1)
        
    run_generic_adsorption_study(config_path)
