import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from examples.DIPAS_on_Si110.run_adsorption import run_generic_adsorption_study

if __name__ == "__main__":
    print("--- Starting TiN(111) + TiCl4 (MACE-MP) Study ---")
    print("This study simulates TiCl4 adsorption and potential dissociation on TiN(111).")
    print("Potential: MACE-MP (medium) on CPU")
    print("Estimated time: ~20-60 minutes depending on CPU speed.")
    
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        sys.exit(1)
        
    run_generic_adsorption_study(config_path)
