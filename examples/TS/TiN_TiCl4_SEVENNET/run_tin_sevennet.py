"""TiN(111) + TiCl4 adsorption study using SevenNet-0 (7net-0).

Re-run of the TiN_TiCl4_MACE example with SevenNet-0 to validate
TiCl4 adsorption energetics on TiN(111).  SevenNet-0 is recommended for
CPU-only environments and covers the Ti-N-Cl chemical space.

Usage:
    cd examples/TS/TiN_TiCl4_SEVENNET
    python run_tin_sevennet.py              # uses config.yaml
    python run_tin_sevennet.py my.yaml      # custom config
"""
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from examples.DIPAS_on_Si110.run_adsorption import run_generic_adsorption_study

if __name__ == "__main__":
    print("--- Starting TiN(111) + TiCl4 Study (SevenNet-0) ---")
    print("Potential  : SevenNet 7net-0 on CPU")
    print("Surface    : TiN(111) 2x2 slab")
    print("Molecule   : TiCl4")
    print("Stages     : slab relax -> physi/chem candidates -> verification -> NEB+ARTn TS search")
    print("Est. time  : ~30-90 min depending on CPU (SevenNet is faster than MACE-MP on CPU)")
    print()

    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    if not os.path.exists(config_path):
        print(f"Error: config file '{config_path}' not found.")
        sys.exit(1)

    run_generic_adsorption_study(config_path)
