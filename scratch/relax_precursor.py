import os
import sys
from ase.io import read, write
from ase.optimize import BFGS

# Add project root to path
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root)

try:
    from mace.calculators import mace_mp
    HAS_MACE = True
except ImportError:
    HAS_MACE = False

def relax_precursor():
    print("--- Relaxing AllylCpNi with MACE-MP ---")
    
    prec_path = "structures/AllylCpNi.vasp"
    if not os.path.exists(prec_path):
        print(f"Error: {prec_path} not found.")
        return

    atoms = read(prec_path)
    
    if not HAS_MACE:
        print("Error: MACE calculator not found. Please ensure 'mace-torch' is installed.")
        return

    # Initialize MACE-MP Calculator
    print("Loading MACE-MP (medium, float64)...")
    calc = mace_mp(model="medium", device="cpu", default_dtype="float64")
    atoms.calc = calc
    
    # Run Relaxation
    print("Starting BFGS relaxation (fmax=0.01)...")
    dyn = BFGS(atoms, trajectory="scratch/relax_prec.traj", logfile="scratch/relax_prec.log")
    dyn.run(fmax=0.01, steps=100)
    
    # Save Final Structure
    print(f"Relaxation complete. Saving to {prec_path}")
    # Update back to original VASP file
    write(prec_path, atoms, format="vasp", vasp5=True, direct=True)
    print("DONE.")

if __name__ == "__main__":
    relax_precursor()
