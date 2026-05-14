import os
import sys
import numpy as np
from ase.io import read

# Add project root to path
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root)

from autoflow_srxn.surface.ads_workflow_mgr import AdsorptionWorkflowManager

def test_alignment_logic():
    print("--- Testing Physisorption Alignment (PCA + H-up Flip) ---")
    
    # 1. Load Inhibitor
    prec_path = "structures/secret_inhibitor.vasp"
    molecule = read(prec_path)
    print(f"Loaded Molecule: {molecule.get_chemical_formula()}")

    # 2. Setup Manager (Dummy slab is fine for internal alignment test)
    from ase import Atoms
    dummy_slab = Atoms('Si', positions=[[0,0,0]], cell=[10,10,10], pbc=True)
    mgr = AdsorptionWorkflowManager(dummy_slab)
    
    # 3. Apply Alignment
    print("Applying _get_physi_alignment(mode='com')...")
    aligned = mgr._get_physi_alignment(molecule, mode="com")
    
    # 4. Analyze Results
    h_indices = [a.index for a in aligned if a.symbol == "H"]
    h_pos = aligned.positions[h_indices]
    avg_h_z = np.mean(h_pos[:, 2])
    
    print(f"Average H-Z coordinate: {avg_h_z:.6f} A")
    
    # Check Planarity (Z-variance should be minimal)
    z_std = np.std(aligned.positions[:, 2])
    print(f"Z-axis Standard Deviation (Flatness): {z_std:.6f} A")

    if avg_h_z > 0:
        print("SUCCESS: Hydrogens are pointing UP (+z). Flip logic worked!")
    else:
        print("FAILURE: Hydrogens are pointing DOWN (-z).")

    if z_std < 2.0: # Arbitrary threshold for flatness
         print("SUCCESS: Molecule is aligned flat along the Z-axis.")

if __name__ == "__main__":
    test_alignment_logic()
