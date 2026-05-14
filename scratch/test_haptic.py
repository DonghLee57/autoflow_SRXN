import os
import sys
import numpy as np
from ase.io import read
from ase.build import surface, add_adsorbate

# Add project root to path
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root)

from autoflow_srxn.surface.ads_workflow_mgr import AdsorptionWorkflowManager

def test_allyl_cp_ni():
    print("--- Testing AllylCpNi Chemisorption Generation ---")
    
    # 1. Load Precursor
    prec_path = "structures/AllylCpNi.vasp"
    molecule = read(prec_path)
    print(f"Loaded Precursor: {molecule.get_chemical_formula()}")

    # 2. Setup Si(100) Slab (Simple 2x2)
    from ase.build import bulk
    si_bulk = bulk('Si', 'diamond', a=5.43)
    slab = surface(si_bulk, (1, 0, 0), layers=4, vacuum=15)
    slab = slab.repeat((2, 2, 1))
    slab.set_tags(0) # Mark as substrate
    print(f"Created Si(100) Slab: {len(slab)} atoms")

    # 3. Initialize Manager
    mgr = AdsorptionWorkflowManager(slab)
    
    # 4. Discover Ligands
    print("\n[Step 1] Ligand Discovery Test")
    c_idx, ligands = mgr.discover_ligands(molecule, center_target="Ni", verbose=True)
    
    for i, l in enumerate(ligands):
        print(f"  Ligand {i}: {l['formula']}, Hapticity: {l['hapticity']}, Binding Atoms: {l['binding_atoms']}")

    # Verification: Should find C3H5 (Allyl, h=3) and C5H5 (Cp, h=5)
    hapticities = [l['hapticity'] for l in ligands]
    if 3 in hapticities and 5 in hapticities:
        print("SUCCESS: Haptic ligands (Allyl & Cp) correctly identified!")
    else:
        print("FAILURE: Haptic ligands not correctly identified.")

    # 5. Generate Chemisorption Candidates
    print("\n[Step 2] Chemisorption Candidate Generation Test")
    from autoflow_srxn.surface.chemisorption_builder import build_chemisorption_structures
    
    # We'll use a dummy config
    config = {
        "reaction_search": {
            "mechanisms": {
                "precursor": {
                    "chemisorption": {"verbose": True}
                },
                "dissociation": {"enabled": True},
                "protector": {"enabled": False}
            },
            "candidate_filter": {"overlap_scale": 0.65}
        }
    }
    
    candidates = build_chemisorption_structures(molecule, center_target="Ni", surface=slab, config=config)
    
    print(f"\nGenerated {len(candidates)} Chemisorption candidates.")
    
    if len(candidates) > 0:
        print("SUCCESS: Chemisorption structures generated for haptic ligands!")
        # Save one candidate for visual check
        os.makedirs("scratch", exist_ok=True)
        candidates[0].write("scratch/AllylCpNi_chem_test.vasp")
        print("Saved sample structure to scratch/AllylCpNi_chem_test.vasp")
    else:
        print("FAILURE: No chemisorption structures generated. Check overlap or placement logic.")

if __name__ == "__main__":
    test_allyl_cp_ni()
