import numpy as np
import os
import sys
from ase.io import read

# Add project root to path
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root)
print(f"DEBUG: Project Root = {root}")
print(f"DEBUG: sys.path[0] = {sys.path[0]}")

from autoflow_srxn.utils.mapping import match_atoms_geometric, reorder_atoms

def run_test():
    print("--- Starting Atom Mapping Algorithm Test ---")
    
    # Path to TiN example results (assuming previous run succeeded or files exist)
    res_dir = "c:/Users/user/Downloads/dev_w_antigravity/auto_surface_reaction/autoflow_SRXN/examples/TiN_TiCl4_MACE/results/clean_on_TiCl4/ts_search_0"
    init_path = os.path.join(res_dir, "init_aligned.vasp")
    final_path = os.path.join(res_dir, "final_state.vasp")
    
    if not os.path.exists(init_path):
        print(f"Error: Test files not found in {res_dir}")
        return

    init = read(init_path)
    final = read(final_path)
    
    print(f"Loaded {len(init)} atoms. Elements: {set(init.symbols)}")

    # 1. Randomize Initial Structure Order
    shuffled_indices = np.random.permutation(len(init))
    shuffled_init = init[shuffled_indices]
    print("Step 1: Randomly shuffled initial atoms.")

    # 1.5. Self-Mapping Test (Should be perfect)
    print("Step 1.5: Running self-mapping (init vs shuffled_init)...")
    mapping_self = match_atoms_geometric(init, shuffled_init)
    recovered_self = reorder_atoms(shuffled_init, mapping_self)
    diff_self = recovered_self.positions - init.positions
    rmsd_self = np.sqrt(np.mean(np.square(diff_self)))
    print(f"Self-mapping RMSD = {rmsd_self:.10f} A")

    # 2. Apply Mapping Algorithm (init vs final)
    print("Step 2: Running cross-mapping (final vs shuffled_init)...")
    try:
        mapping = match_atoms_geometric(final, shuffled_init)
        recovered_init = reorder_atoms(shuffled_init, mapping)
    except Exception as e:
        print(f"Mapping FAILED: {e}")
        return

    # 3. Verify Results
    # Calculate displacement after MIC correction
    from ase.geometry import find_mic
    diff = recovered_init.positions - final.positions
    diff_mic, _ = find_mic(diff, final.cell, final.pbc)
    rmsd = np.sqrt(np.mean(np.square(diff_mic)))
    
    print(f"Step 3: Verification. RMSD = {rmsd:.6f} A")
    if rmsd < 0.1:
        print("SUCCESS: Indices recovered and geometry matches!")
    else:
        print("FAILURE: Geometry mismatch after mapping.")

    # 4. Stress Test: Requirement C (Inconsistency Check)
    print("\nStep 4: Testing Requirement C (Inconsistency Check)...")
    bad_init = init[:-1].copy() # One atom missing
    try:
        match_atoms_geometric(final, bad_init)
        print("FAILURE: Did not catch atom count mismatch.")
    except ValueError as e:
        print(f"SUCCESS: Caught mismatch as expected: {e}")

    bad_init_sym = init.copy()
    bad_init_sym[0].symbol = "Au" # Wrong element
    try:
        match_atoms_geometric(final, bad_init_sym)
        print("FAILURE: Did not catch element mismatch.")
    except ValueError as e:
        print(f"SUCCESS: Caught mismatch as expected: {e}")

if __name__ == "__main__":
    run_test()
