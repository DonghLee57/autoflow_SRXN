import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from ase.io import read
from ase.neighborlist import neighbor_list
from autoflow_srxn.surface.surface_utils import (
    create_slab_from_bulk,
    apply_surface_reconstruction,
    passivate_surface_coverage_general,
    standardize_vasp_atoms,
)
from autoflow_srxn.simulation.potentials import SimulationEngine

def measure_buckling(atoms):
    z_max = np.max(atoms.positions[:, 2])
    # Find surface Si atoms within top 1.5 A
    surf_idx = [i for i, (sym, pos) in enumerate(zip(atoms.symbols, atoms.positions)) if sym == "Si" and pos[2] > z_max - 1.5]
    
    i_list, j_list, d_list = neighbor_list("ijd", atoms, 2.6)
    dimers = []
    for i, j, d in zip(i_list, j_list, d_list):
        if i in surf_idx and j in surf_idx and i < j:
            coord_i = np.sum(i_list == i)
            coord_j = np.sum(i_list == j)
            if coord_i < 4 and coord_j < 4:
                buckling = abs(atoms.positions[i, 2] - atoms.positions[j, 2])
                dimers.append((i, j, d, buckling))
    return dimers

def run_test_case(name, dtype, frozen_z, buckle_amplitude, optimizer, thickness=12.0):
    print(f"\n>>> Running Test Case: {name}")
    print(f"    Settings: dtype={dtype}, frozen_z={frozen_z}, initial_buckle={buckle_amplitude}, optimizer={optimizer}, thickness={thickness}")
    
    script_dir = Path(__file__).parent
    bulk_path = script_dir.parent.parent / "structures" / "Si_mp149.vasp"
    if not bulk_path.exists():
        bulk_path = script_dir.parent.parent / "structures" / "Si_relaxed.vasp"
        
    print(f"    Loading bulk from: {bulk_path}")
    slab = create_slab_from_bulk(
        bulk_atoms=read(str(bulk_path)),
        miller_indices=[1, 0, 0],
        thickness=thickness,
        vacuum=15.0,
        target_area=250.0,
        verbose=False,
    )
    
    # 2. Passivate bottom
    valence_map = {"Si": 4}
    slab = passivate_surface_coverage_general(
        slab,
        coverage=1.0,
        valence_map=valence_map,
        element="H",
        side="bottom",
        verbose=False,
    )
    
    # 3. Apply reconstruction seed
    from autoflow_srxn.surface.reconstruction_recipes import reconstruct_si100_2x1_buckled
    slab = reconstruct_si100_2x1_buckled(
        slab,
        side="top",
        buckle=buckle_amplitude,
        verbose=False,
    )
    
    initial_dimers = measure_buckling(slab)
    init_buckle_vals = [d[3] for d in initial_dimers]
    print(f"    Initial Buckling values: {init_buckle_vals}")
    
    # 4. Set up engine
    config = {
        "engine": {
            "potential": {
                "backend": "sevennet",
                "model": "7net-0",
                "device": "cpu",
                "dtype": dtype,
            }
        },
        "relaxation": {
            "fmax": 0.05,
            "steps": 300,
            "optimizer": optimizer,
        }
    }
    engine = SimulationEngine(config)
    
    # 5. Relax slab
    try:
        engine.relax(slab, frozen_z_ang=frozen_z, verbose=False)
        final_dimers = measure_buckling(slab)
        final_buckle_vals = [d[3] for d in final_dimers]
        avg_final_buckle = np.mean(final_buckle_vals) if final_buckle_vals else 0.0
        print(f"    Final Buckling values: {final_buckle_vals}")
        print(f"    Average Final Buckling: {avg_final_buckle:.6f} A")
        return avg_final_buckle
    except Exception as e:
        print(f"    Relaxation failed: {e}")
        return None

if __name__ == "__main__":
    results = {}
    
    # Base reference (matches prepare_slabs.py settings)
    results["Base (float32, frozen=5.5, buckle=0.7, FIRE)"] = run_test_case(
        "Base Reference", dtype="float32", frozen_z=5.5, buckle_amplitude=0.7, optimizer="FIRE"
    )
    
    # Test float64 precision
    results["Float64 (frozen=5.5, buckle=0.7, FIRE)"] = run_test_case(
        "Float64", dtype="float64", frozen_z=5.5, buckle_amplitude=0.7, optimizer="FIRE"
    )
    
    # Test BFGS optimizer
    results["BFGS (float64, frozen=5.5, buckle=0.7)"] = run_test_case(
        "BFGS Optimizer", dtype="float64", frozen_z=5.5, buckle_amplitude=0.7, optimizer="BFGS"
    )
    
    # Test higher buckling amplitude (1.0 A)
    results["Buckle 1.0 (float64, frozen=5.5, BFGS)"] = run_test_case(
        "Higher Initial Buckle", dtype="float64", frozen_z=5.5, buckle_amplitude=1.0, optimizer="BFGS"
    )
    
    # Test thicker slab (16.0 A)
    results["Thicker Slab 16A (float64, frozen=5.5, buckle=1.0, BFGS)"] = run_test_case(
        "Thicker Slab", dtype="float64", frozen_z=5.5, buckle_amplitude=1.0, optimizer="BFGS", thickness=16.0
    )
    
    # Test freezing fewer layers (3.5 A)
    results["Frozen 3.5A (float64, buckle=1.0, BFGS)"] = run_test_case(
        "Less Constraints", dtype="float64", frozen_z=3.5, buckle_amplitude=1.0, optimizer="BFGS"
    )

    print("\n==================================================")
    print("SUMMARY OF TEST RESULTS")
    print("==================================================")
    for k, v in results.items():
        if v is not None:
            print(f"{k}: {v:.6f} A")
        else:
            print(f"{k}: FAILED")
