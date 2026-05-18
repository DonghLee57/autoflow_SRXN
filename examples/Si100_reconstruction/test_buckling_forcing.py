import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from ase.io import read, write
from ase.neighborlist import neighbor_list
from ase.constraints import FixCartesian, FixAtoms
from autoflow_srxn.surface.surface_utils import (
    create_slab_from_bulk,
    passivate_surface_coverage_general,
)
from autoflow_srxn.surface.reconstruction_recipes import reconstruct_si100_2x1_buckled
from autoflow_srxn.simulation.potentials import SimulationEngine

def measure_buckling(atoms):
    z_max = np.max(atoms.positions[:, 2])
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

def get_7net_equilibrium_bulk():
    print("\n--- Finding 7net-0 equilibrium lattice parameter for bulk Si ---")
    script_dir = Path(__file__).parent
    bulk_path = script_dir.parent.parent / "structures" / "Si_mp149.vasp"
    bulk = read(str(bulk_path))
    
    config = {
        "engine": {
            "potential": {
                "backend": "sevennet",
                "model": "7net-0",
                "device": "cpu",
                "dtype": "float64",
            }
        }
    }
    engine = SimulationEngine(config)
    
    # Simple 1D equation of state
    scales = np.linspace(0.95, 1.05, 7)
    energies = []
    for s in scales:
        b_temp = bulk.copy()
        b_temp.set_cell(bulk.cell * s, scale_atoms=True)
        b_temp.calc = engine.get_calculator()
        energies.append(b_temp.get_potential_energy())
        
    coeffs = np.polyfit(scales, energies, 2)
    opt_scale = -coeffs[1] / (2 * coeffs[0])
    print(f"    Optimal scale relative to MP bulk: {opt_scale:.4f}")
    
    opt_bulk = bulk.copy()
    opt_bulk.set_cell(bulk.cell * opt_scale, scale_atoms=True)
    return opt_bulk

def test_forcing_methods():
    script_dir = Path(__file__).parent
    bulk_opt = get_7net_equilibrium_bulk()
    
    config = {
        "engine": {
            "potential": {
                "backend": "sevennet",
                "model": "7net-0",
                "device": "cpu",
                "dtype": "float64",
            }
        },
        "relaxation": {
            "fmax": 0.05,
            "steps": 200,
            "optimizer": "BFGS",
        }
    }
    engine = SimulationEngine(config)
    
    results = {}
    
    # ----------------------------------------------------
    # Method 1: Biaxial Strain Test
    # ----------------------------------------------------
    for strain in [-0.03, -0.01, 0.0, 0.01, 0.03]:
        print(f"\n>>> Method 1: Biaxial Strain {strain:+.1%} (No Constraints on top)")
        b_strained = bulk_opt.copy()
        b_strained.set_cell(bulk_opt.cell * (1.0 + strain), scale_atoms=True)
        
        slab = create_slab_from_bulk(b_strained, [1, 0, 0], thickness=12.0, vacuum=15.0, target_area=250.0, verbose=False)
        slab = passivate_surface_coverage_general(
            slab,
            coverage=1.0,
            valence_map={"Si": 4},
            element="H",
            side="bottom",
            verbose=False,
        )
        slab = reconstruct_si100_2x1_buckled(slab, "top", buckle=0.7, verbose=False)
        
        # Freeze bottom 5.5 A
        z_min = slab.positions[:, 2].min()
        bottom_idx = np.where(slab.positions[:, 2] < z_min + 5.5)[0].tolist()
        slab.set_constraint(FixAtoms(bottom_idx))
        
        try:
            engine.relax(slab, verbose=False)
            dimers = measure_buckling(slab)
            avg_buckle = np.mean([d[3] for d in dimers]) if dimers else 0.0
            print(f"    Average Final Buckling: {avg_buckle:.6f} A")
            results[f"Strain {strain:+.1%}"] = avg_buckle
        except Exception as e:
            print(f"    Failed: {e}")
            
    # ----------------------------------------------------
    # Method 2: FixCartesian Z-pinning
    # ----------------------------------------------------
    print("\n>>> Method 2: FixCartesian Z-pinning (Fix dimer Z coordinates, relax X and Y)")
    slab = create_slab_from_bulk(bulk_opt, [1, 0, 0], thickness=12.0, vacuum=15.0, target_area=250.0, verbose=False)
    slab = passivate_surface_coverage_general(
        slab,
        coverage=1.0,
        valence_map={"Si": 4},
        element="H",
        side="bottom",
        verbose=False,
    )
    slab = reconstruct_si100_2x1_buckled(slab, "top", buckle=0.7, verbose=False)
    
    # Find top surface dimer atoms to pin in Z
    z_max = np.max(slab.positions[:, 2])
    surf_idx = [i for i, (sym, pos) in enumerate(zip(slab.symbols, slab.positions)) if sym == "Si" and pos[2] > z_max - 1.5]
    
    i_list, j_list, d_list = neighbor_list("ijd", slab, 2.6)
    dimers_idx = []
    for i, j, d in zip(i_list, j_list, d_list):
        if i in surf_idx and j in surf_idx and i < j:
            coord_i = np.sum(i_list == i)
            coord_j = np.sum(i_list == j)
            if coord_i < 4 and coord_j < 4:
                dimers_idx.extend([i, j])
                
    # Freeze bottom 5.5 A in all directions, and freeze top dimers in Z direction only!
    z_min = slab.positions[:, 2].min()
    bottom_idx = np.where(slab.positions[:, 2] < z_min + 5.5)[0].tolist()
    
    constraints = []
    constraints.append(FixAtoms(bottom_idx))
    constraints.append(FixCartesian(dimers_idx, mask=(False, False, True))) # Fix Z, allow X, Y
    slab.set_constraint(constraints)
    
    try:
        engine.relax(slab, verbose=False)
        dimers = measure_buckling(slab)
        avg_buckle = np.mean([d[3] for d in dimers]) if dimers else 0.0
        print(f"    Average Final Buckling: {avg_buckle:.6f} A")
        results["FixCartesian Z-pinning"] = avg_buckle
        
        # Save this structure for inhibitor tests!
        write("Si100_buckled_pinned.vasp", slab)
        print("    Saved buckled pinned structure to 'Si100_buckled_pinned.vasp'")
    except Exception as e:
        print(f"    Failed: {e}")

    print("\n==================================================")
    print("FORCING METHODS SUMMARY")
    print("==================================================")
    for k, v in results.items():
        print(f"{k}: {v:.6f} A")

if __name__ == "__main__":
    test_forcing_methods()
