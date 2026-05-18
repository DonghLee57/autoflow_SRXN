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
from autoflow_srxn.simulation.potentials import SimulationEngine

def find_highest_surface_si(slab):
    z_max = np.max(slab.positions[:, 2])
    surf_idx = [i for i, (sym, pos) in enumerate(zip(slab.symbols, slab.positions)) if sym == "Si" and pos[2] > z_max - 1.5]
    highest_idx = surf_idx[np.argmax(slab.positions[surf_idx, 2])]
    print(f"    Highest surface Si atom index: {highest_idx} at position {slab.positions[highest_idx]}")
    return highest_idx

def setup_constraints(combined_atoms, slab_len, dimer_idxs):
    # Fix bottom 5.5 A in all directions
    z_min = combined_atoms.positions[:slab_len, 2].min()
    bottom_idx = np.where(combined_atoms.positions[:slab_len, 2] < z_min + 5.5)[0].tolist()
    
    # Fix top dimer Z coordinates to preserve buckled shape
    constraints = []
    constraints.append(FixAtoms(bottom_idx))
    constraints.append(FixCartesian(dimer_idxs, mask=(False, False, True)))
    
    combined_atoms.set_constraint(constraints)

def run_physisorption_case(name, slab, mol, target_atom_idx, height=3.0):
    print(f"\n>>> Running Physisorption search: {name}")
    
    # Find active surface Si site
    site_idx = find_highest_surface_si(slab)
    site_pos = slab.positions[site_idx]
    
    # Position the inhibitor molecule
    mol_copy = mol.copy()
    offset = site_pos + np.array([0.0, 0.0, height]) - mol_copy.positions[target_atom_idx]
    mol_copy.positions += offset
    
    # Merge structures
    combined = slab.copy()
    combined += mol_copy
    
    # Tagging: slab is tag 0/1, inhibitor is tag 2
    tags = combined.get_tags()
    tags[len(slab):] = 2
    combined.set_tags(tags)
    
    # Setup constraints on the slab part of the combined system
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
                
    setup_constraints(combined, len(slab), dimers_idx)
    
    # Set up engine
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
            "steps": 250,
            "optimizer": "BFGS",
        }
    }
    engine = SimulationEngine(config)
    
    # Relax combined structure
    try:
        # Calculate isolated energies first
        calc = engine.get_calculator()
        slab.calc = calc
        mol.calc = calc
        e_slab = slab.get_potential_energy()
        e_mol = mol.get_potential_energy()
        
        combined.calc = calc
        e_init = combined.get_potential_energy()
        engine.relax(combined, verbose=False)
        e_final = combined.get_potential_energy()
        
        # Calculate relaxed distance
        relaxed_pos_site = combined.positions[site_idx]
        relaxed_pos_target = combined.positions[len(slab) + target_atom_idx]
        final_dist = np.linalg.norm(relaxed_pos_target - relaxed_pos_site)
        ads_energy = e_final - e_slab - e_mol
        
        print(f"    Initial Energy: {e_init:.4f} eV")
        print(f"    Final Energy: {e_final:.4f} eV")
        print(f"    Relaxed adsorption energy: {ads_energy:.4f} eV")
        print(f"    Relaxed target-Si distance: {final_dist:.4f} A")
        
        filename = f"Si100_{name}_relaxed.vasp"
        write(filename, combined)
        print(f"    Saved relaxed structure to '{filename}'")
        return ads_energy, final_dist
    except Exception as e:
        print(f"    Failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    script_dir = Path(__file__).parent
    
    # Load buckled pinned slab
    slab_path = script_dir / "Si100_buckled_pinned.vasp"
    if not slab_path.exists():
        slab_path = Path("Si100_buckled_pinned.vasp")
        
    if not slab_path.exists():
        # Fallback to creating it if not present
        from test_buckling_forcing import get_7net_equilibrium_bulk
        bulk_opt = get_7net_equilibrium_bulk()
        slab = create_slab_from_bulk(bulk_opt, [1, 0, 0], thickness=12.0, vacuum=15.0, target_area=250.0, verbose=False)
        slab = passivate_surface_coverage_general(slab, coverage=1.0, valence_map={"Si": 4}, element="H", side="bottom", verbose=False)
        from autoflow_srxn.surface.reconstruction_recipes import reconstruct_si100_2x1_buckled
        slab = reconstruct_si100_2x1_buckled(slab, "top", buckle=0.7, verbose=False)
        write("Si100_buckled_pinned.vasp", slab)
    else:
        slab = read(str(slab_path))
        
    # Load inhibitor
    mol_path = script_dir.parent.parent / "structures" / "inhibitor_relaxed.vasp"
    mol = read(str(mol_path))
    
    results = {}
    
    # Run Case A: Central Carbon (Atom 13) down
    results["Inhibitor_C_down"] = run_physisorption_case(
        "Inhibitor_C_down", slab, mol, target_atom_idx=13, height=3.0
    )
    
    # Run Case B: Nitrogen (Atom 18) down
    results["Inhibitor_N_down"] = run_physisorption_case(
        "Inhibitor_N_down", slab, mol, target_atom_idx=18, height=3.0
    )
    
    print("\n==================================================")
    print("PHYSISORPTION RESULTS SUMMARY")
    print("==================================================")
    for k, v in results.items():
        if v is not None:
            ads_e, dist = v
            print(f"{k}: Adsorption Energy = {ads_e:+.4f} eV, Dimer-Inhibitor Distance = {dist:.4f} A")
        else:
            print(f"{k}: FAILED")
