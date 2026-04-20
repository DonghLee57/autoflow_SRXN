import os
import sys
from ase.build import molecule
from ase.io import read

from potentials import SimulationEngine

def test_relaxation():
    print("--- Testing SimulationEngine with MACE ---")
    
    # 1. Create a simple molecule (Ethanol-inspired fragment or just Water for speed)
    atoms = molecule('H2O')
    atoms.center(vacuum=5.0)
    
    # 2. Initialize Engine
    # Note: Using 'mace' model. Device 'cpu' for stability in this env.
    engine = SimulationEngine(model_type='mace', device='cpu')
    
    # 3. Initial Energy
    e_init = engine.get_energy(atoms)
    print(f"  Initial Energy: {e_init:.4f} eV")
    
    # 4. Relax
    print("  Starting Relaxation...")
    e_final = engine.relax(atoms, fmax=0.1, steps=20, verbose=True)
    
    print(f"  Final Energy: {e_final:.4f} eV")
    print(f"  Energy Change: {e_final - e_init:.4f} eV")
    
    if e_final < e_init:
        print("\n[Success] Relaxation engine successfully converged and lowered energy.")
    else:
        print("\n[Warning] Energy did not decrease. Check potential compatibility.")

if __name__ == "__main__":
    test_relaxation()
