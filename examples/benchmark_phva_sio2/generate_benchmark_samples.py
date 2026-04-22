import os
import sys
import numpy as np
from ase.io import read, write
from ase.build import surface, add_adsorbate
from ase.constraints import FixAtoms
from ase.md.langevin import Langevin
from ase import units
import time

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from potentials import SimulationEngine
from surface_utils import passivate_surface_coverage_general
from si_surface_utils import SI_VALENCE_MAP

def generate_benchmark_samples():
    print("--- Starting High-Fidelity Benchmark Sample Generation (Optimized) ---")
    
    # 1. Setup Engine
    engine = SimulationEngine(model_type='mace', device='cpu')
    calc = engine.get_calculator()
    
    # 2. Substrate Preparation
    print("[Step 1] Building SiO2(001) Slab...")
    bulk_sio2 = read('../../structures/SiO2_mp-546794.vasp')
    
    # Create (001) slab - 2x2 for speed
    slab = surface(bulk_sio2, (0, 0, 1), layers=3, vacuum=15.0)
    slab = slab * (2, 2, 1)
    
    # Passivate with Silanols
    print("[Step 2] Applying Silanol Passivation...")
    slab = passivate_surface_coverage_general(slab, h_coverage=1.0, valence_map=SI_VALENCE_MAP, side='top', verbose=True)
    slab = passivate_surface_coverage_general(slab, h_coverage=1.0, valence_map=SI_VALENCE_MAP, side='bottom', verbose=True)
    
    # 3. Equilibration
    print("[Step 3] Substrate Equilibration...")
    z_min = slab.positions[:, 2].min()
    mask = slab.positions[:, 2] < z_min + 5.5
    slab.set_constraint(FixAtoms(mask=mask))
    
    slab.calc = calc
    print("  -> Initial Relaxation...")
    engine.relax(slab, fmax=0.1, steps=50) # Faster relax
    
    print("  -> Running MD (500K, 0.5ps)...")
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    MaxwellBoltzmannDistribution(slab, temperature_K=500)
    dyn = Langevin(slab, 1 * units.fs, temperature_K=500, friction=0.01)
    dyn.run(500) # 0.5ps
    
    print("  -> Final Relaxation...")
    engine.relax(slab, fmax=0.05, steps=50)
    
    # 4. Adsorption
    print("[Step 4] DIPAS Adsorption...")
    dipas = read('../../structures/DIPAS.vasp')
    dipas.calc = calc
    engine.relax(dipas, fmax=0.05)
    
    z_top = slab.positions[:, 2].max()
    lx, ly = slab.cell[0,0], slab.cell[1,1]
    
    # Place at center
    combined = slab.copy()
    add_adsorbate(combined, dipas, height=3.2, position=(lx/2, ly/2))
    
    z_min_final = combined.positions[:, 2].min()
    mask_final = combined.positions[:, 2] < z_min_final + 5.5
    combined.set_constraint(FixAtoms(mask=mask_final))
    
    print("  -> Final Adsorption Relaxation...")
    engine.relax(combined, fmax=0.05, steps=100)
    
    write('SiO2_DIPAS_adsorbed.vasp', combined)
    write('SiO2_substrate_final.vasp', slab)
    print("\n--- Benchmark Samples Generated Successfully ---")

if __name__ == "__main__":
    generate_benchmark_samples()
