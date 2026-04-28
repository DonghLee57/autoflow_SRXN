import os
import sys
import numpy as np
import yaml
from ase.io import read, write
from ase.build import add_adsorbate
from ase.constraints import FixAtoms
from ase.neighborlist import neighbor_list
from ase import Atoms

_HERE = os.path.dirname(os.path.abspath(__file__))
from autoflow_srxn.potentials import SimulationEngine
from autoflow_srxn.vibrational_analyzer import VibrationalAnalyzer

def passivate_sio2(atoms):
    i, j = neighbor_list('ij', atoms, 2.0)
    h_atoms = []
    for a in range(len(atoms)):
        neighbors = j[i == a]
        if atoms[a].symbol == 'O' and len(neighbors) < 2:
            if len(neighbors) > 0:
                neighbor_idx = neighbors[0]
                vec = atoms[a].position - atoms[neighbor_idx].position
                vec /= np.linalg.norm(vec)
                h_pos = atoms[a].position + vec * 0.96
                h_atoms.append(Atoms('H', positions=[h_pos]))
    if h_atoms:
        for h in h_atoms:
            atoms += h
    return atoms

def build_sio2_slab():
    bulk_path = os.path.abspath(os.path.join(_HERE, '../../structures/SiO2_mp-546794.vasp'))
    bulk = read(bulk_path)
    slab = bulk.repeat((1, 1, 1)) 
    slab.center(vacuum=10, axis=2)
    slab = passivate_sio2(slab)
    return slab

def run_physisorption_vibration_test():
    print("Initializing test (Tighter Convergence)...", flush=True)
    config_path = os.path.join(_HERE, 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    config['engine']['potential']['model'] = 'small'
    engine = SimulationEngine(config=config)
    
    slab = build_sio2_slab()
    dipas_path = os.path.abspath(os.path.join(_HERE, '../../structures/DIPAS.vasp'))
    dipas = read(dipas_path)
    
    add_adsorbate(slab, dipas, height=3.2, position=(slab.cell[0,0]/2, slab.cell[1,1]/2))
    atoms = slab
    
    z_min = atoms.positions[:, 2].min()
    fixed_indices = [a.index for a in atoms if a.position[2] < z_min + 1.0]
    atoms.set_constraint(FixAtoms(indices=fixed_indices))
    
    print(f"Total atoms: {len(atoms)}", flush=True)
    
    # 1. First relax molecule and surface roughly
    print("Relaxation Stage 1 (fmax=0.05)...", flush=True)
    engine.relax(atoms, fmax=0.05, steps=300)
    
    # 2. Second relax very tightly
    print("Relaxation Stage 2 (fmax=0.01)...", flush=True)
    engine.relax(atoms, fmax=0.01, steps=200)
    
    n_dipas = len(dipas)
    n_total = len(atoms)
    ads_indices = list(range(n_total - n_dipas, n_total))
    
    print(f"Running PHVA on {len(ads_indices)} atoms...", flush=True)
    analyzer = VibrationalAnalyzer(
        atoms=atoms,
        engine=engine,
        indices=ads_indices,
        displacement=0.01,
        name='vib_analysis_tight'
    )
    
    freqs_thz, _ = analyzer.run_analysis(overwrite=True)
    
    # 8. Report
    imag = [f for f in freqs_thz if f < -0.01]
    print("\n" + "="*40, flush=True)
    print(f"RESULTS: {len(imag)} imaginary modes found", flush=True)
    for f in sorted(imag):
        print(f"  {f:.4f} THz", flush=True)
    print("="*40, flush=True)

    report_path = os.path.join(_HERE, 'report.md')
    with open(report_path, 'w') as f:
        f.write("# Physisorption Vibration Analysis Report (Fast Test)\n\n")
        f.write(f"System: DIPAS on SiO2 (001) slab\n")
        f.write(f"Slab: 1x1x1 unit cell, H-passivated, bottom fixed\n")
        f.write(f"Potential: MACE (small)\n")
        f.write(f"Relaxation: Stage 1 (0.05) + Stage 2 (0.01)\n\n")
        f.write("## Results\n")
        f.write(f"- Total modes: {len(freqs_thz)}\n")
        f.write(f"- Imaginary modes: {len(imag)}\n\n")
        if imag:
            f.write("### Imaginary Frequencies (THz)\n")
            for val in sorted(imag):
                f.write(f"- {val:.4f}\n")
        else:
            f.write("No imaginary frequencies detected.\n")
    print(f"Report generated: {report_path}", flush=True)

if __name__ == "__main__":
    run_physisorption_vibration_test()
