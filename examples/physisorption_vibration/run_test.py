import os
import sys
import yaml
import numpy as np
from ase.io import read
from ase.constraints import FixAtoms

# Path setup
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, '../../src')))

from potentials import SimulationEngine
from surface_utils import create_slab_from_bulk, passivate_surface_coverage_general, write_standardized_vasp
from si_surface_utils import SI_VALENCE_MAP
from logger_utils import setup_logger
from vibrational_analyzer import VibrationalAnalyzer

def run_test():
    # 1. Setup
    with open(os.path.join(script_dir, 'config.yaml'), 'r') as f:
        config = yaml.safe_load(f)
    
    out_dir = os.path.join(script_dir, config['paths'].get('output_dir', 'results'))
    os.makedirs(out_dir, exist_ok=True)
    
    log_file = os.path.join(out_dir, 'physisorption_comparison.log')
    logger = setup_logger(log_path=log_file, verbose=True, mode='w')
    
    logger.info("Starting FHVA vs PHVA Comparison Test")
    
    engine = SimulationEngine(config=config)
    
    # 2. Build Slab
    bulk_path = os.path.abspath(os.path.join(script_dir, config['paths']['substrate_bulk']))
    bulk = read(bulk_path)
    sg = config['surface_prep']['slab_generation']
    slab = create_slab_from_bulk(
        bulk,
        miller_indices=sg['miller'],
        thickness=sg['thickness_ang'],
        vacuum=sg['vacuum_ang'],
        target_area=sg['target_area_ang2'],
        top_termination=sg['top_termination'],
        bottom_termination=sg['bottom_termination'],
        verbose=False
    )
    
    # 3. Passivation
    pv = config['surface_prep']['passivation']
    slab = passivate_surface_coverage_general(
        slab,
        h_coverage=pv['coverage'],
        valence_map=SI_VALENCE_MAP,
        side=pv['side'],
        verbose=False
    )
    
    # 4. Add Adsorbate
    ads_path = os.path.abspath(os.path.join(script_dir, config['paths']['adsorbate']))
    dipas = read(ads_path)
    from ase.build import add_adsorbate
    add_adsorbate(slab, dipas, height=3.5, position=(slab.cell[0,0]/2, slab.cell[1,1]/2))
    atoms = slab
    
    # 5. Fix bottom
    z_min = atoms.positions[:, 2].min()
    frozen_z = config['surface_prep']['equilibration']['frozen_z_ang']
    atoms.set_constraint(FixAtoms(mask=atoms.positions[:, 2] < z_min + frozen_z))
    
    # 6. Relax
    logger.info("Optimizing structure...")
    engine.relax(atoms, verbose=False)
    opt_path = os.path.join(out_dir, 'optimized.vasp')
    write_standardized_vasp(opt_path, atoms)
    
    # 7. FHVA
    logger.info("--- STAGE 1: Full Hessian Vibrational Analysis (FHVA) ---")
    fhva_dir = os.path.join(out_dir, 'fhva')
    os.makedirs(fhva_dir, exist_ok=True)
    
    analyzer_fhva = VibrationalAnalyzer(
        atoms=atoms,
        engine=engine,
        indices=list(range(len(atoms))), # Full Hessian
        displacement=config['analysis']['vibrational']['displacement_ang'],
        name=os.path.join(fhva_dir, 'vib_fhva_cache')
    )
    # run_analysis will now automatically call generate_qpoints_file in fhva_dir
    freqs_fhva, _ = analyzer_fhva.run_analysis(overwrite=True)
    
    # 8. PHVA
    logger.info("--- STAGE 2: Partial Hessian Vibrational Analysis (PHVA) ---")
    phva_dir = os.path.join(out_dir, 'phva')
    os.makedirs(phva_dir, exist_ok=True)
    
    analyzer_phva = VibrationalAnalyzer(
        atoms=atoms,
        engine=engine,
        indices=None, # Automatic resolution (radius + height)
        displacement=config['analysis']['vibrational']['displacement_ang'],
        name=os.path.join(phva_dir, 'vib_phva_cache')
    )
    # run_analysis will now automatically call generate_qpoints_file in phva_dir
    freqs_phva, _ = analyzer_phva.run_analysis(overwrite=True)
    
    # 9. Comparison & Report
    logger.info("Generating comparison report...")
    report_path = os.path.join(out_dir, 'comparison_report.md')
    
    with open(report_path, 'w') as f:
        f.write("# FHVA vs PHVA Comparison Report\n\n")
        f.write(f"- **System**: DIPAS on SiO2 (001) slab\n")
        f.write(f"- **Total Atoms**: {len(atoms)}\n")
        f.write(f"- **FHVA Active Atoms**: {len(analyzer_fhva.indices)}\n")
        f.write(f"- **PHVA Active Atoms**: {len(analyzer_phva.indices)}\n\n")
        
        f.write("## Frequency Summary (Top 10 lowest real modes)\n")
        f.write("| Mode | FHVA (THz) | PHVA (THz) | Diff (%) |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        
        # Compare real frequencies
        real_fhva = sorted([f for f in freqs_fhva if f > 0])
        real_phva = sorted([f for f in freqs_phva if f > 0])
        
        for i in range(min(10, len(real_fhva), len(real_phva))):
            f1, f2 = real_fhva[i], real_phva[i]
            diff = abs(f1 - f2) / f1 * 100 if f1 > 1e-4 else 0
            f.write(f"| {i+1} | {f1:.4f} | {f2:.4f} | {diff:.2f}% |\n")
            
        f.write("\n## Imaginary Modes\n")
        imag_fhva = [f for f in freqs_fhva if f < -0.01]
        imag_phva = [f for f in freqs_phva if f < -0.01]
        f.write(f"- **FHVA**: {len(imag_fhva)} modes\n")
        f.write(f"- **PHVA**: {len(imag_phva)} modes\n")

    logger.info(f"Comparison report saved to: {report_path}")

if __name__ == "__main__":
    run_test()
