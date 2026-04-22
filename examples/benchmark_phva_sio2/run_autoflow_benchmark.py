import os
import sys
import yaml
import numpy as np
from ase.io import read, write
from ase.constraints import FixAtoms
from ase.md.langevin import Langevin
from ase import units
import shutil

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from potentials import SimulationEngine
from surface_utils import create_slab_from_bulk, passivate_surface_coverage_general
from ads_workflow_mgr import AdsorptionWorkflowManager
from vibrational_analyzer import VibrationalAnalyzer
from thermo_engine import ThermoCalculator, eV_to_J_mol
from logger_utils import setup_logger

def run_standard_benchmark():
    # 1. Load Config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    logger = setup_logger(log_path="autoflow_benchmark.log", verbose=True)
    logger.info("Starting Standardized PHVA Benchmark Workflow")
    
    # 2. Setup Engine
    engine = SimulationEngine(
        model_type=config['potentials']['model_type'], 
        device=config['potentials']['device']
    )
    calc = engine.get_calculator()
    
    if os.path.exists('SiO2_substrate_standard.vasp'):
        logger.info("[Step 3] Loading pre-equilibrated substrate...")
        slab = read('SiO2_substrate_standard.vasp')
        slab.calc = calc
    else:
        # 3. Substrate Generation
        logger.info("[Step 1] Generating SiO2(001) Substrate...")
        bulk_atoms = read(config['paths']['bulk_substrate'])
        slab = create_slab_from_bulk(
            bulk_atoms,
            miller_indices=config['substrate_generation']['miller_indices'],
            thickness=config['substrate_generation']['thickness'],
            vacuum=config['substrate_generation']['vacuum'],
            target_area=config['substrate_generation']['target_area'],
            top_termination=config['substrate_generation']['top_termination'],
            bottom_termination=config['substrate_generation']['bottom_termination'],
            verbose=True
        )
        
        # 4. Asymmetric Passivation (Bottom only)
        logger.info("[Step 2] Applying Bottom-side Passivation...")
        from si_surface_utils import SI_VALENCE_MAP
        slab = passivate_surface_coverage_general(
            slab, 
            h_coverage=config['passivation']['coverage'],
            valence_map=SI_VALENCE_MAP,
            side=config['passivation']['side'],
            verbose=True
        )
        
        # 5. Equilibration
        logger.info("[Step 3] Substrate Equilibration (MD 500K)...")
        z_min = slab.positions[:, 2].min()
        fixed_mask = slab.positions[:, 2] < z_min + config['equilibration']['fixed_z_threshold']
        slab.set_constraint(FixAtoms(mask=fixed_mask))
        slab.calc = calc
        
        engine.relax(slab, fmax=config['relaxation']['fmax'], steps=config['relaxation']['steps'])
        
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
        MaxwellBoltzmannDistribution(slab, temperature_K=config['equilibration']['temp_K'])
        dyn = Langevin(slab, 1 * units.fs, temperature_K=config['equilibration']['temp_K'], friction=0.01, logfile=sys.stdout, loginterval=100)
        dyn.run(config['equilibration']['md_steps'])
        
        engine.relax(slab, fmax=config['relaxation']['fmax'], steps=config['relaxation']['steps'])
        write('SiO2_substrate_standard.vasp', slab)

    # 6. Adsorption Workflow
    logger.info("[Step 4] DIPAS Physisorption Search (AdsorptionWorkflowManager)...")
    mgr = AdsorptionWorkflowManager(slab, config=config, verbose=True)
    dipas = read(config['paths']['adsorbate'])
    
    candidates = mgr.generate_physisorption_candidates(
        dipas, 
        height=5.0,
        n_rot=config['adsorption']['sampling_rotations'],
        rot_center='com'
    )
    
    if not candidates:
        logger.error("No valid adsorption candidates found!")
        return

    # Relax and find global minimum
    best_atoms = None
    best_e = 1e9
    all_relaxed = []
    
    logger.info(f"{'Candidate':<12} | {'Initial E (eV)':<15} | {'Final E (eV)':<15} | {'Delta E (meV)':<15}")
    logger.info("-" * 65)

    for i, cand in enumerate(candidates[:5]):
        cand.calc = calc
        e_init = cand.get_potential_energy()
        e_final = engine.relax(cand, fmax=config['relaxation']['fmax'], steps=config['relaxation']['steps'])
        
        delta_e = (e_final - e_init) * 1000
        logger.info(f"{i+1:<12} | {e_init:<15.4f} | {e_final:<15.4f} | {delta_e:<15.2f}")
        
        cand.info['candidate_id'] = i + 1
        cand.info['energy'] = e_final
        all_relaxed.append(cand.copy())
        
        if e_final < best_e:
            best_e = e_final
            best_atoms = cand.copy()
            
    # Save all candidates for visualization
    write('all_relaxed_candidates.extxyz', all_relaxed)
    logger.info(f"Saved all {len(all_relaxed)} relaxed candidates to all_relaxed_candidates.extxyz")
    
    if best_atoms is None:
        logger.error("Relaxation failed for all candidates.")
        return

    write('SiO2_DIPAS_final.vasp', best_atoms)
    
    # 7. PHVA vs FHVA Benchmark
    logger.info("[Step 5] PHVA vs FHVA Comparison...")
    
    # FHVA
    logger.info("  Running FHVA...")
    fhva = VibrationalAnalyzer(best_atoms, engine, name="vibrations/fhva")
    freqs_fhva = fhva.run_analysis()
    
    # PHVA
    n_dipas = len(dipas)
    active_indices = list(range(len(best_atoms) - n_dipas, len(best_atoms)))
    from ase.neighborlist import neighbor_list
    i_list, j_list = neighbor_list('ij', best_atoms, config['thermo']['phva_cutoff_dist'])
    neighbors = []
    for idx in active_indices:
        neighbors.extend([n for n in j_list[i_list == idx] if n not in active_indices])
    active_indices = sorted(list(set(active_indices + neighbors)))
    
    logger.info(f"  Running PHVA (Active: {len(active_indices)} atoms)...")
    phva = VibrationalAnalyzer(best_atoms, engine, indices=active_indices, name="vibrations/phva")
    freqs_phva = phva.run_analysis()
    
    # 8. Data Export (qpoint.yaml)
    logger.info("[Step 6] Generating qpoint.yaml and summary report...")
    
    qpoint_data = {
        'q-point': [0.0, 0.0, 0.0],
        'weight': 1.0,
        'fhva': {
            'total_modes': len(freqs_fhva),
            'frequencies_cm1': [float(f) for f in freqs_fhva],
            'imaginary_count': sum(1 for f in freqs_fhva if f < 0)
        },
        'phva': {
            'active_atoms': len(active_indices),
            'total_modes': len(freqs_phva),
            'frequencies_cm1': [float(f) for f in freqs_phva],
            'imaginary_count': sum(1 for f in freqs_phva if f < 0)
        },
        'thermo': {
            'temperature_K': config['thermo']['temperature_K'],
            'G_vib_fhva_eV': float(ThermoCalculator([f/33.356 for f in freqs_fhva]).calculate_vib_free_energy(config['thermo']['temperature_K']) / eV_to_J_mol),
            'G_vib_phva_eV': float(ThermoCalculator([f/33.356 for f in freqs_phva]).calculate_vib_free_energy(config['thermo']['temperature_K']) / eV_to_J_mol)
        }
    }
    
    with open('qpoint.yaml', 'w') as yf:
        yaml.dump(qpoint_data, yf, default_flow_style=False)
    
    logger.info("Generated qpoint.yaml with structured vibrational data.")
    
    # Cleanup ASE JSON caches to keep dir clean
    # But keep the qpoint.yaml as the human-readable result.
    logger.info("Benchmark Workflow Complete.")

if __name__ == "__main__":
    run_standard_benchmark()
