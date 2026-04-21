import os
import sys
import yaml
import numpy as np
from ase.io import read

# Add the src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from potentials import SimulationEngine
from vibrational_analyzer import VibrationalAnalyzer, MultiModeFollower
from logger_utils import setup_logger

def run_enhanced_phonon_refinement(config_path='config.yaml', displacement=None):
    # 1. Load Config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    vib_cfg = config['vibrational_analysis']
    u = displacement if displacement is not None else vib_cfg.get('phonopy_displacement', 0.01)
    
    log_file = f"stability_u{str(u).replace('.', '')}.log"
    logger = setup_logger(log_path=log_file, verbose=True)
    logger.info(f"--- Starting Enhanced DIPAS Phonon Refinement (Phonopy u={u} A) ---")
    
    # 2. Setup Potentials
    engine = SimulationEngine(
        model_type=config['potentials']['model_type'],
        device=config['potentials']['device'],
        config=config
    )
    
    # 3. Load Structure
    mol_path = config['paths']['molecule']
    logger.info(f"Loading molecule from: {mol_path}")
    atoms = read(mol_path)
    atoms.set_cell([20, 20, 20])
    atoms.center()
    
    # 4. Perform Initial Relaxation
    logger.info("Performing initial relaxation with MACE (ultra-tight CG+FIRE)...")
    target_fmax = vib_cfg.get('fmax', 0.001)
    engine.relax(atoms, fmax=target_fmax, optimizer='CG_FIRE')
    
    # 5. Iterative Stability Loop with Adaptive Control
    max_iter = vib_cfg.get('max_iter', 10)
    stability_goal = vib_cfg.get('stability_threshold', -0.1)
    
    # Selection and Constraints
    selection_cfg = vib_cfg.get('selection', {})
    const_cfg = vib_cfg.get('constraints', {})
    stag_eps = const_cfg.get('stagnation_epsilon', 0.05)
    stag_factor = const_cfg.get('stagnation_factor', 0.5)
    
    # State tracking
    current_atoms = atoms.copy()
    current_alpha = vib_cfg.get('perturbation', {}).get('alpha', 0.1)
    history = [] # List of dicts: cycle, energy, min_freq, alpha
    
    logger.info(f"\nOptimization Goal: min_freq >= {stability_goal} THz")
    logger.info(f"Initial Alpha: {current_alpha} A | Phonopy u: {u} A")
    
    for cycle in range(1, max_iter + 1):
        logger.info(f"\n{'='*20} CYCLE {cycle}/{max_iter} {'='*20}")
        
        # Ensure calculator is attached
        current_atoms.calc = engine.get_calculator()
        
        # A. Phonon Calculation
        analyzer = VibrationalAnalyzer(
            atoms=current_atoms, 
            engine=engine, 
            displacement=u,
            # ... remaining params
            is_symmetry=vib_cfg.get('symmetry', {}).get('enabled', True),
            symprec=vib_cfg.get('symmetry', {}).get('symprec', 1e-5)
        )
        qpath = "qpoints.yaml"
        analyzer.generate_qpoints_file(filename=qpath)
        
        # B. Analyze Results
        from qpoint_handler import QPointParser
        parser = QPointParser(qpath)
        # Scan ALL frequency to find minimum
        all_freqs = []
        for phon in parser.data['phonon']:
            for b in phon['band']:
                all_freqs.append(b['frequency'])
        
        min_freq = min(all_freqs)
        energy = current_atoms.get_potential_energy()
        
        logger.info(f"  [Status] Energy: {energy:.6f} eV | Min Freq: {min_freq:.4f} THz")
        history.append({
            'cycle': cycle,
            'energy': energy,
            'min_freq': min_freq,
            'alpha': current_alpha
        })
        
        # C. Convergence Check
        if min_freq >= stability_goal:
            logger.info(f"  [Success] Stability goal reached (min_freq {min_freq:.4f} >= {stability_goal}).")
            break
            
        # D. Stagnation Check & Adaptive Alpha
        if cycle > 1:
            improvement = min_freq - history[-2]['min_freq']
            if improvement < stag_eps:
                new_alpha = current_alpha * stag_factor
                logger.warning(f"  [Stagnation] Improvement ({improvement:.4f} THz) < Epsilon ({stag_eps} THz).")
                logger.warning(f"               Reducing alpha: {current_alpha:.3f} -> {new_alpha:.3f} A")
                current_alpha = new_alpha
                
        # E. Multi-Mode Refinement
        logger.info(f"  [Action] Starting multi-mode following with alpha={current_alpha:.3f}...")
        
        # Inject local alpha into vibrational_analysis config for this iteration
        iter_cfg = vib_cfg.copy()
        if 'perturbation' not in iter_cfg: iter_cfg['perturbation'] = {}
        iter_cfg['perturbation']['alpha'] = current_alpha
        
        follower = MultiModeFollower(engine, config=iter_cfg)
        # Use tighter fmax and CG_FIRE for refinement cycles
        current_atoms = follower.optimize(current_atoms, fmax=target_fmax, optimizer='CG_FIRE')

    # 6. Final Summary
    logger.info(f"\n{'='*20} REFINEMENT SUMMARY (u={u}) {'='*20}")
    logger.info(f"{'Cycle':>5} | {'Energy (eV)':>12} | {'Min Freq (THz)':>14} | {'Alpha (A)':>10}")
    logger.info("-" * 55)
    for h in history:
        logger.info(f"{h['cycle']:5d} | {h['energy']:12.6f} | {h['min_freq']:14.4f} | {h['alpha']:10.3f}")
    
    if history[-1]['min_freq'] < stability_goal:
        logger.error(f"\n[Conclusion] Stability goal NOT met. Final Min Freq: {history[-1]['min_freq']:.4f} THz.")
    else:
        logger.info(f"\n[Conclusion] Structure successfully stabilized to {history[-1]['min_freq']:.4f} THz.")
        
    # Save final structure
    out_prefix = config['paths'].get('output_prefix', 'refined')
    out_name = f"{out_prefix}_u{str(u).replace('.', '')}_final.vasp"
    current_atoms.write(out_name)
    logger.info(f"Saved final structure to: {out_name}")
    
    return current_atoms

if __name__ == "__main__":
    # Usage: python run_phonon_refinement.py [config_path] [displacement]
    c_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), 'config.yaml')
    u_val = float(sys.argv[2]) if len(sys.argv) > 2 else None
    run_enhanced_phonon_refinement(c_path, u_val)
