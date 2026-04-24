"""
DIPAS precursor stability analysis:
  iterative mode-following structural refinement using MultiModeFollower.

Usage:
    python run_phonon_refinement.py [config.yaml] [displacement_ang]
"""
import copy
import os
import sys
import yaml
import numpy as np
from ase.io import read

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from potentials import SimulationEngine
from vibrational_analyzer import VibrationalAnalyzer, MultiModeFollower
from logger_utils import setup_logger


def run_enhanced_phonon_refinement(config_path='config.yaml', displacement=None):
    # ── Config ─────────────────────────────────────────────────────────────────
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    vib_cfg     = config['analysis']['vibrational']
    mode_ref    = vib_cfg.get('mode_refinement', {})
    rel         = config['engine']['relaxation']

    u           = displacement if displacement is not None else vib_cfg.get('displacement_ang', 0.01)
    target_fmax = mode_ref.get('fmax', rel.get('fmax', 0.001))
    target_steps= mode_ref.get('steps', rel.get('steps', 200))
    optimizer   = rel.get('optimizer', 'CG_FIRE')
    max_iter    = mode_ref.get('max_iter', 10)
    stability_goal  = mode_ref.get('freq_threshold_thz', -0.1)
    stag_eps    = mode_ref.get('stagnation_epsilon', 0.05)
    stag_factor = mode_ref.get('stagnation_factor', 0.5)
    init_alpha  = mode_ref.get('perturbation_alpha', 0.1)

    out_prefix  = config['paths'].get('output_prefix', 'refined')
    log_file    = f"{out_prefix}_u{str(u).replace('.', '')}.log"
    logger      = setup_logger(log_path=log_file, verbose=True)
    logger.info(f"--- Starting Enhanced Phonon Refinement (u={u} Ang) ---")

    # ── Engine and structure ────────────────────────────────────────────────────
    engine   = SimulationEngine(config=config)
    mol_path = config['paths']['adsorbate']
    logger.info(f"Loading molecule from: {mol_path}")
    atoms = read(mol_path)
    atoms.center(vacuum=10.0)
    cell  = atoms.get_cell()
    logger.info(f"Cell: {cell[0,0]:.2f} x {cell[1,1]:.2f} x {cell[2,2]:.2f} Ang")

    # ── Initial relaxation ──────────────────────────────────────────────────────
    logger.info(f"Initial relaxation (optimizer={optimizer}, fmax={target_fmax})...")
    current_atoms = atoms.copy()
    engine.relax(current_atoms, fmax=target_fmax, steps=target_steps, optimizer=optimizer)

    # ── Iterative stability loop ────────────────────────────────────────────────
    current_alpha = init_alpha
    history       = []

    logger.info(f"\nGoal: min_freq >= {stability_goal} THz | alpha={current_alpha} Ang | u={u} Ang")

    for cycle in range(1, max_iter + 1):
        logger.info(f"\n{'='*20} CYCLE {cycle}/{max_iter} {'='*20}")

        current_atoms.calc = engine.get_calculator()

        # A. Vibrational analysis → qpoints.yaml
        analyzer = VibrationalAnalyzer(
            atoms      = current_atoms,
            engine     = engine,
            displacement = u,
        )
        qpath = vib_cfg.get('qpoints_file') or 'qpoints.yaml'
        analyzer.generate_qpoints_file(filename=qpath)

        # B. Parse frequencies
        from qpoint_handler import QPointParser
        parser    = QPointParser(qpath)
        all_freqs = [b['frequency'] for phon in parser.data['phonon'] for b in phon['band']]
        min_freq  = min(all_freqs)
        energy    = current_atoms.get_potential_energy()

        logger.info(f"  Energy: {energy:.6f} eV | Min freq: {min_freq:.4f} THz")
        history.append({'cycle': cycle, 'energy': energy, 'min_freq': min_freq, 'alpha': current_alpha})

        # C. Convergence check
        if min_freq >= stability_goal:
            logger.info(f"  [Success] Stability goal reached (min_freq {min_freq:.4f} ≥ {stability_goal} THz).")
            break

        # D. Stagnation detection and adaptive alpha
        if cycle > 1:
            improvement = min_freq - history[-2]['min_freq']
            if improvement < stag_eps:
                new_alpha = current_alpha * stag_factor
                logger.warning(f"  [Stagnation] Improvement {improvement:.4f} THz < eps={stag_eps} THz. "
                               f"Alpha: {current_alpha:.3f} -> {new_alpha:.3f} Ang")
                current_alpha = new_alpha

        # E. Multi-mode following with updated alpha
        logger.info(f"  [Action] Mode following (alpha={current_alpha:.3f} Ang)...")
        iter_config = copy.deepcopy(config)
        iter_config['analysis']['vibrational']['mode_refinement']['perturbation_alpha'] = current_alpha

        follower      = MultiModeFollower(engine, config=iter_config)
        current_atoms = follower.optimize(
            current_atoms,
            fmax      = target_fmax,
            steps     = target_steps,
            optimizer = optimizer,
        )

    # ── Summary ─────────────────────────────────────────────────────────────────
    logger.info(f"\n{'='*20} SUMMARY (u={u} Ang) {'='*20}")
    logger.info(f"{'Cycle':>5} | {'Energy (eV)':>12} | {'Min Freq (THz)':>14} | {'Alpha (Ang)':>10}")
    logger.info("-" * 55)
    for h in history:
        logger.info(f"{h['cycle']:5d} | {h['energy']:12.6f} | {h['min_freq']:14.4f} | {h['alpha']:10.3f}")

    final_freq = history[-1]['min_freq'] if history else float('nan')
    if final_freq < stability_goal:
        logger.error(f"\n[Conclusion] Goal NOT met. Final min freq: {final_freq:.4f} THz")
    else:
        logger.info(f"\n[Conclusion] Stabilised to {final_freq:.4f} THz.")

    out_name = f"{out_prefix}_u{str(u).replace('.', '')}_final.vasp"
    current_atoms.write(out_name)
    logger.info(f"Saved final structure to: {out_name}")

    return current_atoms


if __name__ == "__main__":
    c_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), 'config.yaml')
    u_val  = float(sys.argv[2]) if len(sys.argv) > 2 else None
    run_enhanced_phonon_refinement(c_path, u_val)
