"""
Step 3: Adsorption site search and candidate relaxation
  1. Load config
  2. Load stable substrate (step 1) and relaxed gas (step 2)
  3. Generate physisorption candidates
  4. Relax top candidates
  5. Save best structure
"""
import os
import sys
import yaml
from ase.io import read, write
from ase.constraints import FixAtoms

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, '../../src')))

from potentials import SimulationEngine
from ads_workflow_mgr import AdsorptionWorkflowManager
from surface_utils import write_standardized_vasp
from logger_utils import setup_logger

def run_step3():
    os.chdir(script_dir)

    with open('config.yaml', 'r') as fh:
        config = yaml.safe_load(fh)

    out_dir = config['paths'].get('output_dir', 'results')
    os.makedirs(out_dir, exist_ok=True)

    log_file = os.path.join(out_dir, 'step3_adsorption_search.log')
    logger = setup_logger(log_path=log_file, verbose=True, mode='a')

    logger.info("=================================================================")
    logger.info("  Step 3: Adsorption Site Search & Candidate Relaxation          ")
    logger.info("=================================================================")

    engine = SimulationEngine(config=config)
    calc   = engine.get_calculator()

    slab_path = os.path.join(out_dir, 'slab_final.vasp')
    gas_path  = os.path.join(out_dir, 'DIPAS_gas_relaxed.vasp')
    if not os.path.exists(slab_path) or not os.path.exists(gas_path):
        logger.error("Required outputs from Step 1 or Step 2 are missing!")
        return

    logger.info(f"Loading slab from {slab_path} and gas from {gas_path}")
    slab = read(slab_path)
    slab.calc = calc
    E_slab = slab.get_potential_energy()

    gas = read(gas_path)
    gas.calc = calc
    E_gas = gas.get_potential_energy()
    logger.info(f"Reference energies: E_slab = {E_slab:.6f} eV, E_gas = {E_gas:.6f} eV")

    # ── Candidate generation ───────────────────────────────────────────────────
    logger.info('Physisorption candidate generation...')
    symprec   = config['reaction_search']['candidate_filter']['symprec']
    physi_cfg = config['reaction_search']['mechanisms']['physisorption']
    mgr = AdsorptionWorkflowManager(slab, config=config, symprec=symprec, verbose=True)

    candidates = mgr.generate_physisorption_candidates(
        gas,
        height     = physi_cfg['placement_height'],
        n_rot      = physi_cfg['rot_steps'],
        rot_center = 'com',
    )

    if not candidates:
        logger.error('No valid adsorption candidates found. Aborting.')
        return
    logger.info(f'Generated {len(candidates)} candidate poses.')

    # ── Candidate relaxation ───────────────────────────────────────────────────
    logger.info('Relaxing top candidates and selecting minimum-energy structure...')
    rel          = config['engine']['relaxation']
    z_fix_thresh = config['surface_prep']['equilibration']['frozen_z_ang']

    header = f"{'#':>3} | {'E_init (eV)':>12} | {'E_final (eV)':>13} | {'ΔE_ads (eV)':>12}"
    logger.info(header)
    logger.info('-' * 55)

    all_relaxed = []
    best_atoms  = None
    best_energy = float('inf')

    for i, cand in enumerate(candidates[:8]):
        z_cand_min = cand.positions[:, 2].min()
        cand.set_constraint(FixAtoms(mask=cand.positions[:, 2] < z_cand_min + z_fix_thresh))
        cand.calc = calc

        e_init  = cand.get_potential_energy()
        e_final = engine.relax(cand, fmax=rel['fmax'], steps=rel['steps'], optimizer=rel['optimizer'])
        dE_ads  = e_final - E_slab - E_gas

        logger.info(f'{i+1:3d} | {e_init:12.4f} | {e_final:13.4f} | {dE_ads:+12.4f}')
        cand.info.update({'candidate_id': i + 1, 'energy': e_final, 'E_ads_eV': dE_ads})
        all_relaxed.append(cand.copy())

        if e_final < best_energy:
            best_energy = e_final
            best_atoms  = cand.copy()

    write(os.path.join(out_dir, 'all_relaxed_candidates.extxyz'), all_relaxed)

    E_ads = best_energy - E_slab - E_gas
    logger.info(f'\n  Best structure:  E = {best_energy:.6f} eV')
    logger.info(f'  ΔE_ads (pot):    {E_ads:+.4f} eV')

    best_clean = best_atoms.copy()
    best_clean.set_constraint([])
    write(os.path.join(out_dir, 'SiO2_DIPAS_best.extxyz'), best_clean)
    write_standardized_vasp(os.path.join(out_dir, 'SiO2_DIPAS_best.vasp'), best_clean)
    logger.info(f"Saved best structure to results/SiO2_DIPAS_best.*")
    logger.info("=================================================================")

if __name__ == '__main__':
    run_step3()
