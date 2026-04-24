"""
Step 1: Substrate preparation
  1. Load config
  2. Generate SiO2(001) slab from bulk
  3. Passivate bottom layer with H
  4. Relax → MD → Relax
  5. Save outputs into 'results/'
"""
import os
import sys
import yaml
from ase.io import read, write
from ase.constraints import FixAtoms
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, '../../src')))

from potentials import SimulationEngine
from surface_utils import create_slab_from_bulk, passivate_surface_coverage_general, write_standardized_vasp
from si_surface_utils import SI_VALENCE_MAP
from logger_utils import setup_logger

def run_step1():
    os.chdir(script_dir)

    with open('config.yaml', 'r') as fh:
        config = yaml.safe_load(fh)

    out_dir = config['paths'].get('output_dir', 'results')
    os.makedirs(out_dir, exist_ok=True)

    log_file = os.path.join(out_dir, 'step1_substrate_prep.log')
    logger = setup_logger(log_path=log_file, verbose=True, mode='a')

    logger.info("=================================================================")
    logger.info("  Step 1: SiO2(001) Substrate Generation & Equilibration         ")
    logger.info("=================================================================")

    engine = SimulationEngine(config=config)
    calc   = engine.get_calculator()

    # ── Slab generation ────────────────────────────────────────────────────────
    logger.info('Generating SiO2(001) substrate from bulk...')
    bulk   = read(config['paths']['substrate_bulk'])
    sg     = config['surface_prep']['slab_generation']
    slab   = create_slab_from_bulk(
        bulk,
        miller_indices    = sg['miller'],
        thickness         = sg['thickness_ang'],
        vacuum            = sg['vacuum_ang'],
        target_area       = sg['target_area_ang2'],
        top_termination   = sg['top_termination'],
        bottom_termination= sg['bottom_termination'],
        verbose           = True,
    )

    # ── Passivation ────────────────────────────────────────────────────────────
    logger.info('Passivating bottom layer with Hydrogen...')
    pv   = config['surface_prep']['passivation']
    slab = passivate_surface_coverage_general(
        slab,
        h_coverage  = pv['coverage'],
        valence_map = SI_VALENCE_MAP,
        side        = pv['side'],
        verbose     = True,
    )
    write_standardized_vasp(os.path.join(out_dir, 'slab_initial.vasp'), slab)

    # ── Relaxation 1 ───────────────────────────────────────────────────────────
    logger.info('Performing initial relaxation...')
    rel   = config['engine']['relaxation']
    eq    = config['surface_prep']['equilibration']
    z_min = slab.positions[:, 2].min()
    slab.set_constraint(FixAtoms(mask=slab.positions[:, 2] < z_min + eq['frozen_z_ang']))
    slab.calc = calc
    engine.relax(slab, fmax=rel['fmax'], steps=rel['steps'], optimizer=rel['optimizer'])
    write_standardized_vasp(os.path.join(out_dir, 'slab_relax1.vasp'), slab)

    # ── MD equilibration ───────────────────────────────────────────────────────
    logger.info('Running MD equilibration...')
    md_cfg    = config['engine']['md']
    eq        = config['surface_prep']['equilibration']
    timestep  = md_cfg.get('timestep_fs', 1.0)
    temp_K    = eq.get('temperature_K', md_cfg['temperature_K'])
    n_steps   = eq.get('md_steps',      md_cfg['md_steps'])
    friction  = 1.0 / (eq.get('damping', md_cfg['damping']) * units.fs)
    MaxwellBoltzmannDistribution(slab, temperature_K=temp_K)
    md_log = os.path.join(out_dir, 'md_equilibration.log')
    with open(md_log, 'a') as md_f:
        dyn = Langevin(
            slab, timestep * units.fs,
            temperature_K = temp_K,
            friction      = friction,
            logfile       = md_f,
            loginterval   = 100,
        )
        dyn.run(n_steps)
    write_standardized_vasp(os.path.join(out_dir, 'slab_md.vasp'), slab)

    # ── Relaxation 2 ───────────────────────────────────────────────────────────
    logger.info('Performing final relaxation after MD...')
    engine.relax(slab, fmax=rel['fmax'], steps=rel['steps'], optimizer=rel['optimizer'])

    final_path = os.path.join(out_dir, 'slab_final.vasp')
    write_standardized_vasp(final_path, slab)
    E_slab = slab.get_potential_energy()
    logger.info(f'Final relaxed substrate saved to {final_path}')
    logger.info(f'Substrate Energy: E_slab = {E_slab:.6f} eV  ({len(slab)} atoms)')
    logger.info("=================================================================")

if __name__ == '__main__':
    run_step1()
