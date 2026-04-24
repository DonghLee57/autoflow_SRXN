"""
PHVA vs FHVA Benchmark: Free Energy of Adsorption on SiO2(001)
==============================================================
Full all-in-one workflow:
  Step 1 : SiO2(001) slab generation + MD equilibration
  Step 2 : Gas-phase DIPAS relaxation
  Step 3 : Adsorption site search (physisorption grid)
  Step 4 : Relax top candidates, select lowest-energy structure
  Step 5 : FHVA on isolated gas molecule
  Step 6 : FHVA on adsorbed system
  Step 7 : PHVA on adsorbed system
  Step 8 : Compute ΔG_rxn (FHVA and PHVA) and save comparison report

Adsorption reaction modelled:
    DIPAS(g) + SiO2-slab → DIPAS*/SiO2-slab

Rigid-slab approximation: slab phonons assumed to cancel between
reactant and product.

Note: for step-by-step execution see step1_*.py … step4_*.py.
"""

import os
import sys
import shutil
import yaml
import numpy as np
from ase.io import read, write
from ase.constraints import FixAtoms
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from ase.neighborlist import neighbor_list

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, '../../src')))

from potentials import SimulationEngine
from surface_utils import create_slab_from_bulk, passivate_surface_coverage_general
from si_surface_utils import SI_VALENCE_MAP
from ads_workflow_mgr import AdsorptionWorkflowManager
from vibrational_analyzer import VibrationalAnalyzer
from thermo_engine import ThermoCalculator, eV_to_J_mol
from logger_utils import setup_logger


# ── Utilities ────────────────────────────────────────────────────────────────

def save_qpoint_yaml(path, system_name, method, freqs_thz,
                     potential_energy_eV, n_total_atoms, n_active_atoms, T):
    """Save vibrational data as phonopy-compatible qpoint.yaml (Gamma point).

    Parameters
    ----------
    freqs_thz : list[float]
        Frequencies in THz. Negative values indicate imaginary modes.

    Returns
    -------
    G_vib_eV : float   Helmholtz vibrational free energy at T (eV/unit)
    ZPE_eV   : float   Zero-point energy (eV/unit)
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    thermo   = ThermoCalculator(freqs_thz)
    G_vib_J  = thermo.calculate_vib_free_energy(T)
    ZPE_J    = thermo.calculate_zpe()
    U_vib_J  = thermo.calculate_vib_internal_energy(T)
    S_vib    = thermo.calculate_vib_entropy(T)

    G_vib_eV = float(G_vib_J / eV_to_J_mol)
    ZPE_eV   = float(ZPE_J   / eV_to_J_mol)

    data = {
        'system':              system_name,
        'method':              method,
        'potential_energy_eV': float(potential_energy_eV),
        'total_atoms':         int(n_total_atoms),
        'active_atoms':        int(n_active_atoms),
        'phonon': [{
            'q-position': [0.0, 0.0, 0.0],
            'weight':     1,
            'band':       [{'frequency': float(f)} for f in freqs_thz],
        }],
        'thermo': {
            'temperature_K':   float(T),
            'G_vib_eV':        G_vib_eV,
            'ZPE_eV':          ZPE_eV,
            'U_vib_eV':        float(U_vib_J / eV_to_J_mol),
            'TS_vib_eV':       float(T * S_vib / eV_to_J_mol),
            'imaginary_count': int(sum(1 for f in freqs_thz if f < 0)),
        },
    }

    with open(path, 'w') as fh:
        yaml.dump(data, fh, default_flow_style=False, sort_keys=False)

    return G_vib_eV, ZPE_eV


def build_phva_active_indices(atoms, n_adsorbate, cutoff_angstrom):
    """Adsorbate atoms + slab atoms within cutoff of any adsorbate atom."""
    n_total = len(atoms)
    ads_set = set(range(n_total - n_adsorbate, n_total))
    i_arr, j_arr = neighbor_list('ij', atoms, cutoff_angstrom)
    slab_neighbors = {
        int(j_arr[k])
        for k, i in enumerate(i_arr)
        if i in ads_set and j_arr[k] not in ads_set
    }
    return sorted(ads_set | slab_neighbors)


# ── Main workflow ─────────────────────────────────────────────────────────────

def run_phva_benchmark():
    os.chdir(script_dir)

    with open('config.yaml', 'r') as fh:
        config = yaml.safe_load(fh)

    logger = setup_logger(log_path='phva_benchmark.log', verbose=True)
    sep    = '=' * 65
    logger.info(sep)
    logger.info('  PHVA vs FHVA Benchmark: Free Energy of Adsorption (SiO2/DIPAS)')
    logger.info(sep)

    # ── Parse config ──────────────────────────────────────────────────────────
    vib_cfg    = config['analysis']['vibrational']
    thermo_cfg = config['analysis']['thermochemistry']
    rel        = config['engine']['relaxation']
    md_cfg     = config['engine']['md']
    sp_cfg     = config['surface_prep']

    T          = thermo_cfg['temperatures_K'][0]
    phva_cut   = vib_cfg['phva_radius_ang']
    relax_fmax = rel['fmax']
    relax_step = rel['steps']
    relax_opt  = rel['optimizer']
    vib_disp   = vib_cfg['displacement_ang']

    # ── Engine ────────────────────────────────────────────────────────────────
    engine = SimulationEngine(config=config)
    calc   = engine.get_calculator()

    # ── Step 1: Substrate ─────────────────────────────────────────────────────
    slab_path   = 'slab_clean.vasp'
    legacy_path = 'SiO2_substrate_standard.vasp'

    if not os.path.exists(slab_path) and os.path.exists(legacy_path):
        shutil.copy(legacy_path, slab_path)
        logger.info(f'[Step 1] Copied legacy substrate → {slab_path}')

    if os.path.exists(slab_path):
        logger.info(f'[Step 1] Loading pre-existing substrate from {slab_path}')
        slab = read(slab_path)
        slab.calc = calc
    else:
        logger.info('[Step 1] Generating SiO2(001) substrate...')
        bulk   = read(config['paths']['substrate_bulk'])
        sg     = sp_cfg['slab_generation']
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
        pv   = sp_cfg['passivation']
        slab = passivate_surface_coverage_general(
            slab,
            h_coverage  = pv['coverage'],
            valence_map = SI_VALENCE_MAP,
            side        = pv['side'],
            verbose     = True,
        )

        eq    = sp_cfg['equilibration']
        z_min = slab.positions[:, 2].min()
        slab.set_constraint(FixAtoms(mask=slab.positions[:, 2] < z_min + eq['frozen_z_ang']))
        slab.calc = calc
        engine.relax(slab, fmax=relax_fmax, steps=relax_step, optimizer=relax_opt)

        eq_cfg   = sp_cfg['equilibration']
        temp_K   = eq_cfg.get('temperature_K', md_cfg['temperature_K'])
        n_steps  = eq_cfg.get('md_steps',      md_cfg['md_steps'])
        timestep = md_cfg.get('timestep_fs', 1.0)
        friction = 1.0 / (eq_cfg.get('damping', md_cfg['damping']) * units.fs)
        MaxwellBoltzmannDistribution(slab, temperature_K=temp_K)
        dyn = Langevin(
            slab, timestep * units.fs,
            temperature_K = temp_K,
            friction      = friction,
            logfile       = 'md_equilibration.log',
            loginterval   = 100,
        )
        dyn.run(n_steps)

        engine.relax(slab, fmax=relax_fmax, steps=relax_step, optimizer=relax_opt)
        write(slab_path, slab)
        logger.info(f'  Saved: {slab_path}')

    E_slab = slab.get_potential_energy()
    n_slab = len(slab)
    logger.info(f'  E_slab = {E_slab:.6f} eV  ({n_slab} atoms)')

    # ── Step 2: Gas-phase molecule ────────────────────────────────────────────
    gas_path = 'DIPAS_gas_relaxed.vasp'

    if os.path.exists(gas_path):
        logger.info(f'[Step 2] Loading pre-relaxed gas molecule from {gas_path}')
        dipas_gas = read(gas_path)
        dipas_gas.calc = calc
    else:
        logger.info('[Step 2] Relaxing gas-phase DIPAS in 10 Å vacuum cell...')
        dipas_gas = read(config['paths']['adsorbate'])
        dipas_gas.center(vacuum=10.0)
        dipas_gas.calc = calc
        engine.relax(dipas_gas, fmax=relax_fmax, steps=relax_step, optimizer=relax_opt)
        write(gas_path, dipas_gas)
        logger.info(f'  Saved: {gas_path}')

    E_gas = dipas_gas.get_potential_energy()
    n_gas = len(dipas_gas)
    logger.info(f'  E_gas = {E_gas:.6f} eV  ({n_gas} atoms)')

    # ── Step 3: Adsorption site search ────────────────────────────────────────
    logger.info('[Step 3] Physisorption candidate generation...')
    slab_ads = slab.copy()
    slab_ads.calc = calc

    physi_cfg  = config['reaction_search']['mechanisms']['physisorption']
    symprec    = config['reaction_search']['candidate_filter']['symprec']
    mgr        = AdsorptionWorkflowManager(slab_ads, config=config, symprec=symprec, verbose=True)
    dipas_ref  = read(config['paths']['adsorbate'])

    candidates = mgr.generate_physisorption_candidates(
        dipas_ref,
        height     = physi_cfg['placement_height'],
        n_rot      = physi_cfg['rot_steps'],
        rot_center = 'com',
    )
    if not candidates:
        logger.error('No valid adsorption candidates found. Aborting.')
        return
    logger.info(f'  Generated {len(candidates)} candidate poses.')

    # ── Step 4: Relax candidates, select best ────────────────────────────────
    logger.info('[Step 4] Relaxing candidates...')
    z_fix_thresh = sp_cfg['equilibration']['frozen_z_ang']
    all_relaxed  = []
    best_atoms   = None
    best_energy  = float('inf')

    logger.info(f"{'#':>3} | {'E_init (eV)':>12} | {'E_final (eV)':>13} | {'ΔE_ads (eV)':>12}")
    logger.info('-' * 55)

    for i, cand in enumerate(candidates[:8]):
        z_cand_min = cand.positions[:, 2].min()
        cand.set_constraint(FixAtoms(mask=cand.positions[:, 2] < z_cand_min + z_fix_thresh))
        cand.calc = calc
        e_init  = cand.get_potential_energy()
        e_final = engine.relax(cand, fmax=relax_fmax, steps=relax_step, optimizer=relax_opt)
        dE_ads  = e_final - E_slab - E_gas
        logger.info(f'{i+1:3d} | {e_init:12.4f} | {e_final:13.4f} | {dE_ads:+12.4f}')
        cand.info.update({'candidate_id': i + 1, 'energy': e_final, 'E_ads_eV': dE_ads})
        all_relaxed.append(cand.copy())
        if e_final < best_energy:
            best_energy = e_final
            best_atoms  = cand.copy()

    write('all_relaxed_candidates.extxyz', all_relaxed)
    E_ads = best_energy - E_slab - E_gas
    logger.info(f'\n  Best structure:  E = {best_energy:.6f} eV  |  ΔE_ads = {E_ads:+.4f} eV')

    # extxyz preserves atom order (critical for PHVA index matching)
    best_clean = best_atoms.copy()
    best_clean.set_constraint([])
    write('SiO2_DIPAS_best.extxyz', best_clean)
    write('SiO2_DIPAS_best.vasp',   best_clean)

    # ── Step 5: FHVA — gas-phase DIPAS ───────────────────────────────────────
    logger.info('\n[Step 5] FHVA on gas-phase DIPAS...')
    gas_vib = dipas_gas.copy()
    gas_vib.set_constraint([])
    gas_vib.calc = calc

    va_gas = VibrationalAnalyzer(
        gas_vib, engine, indices=None,
        name='vibrations/gas_fhva', displacement=vib_disp,
    )
    freqs_gas_thz, _ = va_gas.run_analysis()
    G_vib_gas, ZPE_gas = save_qpoint_yaml(
        'vibrations/gas_fhva/qpoint.yaml',
        'DIPAS_gas', 'fhva', freqs_gas_thz, E_gas, n_gas, n_gas, T,
    )
    logger.info(f'  {len(freqs_gas_thz)} modes | G_vib = {G_vib_gas:+.4f} eV | '
                f'imag = {sum(1 for f in freqs_gas_thz if f < 0)}')

    # ── Step 6: FHVA — adsorbed system ───────────────────────────────────────
    logger.info('[Step 6] FHVA on adsorbed system (full Hessian)...')
    ads_fhva = best_clean.copy()
    ads_fhva.calc = calc

    va_fhva = VibrationalAnalyzer(
        ads_fhva, engine, indices=None,
        name='vibrations/adsorbed_fhva', displacement=vib_disp,
    )
    freqs_fhva_thz, _ = va_fhva.run_analysis()
    G_vib_ads_fhva, ZPE_ads_fhva = save_qpoint_yaml(
        'vibrations/adsorbed_fhva/qpoint.yaml',
        'SiO2_DIPAS', 'fhva', freqs_fhva_thz, best_energy, len(ads_fhva), len(ads_fhva), T,
    )
    logger.info(f'  {len(freqs_fhva_thz)} modes | G_vib = {G_vib_ads_fhva:+.4f} eV | '
                f'imag = {sum(1 for f in freqs_fhva_thz if f < 0)}')

    # ── Step 7: PHVA — adsorbed system ───────────────────────────────────────
    logger.info(f'[Step 7] PHVA on adsorbed system (cutoff = {phva_cut} Ang)...')
    ads_phva   = best_clean.copy()
    ads_phva.calc = calc
    active_idx    = build_phva_active_indices(best_clean, n_gas, phva_cut)
    n_active      = len(active_idx)
    n_slab_active = n_active - n_gas
    logger.info(f'  Active set: {n_active} / {len(ads_phva)} atoms '
                f'({n_gas} adsorbate + {n_slab_active} slab neighbors)')

    va_phva = VibrationalAnalyzer(
        ads_phva, engine, indices=active_idx,
        name='vibrations/adsorbed_phva', displacement=vib_disp,
    )
    freqs_phva_thz, _ = va_phva.run_analysis()
    G_vib_ads_phva, ZPE_ads_phva = save_qpoint_yaml(
        'vibrations/adsorbed_phva/qpoint.yaml',
        'SiO2_DIPAS', 'phva', freqs_phva_thz, best_energy, len(ads_phva), n_active, T,
    )
    logger.info(f'  {len(freqs_phva_thz)} modes | G_vib = {G_vib_ads_phva:+.4f} eV | '
                f'imag = {sum(1 for f in freqs_phva_thz if f < 0)}')

    # ── Step 8: Reaction free energies ────────────────────────────────────────
    dG_vib_fhva = G_vib_ads_fhva - G_vib_gas
    dG_vib_phva = G_vib_ads_phva - G_vib_gas
    dG_rxn_fhva = E_ads + dG_vib_fhva
    dG_rxn_phva = E_ads + dG_vib_phva
    ddG         = dG_rxn_phva - dG_rxn_fhva
    dZPE_fhva   = ZPE_ads_fhva - ZPE_gas
    dZPE_phva   = ZPE_ads_phva - ZPE_gas

    os.makedirs('results', exist_ok=True)
    comparison = {
        'temperature_K': float(T),
        'approximation': 'rigid-slab',
        'potential_energy': {
            'E_slab_eV':     float(E_slab),
            'E_gas_eV':      float(E_gas),
            'E_adsorbed_eV': float(best_energy),
            'dE_ads_eV':     float(E_ads),
        },
        'vibrational_free_energy_eV': {
            'G_vib_gas_fhva':      float(G_vib_gas),
            'G_vib_adsorbed_fhva': float(G_vib_ads_fhva),
            'G_vib_adsorbed_phva': float(G_vib_ads_phva),
        },
        'zpe_correction_eV':   {'dZPE_fhva': float(dZPE_fhva), 'dZPE_phva': float(dZPE_phva)},
        'reaction_free_energy': {
            'dG_vib_correction_fhva_eV': float(dG_vib_fhva),
            'dG_vib_correction_phva_eV': float(dG_vib_phva),
            'dG_rxn_fhva_eV':            float(dG_rxn_fhva),
            'dG_rxn_phva_eV':            float(dG_rxn_phva),
            'ddG_phva_minus_fhva_eV':    float(ddG),
            'ddG_phva_minus_fhva_meV':   float(ddG * 1000),
        },
        'phva_efficiency': {
            'cutoff_dist_A':            float(phva_cut),
            'n_total_atoms':            int(len(ads_fhva)),
            'n_active_atoms':           int(n_active),
            'n_adsorbate_atoms':        int(n_gas),
            'n_slab_active':            int(n_slab_active),
            'force_call_reduction_pct': float(100.0 * (1.0 - n_active / len(ads_fhva))),
        },
    }
    with open('results/phva_fhva_comparison.yaml', 'w') as fh:
        yaml.dump(comparison, fh, default_flow_style=False, sort_keys=False)

    logger.info(f'\n{sep}')
    logger.info(f'  PHVA vs FHVA COMPARISON  (T = {T} K)')
    logger.info(sep)
    logger.info(f'  ΔE_ads   (potential):     {E_ads:+.4f} eV')
    logger.info(f'  ΔG_rxn   (FHVA ref):      {dG_rxn_fhva:+.4f} eV')
    logger.info(f'  ΔG_rxn   (PHVA approx):   {dG_rxn_phva:+.4f} eV')
    logger.info(f'  ΔΔG      (PHVA − FHVA):   {ddG*1000:+.2f} meV')
    logger.info(f'  Force call savings:       '
                f'{comparison["phva_efficiency"]["force_call_reduction_pct"]:.1f} %')
    verdict = 'PASS' if abs(ddG) < 0.010 else 'WARN: >10 meV — consider wider cutoff'
    logger.info(f'  PHVA accuracy:            {verdict}')
    logger.info(sep)


if __name__ == '__main__':
    run_phva_benchmark()
