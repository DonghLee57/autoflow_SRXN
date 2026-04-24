"""
Step 4: Vibrational analysis (FHVA vs PHVA benchmark)
  1. Load config
  2. Load structures from previous steps
  3. FHVA on gas-phase
  4. FHVA on adsorbed system
  5. PHVA on adsorbed system
  6. Save comparison and free energy summary
"""
import os
import sys
import yaml
from ase.io import read
from ase.constraints import FixAtoms

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, '../../src')))

from potentials import SimulationEngine
from vibrational_analyzer import VibrationalAnalyzer, calculate_thermo, build_phva_active_indices
from logger_utils import setup_logger

def run_step4():
    os.chdir(script_dir)

    with open('config.yaml', 'r') as fh:
        config = yaml.safe_load(fh)

    out_dir = config['paths'].get('output_dir', 'results')
    log_file = os.path.join(out_dir, 'step4_vibrational_analysis.log')
    logger = setup_logger(log_path=log_file, verbose=True, mode='a')

    logger.info("=================================================================")
    logger.info("  Step 4: Vibrational Analysis (FHVA vs PHVA Benchmark)          ")
    logger.info("=================================================================")

    engine = SimulationEngine(config=config)
    calc   = engine.get_calculator()

    slab_path = os.path.join(out_dir, 'slab_final.vasp')
    gas_path  = os.path.join(out_dir, 'DIPAS_gas_relaxed.vasp')
    best_path = os.path.join(out_dir, 'SiO2_DIPAS_best.extxyz')
    if not (os.path.exists(slab_path) and os.path.exists(gas_path) and os.path.exists(best_path)):
        logger.error("Required outputs from Step 1, 2, or 3 are missing!")
        return

    logger.info("Calculating reference energies...")
    slab = read(slab_path)
    slab.calc = calc
    E_slab = slab.get_potential_energy()
    n_slab = len(slab)

    gas_vib = read(gas_path)
    gas_vib.set_constraint([])
    gas_vib.calc = calc
    E_gas = gas_vib.get_potential_energy()
    n_gas = len(gas_vib)

    best_clean = read(best_path)
    best_clean.set_constraint([])
    best_clean.calc = calc
    best_energy = best_clean.get_potential_energy()
    E_ads = best_energy - E_slab - E_gas

    logger.info(f"  E_slab = {E_slab:.6f} eV")
    logger.info(f"  E_gas  = {E_gas:.6f} eV")
    logger.info(f"  E_best = {best_energy:.6f} eV")
    logger.info(f"  E_ads  = {E_ads:+.6f} eV")

    # ── Vibrational parameters from new config paths ───────────────────────────
    vib_cfg  = config['analysis']['vibrational']
    thermo_cfg = config['analysis']['thermochemistry']
    T        = thermo_cfg['temperatures_K'][0]
    vib_disp = vib_cfg['displacement_ang']
    phva_cut = vib_cfg['phva_radius_ang']

    vib_dir = os.path.join(out_dir, 'vibrations')
    os.makedirs(vib_dir, exist_ok=True)

    # ── FHVA on gas-phase ──────────────────────────────────────────────────────
    logger.info('\n[FHVA] Gas-phase DIPAS...')
    va_gas = VibrationalAnalyzer(
        gas_vib, engine,
        indices      = None,
        name         = os.path.join(vib_dir, 'gas_fhva'),
        displacement = vib_disp,
    )
    freqs_thz_gas, _ = va_gas.run_analysis()
    va_gas.generate_qpoints_file(os.path.join(vib_dir, 'gas_fhva', 'qpoints.yaml'))
    G_vib_gas, ZPE_gas = calculate_thermo(freqs_thz_gas, T)
    logger.info(f'  {len(freqs_thz_gas)} modes | G_vib = {G_vib_gas:+.4f} eV | '
                f'imag = {sum(1 for f in freqs_thz_gas if f < 0)}')

    # ── FHVA on adsorbed system ────────────────────────────────────────────────
    logger.info('\n[FHVA] Adsorbed system (full Hessian)...')
    ads_fhva = best_clean.copy()
    ads_fhva.calc = calc
    va_fhva = VibrationalAnalyzer(
        ads_fhva, engine,
        indices      = None,
        name         = os.path.join(vib_dir, 'adsorbed_fhva'),
        displacement = vib_disp,
    )
    freqs_thz_fhva, _ = va_fhva.run_analysis()
    va_fhva.generate_qpoints_file(os.path.join(vib_dir, 'adsorbed_fhva', 'qpoints.yaml'))
    G_vib_ads_fhva, ZPE_ads_fhva = calculate_thermo(freqs_thz_fhva, T)
    logger.info(f'  {len(freqs_thz_fhva)} modes | G_vib = {G_vib_ads_fhva:+.4f} eV | '
                f'imag = {sum(1 for f in freqs_thz_fhva if f < 0)}')

    # ── PHVA on adsorbed system ────────────────────────────────────────────────
    logger.info(f'\n[PHVA] Adsorbed system (cutoff = {phva_cut} Å)...')
    ads_phva = best_clean.copy()
    ads_phva.calc = calc
    active_idx    = build_phva_active_indices(best_clean, n_gas, phva_cut)
    n_active      = len(active_idx)
    n_slab_active = n_active - n_gas
    logger.info(f'  Active set: {n_active} / {len(ads_phva)} atoms '
                f'({n_gas} adsorbate + {n_slab_active} slab neighbors)')

    va_phva = VibrationalAnalyzer(
        ads_phva, engine,
        indices      = active_idx,
        name         = os.path.join(vib_dir, 'adsorbed_phva'),
        displacement = vib_disp,
    )
    freqs_thz_phva, _ = va_phva.run_analysis()
    va_phva.generate_qpoints_file(os.path.join(vib_dir, 'adsorbed_phva', 'qpoints.yaml'))
    G_vib_ads_phva, ZPE_ads_phva = calculate_thermo(freqs_thz_phva, T)
    logger.info(f'  {len(freqs_thz_phva)} modes | G_vib = {G_vib_ads_phva:+.4f} eV | '
                f'imag = {sum(1 for f in freqs_thz_phva if f < 0)}')

    # ── Thermochemistry ────────────────────────────────────────────────────────
    dG_vib_fhva = G_vib_ads_fhva - G_vib_gas
    dG_vib_phva = G_vib_ads_phva - G_vib_gas
    dG_rxn_fhva = E_ads + dG_vib_fhva
    dG_rxn_phva = E_ads + dG_vib_phva
    ddG         = dG_rxn_phva - dG_rxn_fhva
    dZPE_fhva   = ZPE_ads_fhva - ZPE_gas
    dZPE_phva   = ZPE_ads_phva - ZPE_gas

    comparison = {
        'temperature_K': float(T),
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
        'zpe_correction_eV': {
            'dZPE_fhva': float(dZPE_fhva),
            'dZPE_phva': float(dZPE_phva),
        },
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

    import yaml as _yaml
    comp_path = os.path.join(out_dir, 'phva_fhva_comparison.yaml')
    with open(comp_path, 'w') as fh:
        _yaml.dump(comparison, fh, default_flow_style=False, sort_keys=False)

    logger.info('\n=================================================================')
    logger.info(f'  PHVA vs FHVA COMPARISON  (T = {T} K)')
    logger.info('=================================================================')
    logger.info(f'  ΔE_ads (static):    {E_ads:+.4f} eV')
    logger.info(f'  ΔG_rxn (FHVA):      {dG_rxn_fhva:+.4f} eV')
    logger.info(f'  ΔG_rxn (PHVA):      {dG_rxn_phva:+.4f} eV')
    logger.info(f'  Error (PHVA-FHVA):  {ddG*1000:+.1f} meV')
    logger.info(f'  Force call savings: {comparison["phva_efficiency"]["force_call_reduction_pct"]:.1f} %\n')
    logger.info(f'Saved benchmark summary to {comp_path}')
    logger.info("=================================================================")

if __name__ == '__main__':
    run_step4()
