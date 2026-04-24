import os
import sys
import yaml
import numpy as np
from ase.io import read, write

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from ads_workflow_mgr import AdsorptionWorkflowManager
from chemisorption_builder import build_chemisorption_structures
from logger_utils import setup_logger
from surface_utils import create_slab_from_bulk, write_standardized_vasp, passivate_surface_coverage_general


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def execute_discovery_stage(slab, mol, config, out_prefix, logger, verbose=True, tag=2, center_target='Si'):
    """Geometry-only candidate generation for physisorption and chemisorption."""
    rs_cfg    = config.get('reaction_search', {})
    mechs_cfg = rs_cfg.get('mechanisms', {})
    physi_cfg = mechs_cfg.get('physisorption', {})
    chem_cfg  = mechs_cfg.get('chemisorption', {})
    symprec   = rs_cfg.get('candidate_filter', {}).get('symprec', 0.2)

    run_phy  = physi_cfg.get('enabled', True)
    run_chem = chem_cfg.get('enabled', True)

    mgr = AdsorptionWorkflowManager(slab, config=config, symprec=symprec, verbose=verbose)

    # Lateral span diagnostic
    d_mol = mgr.calculate_molecule_lateral_extent(mol)
    a_len = np.linalg.norm(slab.cell[0])
    b_len = np.linalg.norm(slab.cell[1])
    logger.info(f"  DIAGNOSTIC: {mol.get_chemical_formula()} span = {d_mol:.2f} Å | "
                f"Substrate = {a_len:.2f} × {b_len:.2f} Å")
    if a_len < d_mol + 3.0 or b_len < d_mol + 3.0:
        logger.warning(f"  PBC CONFLICT: cell ({a_len:.1f}×{b_len:.1f}) may be too small "
                       f"for {mol.get_chemical_formula()} (span={d_mol:.1f} Å).")

    all_cands = []

    if run_phy:
        logger.info(f"  Physisorption search for {mol.get_chemical_formula()}...")
        phy_cands = mgr.generate_physisorption_candidates(
            mol,
            height     = physi_cfg.get('placement_height', 3.5),
            n_rot      = physi_cfg.get('rot_steps', 8),
            config     = config,
            tag        = tag,
        )
        for c in phy_cands:
            c.info['mechanism'] = 'physisorption'
        all_cands.extend(phy_cands)

    if run_chem:
        logger.info(f"  Chemisorption search for {mol.get_chemical_formula()} "
                    f"(center={center_target})...")
        chem_cands = build_chemisorption_structures(
            molecule     = mol,
            center_target= center_target,
            surface      = slab,
            rot_steps    = chem_cfg.get('rot_steps', 8),
            config       = config,
            verbose      = verbose,
            tag          = tag,
        )
        for c in chem_cands:
            c.info['mechanism'] = 'chemisorption'
        all_cands.extend(chem_cands)

        byproducts = [c.info['isolated_byproduct'] for c in chem_cands if 'isolated_byproduct' in c.info]
        if byproducts:
            write(f'{out_prefix}_byproducts.extxyz', byproducts)

    if all_cands:
        write(f'{out_prefix}_all_poses.extxyz', all_cands)

    return all_cands


def run_generic_adsorption_study(config_path='config.yaml'):
    config    = load_config(config_path)
    paths     = config['paths']
    sp_cfg    = config.get('surface_prep', {})
    rs_cfg    = config.get('reaction_search', {})
    mechs_cfg = rs_cfg.get('mechanisms', {})
    inh_cfg   = mechs_cfg.get('inhibition', {})

    logger = setup_logger(
        log_path = paths.get('output_prefix', 'results') + '_workflow.log',
        verbose  = True,
    )
    logger.info(f"--- Starting AutoFlow-SRXN Discovery Study ({config_path}) ---")

    mol_file   = paths['adsorbate']
    inh_file   = paths.get('inhibitor')
    out_prefix = paths.get('output_prefix', 'cands_out')

    mol  = read(mol_file)
    slab = None

    # ── Stage 0: Substrate generation & passivation ────────────────────────────
    sub_gen_cfg = sp_cfg.get('slab_generation', {})
    if sub_gen_cfg.get('enabled', False):
        logger.info("STAGE 0: Generating substrate slab...")
        bulk_atoms = read(paths['substrate_bulk'])
        slab = create_slab_from_bulk(
            bulk_atoms        = bulk_atoms,
            miller_indices    = sub_gen_cfg.get('miller', [1, 0, 0]),
            thickness         = sub_gen_cfg.get('thickness_ang', 10.0),
            vacuum            = sub_gen_cfg.get('vacuum_ang', 10.0),
            target_area       = sub_gen_cfg.get('target_area_ang2'),
            supercell_matrix  = sub_gen_cfg.get('supercell_matrix'),
            top_termination   = sub_gen_cfg.get('top_termination'),
            bottom_termination= sub_gen_cfg.get('bottom_termination'),
            verbose           = True,
        )
        write_standardized_vasp('generated_substrate.vasp', slab)
        logger.info("Saved generated raw substrate to 'generated_substrate.vasp'.")
    else:
        slab_path = paths.get('substrate_slab')
        if not slab_path:
            raise ValueError("Either surface_prep.slab_generation.enabled must be true "
                             "or paths.substrate_slab must point to a pre-built slab.")
        slab = read(slab_path)

    pass_cfg = sp_cfg.get('passivation', {})
    if pass_cfg.get('enabled', False):
        logger.info("Applying geometric passivation...")
        ideal_coord = sp_cfg.get('surface_analysis', {}).get('ideal_coordination', {})
        side  = pass_cfg.get('side', 'bottom')
        sides = [side] if side != 'both' else ['top', 'bottom']
        for s in sides:
            slab = passivate_surface_coverage_general(
                slab,
                h_coverage  = pass_cfg.get('coverage', 1.0),
                valence_map = ideal_coord,
                element     = pass_cfg.get('element', 'H'),
                side        = s,
                verbose     = True,
            )
        write_standardized_vasp('passivated.vasp', slab)
        logger.info("Saved passivated substrate to 'passivated.vasp'.")

    # ── Stage 1: Inhibitor pre-treatment (optional branching) ─────────────────
    base_slabs = [slab]
    if inh_file and os.path.exists(inh_file) and inh_cfg.get('enabled', False):
        logger.info(f"STAGE 1: Inhibitor discovery ({inh_file})")
        inh_mol    = read(inh_file)
        inh_center = inh_cfg.get('inhibitor_center', 'O')
        inh_cands  = execute_discovery_stage(
            slab, inh_mol, config, f"{out_prefix}_inh",
            logger, True, tag=2, center_target=inh_center,
        )
        limit      = inh_cfg.get('branching_limit', 5)
        base_slabs = inh_cands[:limit]
        logger.info(f"  Branching into {len(base_slabs)} inhibited geometries for Stage 2.")

    # ── Stage 2: Main precursor discovery ─────────────────────────────────────
    logger.info(f"STAGE 2: Main precursor discovery ({mol_file})")
    mol_center       = mechs_cfg.get('chemisorption', {}).get('precursor_center', 'Si')
    all_final_results = []
    for i, s in enumerate(base_slabs):
        suffix  = f"_inh{i}" if len(base_slabs) > 1 else ""
        results = execute_discovery_stage(
            s, mol, config, f"{out_prefix}{suffix}",
            logger, True, tag=3, center_target=mol_center,
        )
        all_final_results.extend(results)

    logger.info(f"--- Study Complete. Total unique candidates: {len(all_final_results)} ---")


if __name__ == '__main__':
    default_config = os.path.join(os.path.dirname(__file__), 'config.yaml')
    c_path = sys.argv[1] if len(sys.argv) > 1 else default_config
    run_generic_adsorption_study(c_path)
