import os
import sys
import yaml
import numpy as np
from ase.io import read, write

from autoflow_srxn.ads_workflow_mgr import AdsorptionWorkflowManager
from autoflow_srxn.chemisorption_builder import build_chemisorption_structures
from autoflow_srxn.logger_utils import setup_logger
from autoflow_srxn.surface_utils import create_slab_from_bulk, write_standardized_vasp, passivate_surface_coverage_general


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
    logger.info(f"  DIAGNOSTIC: {mol.get_chemical_formula()} span = {d_mol:.2f} A | "
                f"Substrate = {a_len:.2f} x {b_len:.2f} A")
    if a_len < d_mol + 3.0 or b_len < d_mol + 3.0:
        logger.warning(f"  PBC CONFLICT: cell ({a_len:.1f}x{b_len:.1f}) may be too small "
                       f"for {mol.get_chemical_formula()} (span={d_mol:.1f} A).")

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
        mode     = 'w', # Overwrite logs for clarity
    )
    logger.info(f"--- Starting AutoFlow-SRXN Discovery Study ({config_path}) ---")

    mol_file   = paths.get('adsorbate')
    inh_file   = paths.get('inhibitor')
    out_prefix = paths.get('output_prefix', 'cands_out')

    mol  = None
    if mol_file:
        if os.path.exists(mol_file):
            mol = read(mol_file)
        else:
            logger.warning(f"Adsorbate file not found: {mol_file}")
    
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

        # [Optional Reconstruction]
        recon_cfg = sub_gen_cfg.get('reconstruction', {})
        if recon_cfg.get('enabled', False):
            from autoflow_srxn.surface_utils import apply_surface_reconstruction
            strategy = recon_cfg.get('strategy', 'auto')
            side = recon_cfg.get('side', 'top')
            logger.info(f"Applying surface reconstruction strategy: {strategy} on {side}...")
            
            # Extract hyperparameters for fine-tuning
            recon_params = {
                'buckling_dist': recon_cfg.get('buckling_dist', 0.4),
                'dimer_dist':    recon_cfg.get('dimer_dist', 0.6),
                'amplitude':     recon_cfg.get('amplitude', 0.1)
            }
            
            slab = apply_surface_reconstruction(slab, strategy=strategy, side=side, verbose=True, **recon_params)
            write_standardized_vasp('reconstructed_substrate.vasp', slab)
            logger.info(f"Saved reconstructed substrate to 'reconstructed_substrate.vasp'.")
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
    if inh_cfg.get('enabled', False):
        if not inh_file:
            logger.info("STAGE 1: Inhibitor discovery skipped (inhibitor path is null).")
        elif not os.path.exists(inh_file):
            logger.warning(f"STAGE 1: Inhibitor discovery skipped (file not found: {inh_file}).")
        else:
            logger.info(f"STAGE 1: Inhibitor discovery ({inh_file})")
            inh_mol    = read(inh_file)
            inh_center = inh_cfg.get('inhibitor_center', 'O')
            inh_cands  = execute_discovery_stage(
                slab, inh_mol, config, f"{out_prefix}_inh",
                logger, True, tag=2, center_target=inh_center,
            )
            limit      = inh_cfg.get('branching_limit', 5)
            if inh_cands:
                base_slabs = inh_cands[:limit]
                logger.info(f"  Branching into {len(base_slabs)} inhibited geometries for Stage 2.")
            else:
                logger.info("  No inhibitor candidates found. Proceeding with clean slab.")

    # ── Stage 2: Main precursor discovery ─────────────────────────────────────
    if not mol:
        logger.info("STAGE 2: Main precursor discovery skipped (adsorbate is null or missing).")
        all_final_results = []
    else:
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

    # ── Stage 3: Short Relaxation (Verification) ──────────────────────────────
    relax_cfg = rs_cfg.get('candidate_relaxation', {})
    if all_final_results and relax_cfg.get('enabled', False):
        from autoflow_srxn.potentials import SimulationEngine
        n_steps = relax_cfg.get('steps', 10)
        fmax_val = relax_cfg.get('fmax', 0.01)
        v_flag = relax_cfg.get('verbose', False)
        sel_idx = relax_cfg.get('selected_indices', None)
        
        # Support dynamic evaluation of list expressions or numpy arrays
        if isinstance(sel_idx, str):
            try:
                import numpy as np
                # Provide a safe namespace with common utilities
                allowed_names = {"range": range, "list": list, "np": np, "numpy": np, "abs": abs}
                # Evaluate string expression safely
                sel_idx = eval(sel_idx, {"__builtins__": {}}, allowed_names)
                # Convert to list if it's a numpy array or range object
                if hasattr(sel_idx, 'tolist'):
                    sel_idx = sel_idx.tolist()
                elif not isinstance(sel_idx, list):
                    sel_idx = list(sel_idx)
                logger.info(f"  [Relaxation] Evaluated selected_indices: {len(sel_idx)} sites selected.")
            except Exception as e:
                logger.error(f"  [Relaxation] Failed to evaluate 'selected_indices' expression '{sel_idx}': {e}")
                sel_idx = None

        n_total = len(all_final_results)
        n_target = len(sel_idx) if sel_idx is not None else n_total
        logger.info(f"STAGE 3: Performing short relaxation ({n_steps} steps) on {n_target}/{n_total} candidates...")
        
        # We rely on the engine block in config.yaml. SimulationEngine handles defaults if missing.
        if 'engine' in config:
            backend = config['engine'].get('potential', {}).get('backend', 'mace')
            logger.info(f"  [Relaxation] Using configured engine backend: {backend} (fmax={fmax_val})")
        else:
            logger.warning("  [Relaxation] 'engine' block missing in config. Falling back to internal defaults.")
            
        engine = SimulationEngine(config)
        calc = engine.get_calculator() # Load engine once before the table starts
        
        relaxed_cands = []
        summary_data = [] # To store results for clean table output at the end

        for i, atoms in enumerate(all_final_results):
            # Skip if not in selected_indices
            if sel_idx is not None and i not in sel_idx:
                continue

            atoms_relaxed = atoms.copy()
            atoms_relaxed.info = atoms.info.copy()
            
            try:
                # Attach calculator to get initial energy
                atoms_relaxed.calc = calc
                e_init = atoms_relaxed.get_potential_energy()
                
                # Perform relaxation using parameters from config
                engine.relax(atoms_relaxed, steps=n_steps, verbose=v_flag, fmax=fmax_val)
                
                e_final = atoms_relaxed.get_potential_energy()
                delta_e = e_final - e_init
                mech = atoms.info.get('mechanism', 'unknown')
                
                summary_data.append({
                    'id': i, 'mech': mech, 'e_init': e_init, 'e_final': e_final, 'delta': delta_e
                })
                
                atoms_relaxed.info['e_initial'] = e_init
                atoms_relaxed.info['e_final'] = e_final
                atoms_relaxed.info['relaxation'] = f'short_relax_{n_steps}_steps'
            except Exception as e:
                logger.warning(f"Candidate {i} relaxation failed: {e}")
                atoms_relaxed.info['relaxation'] = 'failed'
            
            relaxed_cands.append(atoms_relaxed)

        # --- Print Visual Summary Table ---
        if summary_data:
            # Find best (lowest e_final) for each mechanism group
            best_by_mech = {}
            for row in summary_data:
                m = row['mech']
                if m not in best_by_mech or row['e_final'] < best_by_mech[m]['e_final']:
                    best_by_mech[m] = row

            best_ids = {res['id'] for res in best_by_mech.values()}

            logger.info("\n" + "="*95)
            logger.info(f"{'ID':<4} | {'Mechanism':<15} | {'E_initial (eV)':<15} | {'E_final (eV)':<15} | {'Delta (eV)':<10} | {'Note'}")
            logger.info("-" * 95)
            for row in summary_data:
                marker = "* (Best Pose)" if row['id'] in best_ids else ""
                logger.info(f"{row['id']:<4} | {row['mech'][:15]:<15} | {row['e_init']:15.4f} | {row['e_final']:15.4f} | {row['delta']:10.4f} | {marker}")
            logger.info("="*95 + "\n")
            
        if relaxed_cands:
            write(f'{out_prefix}_relaxed_poses.extxyz', relaxed_cands)
            logger.info(f"Saved relaxed candidates to '{out_prefix}_relaxed_poses.extxyz'.")

    logger.info(f"--- Study Complete. Total unique candidates: {len(all_final_results)} ---")


if __name__ == '__main__':
    default_config = os.path.join(os.path.dirname(__file__), 'config.yaml')
    c_path = sys.argv[1] if len(sys.argv) > 1 else default_config
    run_generic_adsorption_study(c_path)
