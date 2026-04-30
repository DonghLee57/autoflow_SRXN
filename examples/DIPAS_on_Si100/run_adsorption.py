import os
import sys

import numpy as np
import yaml
from ase.io import read, write

from autoflow_srxn.ads_workflow_mgr import AdsorptionWorkflowManager
from autoflow_srxn.chemisorption_builder import build_chemisorption_structures
from autoflow_srxn.logger_utils import log_energy_comparison, log_results_table, log_stage_title, setup_logger
from autoflow_srxn.surface_utils import (
    create_slab_from_bulk,
    passivate_surface_coverage_general,
    write_standardized_vasp,
)


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def execute_discovery_stage(slab, mol, config, out_prefix, logger, verbose=True, tag=2, center_target="Si"):
    """Geometry-only candidate generation for physisorption and chemisorption."""
    rs_cfg = config.get("reaction_search", {})
    mechs_cfg = rs_cfg.get("mechanisms", {})
    physi_cfg = mechs_cfg.get("physisorption", {})
    chem_cfg = mechs_cfg.get("chemisorption", {})
    symprec = rs_cfg.get("candidate_filter", {}).get("symprec", 0.2)

    run_phy = physi_cfg.get("enabled", True)
    run_chem = chem_cfg.get("enabled", True)

    mgr = AdsorptionWorkflowManager(slab, config=config, symprec=symprec, verbose=verbose)

    # Lateral span diagnostic
    d_mol = mgr.calculate_molecule_lateral_extent(mol)
    a_len = np.linalg.norm(slab.cell[0])
    b_len = np.linalg.norm(slab.cell[1])
    logger.info(
        f"  DIAGNOSTIC: {mol.get_chemical_formula()} span = {d_mol:.2f} A | Substrate = {a_len:.2f} x {b_len:.2f} A"
    )
    if a_len < d_mol + 3.0 or b_len < d_mol + 3.0:
        logger.warning(
            f"  PBC CONFLICT: cell ({a_len:.1f}x{b_len:.1f}) may be too small "
            f"for {mol.get_chemical_formula()} (span={d_mol:.1f} A)."
        )

    all_cands = []

    if run_phy:
        logger.info(f"  Physisorption search for {mol.get_chemical_formula()}...")
        phy_cands = mgr.generate_physisorption_candidates(
            mol,
            height=physi_cfg.get("placement_height", 3.5),
            n_rot=physi_cfg.get("rot_steps", 8),
            rot_center=physi_cfg.get("center", "com"),
            config=config,
            tag=tag,
        )
        for c in phy_cands:
            c.info["mechanism"] = "physisorption"
        all_cands.extend(phy_cands)

    if run_chem:
        logger.info(f"  Chemisorption search for {mol.get_chemical_formula()} (center={center_target})...")
        chem_cands = build_chemisorption_structures(
            molecule=mol,
            center_target=center_target,
            surface=slab,
            rot_steps=chem_cfg.get("rot_steps", 8),
            config=config,
            verbose=verbose,
            tag=tag,
        )
        for c in chem_cands:
            c.info["mechanism"] = "chemisorption"
        all_cands.extend(chem_cands)

        byproducts = [c.info["isolated_byproduct"] for c in chem_cands if "isolated_byproduct" in c.info]
        if byproducts:
            write(f"{out_prefix}_byproducts.extxyz", byproducts)

    if all_cands:
        write(f"{out_prefix}_all_poses.extxyz", all_cands)

    return all_cands


def run_generic_adsorption_study(config_path="config.yaml"):
    config = load_config(config_path)
    paths = config["paths"]
    sp_cfg = config.get("surface_prep", {})
    rs_cfg = config.get("reaction_search", {})
    mechs_cfg = rs_cfg.get("mechanisms", {})
    inh_cfg = mechs_cfg.get("inhibition", {})

    out_prefix = paths.get("output_prefix")
    log_file = f"{out_prefix}_workflow.log" if out_prefix else "workflow.log"

    logger = setup_logger(
        log_path=log_file,
        verbose=True,
        mode="w",  # Overwrite logs for clarity
    )
    logger.info(f"--- Starting AutoFlow-SRXN Discovery Study ({config_path}) ---")

    mol_file = paths.get("adsorbate")
    inh_file = paths.get("inhibitor")
    out_prefix = paths.get("output_prefix", "cands_out")

    mol = None
    if mol_file:
        if os.path.exists(mol_file):
            mol = read(mol_file)
        else:
            logger.warning(f"Adsorbate file not found: {mol_file}")

    slab = None

    # ── Stage 0: Substrate generation & passivation ────────────────────────────
    sub_gen_cfg = sp_cfg.get("slab_generation", {})
    if sub_gen_cfg.get("enabled", False):
        log_stage_title(logger, "STAGE 0", "Generating substrate slab...")
        bulk_atoms = read(paths["substrate_bulk"])
        slab = create_slab_from_bulk(
            bulk_atoms=bulk_atoms,
            miller_indices=sub_gen_cfg.get("miller", [1, 0, 0]),
            thickness=sub_gen_cfg.get("thickness_ang", 10.0),
            vacuum=sub_gen_cfg.get("vacuum_ang", 10.0),
            target_area=sub_gen_cfg.get("target_area_ang2"),
            supercell_matrix=sub_gen_cfg.get("supercell_matrix"),
            top_termination=sub_gen_cfg.get("top_termination"),
            bottom_termination=sub_gen_cfg.get("bottom_termination"),
            verbose=True,
        )
        write_standardized_vasp("generated_substrate.vasp", slab)
        logger.info("Saved generated raw substrate to 'generated_substrate.vasp'.")

        # [Optional Reconstruction]
        recon_cfg = sub_gen_cfg.get("reconstruction", {})
        if recon_cfg.get("enabled", False):
            from autoflow_srxn.surface_utils import apply_surface_reconstruction

            strategy = recon_cfg.get("strategy", "auto")
            side = recon_cfg.get("side", "top")
            logger.info(f"Applying surface reconstruction strategy: {strategy} on {side}...")

            # Extract hyperparameters for fine-tuning
            recon_params = {
                "buckling_dist": recon_cfg.get("buckling_dist", 0.4),
                "dimer_dist": recon_cfg.get("dimer_dist", 0.6),
                "amplitude": recon_cfg.get("amplitude", 0.1),
            }

            slab = apply_surface_reconstruction(slab, strategy=strategy, side=side, verbose=True, **recon_params)
            write_standardized_vasp("reconstructed_substrate.vasp", slab)
            logger.info("Saved reconstructed substrate to 'reconstructed_substrate.vasp'.")
    else:
        slab_path = paths.get("substrate_slab")
        if not slab_path:
            raise ValueError(
                "Either surface_prep.slab_generation.enabled must be true "
                "or paths.substrate_slab must point to a pre-built slab."
            )
        slab = read(slab_path)

    pass_cfg = sp_cfg.get("passivation", {})
    if pass_cfg.get("enabled", False):
        logger.info("Applying geometric passivation...")
        ideal_coord = sp_cfg.get("surface_analysis", {}).get("ideal_coordination", {})
        side = pass_cfg.get("side", "bottom")
        sides = [side] if side != "both" else ["top", "bottom"]
        for s in sides:
            slab = passivate_surface_coverage_general(
                slab,
                h_coverage=pass_cfg.get("coverage", 1.0),
                valence_map=ideal_coord,
                element=pass_cfg.get("element", "H"),
                side=s,
                verbose=True,
            )
        write_standardized_vasp("passivated.vasp", slab)
        logger.info("Saved passivated substrate to 'passivated.vasp'.")

    slab_relax_cfg = sp_cfg.get("slab_relaxation", {})
    if slab_relax_cfg.get("enabled", False):
        from autoflow_srxn.potentials import SimulationEngine

        log_stage_title(logger, "STAGE 0.5", "Performing slab relaxation...")
        try:
            engine = SimulationEngine(config)
            n_steps = slab_relax_cfg.get("steps", 200)
            fmax_val = slab_relax_cfg.get("fmax", 0.05)
            frozen_z = slab_relax_cfg.get("frozen_z_ang")

            # Calculate initial energy
            slab.calc = engine.get_calculator()
            e_init = slab.get_potential_energy()

            engine.relax(slab, fmax=fmax_val, steps=n_steps, frozen_z_ang=frozen_z, verbose=True)

            e_final = slab.get_potential_energy()
            log_energy_comparison(logger, "Slab Relax", e_init, e_final)

            write_standardized_vasp("relaxed_slab.vasp", slab)
            logger.info("Saved relaxed slab to 'relaxed_slab.vasp'.")
        except Exception as e:
            logger.error(f"Slab relaxation failed: {e}")

    # ── Stage 1: Inhibitor pre-treatment (optional branching) ─────────────────
    base_slabs = [slab]
    if inh_cfg.get("enabled", False):
        if not inh_file:
            logger.info("STAGE 1: Inhibitor discovery skipped (inhibitor path is null).")
        elif not os.path.exists(inh_file):
            logger.warning(f"STAGE 1: Inhibitor discovery skipped (file not found: {inh_file}).")
        else:
            log_stage_title(logger, "STAGE 1", f"Inhibitor discovery ({inh_file})")
            inh_mol = read(inh_file)
            inh_center = inh_cfg.get("inhibitor_center", "O")
            inh_cands = execute_discovery_stage(
                slab,
                inh_mol,
                config,
                f"{out_prefix}_inh",
                logger,
                True,
                tag=2,
                center_target=inh_center,
            )
            limit = inh_cfg.get("branching_limit", 5)
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
        log_stage_title(logger, "STAGE 2", f"Main precursor discovery ({mol_file})")
        mol_center = mechs_cfg.get("chemisorption", {}).get("precursor_center", "Si")
        all_final_results = []
        for i, s in enumerate(base_slabs):
            suffix = f"_inh{i}" if len(base_slabs) > 1 else ""
            results = execute_discovery_stage(
                s,
                mol,
                config,
                f"{out_prefix}{suffix}",
                logger,
                True,
                tag=3,
                center_target=mol_center,
            )
            all_final_results.extend(results)

    # ── Stage 3: Verification (Relaxation & MD Sampling) ──────────────────────
    verify_cfg = rs_cfg.get("verification", {})
    run_relax = verify_cfg.get("relaxation", {}).get("enabled", False)
    run_md = verify_cfg.get("md_sampling", {}).get("enabled", False)

    if all_final_results and (run_relax or run_md):
        from autoflow_srxn.potentials import SimulationEngine

        sel_idx = verify_cfg.get("selected_indices", None)

        # Support dynamic evaluation of list expressions or numpy arrays
        if isinstance(sel_idx, str):
            try:
                import numpy as np

                # Provide a safe namespace with common utilities
                allowed_names = {"range": range, "list": list, "np": np, "numpy": np, "abs": abs}
                # Evaluate string expression safely
                sel_idx = eval(sel_idx, {"__builtins__": {}}, allowed_names)
                # Convert to list if it's a numpy array or range object
                if hasattr(sel_idx, "tolist"):
                    sel_idx = sel_idx.tolist()
                elif not isinstance(sel_idx, list):
                    sel_idx = list(sel_idx)
                logger.info(f"  [Verification] Evaluated selected_indices: {len(sel_idx)} sites selected.")
            except Exception as e:
                logger.error(f"  [Verification] Failed to evaluate 'selected_indices' expression '{sel_idx}': {e}")
                sel_idx = None

        n_total = len(all_final_results)
        n_target = len(sel_idx) if sel_idx is not None else n_total
        log_stage_title(
            logger,
            "STAGE 3",
            f"Performing verification (Relax={run_relax}, MD={run_md}) on {n_target}/{n_total} sites.",
        )

        engine = SimulationEngine(config)
        calc = engine.get_calculator()

        processed_cands = []
        summary_data = []

        for i, atoms in enumerate(all_final_results):
            if sel_idx is not None and i not in sel_idx:
                continue

            atoms_proc = atoms.copy()
            atoms_proc.info = atoms.info.copy()
            atoms_proc.calc = calc

            try:
                e_init = atoms_proc.get_potential_energy()

                # --- 1. Relaxation ---
                if run_relax:
                    r_cfg = verify_cfg.get("relaxation", {})
                    engine.relax(
                        atoms_proc,
                        steps=r_cfg.get("steps", 50),
                        fmax=r_cfg.get("fmax", 0.05),
                        verbose=r_cfg.get("verbose", False),
                    )

                # --- 2. MD Sampling ---
                if run_md:
                    m_cfg = verify_cfg.get("md_sampling", {})
                    engine.run_md(
                        atoms_proc,
                        temp_K=m_cfg.get("temperature_K", 300),
                        md_steps=m_cfg.get("md_steps", 1000),
                        timestep_fs=m_cfg.get("timestep_fs", 1.0),
                        damping=m_cfg.get("damping", 100.0),
                        frozen_z_ang=m_cfg.get("frozen_z_ang"),
                    )

                e_final = atoms_proc.get_potential_energy()
                delta_e = e_final - e_init
                mech = atoms.info.get("mechanism", "unknown")

                summary_data.append({"id": i, "mech": mech, "e_init": e_init, "e_final": e_final, "delta": delta_e})

                atoms_proc.info["e_initial"] = e_init
                atoms_proc.info["e_final"] = e_final
                atoms_proc.info["verification"] = f"relax={run_relax}_md={run_md}"
            except Exception as e:
                logger.warning(f"Candidate {i} verification failed: {e}")
                atoms_proc.info["verification"] = "failed"

            processed_cands.append(atoms_proc)

        # --- Print Visual Summary Table ---
        log_results_table(logger, summary_data, title="Candidate Verification Summary")

        if processed_cands:
            write(f"{out_prefix}_verified_poses.extxyz", processed_cands)
            logger.info(f"Saved verified candidates to '{out_prefix}_verified_poses.extxyz'.")

    logger.info(f"--- Study Complete. Total unique candidates: {len(all_final_results)} ---")


if __name__ == "__main__":
    default_config = os.path.join(os.path.dirname(__file__), "config.yaml")
    c_path = sys.argv[1] if len(sys.argv) > 1 else default_config
    run_generic_adsorption_study(c_path)
