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


def calculate_gas_energy(mol, config, logger):
    """Calculates the potential energy of a molecule in vacuum after relaxation."""
    from autoflow_srxn.potentials import SimulationEngine

    mol_copy = mol.copy()
    mol_copy.center(vacuum=10.0)
    engine = SimulationEngine(config)
    try:
        mol_copy.calc = engine.get_calculator()
        engine.relax(mol_copy, steps=100, fmax=0.02, verbose=False)
        e_gas = mol_copy.get_potential_energy()
        logger.info(f"  [Gas Phase] {mol.get_chemical_formula()} optimized energy: {e_gas:.4f} eV")
        return e_gas
    except Exception as e:
        logger.error(f"  [Gas Phase] Failed to calculate energy for {mol.get_chemical_formula()}: {e}")
        return 0.0


def execute_verification_stage(candidates, config, logger, out_prefix, tag=3, e_gas=0.0, e_base=0.0):
    """Performs Relaxation, Equilibration (MD), and Optional Post-Relax on candidates.
    Also calculates and logs the adsorption energy (E_ads).
    """
    rs_cfg = config.get("reaction_search", {})
    verify_cfg = rs_cfg.get("verification", {})
    run_relax = verify_cfg.get("relaxation", {}).get("enabled", False)
    run_equil = verify_cfg.get("equilibration", {}).get("enabled", False)

    if not candidates or not (run_relax or run_equil):
        return candidates

    from autoflow_srxn.potentials import SimulationEngine

    sel_idx = verify_cfg.get("selected_indices", None)
    if isinstance(sel_idx, str):
        try:
            import numpy as np

            allowed_names = {"range": range, "list": list, "np": np, "numpy": np, "abs": abs}
            sel_idx = eval(sel_idx, {"__builtins__": {}}, allowed_names)
            if hasattr(sel_idx, "tolist"):
                sel_idx = sel_idx.tolist()
            elif not isinstance(sel_idx, list):
                sel_idx = list(sel_idx)
        except Exception as e:
            logger.error(f"  [Verification] Failed to evaluate 'selected_indices' expression: {e}")
            sel_idx = None

    n_total = len(candidates)
    n_target = len(sel_idx) if sel_idx is not None else n_total
    log_stage_title(
        logger,
        "VERIFICATION",
        f"Processing {n_target}/{n_total} sites (Relax={run_relax}, Equil={run_equil})",
    )

    engine = SimulationEngine(config)
    calc = engine.get_calculator()

    processed_cands = []
    summary_data = []

    for i, atoms in enumerate(candidates):
        if sel_idx is not None and i not in sel_idx:
            continue

        atoms_proc = atoms.copy()
        atoms_proc.info = atoms.info.copy()
        atoms_proc.calc = calc

        try:
            e_init = atoms_proc.get_potential_energy()

            # --- 1. Initial Relaxation ---
            if run_relax:
                r_cfg = verify_cfg.get("relaxation", {})
                engine.relax(
                    atoms_proc,
                    steps=r_cfg.get("steps", 50),
                    fmax=r_cfg.get("fmax", 0.05),
                    verbose=r_cfg.get("verbose", False),
                )

            # --- 2. Thermal Equilibration (MD) ---
            if run_equil:
                e_cfg = verify_cfg.get("equilibration", {})
                engine.run_md(
                    atoms_proc,
                    temp_K=e_cfg.get("temperature_K", 300),
                    md_steps=e_cfg.get("md_steps", 1000),
                    timestep_fs=e_cfg.get("timestep_fs", 1.0),
                    damping=e_cfg.get("damping", 100.0),
                    frozen_z_ang=e_cfg.get("frozen_z_ang"),
                )

                # --- 3. Post-Equilibration Relaxation ---
                if e_cfg.get("post_relax", True):
                    engine.relax(atoms_proc, steps=50, fmax=0.05, verbose=False)

            e_final = atoms_proc.get_potential_energy()
            # E_ads = E_total - (E_gas + E_base)
            e_ads = e_final - (e_gas + e_base)
            mech = atoms.info.get("mechanism", "unknown")

            summary_data.append(
                {"id": i, "mech": mech, "e_init": e_init, "e_final": e_final, "delta": e_final - e_init, "e_ads": e_ads}
            )

            atoms_proc.info["e_initial"] = e_init
            atoms_proc.info["e_final"] = e_final
            atoms_proc.info["e_ads"] = e_ads
            atoms_proc.info["verification"] = f"relax={run_relax}_equil={run_equil}"
        except Exception as e:
            logger.warning(f"  [Verification] Candidate {i} failed: {e}")
            atoms_proc.info["verification"] = "failed"

        processed_cands.append(atoms_proc)

    log_results_table(logger, summary_data, title=f"Verification Summary (tag={tag})")

    if processed_cands:
        write(f"{out_prefix}_verified_poses.extxyz", processed_cands)

    return processed_cands


def execute_discovery_stage(slab, mol, config, out_prefix, logger, tag=2, center_target="Si", e_gas=0.0, e_base=0.0):
    """Orchestrates candidate generation and subsequent verification."""
    rs_cfg = config.get("reaction_search", {})
    mechs_cfg = rs_cfg.get("mechanisms", {})
    physi_cfg = mechs_cfg.get("physisorption", {})
    chem_cfg = mechs_cfg.get("chemisorption", {})
    symprec = rs_cfg.get("candidate_filter", {}).get("symprec", 0.2)

    run_phy = physi_cfg.get("enabled", True)
    run_chem = chem_cfg.get("enabled", True)

    mgr = AdsorptionWorkflowManager(slab, config=config, symprec=symprec, verbose=False)
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
            verbose=False,
            tag=tag,
        )
        for c in chem_cands:
            c.info["mechanism"] = "chemisorption"
        all_cands.extend(chem_cands)

    if all_cands:
        write(f"{out_prefix}_all_poses.extxyz", all_cands)

    # Automated Verification (Relax + Equil + Post-Relax) using pre-calculated energies
    verified_cands = execute_verification_stage(
        all_cands, config, logger, out_prefix, tag=tag, e_gas=e_gas, e_base=e_base
    )
    return verified_cands


def execute_discovery_workflow(config, logger, gas_energy_map=None, slab_base_energy=0.0):
    """Core logic for a single discovery run (one precursor, one inhibitor)."""
    paths = config["paths"]
    sp_cfg = config.get("surface_prep", {})
    rs_cfg = config.get("reaction_search", {})
    mechs_cfg = rs_cfg.get("mechanisms", {})
    inh_cfg = mechs_cfg.get("inhibition", {})

    mol_file = paths.get("adsorbate")
    inh_file = paths.get("inhibitor")
    out_prefix = paths.get("output_prefix", "results")

    mol = read(mol_file) if mol_file and os.path.exists(mol_file) else None
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
        recon_cfg = sub_gen_cfg.get("reconstruction", {})
        if recon_cfg.get("enabled", False):
            from autoflow_srxn.surface_utils import apply_surface_reconstruction

            slab = apply_surface_reconstruction(
                slab,
                strategy=recon_cfg.get("strategy", "auto"),
                side=recon_cfg.get("side", "top"),
                buckling_dist=recon_cfg.get("buckling_dist", 0.4),
                dimer_dist=recon_cfg.get("dimer_dist", 0.6),
                amplitude=recon_cfg.get("amplitude", 0.1),
            )
    else:
        slab = read(paths["substrate_slab"])

    pass_cfg = sp_cfg.get("passivation", {})
    if pass_cfg.get("enabled", False):
        ideal_coord = sp_cfg.get("surface_analysis", {}).get("ideal_coordination", {})
        slab = passivate_surface_coverage_general(
            slab,
            h_coverage=pass_cfg.get("coverage", 1.0),
            valence_map=ideal_coord,
            element=pass_cfg.get("element", "H"),
            side=pass_cfg.get("side", "bottom"),
        )

    slab_relax_cfg = sp_cfg.get("slab_relaxation", {})
    if slab_relax_cfg.get("enabled", False):
        from autoflow_srxn.potentials import SimulationEngine

        log_stage_title(logger, "STAGE 0.5", "Performing slab relaxation...")
        engine = SimulationEngine(config)
        slab.calc = engine.get_calculator()
        e_init = slab.get_potential_energy()
        engine.relax(
            slab,
            fmax=slab_relax_cfg.get("fmax", 0.05),
            steps=slab_relax_cfg.get("steps", 200),
            frozen_z_ang=slab_relax_cfg.get("frozen_z_ang"),
        )
        slab_base_energy = slab.get_potential_energy()  # Update base energy after relax
        log_energy_comparison(logger, "Slab Relax", e_init, slab_base_energy)

    # ── Stage 1: Inhibitor pre-treatment ──────────────────────────────────────
    base_slabs = [slab]
    if inh_cfg.get("enabled", False) and inh_file and os.path.exists(inh_file):
        log_stage_title(logger, "STAGE 1", f"Inhibitor Discovery ({os.path.basename(inh_file)})")
        inh_mol = read(inh_file)
        # Use cached gas energy if available
        e_gas_inh = (
            gas_energy_map.get(inh_file, 0.0) if gas_energy_map else calculate_gas_energy(inh_mol, config, logger)
        )

        inh_cands = execute_discovery_stage(
            slab,
            inh_mol,
            config,
            f"{out_prefix}_inh",
            logger,
            tag=2,
            center_target=inh_cfg.get("inhibitor_center", "O"),
            e_gas=e_gas_inh,
            e_base=slab_base_energy,
        )
        if inh_cands:
            if any("e_final" in c.info for c in inh_cands):
                inh_cands.sort(key=lambda x: x.info.get("e_final", 1e10))
            limit = inh_cfg.get("branching_limit", 3)
            base_slabs = inh_cands[:limit]
            logger.info(f"  Selected top {len(base_slabs)} inhibited surfaces for Stage 2.")

    # ── Stage 2: Main precursor discovery ─────────────────────────────────────
    if mol:
        log_stage_title(logger, "STAGE 2", f"Main Precursor Discovery ({os.path.basename(mol_file)})")
        mol_center = mechs_cfg.get("chemisorption", {}).get("precursor_center", "Si")
        # Use cached gas energy if available
        e_gas_mol = gas_energy_map.get(mol_file, 0.0) if gas_energy_map else calculate_gas_energy(mol, config, logger)

        all_final_results = []
        for i, s in enumerate(base_slabs):
            try:
                # For Stage 2, base is the inhibited surface
                e_base_stage2 = s.info.get("e_final", s.get_potential_energy())
            except Exception:
                e_base_stage2 = slab_base_energy  # fallback

            suffix = f"_inh{i}" if len(base_slabs) > 1 else ""
            results = execute_discovery_stage(
                s,
                mol,
                config,
                f"{out_prefix}{suffix}",
                logger,
                tag=3,
                center_target=mol_center,
                e_gas=e_gas_mol,
                e_base=e_base_stage2,
            )
            all_final_results.extend(results)

        if all_final_results:
            write(f"{out_prefix}_final_verified.extxyz", all_final_results)


def run_generic_adsorption_study(config_path="config.yaml"):
    import copy
    import glob

    config = load_config(config_path)
    paths = config.get("paths", {})

    def get_structure_files(path_or_dir):
        if not path_or_dir:
            return [None]
        if os.path.isdir(path_or_dir):
            files = []
            for ext in ["*.vasp", "*.xyz", "*.extxyz"]:
                files.extend(glob.glob(os.path.join(path_or_dir, ext)))
            return sorted(files)
        if isinstance(path_or_dir, list):
            return path_or_dir
        return [path_or_dir]

    # 1. Resolve adsorbates
    ads_input = paths.get("adsorbates_dir") or paths.get("adsorbate")
    adsorbates = get_structure_files(ads_input)

    # 2. Resolve inhibitors
    inh_input = paths.get("inhibitors_dir") or paths.get("inhibitor")
    inhibitors = get_structure_files(inh_input)

    # NEW: Automatically include a 'no-inhibitor' baseline if requested
    if paths.get("include_no_inhibitor", False):
        if None not in inhibitors:
            inhibitors = [None] + inhibitors
    elif not inhibitors:
        inhibitors = [None]

    global_prefix = paths.get("output_prefix", "discovery")

    # ── E_ads Optimization: Pre-calculate Reference Energies ───────────────────
    # We calculate Gas Energies once for all unique molecules to avoid redundancy.
    unique_mols = list(set([f for f in adsorbates + inhibitors if f and os.path.exists(f)]))
    gas_energy_map = {}
    if unique_mols:
        # Use a temporary logger for pre-calculation
        tmp_logger = setup_logger(log_path=os.path.join(global_prefix, "ref_energies.log"), mode="w")
        log_stage_title(tmp_logger, "PRE-CALC", "Calculating Reference Gas Energies...")
        for m_path in unique_mols:
            mol_atoms = read(m_path)
            gas_energy_map[m_path] = calculate_gas_energy(mol_atoms, config, tmp_logger)

    # Pre-calculate Slab Base Energy if slab is already provided
    slab_base_energy = 0.0
    if paths.get("substrate_slab") and os.path.exists(paths["substrate_slab"]):
        from autoflow_srxn.potentials import SimulationEngine

        tmp_slab = read(paths["substrate_slab"])
        engine = SimulationEngine(config)
        tmp_slab.calc = engine.get_calculator()
        slab_base_energy = tmp_slab.get_potential_energy()

    # ── Main Batch Loop ────────────────────────────────────────────────────────
    for inh_path in inhibitors:
        for ads_path in adsorbates:
            if not ads_path:
                continue

            inh_name = os.path.splitext(os.path.basename(inh_path))[0] if inh_path else ""
            ads_name = os.path.splitext(os.path.basename(ads_path))[0] if ads_path else "none"

            # Cleaner naming: 'precursor' instead of 'none_on_precursor'
            run_name = f"{inh_name}_on_{ads_name}" if inh_name else ads_name
            run_dir = os.path.join(global_prefix, run_name)
            os.makedirs(run_dir, exist_ok=True)

            log_file = os.path.join(run_dir, "workflow.log")
            logger = setup_logger(log_path=log_file, verbose=True, mode="w")

            log_stage_title(logger, "BATCH RUN", f"Pair: {inh_name} + {ads_name}")
            logger.info(f"  Inhibitor: {inh_path if inh_path else 'None'}")
            logger.info(f"  Adsorbate: {ads_path}")

            run_config = copy.deepcopy(config)
            run_config["paths"]["adsorbate"] = ads_path
            run_config["paths"]["inhibitor"] = inh_path
            run_config["paths"]["output_prefix"] = os.path.join(run_dir, "results")

            try:
                execute_discovery_workflow(
                    run_config, logger, gas_energy_map=gas_energy_map, slab_base_energy=slab_base_energy
                )
            except Exception as e:
                logger.error(f"Discovery workflow failed for {run_name}: {e}")

    print(f"\n--- Batch discovery complete. Results saved in '{global_prefix}/' ---")


if __name__ == "__main__":
    default_config = os.path.join(os.path.dirname(__file__), "config.yaml")
    c_path = sys.argv[1] if len(sys.argv) > 1 else default_config
    run_generic_adsorption_study(c_path)
