import os
import sys
import copy
import yaml
import numpy as np
from ase.io import read, write
from ase import Atoms

from autoflow_srxn.surface.ads_workflow_mgr import AdsorptionWorkflowManager
from autoflow_srxn.surface.chemisorption_builder import build_chemisorption_structures
from autoflow_srxn.utils.logger_utils import log_energy_comparison, log_results_table, log_stage_title, setup_logger
from autoflow_srxn.surface.surface_utils import (
    create_slab_from_bulk,
    passivate_surface_coverage_general,
    standardize_vasp_atoms,
)


# =============================================================================
# Config helpers — support new unified format AND legacy split format
# =============================================================================

def load_config(config_path):
    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Resolve relative paths in 'paths' section
    config_dir = os.path.dirname(os.path.abspath(config_path))
    paths = config.get("paths", {})
    for key in ["precursor", "inhibitor", "substrate_bulk", "input_structure"]:
        val = paths.get(key)
        if val and isinstance(val, str) and not os.path.isabs(val):
            if not os.path.exists(val):
                alt = os.path.join(config_dir, val)
                if os.path.exists(alt):
                    paths[key] = alt
    return config


def _resolve_workflow(config):
    """Return pipeline enable-flags.

    New format  (config.workflow.*):
        workflow:
          slab_relax:       false
          candidate_relax:  true
          md_equilibrate:   false
          post_md_relax:    true

    Legacy format (still accepted):
        surface_prep.slab_relaxation.enabled
        verification.relaxation.enabled
        verification.equilibration.enabled / post_relax
    """
    wf = config.get("workflow", {})
    # Legacy fall-backs
    legacy_sr  = config.get("surface_prep", {}).get("slab_relaxation", {}).get("enabled", False)
    legacy_cr  = config.get("verification", {}).get("relaxation", {}).get("enabled", False)
    legacy_md  = config.get("verification", {}).get("equilibration", {}).get("enabled", False)
    legacy_pmd = config.get("verification", {}).get("equilibration", {}).get("post_relax", True)
    return {
        "slab_relax":      wf.get("slab_relax",      legacy_sr),
        "candidate_relax": wf.get("candidate_relax",  legacy_cr),
        "md_equilibrate":  wf.get("md_equilibrate",   legacy_md),
        "post_md_relax":   wf.get("post_md_relax",    legacy_pmd),
    }


def _resolve_relax_params(config):
    """Return relaxation hyper-parameters.

    New format  (config.relaxation.*):
        relaxation:
          fmax:         0.05
          steps:        100
          frozen_z_ang: 5.5

    Legacy: config.verification.relaxation / config.engine.relaxation
    """
    new  = config.get("relaxation", {})
    lv   = config.get("verification", {}).get("relaxation", {})
    le   = config.get("engine", {}).get("relaxation", {})
    return {
        "fmax":         new.get("fmax",         lv.get("fmax",  le.get("fmax",  0.05))),
        "steps":        new.get("steps",        lv.get("steps", le.get("steps", 100))),
        "frozen_z_ang": new.get("frozen_z_ang", lv.get("frozen_z_ang",
                                                le.get("frozen_z_ang", None))),
    }


def _resolve_equil_params(config):
    """Return MD equilibration parameters.

    New format  (config.equilibration.*):
        equilibration:
          temperature_K: 300
          md_steps:      1000

    Legacy: config.verification.equilibration
    """
    new = config.get("equilibration", {})
    lv  = config.get("verification", {}).get("equilibration", {})
    return {
        "temperature_K": new.get("temperature_K", lv.get("temperature_K", 300)),
        "md_steps":      new.get("md_steps",      lv.get("md_steps",      1000)),
        "timestep_fs":   new.get("timestep_fs",   lv.get("timestep_fs",   1.0)),
        "damping":       new.get("damping",        lv.get("damping",       100.0)),
        "frozen_z_ang":  new.get("frozen_z_ang",  lv.get("frozen_z_ang",  None)),
    }


# =============================================================================
# Utility helpers
# =============================================================================

def calculate_gas_energy(mol, config, logger):
    """Calculates the potential energy of a molecule in vacuum after relaxation."""
    wf = _resolve_workflow(config)
    if not wf["candidate_relax"]:
        logger.info(
            f"  [Gas Phase] candidate_relax disabled — skipping energy calc for {mol.get_chemical_formula()}."
        )
        return 0.0

    from autoflow_srxn.simulation.potentials import SimulationEngine
    rp = _resolve_relax_params(config)
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


def log_to_csv(csv_path, summary_data):
    """Appends verification results to a CSV file."""
    import csv
    if not summary_data:
        return
    log_dir = os.path.dirname(os.path.abspath(csv_path))
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    file_exists = os.path.isfile(csv_path)
    keys = summary_data[0].keys()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        if not file_exists:
            writer.writeheader()
        writer.writerows(summary_data)


# =============================================================================
# Verification stage
# =============================================================================

def execute_verification_stage(candidates, config, logger, out_prefix, tag=3, e_gas=0.0, e_base=0.0):
    """Geometry-optimize, optionally MD-equilibrate, and score each candidate.

    Reads pipeline flags from ``workflow`` block (new format) or legacy
    ``verification`` block.  Relaxation hyper-parameters come from the unified
    ``relaxation`` block (new) or ``verification.relaxation`` (legacy).
    """
    wf       = _resolve_workflow(config)
    rp       = _resolve_relax_params(config)
    ep       = _resolve_equil_params(config)
    run_relax = wf["candidate_relax"]
    run_equil = wf["md_equilibrate"]
    run_post  = wf["post_md_relax"] if run_equil else False

    if not candidates:
        return []

    from autoflow_srxn.simulation.potentials import SimulationEngine

    # selected_indices: still read from verification block for backward-compat
    sel_idx = config.get("verification", {}).get("selected_indices", None)
    if isinstance(sel_idx, str):
        try:
            allowed = {"range": range, "list": list, "np": np, "numpy": np, "abs": abs}
            sel_idx = eval(sel_idx, {"__builtins__": {}}, allowed)
            if hasattr(sel_idx, "tolist"):
                sel_idx = sel_idx.tolist()
            elif not isinstance(sel_idx, list):
                sel_idx = list(sel_idx)
        except Exception as exc:
            logger.error(f"  [Verification] Failed to evaluate 'selected_indices': {exc}")
            sel_idx = None

    n_total  = len(candidates)
    n_target = len(sel_idx) if sel_idx is not None else n_total
    log_stage_title(logger, "VERIFICATION", f"Processing {n_target}/{n_total} candidates")
    logger.info(
        f"  Pipeline: candidate_relax={run_relax}, md_equilibrate={run_equil}, "
        f"post_md_relax={run_post} | fmax={rp['fmax']}, steps={rp['steps']}, "
        f"frozen_z={rp['frozen_z_ang']}"
    )

    engine = None
    calc   = None
    if run_relax:
        engine = SimulationEngine(config)
        calc   = engine.get_calculator()

    processed_cands = []
    summary_data    = []

    for i, atoms in enumerate(candidates):
        if sel_idx is not None and i not in sel_idx:
            continue

        atoms_proc = standardize_vasp_atoms(atoms.copy(), z_min_offset=0.5)
        atoms_proc.info = atoms.info.copy()

        try:
            if run_relax:
                atoms_proc.calc = calc
                e_init = atoms_proc.get_potential_energy()

                # 1. Geometry relaxation
                engine.relax(
                    atoms_proc,
                    steps=rp["steps"],
                    fmax=rp["fmax"],
                    frozen_z_ang=rp["frozen_z_ang"],
                    verbose=False,
                )

                # 2. Optional MD equilibration
                if run_equil:
                    engine.run_md(
                        atoms_proc,
                        temp_K=ep["temperature_K"],
                        md_steps=ep["md_steps"],
                    )
                    if run_post:
                        engine.relax(
                            atoms_proc,
                            steps=rp["steps"],
                            fmax=rp["fmax"],
                            frozen_z_ang=rp["frozen_z_ang"],
                            verbose=False,
                        )

                atoms_proc = standardize_vasp_atoms(atoms_proc, z_min_offset=0.5)
                atoms_proc.calc = calc
                e_final = atoms_proc.get_potential_energy()
                delta   = e_final - e_init
                e_ads   = e_final - (e_gas + e_base)
            else:
                e_init = e_final = delta = e_ads = 0.0

            mech = atoms.info.get("mechanism", "unknown")
            summary_data.append({
                "id": i, "mech": mech,
                "e_initial": e_init, "e_final": e_final,
                "delta": delta, "e_ads": e_ads,
            })
            log_to_csv(
                os.path.join(os.path.dirname(out_prefix), "energylog.csv"),
                [{"tag": tag, "id": i, "mechanism": mech,
                  "e_initial": e_init, "e_final": e_final,
                  "delta": delta, "e_ads": e_ads}],
            )
            atoms_proc.info.update(
                {"e_initial": e_init, "e_final": e_final, "delta": delta, "e_ads": e_ads}
            )
            processed_cands.append(atoms_proc)

        except Exception as exc:
            logger.error(f"  [Verification] Candidate {i} failed: {exc}")

    log_results_table(logger, summary_data, title=f"Verification Summary (tag={tag})")

    if processed_cands:
        suffix = "_relaxed" if run_relax else "_evaluated"
        write(f"{out_prefix}{suffix}.extxyz", processed_cands)
    return processed_cands


# =============================================================================
# Discovery stage
# =============================================================================

def execute_discovery_stage(slab, mol, config, out_prefix, logger,
                            tag=2, center_target="Si", e_gas=0.0, e_base=0.0,
                            stage_type="precursor"):
    """Generate candidates (physi + chemi) then run verification."""
    rs_cfg    = config.get("reaction_search", {})
    mechs_cfg = rs_cfg.get("mechanisms", {})
    stage_cfg = mechs_cfg.get(stage_type, {})
    physi_cfg = stage_cfg.get("physisorption", {"enabled": False})
    chem_cfg  = stage_cfg.get("chemisorption",  {"enabled": False})
    symprec   = rs_cfg.get("candidate_filter", {}).get("symprec", 0.2)

    mgr       = AdsorptionWorkflowManager(slab, config=config, symprec=symprec, verbose=True)
    all_cands = []

    if physi_cfg.get("enabled", False):
        logger.info(f"  [Stage: {stage_type}] Physisorption search for {mol.get_chemical_formula()}...")
        phy_cands = mgr.generate_physisorption_candidates(
            mol,
            height=physi_cfg.get("placement_height", 3.5),
            tag=tag,
            rot_center=physi_cfg.get("rotation_center", center_target),
            height_mode=physi_cfg.get("height_mode", "clearance"),
            gravity_pull=physi_cfg.get("gravity_pull", {"enabled": False}),
        )
        for c in phy_cands:
            c.info["mechanism"] = "physisorption"
        all_cands.extend(phy_cands)

    if chem_cfg.get("enabled", False):
        logger.info(
            f"  [Stage: {stage_type}] Chemisorption search for "
            f"{mol.get_chemical_formula()} (center={center_target})..."
        )
        chem_cands = build_chemisorption_structures(
            molecule=mol, center_target=center_target, surface=slab,
            config=config, tag=tag,
            results_dir=os.path.dirname(out_prefix),
        )
        for c in chem_cands:
            c.info["mechanism"] = "chemisorption"
        all_cands.extend(chem_cands)

    if all_cands:
        write(f"{out_prefix}_candidates.extxyz", all_cands)

    return execute_verification_stage(
        all_cands, config, logger, out_prefix,
        tag=tag, e_gas=e_gas, e_base=e_base,
    )


# =============================================================================
# Top-level workflow
# =============================================================================

def execute_discovery_workflow(config, logger, gas_energy_map=None, slab_base_energy=0.0):
    """Main workflow: slab → inhibitor → precursor."""
    paths     = config["paths"]
    sp_cfg    = config.get("surface_prep", {})
    rs_cfg    = config.get("reaction_search", {})
    mechs_cfg = rs_cfg.get("mechanisms", {})
    inh_cfg   = mechs_cfg.get("inhibitor", {})
    wf        = _resolve_workflow(config)
    rp        = _resolve_relax_params(config)

    precursor_file = paths.get("precursor")
    inh_file       = paths.get("inhibitor")
    out_dir        = paths.get("output_prefix", "results")
    mol = read(precursor_file) if precursor_file and os.path.exists(precursor_file) else None

    # --- Stage 0: Slab generation ---
    sub_gen_cfg = sp_cfg.get("slab_generation", {})
    if sub_gen_cfg.get("enabled", False):
        log_stage_title(logger, "STAGE 0", "Generating substrate slab...")
        slab = create_slab_from_bulk(
            bulk_atoms=read(paths["substrate_bulk"]),
            miller_indices=sub_gen_cfg.get("miller", [1, 0, 0]),
            thickness=sub_gen_cfg.get("thickness_ang", 10.0),
            vacuum=sub_gen_cfg.get("vacuum_ang", 10.0),
            target_area=sub_gen_cfg.get("target_area_ang2"),
            verbose=True,
        )
    else:
        slab = standardize_vasp_atoms(read(paths["input_structure"]), z_min_offset=0.5)
    slab.set_tags(0)

    # --- Stage 0.5: Slab relaxation ---
    if wf["slab_relax"]:
        from autoflow_srxn.simulation.potentials import SimulationEngine
        log_stage_title(logger, "STAGE 0.5", "Slab relaxation...")
        engine = SimulationEngine(config)
        slab.calc = engine.get_calculator()
        e_init = slab.get_potential_energy()
        engine.relax(
            slab,
            fmax=rp["fmax"],
            steps=200,
            frozen_z_ang=rp["frozen_z_ang"],
        )
        slab = standardize_vasp_atoms(slab, z_min_offset=0.5)
        slab_base_energy = slab.get_potential_energy()
        log_energy_comparison(logger, "Slab Relax", e_init, slab_base_energy)
    else:
        slab = standardize_vasp_atoms(slab, z_min_offset=0.5)
        if not slab_base_energy and wf["candidate_relax"]:
            from autoflow_srxn.simulation.potentials import SimulationEngine
            engine = SimulationEngine(config)
            slab.calc = engine.get_calculator()
            slab_base_energy = slab.get_potential_energy()
        else:
            slab_base_energy = slab_base_energy or 0.0

    # --- Stage 1: Inhibitor discovery ---
    base_slabs = [slab]
    if inh_cfg.get("enabled", False) and inh_file and os.path.exists(inh_file):
        log_stage_title(logger, "STAGE 1", f"Inhibitor Discovery ({os.path.basename(inh_file)})")
        e_gas_inh = (gas_energy_map.get(inh_file, 0.0) if gas_energy_map
                     else calculate_gas_energy(read(inh_file), config, logger))
        inh_cands = execute_discovery_stage(
            slab, read(inh_file), config,
            os.path.join(out_dir, "stage1_inhibitor"), logger,
            tag=2, center_target=inh_cfg.get("center", "O"),
            e_gas=e_gas_inh, e_base=slab_base_energy,
            stage_type="inhibitor",
        )
        if inh_cands:
            inh_cands.sort(key=lambda x: x.info.get("e_final", 1e10))
            base_slabs = inh_cands[:inh_cfg.get("branching_limit", 3)]
            logger.info(f"  Selected top {len(base_slabs)} inhibited surfaces for Stage 2.")

    # --- Stage 2: Precursor discovery ---
    if mol:
        log_stage_title(logger, "STAGE 2", f"Precursor Discovery ({os.path.basename(precursor_file)})")
        e_gas_mol = (gas_energy_map.get(precursor_file, 0.0) if gas_energy_map
                     else calculate_gas_energy(mol, config, logger))
        pre_cfg    = mechs_cfg.get("precursor", {})
        pre_center = pre_cfg.get("center", "Si")
        all_final  = []

        for i, s in enumerate(base_slabs):
            e_base_s2 = s.info.get("e_final")
            if e_base_s2 is None:
                e_base_s2 = s.get_potential_energy() if s.calc is not None else slab_base_energy
            suffix  = f"_branch{i}" if len(base_slabs) > 1 else ""
            results = execute_discovery_stage(
                s, mol, config,
                os.path.join(out_dir, f"stage2_precursor{suffix}"), logger,
                tag=3, center_target=pre_center,
                e_gas=e_gas_mol, e_base=e_base_s2,
                stage_type="precursor",
            )
            for r in results:
                r.info["inh_id"] = i
            all_final.extend(results)

        if all_final:
            write(os.path.join(out_dir, "final_results.extxyz"), all_final)


def run_generic_adsorption_study(config_path="config.yaml"):
    config = load_config(config_path)
    paths  = config["paths"]

    def get_files(p):
        if not p:
            return [None]
        if os.path.isdir(p):
            import glob
            files = []
            for ext in ["*.vasp", "*.xyz", "*.extxyz"]:
                files.extend(glob.glob(os.path.join(p, ext)))
            return sorted(files)
        return [p]

    precursors = get_files(paths.get("precursor") or paths.get("precursors_dir"))
    inhibitors = get_files(paths.get("inhibitor"))
    if paths.get("include_no_inhibitor", False):
        inhibitors = [None] + inhibitors
    elif not inhibitors:
        inhibitors = [None]

    global_prefix = paths.get("output_prefix", "discovery")
    unique_mols   = list(set(f for f in precursors + inhibitors if f and os.path.exists(f)))
    gas_energy_map = {}
    if unique_mols:
        tmp_logger = setup_logger(log_path=os.path.join(global_prefix, "ref_energies.log"), mode="w")
        for m_path in unique_mols:
            gas_energy_map[m_path] = calculate_gas_energy(read(m_path), config, tmp_logger)

    for inh_path in inhibitors:
        for pre_path in precursors:
            if not pre_path:
                continue
            inh_name = os.path.splitext(os.path.basename(inh_path))[0] if inh_path else "clean"
            pre_name = os.path.splitext(os.path.basename(pre_path))[0]
            run_name = f"{inh_name}_pretreated_{pre_name}"
            run_dir  = os.path.join(global_prefix, run_name)
            os.makedirs(run_dir, exist_ok=True)

            logger = setup_logger(log_path=os.path.join(run_dir, "workflow.log"), mode="w")
            log_stage_title(logger, "BATCH RUN", f"Sequence: {inh_name} -> {pre_name}")

            run_config = copy.deepcopy(config)
            run_config["paths"]["precursor"]     = pre_path
            run_config["paths"]["inhibitor"]     = inh_path
            run_config["paths"]["output_prefix"] = run_dir

            try:
                execute_discovery_workflow(run_config, logger, gas_energy_map=gas_energy_map)
            except Exception as exc:
                logger.error(f"Discovery workflow failed for {run_name}: {exc}")


if __name__ == "__main__":
    run_generic_adsorption_study(sys.argv[1] if len(sys.argv) > 1 else "config.yaml")
