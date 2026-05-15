import os
import sys
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)
import copy
import yaml
import numpy as np
from ase.io import read, write
from ase import Atoms

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None

from autoflow_srxn.surface.ads_workflow_mgr import AdsorptionWorkflowManager
from autoflow_srxn.surface.chemisorption_builder import build_chemisorption_structures
from autoflow_srxn.surface.site_map import generate_and_plot_site_map
from autoflow_srxn.utils.logger_utils import log_energy_comparison, log_results_table, log_stage_title, setup_logger
from autoflow_srxn.utils.perf_tracker import PerfTracker, perf_stage, set_perf_tracker
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
    """
    wf = config.get("workflow", {})
    return {
        "slab_relax":      wf.get("slab_relax",      False),
        "candidate_relax": wf.get("candidate_relax",  False),
        "md_equilibrate":  wf.get("md_equilibrate",   False),
        "post_md_relax":   wf.get("post_md_relax",    True),
    }


def _resolve_relax_params(config):
    """Return relaxation hyper-parameters.

    New format  (config.relaxation.*):
        relaxation:
          fmax:         0.05
          steps:        100
          frozen_z_ang: 5.5
    """
    new = config.get("relaxation", {})
    return {
        "fmax":         new.get("fmax",         0.05),
        "steps":        new.get("steps",        100),
        "frozen_z_ang": new.get("frozen_z_ang", None),
    }


def _resolve_equil_params(config):
    """Return MD equilibration parameters.

    New format  (config.equilibration.*):
        equilibration:
          temperature_K: 300
          md_steps:      1000
    """
    new = config.get("equilibration", {})
    return {
        "temperature_K": new.get("temperature_K", 300),
        "md_steps":      new.get("md_steps",      1000),
        "timestep_fs":   new.get("timestep_fs",   1.0),
        "damping":       new.get("damping",       100.0),
        "frozen_z_ang":  new.get("frozen_z_ang",  None),
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

    cand_iter = (
        _tqdm(
            enumerate(candidates),
            total=len(candidates),
            desc="[Verification] candidates",
            unit="struct",
            leave=True,
            dynamic_ncols=True,
        )
        if _tqdm else enumerate(candidates)
    )

    for i, atoms in cand_iter:
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


# Thin timing shim — wraps execute_verification_stage with a perf_stage block
_orig_verification = execute_verification_stage


def execute_verification_stage(candidates, config, logger, out_prefix, tag=3, e_gas=0.0, e_base=0.0):
    label = f"verification ({len(candidates)} candidates)"
    with perf_stage(label):
        return _orig_verification(candidates, config, logger, out_prefix, tag=tag, e_gas=e_gas, e_base=e_base)


# =============================================================================
# Discovery stage
# =============================================================================

def execute_ts_search_stage(results, config, logger, out_prefix):
    """Pairs physisorption and chemisorption results and runs automated TS search."""
    rs_cfg = config.get("reaction_search", {})
    mechs_cfg = rs_cfg.get("mechanisms", {})
    # Look for ts_search config in either precursor or generic block
    ts_cfg = mechs_cfg.get("precursor", {}).get("ts_search", {"enabled": False})
    
    if not ts_cfg.get("enabled", False):
        return []

    phy_results = [c for c in results if str(c.info.get("mechanism", "")).lower().startswith("physisorption")]
    chem_results = [c for c in results if str(c.info.get("mechanism", "")).lower().startswith("chemisorption")]

    if not phy_results or not chem_results:
        logger.info("  [TS Search] Skipping: Need both physisorption and chemisorption candidates.")
        return []

    from autoflow_srxn.simulation.potentials import SimulationEngine
    from autoflow_srxn.transition.workflow import TransitionStateWorkflow

    engine = SimulationEngine(config)
    workflow = TransitionStateWorkflow(engine, config=ts_cfg)
    
    # Strategy: Match each chemisorption candidate with the BEST physisorption candidate
    # (In a more advanced version, we could match by site proximity)
    best_phy = min(phy_results, key=lambda x: x.info.get("e_final", 1e10))
    
    ts_results = []
    log_stage_title(logger, "TS SEARCH", f"Running NEB+ARTn for {len(chem_results)} chemisorption candidates")

    neb_cfg = ts_cfg.get("neb", {})
    art_cfg = ts_cfg.get("artn", {})

    for i, chem in enumerate(chem_results):
        ts_dir = os.path.join(os.path.dirname(out_prefix), f"ts_search_{i}")
        with perf_stage(f"ts_search_{i} (NEB+ARTn+Vib)"):
            try:
                ts_structure = workflow.run_ts_search(
                    best_phy, chem,
                    n_images       = neb_cfg.get("n_images",    ts_cfg.get("n_images", 7)),
                    fmax_neb       = neb_cfg.get("fmax",        ts_cfg.get("fmax",     0.05)),
                    steps_neb      = neb_cfg.get("steps",       ts_cfg.get("steps",    100)),
                    interpolate    = neb_cfg.get("interpolate", "idpp"),
                    climbing_image = neb_cfg.get("climbing_image", False),
                    fmax_art       = art_cfg.get("fmax",        ts_cfg.get("fmax",     0.05)),
                    steps_art      = art_cfg.get("steps",       ts_cfg.get("steps",    200)),
                    displacement_ang = art_cfg.get("displacement_ang", 0.1),
                    output_dir=ts_dir
                )
                if ts_structure:
                    ts_structure.info["mechanism"] = f"TS_{chem.info.get('mechanism')}"
                    ts_results.append(ts_structure)
            except Exception as e:
                logger.error(f"  [TS Search] Candidate {i} failed: {e}")

    if ts_results:
        write(f"{out_prefix}_ts_results.extxyz", ts_results)
    return ts_results


def execute_discovery_stage(slab, mol, config, out_prefix, logger,
                            tag=2, center_target="Si", e_gas=0.0, e_base=0.0,
                            stage_type="precursor"):
    """Generate candidates (physi + chemi) then run verification."""
    rs_cfg    = config.get("reaction_search", {})
    mechs_cfg = rs_cfg.get("mechanisms", {})
    stage_cfg = mechs_cfg.get(stage_type, {})
    physi_cfg = stage_cfg.get("physisorption", {"enabled": False})
    chem_cfg  = stage_cfg.get("chemisorption",  {"enabled": False})
    symprec   = rs_cfg.get("symprec", 0.2)

    # --- Intelligent Center Selection ---
    actual_center = center_target
    if stage_type == "inhibitor":
        # Default inhibitor center to COM (Center of Mass)
        actual_center = "com"
    else:
        # Precursor logic:
        # 1. If center_target is a list, find first match in molecule
        # 2. If no match or not a list, look for non-HCNO elements
        # 3. Fallback to 'com' if still nothing found
        mol_symbols = set(mol.get_chemical_symbols())
        if isinstance(center_target, list):
            found = False
            for c in center_target:
                if c in mol_symbols:
                    actual_center = c
                    found = True
                    break
            if not found:
                others = [s for s in mol_symbols if s not in ["H", "C", "N", "O"]]
                actual_center = others[0] if others else "com"
        elif isinstance(center_target, str) and center_target not in mol_symbols and center_target != "com":
             others = [s for s in mol_symbols if s not in ["H", "C", "N", "O"]]
             actual_center = others[0] if others else "com"
             if others:
                 logger.info(f"  [Stage: {stage_type}] '{center_target}' not found. Auto-selected '{actual_center}' as center.")
 
    mgr       = AdsorptionWorkflowManager(slab, config=config, symprec=symprec, verbose=True)
    all_cands = []

    if physi_cfg.get("enabled", False):
        logger.info(f"  [Stage: {stage_type}] Physisorption search for {mol.get_chemical_formula()}...")

        # Save site map before generating candidates
        site_map_path = os.path.join(os.path.dirname(out_prefix), "site_map.png")
        try:
            generate_and_plot_site_map(
                slab,
                site_map_path,
                symprec=symprec,
                mgr=mgr,
                title=f"Adsorption site map — {stage_type} ({mol.get_chemical_formula()})",
            )
            logger.info(f"  [SiteMap] Saved: {os.path.relpath(site_map_path)}")
        except Exception as _sm_exc:
            logger.warning(f"  [SiteMap] Could not generate site map: {_sm_exc}")

        with perf_stage(f"physi_candidates ({stage_type}/{mol.get_chemical_formula()})"):
            phy_cands = mgr.generate_physisorption_candidates(
                mol,
                height=physi_cfg.get("placement_height", 3.5),
                n_rot=physi_cfg.get("n_rot", 32),
                tag=tag,
                rot_center=actual_center,
                height_mode=physi_cfg.get("height_mode", "clearance"),
                gravity_pull=physi_cfg.get("gravity_pull", {"enabled": False}),
            )
        for c in phy_cands:
            c.info.setdefault("mechanism", "physisorption")
        all_cands.extend(phy_cands)

    if chem_cfg.get("enabled", False):
        logger.info(
            f"  [Stage: {stage_type}] Chemisorption search for "
            f"{mol.get_chemical_formula()} (center={actual_center})..."
        )
        with perf_stage(f"chem_candidates ({stage_type}/{mol.get_chemical_formula()})"):
            chem_cands = build_chemisorption_structures(
                molecule=mol, center_target=actual_center, surface=slab,
                rot_steps=chem_cfg.get("rot_steps", 8),
                config=config, tag=tag,
                results_dir=os.path.dirname(out_prefix),
            )
        for c in chem_cands:
            c.info["mechanism"] = "chemisorption"
        all_cands.extend(chem_cands)

    if all_cands:
        write(f"{out_prefix}_candidates.extxyz", all_cands)

    results = execute_verification_stage(
        all_cands, config, logger, out_prefix,
        tag=tag, e_gas=e_gas, e_base=e_base,
    )

    # --- NEW: Automated TS Search ---
    if stage_type == "precursor":
        execute_ts_search_stage(results, config, logger, out_prefix)

    return results


# =============================================================================
# Top-level workflow
# =============================================================================

def execute_discovery_workflow(config, logger, slab, gas_energy_map=None, slab_base_energy=0.0):
    """Main workflow: inhibitor → precursor using a pre-prepared slab."""
    paths     = config["paths"]
    rs_cfg    = config.get("reaction_search", {})
    mechs_cfg = rs_cfg.get("mechanisms", {})
    inh_cfg   = mechs_cfg.get("inhibitor", {})
    wf        = _resolve_workflow(config)

    precursor_file = paths.get("precursor")
    inh_file       = paths.get("inhibitor")
    out_dir        = paths.get("output_prefix", "results")
    mol = read(precursor_file) if precursor_file and os.path.exists(precursor_file) else None

    # --- Stage 1: Inhibitor discovery ---
    base_slabs = [slab.copy()]
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
                try:
                    e_base_s2 = s.get_potential_energy() if s.calc is not None else slab_base_energy
                except Exception:
                    e_base_s2 = slab_base_energy
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
    sp_cfg = config.get("surface_prep", {})
    wf     = _resolve_workflow(config)
    rp     = _resolve_relax_params(config)

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
    os.makedirs(global_prefix, exist_ok=True)

    # Setup global logger for common stages
    master_logger = setup_logger(log_path=os.path.join(global_prefix, "master_workflow.log"), mode="w")

    # --- Init performance tracker ---
    tracker = PerfTracker(sample_interval=1.0, log_on_exit=True)
    set_perf_tracker(tracker)

    # --- SHARED STAGE: Slab preparation (only once) ---
    sub_gen_cfg = sp_cfg.get("slab_generation", {})
    with perf_stage("slab_generation"):
        if sub_gen_cfg.get("enabled", False):
            log_stage_title(master_logger, "GLOBAL STAGE 0", "Generating substrate slab...")
            slab = create_slab_from_bulk(
                bulk_atoms=read(paths["substrate_bulk"]),
                miller_indices=sub_gen_cfg.get("miller", [1, 0, 0]),
                thickness=sub_gen_cfg.get("thickness_ang", 10.0),
                vacuum=sub_gen_cfg.get("vacuum_ang", 10.0),
                target_area=sub_gen_cfg.get("target_area_ang2"),
                supercell_matrix=sub_gen_cfg.get("supercell_matrix"),
                verbose=True,
            )
        else:
            slab = standardize_vasp_atoms(read(paths["input_structure"]), z_min_offset=0.5)
        slab.set_tags(0)

        # --- GLOBAL STAGE 0.1: Passivation ---
        pass_cfg = sp_cfg.get("passivation", {})
        if pass_cfg.get("enabled", False):
            log_stage_title(master_logger, "GLOBAL STAGE 0.1", f"Passivating surface with {pass_cfg.get('element', 'H')}...")
            valence_map = sp_cfg.get("surface_analysis", {}).get("ideal_coordination", {})
            slab = passivate_surface_coverage_general(
                slab,
                coverage=pass_cfg.get("coverage", 1.0),
                valence_map=valence_map,
                element=pass_cfg.get("element", "H"),
                side=pass_cfg.get("side", "bottom"),
                verbose=True,
            )

    # --- GLOBAL STAGE 0.5: Slab relaxation ---
    # Open ref_energies logger early so slab energy is also captured there
    ref_logger = setup_logger(log_path=os.path.join(global_prefix, "ref_energies.log"), mode="w")

    slab_base_energy = 0.0
    if wf["slab_relax"]:
        from autoflow_srxn.simulation.potentials import SimulationEngine
        log_stage_title(master_logger, "GLOBAL STAGE 0.5", "Slab relaxation...")
        with perf_stage("slab_relax"):
            engine = SimulationEngine(config)
            slab.calc = engine.get_calculator()
            e_init = slab.get_potential_energy()
            engine.relax(slab, fmax=rp["fmax"], steps=200, frozen_z_ang=rp["frozen_z_ang"])
            slab = standardize_vasp_atoms(slab, z_min_offset=0.5)
            slab_base_energy = slab.get_potential_energy()
        log_energy_comparison(master_logger, "Slab Relax", e_init, slab_base_energy)
        ref_logger.info(f"  [Slab] E_relaxed: {slab_base_energy:12.4f} eV  ({slab.get_chemical_formula()})")
    else:
        slab = standardize_vasp_atoms(slab, z_min_offset=0.5)
        if wf["candidate_relax"]:
            from autoflow_srxn.simulation.potentials import SimulationEngine
            engine = SimulationEngine(config)
            slab.calc = engine.get_calculator()
            try:
                slab_base_energy = slab.get_potential_energy()
                ref_logger.info(f"  [Slab] E_static:  {slab_base_energy:12.4f} eV  ({slab.get_chemical_formula()})")
            except Exception:
                slab_base_energy = 0.0

    write(os.path.join(global_prefix, "prepared_slab.extxyz"), slab)

    # --- Pre-calculate gas phase energies ---
    unique_mols = list(set(f for f in precursors + inhibitors if f and os.path.exists(f)))
    gas_energy_map = {}
    with perf_stage(f"gas_phase_energies ({len(unique_mols)} molecules)"):
        for m_path in unique_mols:
            gas_energy_map[m_path] = calculate_gas_energy(read(m_path), config, ref_logger)

    # --- BATCH LOOP ---
    for inh_path in inhibitors:
        for pre_path in precursors:
            inh_name = os.path.splitext(os.path.basename(inh_path))[0] if inh_path else "clean"
            pre_name = os.path.splitext(os.path.basename(pre_path))[0] if pre_path else "none"

            if not pre_path and inh_name == "clean":
                run_name = "bare_slab"
            else:
                run_name = f"{inh_name}_on_{pre_name}"

            run_dir = os.path.join(global_prefix, run_name)
            os.makedirs(run_dir, exist_ok=True)

            logger = setup_logger(log_path=os.path.join(run_dir, "workflow.log"), mode="w")
            log_stage_title(logger, "BATCH RUN", f"Sequence: {inh_name} -> {pre_name}")

            run_config = copy.deepcopy(config)
            run_config["paths"]["precursor"]     = pre_path
            run_config["paths"]["inhibitor"]     = inh_path
            run_config["paths"]["output_prefix"] = run_dir

            master_logger.info(f"BATCH START: {run_name}  →  {run_dir}")
            try:
                execute_discovery_workflow(run_config, logger, slab=slab,
                                           gas_energy_map=gas_energy_map,
                                           slab_base_energy=slab_base_energy)
                master_logger.info(f"BATCH DONE:  {run_name}  [OK]")
            except Exception as exc:
                logger.error(f"Discovery workflow failed for {run_name}: {exc}")
                import traceback
                logger.error(traceback.format_exc())
                master_logger.error(f"BATCH FAIL:  {run_name}  — {exc}")

    # --- Write performance report ---
    perf_path = os.path.join(global_prefix, "perf_report.log")
    tracker.write_report(perf_path)
    tracker.log_report(master_logger)
    master_logger.info(f"Performance report written to {os.path.relpath(perf_path)}")
    set_perf_tracker(None)  # reset singleton



if __name__ == "__main__":
    run_generic_adsorption_study(sys.argv[1] if len(sys.argv) > 1 else "config.yaml")
