import os
import sys
import copy
import yaml
import numpy as np
from ase.io import read, write
from ase import Atoms

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None

from .ads_workflow_mgr import AdsorptionWorkflowManager
from .chemisorption_builder import build_chemisorption_structures
from .site_map import generate_and_plot_site_map
from ..utils import (
    log_energy_comparison,
    log_results_table,
    log_stage_title,
    setup_logger,
    load_yaml_config as load_config,
)
from ..utils.perf_tracker import PerfTracker, perf_stage, set_perf_tracker
from .surface_utils import (
    create_slab_from_bulk,
    passivate_surface_coverage_general,
    standardize_vasp_atoms,
)


# =============================================================================
# Config helpers — support new unified format AND legacy split format
# =============================================================================

# (load_config is now imported from autoflow_srxn.utils)


def _resolve_workflow(config):
    """Return pipeline enable-flags."""
    wf = config.get("workflow", {})
    return {
        "slab_relax":      wf.get("slab_relax",      False),
        "candidate_relax": wf.get("candidate_relax",  False),
        "md_equilibrate":  wf.get("md_equilibrate",   False),
        "post_md_relax":   wf.get("post_md_relax",    True),
    }


def _resolve_relax_params(config):
    """Return relaxation hyper-parameters."""
    new = config.get("relaxation", {})
    return {
        "fmax":         new.get("fmax",         0.05),
        "steps":        new.get("steps",        100),
        "frozen_z_ang": new.get("frozen_z_ang", None),
    }


def _resolve_equil_params(config):
    """Return MD equilibration parameters."""
    new = config.get("equilibration", {})
    return {
        "temperature_K": new.get("temperature_K", 300),
        "md_steps":      new.get("md_steps",      1000),
        "timestep_fs":   new.get("timestep_fs",   1.0),
        "damping":       new.get("damping",       100.0),
        "frozen_z_ang":  new.get("frozen_z_ang",  None),
    }


# =============================================================================
# Workflow Stages (Modularized)
# =============================================================================

def prepare_slab_stage(config, logger):
    """Handles Stage 0: Slab generation and passivation."""
    paths = config.get("paths", {})
    sp_cfg = config.get("surface_prep", {})
    sub_gen_cfg = sp_cfg.get("slab_generation", {})
    
    with perf_stage("slab_generation"):
        if sub_gen_cfg.get("enabled", False):
            log_stage_title(logger, "GLOBAL STAGE 0", "Generating substrate slab...")
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
            slab_file = paths.get("input_structure")
            if not slab_file or not os.path.exists(slab_file):
                raise FileNotFoundError(f"Input structure file not found: {slab_file}")
            slab = standardize_vasp_atoms(read(slab_file), z_min_offset=0.5)
        
        slab.set_tags(0)

        # --- GLOBAL STAGE 0.1: Passivation ---
        pass_cfg = sp_cfg.get("passivation", {})
        if pass_cfg.get("enabled", False):
            log_stage_title(logger, "GLOBAL STAGE 0.1", f"Passivating surface with {pass_cfg.get('element', 'H')}...")
            valence_map = sp_cfg.get("surface_analysis", {}).get("ideal_coordination", {})
            slab = passivate_surface_coverage_general(
                slab,
                coverage=pass_cfg.get("coverage", 1.0),
                valence_map=valence_map,
                element=pass_cfg.get("element", "H"),
                side=pass_cfg.get("side", "bottom"),
                verbose=True,
            )
    return slab


def relax_slab_stage(slab, config, logger):
    """Handles Stage 0.5: Slab relaxation. Returns (relaxed_slab, base_energy)."""
    wf = _resolve_workflow(config)
    rp = _resolve_relax_params(config)
    
    slab_base_energy = 0.0
    from ..simulation.potentials import SimulationEngine
    
    if wf["slab_relax"]:
        log_stage_title(logger, "GLOBAL STAGE 0.5", "Slab relaxation...")
        with perf_stage("slab_relax"):
            engine = SimulationEngine(config)
            slab.calc = engine.get_calculator()
            e_init = slab.get_potential_energy()
            engine.relax(slab, fmax=rp["fmax"], steps=200, frozen_z_ang=rp["frozen_z_ang"])
            slab = standardize_vasp_atoms(slab, z_min_offset=0.5)
            slab_base_energy = slab.get_potential_energy()
        log_energy_comparison(logger, "Slab Relax", e_init, slab_base_energy)
    else:
        slab = standardize_vasp_atoms(slab, z_min_offset=0.5)
        if wf["candidate_relax"]:
            engine = SimulationEngine(config)
            slab.calc = engine.get_calculator()
            try:
                slab_base_energy = slab.get_potential_energy()
            except Exception:
                slab_base_energy = 0.0
    
    return slab, slab_base_energy


def calculate_gas_energy(mol, config, logger):
    """Calculates the potential energy of a molecule in vacuum after relaxation."""
    wf = _resolve_workflow(config)
    if not wf["candidate_relax"]:
        logger.info(f"  [Gas Phase] candidate_relax disabled — skipping energy calc for {mol.get_chemical_formula()}.")
        return 0.0

    from ..simulation.potentials import SimulationEngine
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


# =============================================================================
# Reaction Search Stages
# =============================================================================

def execute_verification_stage(candidates, config, logger, out_prefix, tag=3, e_gas=0.0, e_base=0.0):
    """Geometry-optimize, optionally MD-equilibrate, and score each candidate."""
    wf       = _resolve_workflow(config)
    rp       = _resolve_relax_params(config)
    ep       = _resolve_equil_params(config)
    run_relax = wf["candidate_relax"]
    run_equil = wf["md_equilibrate"]
    run_post  = wf["post_md_relax"] if run_equil else False

    if not candidates:
        return []

    from ..simulation.potentials import SimulationEngine
    sel_idx = config.get("verification", {}).get("selected_indices", None)
    
    # ... (Evaluation logic for selected_indices) ...
    if isinstance(sel_idx, str):
        try:
            allowed = {"range": range, "list": list, "np": np, "numpy": np, "abs": abs}
            sel_idx = eval(sel_idx, {"__builtins__": {}}, allowed)
        except Exception: sel_idx = None

    n_total  = len(candidates)
    n_target = len(sel_idx) if sel_idx is not None else n_total
    log_stage_title(logger, "VERIFICATION", f"Processing {n_target}/{n_total} candidates")

    engine = SimulationEngine(config) if run_relax else None
    calc = engine.get_calculator() if engine else None

    processed_cands = []
    summary_data    = []

    for i, atoms in enumerate(candidates):
        if sel_idx is not None and i not in sel_idx: continue
        atoms_proc = standardize_vasp_atoms(atoms.copy(), z_min_offset=0.5)
        atoms_proc.info = atoms.info.copy()
        try:
            if run_relax:
                atoms_proc.calc = calc
                e_init = atoms_proc.get_potential_energy()
                engine.relax(atoms_proc, steps=rp["steps"], fmax=rp["fmax"], 
                             frozen_z_ang=rp["frozen_z_ang"], verbose=False)
                if run_equil:
                    engine.run_md(atoms_proc, temp_K=ep["temperature_K"], md_steps=ep["md_steps"])
                    if run_post:
                        engine.relax(atoms_proc, steps=rp["steps"], fmax=rp["fmax"], 
                                     frozen_z_ang=rp["frozen_z_ang"], verbose=False)
                e_final = atoms_proc.get_potential_energy()
                delta, e_ads = e_final - e_init, e_final - (e_gas + e_base)
            else:
                e_init = e_final = delta = e_ads = 0.0

            summary_data.append({"id": i, "mech": atoms.info.get("mechanism", "unknown"),
                                 "e_initial": e_init, "e_final": e_final, "e_ads": e_ads})
            atoms_proc.info.update({"e_initial": e_init, "e_final": e_final, "e_ads": e_ads})
            processed_cands.append(atoms_proc)
        except Exception as exc:
            logger.error(f"  [Verification] Candidate {i} failed: {exc}")

    if processed_cands:
        write(f"{out_prefix}_{'relaxed' if run_relax else 'evaluated'}.extxyz", processed_cands)
    return processed_cands


def execute_ts_search_stage(results, config, logger, out_prefix):
    """Pairs physisorption and chemisorption results and runs automated TS search."""
    ts_cfg = config.get("reaction_search", {}).get("mechanisms", {}).get("precursor", {}).get("ts_search", {"enabled": False})
    if not ts_cfg.get("enabled", False): return []

    phy_results = [c for c in results if str(c.info.get("mechanism", "")).lower().startswith("physisorption")]
    chem_results = [c for c in results if str(c.info.get("mechanism", "")).lower().startswith("chemisorption")]
    if not phy_results or not chem_results: return []

    from ..simulation.potentials import SimulationEngine
    from ..transition.workflow import TransitionStateWorkflow
    engine = SimulationEngine(config)
    workflow = TransitionStateWorkflow(engine, config=ts_cfg)
    
    best_phy = min(phy_results, key=lambda x: x.info.get("e_final", 1e10))
    ts_results = []
    
    neb_cfg, art_cfg = ts_cfg.get("neb", {}), ts_cfg.get("artn", {})
    for i, chem in enumerate(chem_results):
        ts_dir = os.path.join(os.path.dirname(out_prefix), f"ts_search_{i}")
        try:
            ts_structure = workflow.run_ts_search(
                best_phy, chem,
                n_images=neb_cfg.get("n_images", 7), fmax_neb=neb_cfg.get("fmax", 0.05),
                steps_neb=neb_cfg.get("steps", 100), output_dir=ts_dir
            )
            if ts_structure:
                ts_structure.info["mechanism"] = f"TS_{chem.info.get('mechanism')}"
                ts_results.append(ts_structure)
        except Exception as e: logger.error(f"  [TS Search] Candidate {i} failed: {e}")

    if ts_results: write(f"{out_prefix}_ts_results.extxyz", ts_results)
    return ts_results


def execute_discovery_stage(slab, mol, config, out_prefix, logger, tag=2, center_target="Si", e_gas=0.0, e_base=0.0, stage_type="precursor"):
    """Generate candidates then run verification and TS search."""
    rs_cfg = config.get("reaction_search", {})
    mechs_cfg = rs_cfg.get("mechanisms", {})
    stage_cfg = mechs_cfg.get(stage_type, {})
    physi_cfg = stage_cfg.get("physisorption", {"enabled": False})
    chem_cfg  = stage_cfg.get("chemisorption",  {"enabled": False})
    symprec   = rs_cfg.get("symprec", 0.2)

    # --- Intelligent Center Selection ---
    actual_center = center_target
    if stage_type == "inhibitor":
        actual_center = "com"
    else:
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

    mgr = AdsorptionWorkflowManager(slab, config=config, symprec=symprec, verbose=True)
    all_cands = []

    if physi_cfg.get("enabled", False):
        logger.info(f"  [Stage: {stage_type}] Physisorption search for {mol.get_chemical_formula()}...")
        
        # Site Map Generation
        site_map_path = os.path.join(os.path.dirname(out_prefix), "site_map.png")
        try:
            generate_and_plot_site_map(
                slab, site_map_path, symprec=symprec, mgr=mgr,
                title=f"Adsorption site map — {stage_type} ({mol.get_chemical_formula()})"
            )
            logger.info(f"  [SiteMap] Saved: {os.path.relpath(site_map_path)}")
        except Exception as _sm_exc:
            logger.warning(f"  [SiteMap] Could not generate site map: {_sm_exc}")

        with perf_stage(f"physi_candidates ({stage_type})"):
            phy_cands = mgr.generate_physisorption_candidates(
                mol, 
                height=physi_cfg.get("placement_height", 3.5), 
                tag=tag,
                n_rot=physi_cfg.get("n_rot", 32),
                rot_center=actual_center,
                height_mode=physi_cfg.get("height_mode", "clearance"),
                gravity_pull=physi_cfg.get("gravity_pull", {"enabled": False})
            )
        for c in phy_cands: c.info.setdefault("mechanism", "physisorption")
        all_cands.extend(phy_cands)
    
    if chem_cfg.get("enabled", False):
        logger.info(f"  [Stage: {stage_type}] Chemisorption search for {mol.get_chemical_formula()}...")
        with perf_stage(f"chem_candidates ({stage_type})"):
            chem_cands = build_chemisorption_structures(
                molecule=mol, center_target=actual_center, surface=slab, 
                rot_steps=chem_cfg.get("rot_steps", 8),
                config=config, tag=tag, results_dir=os.path.dirname(out_prefix)
            )
        for c in chem_cands: c.info["mechanism"] = "chemisorption"
        all_cands.extend(chem_cands)

    if all_cands:
        write(f"{out_prefix}_candidates.extxyz", all_cands)

    results = execute_verification_stage(all_cands, config, logger, out_prefix, tag=tag, e_gas=e_gas, e_base=e_base)
    if stage_type == "precursor":
        execute_ts_search_stage(results, config, logger, out_prefix)
    return results


# =============================================================================
# Main Workflow Coordinator
# =============================================================================

def execute_discovery_workflow(config, logger, slab, gas_energy_map=None, slab_base_energy=0.0):
    """Main workflow: inhibitor → precursor using a pre-prepared slab."""
    paths = config["paths"]
    mechs_cfg = config.get("reaction_search", {}).get("mechanisms", {})
    inh_cfg = mechs_cfg.get("inhibitor", {})
    
    precursor_file, inh_file = paths.get("precursor"), paths.get("inhibitor")
    out_dir = paths.get("output_prefix", "results")
    
    base_slabs = [slab.copy()]
    if inh_cfg.get("enabled", False) and inh_file:
        e_gas_inh = gas_energy_map.get(inh_file, 0.0) if gas_energy_map else calculate_gas_energy(read(inh_file), config, logger)
        inh_cands = execute_discovery_stage(slab, read(inh_file), config, os.path.join(out_dir, "stage1_inhibitor"), logger, 
                                            tag=2, center_target=inh_cfg.get("center", "O"),
                                            e_gas=e_gas_inh, e_base=slab_base_energy, stage_type="inhibitor")
        if inh_cands:
            inh_cands.sort(key=lambda x: x.info.get("e_final", 1e10))
            base_slabs = inh_cands[:inh_cfg.get("branching_limit", 1)]

    pre_cfg = mechs_cfg.get("precursor", {})
    pre_center = pre_cfg.get("center", "Si")
    
    mol = read(precursor_file) if precursor_file else None
    if mol:
        e_gas_mol = gas_energy_map.get(precursor_file, 0.0) if gas_energy_map else calculate_gas_energy(mol, config, logger)
        for i, s in enumerate(base_slabs):
            suffix = f"_branch{i}" if len(base_slabs) > 1 else ""
            execute_discovery_stage(s, mol, config, os.path.join(out_dir, f"stage2_precursor{suffix}"), logger, 
                                     tag=3, center_target=pre_center,
                                     e_gas=e_gas_mol, e_base=s.info.get("e_final", slab_base_energy), 
                                     stage_type="precursor")


def run_generic_adsorption_study(config_path="config.yaml"):
    """Top-level batch driver."""
    if isinstance(config_path, dict):
        config = config_path
    else:
        config = load_config(config_path)
    paths = config["paths"]
    
    pre_path_raw = paths.get("precursor")
    inh_path_raw = paths.get("inhibitor")
    
    # Auto-detect if singular precursor/inhibitor points to a directory
    is_pre_dir = pre_path_raw and os.path.exists(pre_path_raw) and os.path.isdir(pre_path_raw)
    is_inh_dir = inh_path_raw and os.path.exists(inh_path_raw) and os.path.isdir(inh_path_raw)
    
    if is_pre_dir or is_inh_dir:
        # Resolve precursors
        precursor_files = []
        if is_pre_dir:
            for f in sorted(os.listdir(pre_path_raw)):
                if f.endswith((".vasp", ".xyz", ".extxyz")):
                    precursor_files.append(os.path.abspath(os.path.join(pre_path_raw, f)))
        elif pre_path_raw:
            precursor_files.append(os.path.abspath(pre_path_raw))
            
        # Resolve inhibitors
        inhibitor_files = []
        if is_inh_dir:
            for f in sorted(os.listdir(inh_path_raw)):
                if f.endswith((".vasp", ".xyz", ".extxyz")):
                    inhibitor_files.append(os.path.abspath(os.path.join(inh_path_raw, f)))
        elif inh_path_raw:
            inhibitor_files.append(os.path.abspath(inh_path_raw))
            
        # Include baseline 'no inhibitor' if requested
        if paths.get("include_no_inhibitor", False) or not inhibitor_files:
            if None not in inhibitor_files:
                inhibitor_files.append(None)
                
        if not precursor_files:
            raise ValueError(f"No precursor files found in path: {pre_path_raw}")
            
        output_dir = paths.get("output_dir", "results")
        os.makedirs(output_dir, exist_ok=True)
        
        master_logger = setup_logger(log_path=os.path.join(output_dir, "batch_screening.log"), mode="w")
        master_logger.info(f"Starting batch screening: {len(precursor_files)} precursors x {len(inhibitor_files)} inhibitors")
        
        for prec_path in precursor_files:
            for inh_path in inhibitor_files:
                pre_name = os.path.splitext(os.path.basename(prec_path))[0]
                inh_name = os.path.splitext(os.path.basename(inh_path))[0] if inh_path else "None"
                
                pair_prefix = os.path.join(output_dir, f"{inh_name}_on_{pre_name}")
                master_logger.info(f"Running pair: {inh_name} on {pre_name} -> {pair_prefix}")
                
                # Clone the config and update for this pair
                pair_config = copy.deepcopy(config)
                pair_config["paths"]["precursor"] = prec_path
                pair_config["paths"]["inhibitor"] = inh_path
                pair_config["paths"]["output_prefix"] = pair_prefix
                
                try:
                    run_generic_adsorption_study(pair_config)
                except Exception as e:
                    master_logger.error(f"Failed running pair {inh_name} on {pre_name}: {e}", exc_info=True)
        return

    global_prefix = paths.get("output_prefix", "discovery")
    os.makedirs(global_prefix, exist_ok=True)
    master_logger = setup_logger(log_path=os.path.join(global_prefix, "master_workflow.log"), mode="w")

    # 1. Slab Preparation
    slab = prepare_slab_stage(config, master_logger)
    
    # 2. Slab Relaxation
    slab, slab_base_energy = relax_slab_stage(slab, config, master_logger)
    write(os.path.join(global_prefix, "prepared_slab.extxyz"), slab)

    # 3. Gas Phase (Simplified Batch)
    pre_path, inh_path = paths.get("precursor"), paths.get("inhibitor")
    gas_energy_map = {}
    for p in [pre_path, inh_path]:
        if p: gas_energy_map[p] = calculate_gas_energy(read(p), config, master_logger)

    # 4. Main Workflow
    execute_discovery_workflow(config, master_logger, slab, gas_energy_map, slab_base_energy)
