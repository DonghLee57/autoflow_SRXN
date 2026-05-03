import os
import sys
import copy
import yaml
import numpy as np
from ase.io import read, write
from ase import Atoms

from autoflow_srxn.ads_workflow_mgr import AdsorptionWorkflowManager
from autoflow_srxn.chemisorption_builder import build_chemisorption_structures
from autoflow_srxn.logger_utils import log_energy_comparison, log_results_table, log_stage_title, setup_logger
from autoflow_srxn.surface_utils import (
    create_slab_from_bulk,
    passivate_surface_coverage_general,
    standardize_vasp_atoms,
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

def execute_verification_stage(candidates, config, logger, out_prefix, tag=3, e_gas=0.0, e_base=0.0):
    """Performs Relaxation, Equilibration (MD), and Optional Post-Relax on candidates.
    Enforces strict structural standardization (Sorting, Z=0.5).
    """
    verify_cfg = config.get("verification", {})
    run_relax = verify_cfg.get("relaxation", {}).get("enabled", False)
    run_equil = verify_cfg.get("equilibration", {}).get("enabled", False)
    run_post = verify_cfg.get("equilibration", {}).get("post_relax", True) if run_equil else False

    if not candidates:
        return []

    from autoflow_srxn.potentials import SimulationEngine
    sel_idx = verify_cfg.get("selected_indices", None)
    if isinstance(sel_idx, str):
        try:
            allowed_names = {"range": range, "list": list, "np": np, "numpy": np, "abs": abs}
            sel_idx = eval(sel_idx, {"__builtins__": {}}, allowed_names)
            if hasattr(sel_idx, "tolist"): sel_idx = sel_idx.tolist()
            elif not isinstance(sel_idx, list): sel_idx = list(sel_idx)
        except Exception as e:
            logger.error(f"  [Verification] Failed to evaluate 'selected_indices': {e}")
            sel_idx = None

    n_total = len(candidates)
    n_target = len(sel_idx) if sel_idx is not None else n_total
    log_stage_title(logger, "VERIFICATION", f"Processing {n_target}/{n_total} sites")
    
    logger.info(f"VERIFICATION: Processing {len(candidates)} sites")
    candidates = candidates[:2] # TEMPORARY LIMIT FOR DEBUGGING
    results = []

    engine = SimulationEngine(config)
    calc = engine.get_calculator()
    processed_cands = []
    summary_data = []
    csv_rows = []

    for i, atoms in enumerate(candidates):
        if sel_idx is not None and i not in sel_idx:
            continue

        # Start from a standardized copy
        atoms_proc = standardize_vasp_atoms(atoms.copy(), z_min_offset=0.5)
        atoms_proc.info = atoms.info.copy()
        atoms_proc.calc = calc

        try:
            e_init = atoms_proc.get_potential_energy()

            # --- 1. Initial Relaxation ---
            if run_relax:
                r_cfg = verify_cfg.get("relaxation", {})
                engine.relax(atoms_proc, steps=r_cfg.get("steps", 50), fmax=r_cfg.get("fmax", 0.05), verbose=False)

            # --- 2. Thermal Equilibration (MD) ---
            if run_equil:
                e_cfg = verify_cfg.get("equilibration", {})
                engine.run_md(atoms_proc, temp_K=e_cfg.get("temperature_K", 300), md_steps=e_cfg.get("md_steps", 1000))
                if run_post:
                    engine.relax(atoms_proc, steps=50, fmax=0.05, verbose=False)

            # Final standardization after relaxation/MD
            atoms_proc = standardize_vasp_atoms(atoms_proc, z_min_offset=0.5)
            e_final = atoms_proc.get_potential_energy()
            delta = e_final - e_init
            e_ads = e_final - (e_gas + e_base)
            mech = atoms.info.get("mechanism", "unknown")

            summary_data.append({
                "id": i, "mech": mech, 
                "e_initial": e_init, "e_final": e_final, 
                "delta": delta, "e_ads": e_ads
            })
            csv_rows.append({
                "tag": tag, "id": i, "mechanism": mech, 
                "e_initial": e_init, "e_final": e_final, 
                "delta": delta, "e_ads": e_ads
            })
            
            atoms_proc.info["e_initial"] = e_init
            atoms_proc.info["e_final"] = e_final
            atoms_proc.info["delta"] = delta
            atoms_proc.info["e_ads"] = e_ads
            processed_cands.append(atoms_proc)

        except Exception as e:
            logger.error(f"  [Verification] Candidate {i} failed: {e}")

    log_results_table(logger, summary_data, title=f"Verification Summary (tag={tag})")
    csv_path = os.path.join(os.path.dirname(out_prefix), "energylog.csv")
    log_to_csv(csv_path, csv_rows)

    if processed_cands:
        write(f"{out_prefix}_relaxed.extxyz", processed_cands)
    return processed_cands

def execute_discovery_stage(slab, mol, config, out_prefix, logger, tag=2, center_target="Si", e_gas=0.0, e_base=0.0, stage_type="precursor"):
    """Orchestrates candidate generation and subsequent verification."""
    rs_cfg = config.get("reaction_search", {})
    mechs_cfg = rs_cfg.get("mechanisms", {})
    
    # --- Strict Stage-Specific Mechanism Logic ---
    # Only look at settings under inhibition or precursor
    stage_cfg = mechs_cfg.get(stage_type, {})
    
    physi_cfg = stage_cfg.get("physisorption", {"enabled": False})
    chem_cfg = stage_cfg.get("chemisorption", {"enabled": False})
    
    symprec = rs_cfg.get("candidate_filter", {}).get("symprec", 0.2)

    mgr = AdsorptionWorkflowManager(slab, config=config, symprec=symprec, verbose=True)
    all_cands = []

    if physi_cfg.get("enabled", False):
        logger.info(f"  [Stage: {stage_type}] Physisorption search for {mol.get_chemical_formula()}...")
        phy_cands = mgr.generate_physisorption_candidates(
            mol, height=physi_cfg.get("placement_height", 3.5), tag=tag
        )
        for c in phy_cands: c.info["mechanism"] = "physisorption"
        all_cands.extend(phy_cands)

    if chem_cfg.get("enabled", True):
        logger.info(f"  [Stage: {stage_type}] Chemisorption search for {mol.get_chemical_formula()} (center={center_target})...")
        chem_cands = build_chemisorption_structures(
            molecule=mol, center_target=center_target, surface=slab, config=config, tag=tag
        )
        for c in chem_cands: c.info["mechanism"] = "chemisorption"
        all_cands.extend(chem_cands)

    if all_cands:
        write(f"{out_prefix}_candidates.extxyz", all_cands)
    
    return execute_verification_stage(all_cands, config, logger, out_prefix, tag=tag, e_gas=e_gas, e_base=e_base)

def execute_discovery_workflow(config, logger, gas_energy_map=None, slab_base_energy=0.0):
    """Rigorous Workflow Implementation: Step-by-Step Relaxation and Standardization."""
    paths = config["paths"]
    sp_cfg = config.get("surface_prep", {})
    rs_cfg = config.get("reaction_search", {})
    mechs_cfg = rs_cfg.get("mechanisms", {})
    # Use 'inhibitor' instead of 'inhibition'
    inh_cfg = mechs_cfg.get("inhibitor", {})

    mol_file = paths.get("adsorbate")
    inh_file = paths.get("inhibitor")
    out_dir = paths.get("output_prefix", "results") # Re-purposing as output base
    mol = read(mol_file) if mol_file and os.path.exists(mol_file) else None

    # --- 1. Slab Generation & Standardization ---
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
        slab = standardize_vasp_atoms(read(paths["substrate_slab"]), z_min_offset=0.5)
    
    # Reset all substrate atoms to tag 0 to avoid mis-identification as inhibitors/adsorbates
    slab.set_tags(0)

    # --- 2. Rigorous Slab Relaxation ---
    slab_relax_cfg = sp_cfg.get("slab_relaxation", {})
    if slab_relax_cfg.get("enabled", False):
        from autoflow_srxn.potentials import SimulationEngine
        log_stage_title(logger, "STAGE 0.5", "Performing slab relaxation...")
        engine = SimulationEngine(config)
        slab.calc = engine.get_calculator()
        e_init = slab.get_potential_energy()
        engine.relax(slab, fmax=slab_relax_cfg.get("fmax", 0.05), steps=200, frozen_z_ang=slab_relax_cfg.get("frozen_z_ang"))
        slab = standardize_vasp_atoms(slab, z_min_offset=0.5)
        slab_base_energy = slab.get_potential_energy()
        log_energy_comparison(logger, "Slab Relax", e_init, slab_base_energy)
    else:
        slab = standardize_vasp_atoms(slab, z_min_offset=0.5)
        if not slab_base_energy:
            from autoflow_srxn.potentials import SimulationEngine
            engine = SimulationEngine(config)
            slab.calc = engine.get_calculator()
            slab_base_energy = slab.get_potential_energy()

    # --- 3. Inhibitor Discovery (Stage 1) ---
    base_slabs = [slab]
    if inh_cfg.get("enabled", False) and inh_file and os.path.exists(inh_file):
        log_stage_title(logger, "STAGE 1", f"Inhibitor Discovery ({os.path.basename(inh_file)})")
        e_gas_inh = gas_energy_map.get(inh_file, 0.0) if gas_energy_map else calculate_gas_energy(read(inh_file), config, logger)
        
        # Use stage-specific settings for inhibitor
        inh_center = inh_cfg.get("center", "O")
        inh_cands = execute_discovery_stage(
            slab, read(inh_file), config, os.path.join(out_dir, "stage1_inhibitor"), logger, tag=2,
            center_target=inh_center, e_gas=e_gas_inh, e_base=slab_base_energy,
            stage_type="inhibitor"
        )
        if inh_cands:
            inh_cands.sort(key=lambda x: x.info.get("e_final", 1e10))
            base_slabs = inh_cands[:inh_cfg.get("branching_limit", 3)]
            logger.info(f"  Selected top {len(base_slabs)} inhibited surfaces for Stage 2.")

    # --- 4. Precursor Discovery (Stage 2) ---
    if mol:
        log_stage_title(logger, "STAGE 2", f"Main Precursor Discovery ({os.path.basename(mol_file)})")
        e_gas_mol = gas_energy_map.get(mol_file, 0.0) if gas_energy_map else calculate_gas_energy(mol, config, logger)
        all_final_results = []
        
        # Precursor-specific settings
        pre_cfg = mechs_cfg.get("precursor", {})
        pre_center = pre_cfg.get("center", "Si")
        
        for i, s in enumerate(base_slabs):
            e_base_s2 = s.info.get("e_final", s.get_potential_energy())
            suffix = f"_branch{i}" if len(base_slabs) > 1 else ""
            results = execute_discovery_stage(
                s, mol, config, os.path.join(out_dir, f"stage2_precursor{suffix}"), logger, tag=3,
                center_target=pre_center,
                e_gas=e_gas_mol, e_base=e_base_s2,
                stage_type="precursor"
            )
            for r in results: r.info["inh_id"] = i
            all_final_results.extend(results)

        if all_final_results:
            write(os.path.join(out_dir, "final_results.extxyz"), all_final_results)

def run_generic_adsorption_study(config_path="config.yaml"):
    config = load_config(config_path)
    paths = config["paths"]
    
    def get_files(p):
        if not p: return [None]
        if os.path.isdir(p):
            import glob
            files = []
            for ext in ["*.vasp", "*.xyz", "*.extxyz"]: files.extend(glob.glob(os.path.join(p, ext)))
            return sorted(files)
        return [p]

    adsorbates = get_files(paths.get("adsorbate"))
    inhibitors = get_files(paths.get("inhibitor"))
    if paths.get("include_no_inhibitor", False): inhibitors = [None] + inhibitors
    elif not inhibitors: inhibitors = [None]

    global_prefix = paths.get("output_prefix", "discovery")
    unique_mols = list(set([f for f in adsorbates + inhibitors if f and os.path.exists(f)]))
    gas_energy_map = {}
    if unique_mols:
        tmp_logger = setup_logger(log_path=os.path.join(global_prefix, "ref_energies.log"), mode="w")
        for m_path in unique_mols:
            gas_energy_map[m_path] = calculate_gas_energy(read(m_path), config, tmp_logger)

    for inh_path in inhibitors:
        for ads_path in adsorbates:
            if not ads_path: continue
            inh_name = os.path.splitext(os.path.basename(inh_path))[0] if inh_path else "clean"
            ads_name = os.path.splitext(os.path.basename(ads_path))[0]
            
            # New naming convention: {inhibitor}_pretreated_{precursor}
            run_name = f"{inh_name}_pretreated_{ads_name}"
            run_dir = os.path.join(global_prefix, run_name)
            os.makedirs(run_dir, exist_ok=True)
            
            logger = setup_logger(log_path=os.path.join(run_dir, "workflow.log"), mode="w")
            log_stage_title(logger, "BATCH RUN", f"Sequence: {inh_name} -> {ads_name}")
            
            run_config = copy.deepcopy(config)
            run_config["paths"]["adsorbate"] = ads_path
            run_config["paths"]["inhibitor"] = inh_path
            run_config["paths"]["output_prefix"] = run_dir # Pass run_dir as the base for files
            
            try:
                execute_discovery_workflow(run_config, logger, gas_energy_map=gas_energy_map)
            except Exception as e:
                logger.error(f"Discovery workflow failed for {run_name}: {e}")

if __name__ == "__main__":
    run_generic_adsorption_study(sys.argv[1] if len(sys.argv) > 1 else "config.yaml")
