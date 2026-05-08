# Backward-compatible re-exports.
# All public symbols remain importable from the top-level package so that
# existing scripts (examples/, unittests/) continue to work without change.

# utils
from .utils.knowledge_engine import KnowledgeBase, chem_kb
from .utils.logger_utils import (
    setup_logger,
    get_workflow_logger,
    log_stage_title,
    log_energy_comparison,
    log_results_table,
)

# surface
from .surface.surface_utils import (
    standardize_vasp_atoms,
    write_standardized_vasp,
    find_surface_indices,
    calculate_haptic_vbs,
    calculate_haptic_normal,
    generate_vsepr_vectors,
    get_all_dangling_bonds_general,
    passivate_surface_coverage_general,
    identify_protectors,
    CavityDetector,
    create_slab_from_bulk,
    apply_surface_reconstruction,
    auto_reconstruct_surface,
    reconstruct_si100_2x1_buckled,
    oxidize_si_surface,
    build_si100_slab,
    generate_standard_surfaces,
    get_surface_h_mapping,
)
from .surface.ads_workflow_mgr import AdsorptionWorkflowManager
from .surface.chemisorption_builder import (
    analyze_surface_reactivity,
    analyze_molecule_ligands,
    build_chemisorption_structures,
)

# simulation
from .simulation.potentials import ZBLCalculator, ExplosionMonitor, SimulationEngine
from .simulation.thermo_engine import (
    ThermoCalculator,
    GasThermo,
    thz_to_cm1,
    thz_to_joule,
    eV_to_J_mol,
)
from .simulation.qpoint_handler import QPointParser

# vibrational
from .vibrational.vibrational_analyzer import (
    VibrationalAnalyzer,
    MultiModeFollower,
    GradientFlippingCalculator,
    AdaptiveGradientFlippingCalculator,
    TSSearcher,
    calculate_thermo,
    build_phva_active_indices,
    calculate_mac,
    calculate_atomic_participation,
)
from .vibrational.mode_following import run_mode_following
