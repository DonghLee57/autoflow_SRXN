"""Mode-following structural refinement workflow.

Provides the :func:`run_mode_following` entry point used by the
``examples/mode_following_relaxation/*/run_mode_following.py`` wrappers.

All behaviour is driven by ``config.yaml``; the same script is used for
isolated molecules and periodic slabs.

Config keys that control structure handling
-------------------------------------------
``paths.input_structure``
    Path to the input VASP file (relative to the config file).

``paths.center_in_vacuum``  (default: false)
    ``true``  – re-centre the structure in a 10 Å vacuum box before
               relaxation.  Use for isolated molecules.
    ``false`` – keep the cell exactly as read (periodic slabs, clusters …).

``analysis.vibrational.phva.enabled``  (default: false)
    ``true``  – Partial Hessian.  ``phva.frozen_z_ang`` determines the
               frozen zone.  A :class:`~ase.constraints.FixAtoms` constraint
               is automatically applied so frozen atoms stay fixed during
               every relaxation step.
    ``false`` – Full Hessian; no atoms are frozen.
"""

from __future__ import annotations

import copy
import os
import shutil

import numpy as np
import yaml
from ase.constraints import FixAtoms
from ase.io import read

from .vibrational_analyzer import MultiModeFollower, VibrationalAnalyzer
from ..simulation.potentials import SimulationEngine
from ..simulation.qpoint_handler import QPointParser
from ..utils.logger_utils import setup_logger


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_structure(config: dict, config_dir: str, engine, logger):
    """Load atoms and apply config-driven preprocessing (including slab generation)."""
    paths = config["paths"]
    struct_path = paths["input_structure"]
    if not os.path.isabs(struct_path):
        struct_path = os.path.join(config_dir, struct_path)
    if not os.path.exists(struct_path):
        raise FileNotFoundError(f"Structure not found: {struct_path}")

    atoms = read(struct_path)
    logger.info(f"  Loaded : {os.path.relpath(struct_path)} ({len(atoms)} atoms)")

    # 1. Optional Slab Generation from Bulk (Standardized to config_full.yaml)
    prep_cfg = config.get("surface_prep", {})
    gen_cfg = prep_cfg.get("slab_generation", {})
    if gen_cfg.get("enabled", False):
        from ..surface.surface_utils import create_slab_from_bulk
        miller = gen_cfg.get("miller", gen_cfg.get("miller_indices", (0,0,1)))
        logger.info(f"  [Workflow] Generating slab from bulk: miller={miller}")
        atoms = create_slab_from_bulk(
            atoms,
            miller_indices=miller,
            thickness=gen_cfg.get("thickness_ang", 15.0),
            vacuum=gen_cfg.get("vacuum_ang", 15.0),
            target_area=gen_cfg.get("target_area_ang2", 100.0)
        )
        logger.info(f"  [Workflow] Slab generated: {len(atoms)} atoms")

        # Optional Slab Relaxation
        if config.get("workflow", {}).get("slab_relax", False):
            logger.info("  [Workflow] Relaxing bare slab before vibrational analysis...")
            # Use top-level relaxation settings for slab relax
            engine.relax(atoms, verbose=False)

    # 2. Vacuum centering (isolated molecule)
    if paths.get("center_in_vacuum", False):
        atoms.center(vacuum=10.0)
        logger.info(f"  center_in_vacuum=true")

    # 3. FixAtoms for frozen zone (slab PHVA)
    frozen_idx = None
    phva_cfg = config["analysis"]["vibrational"].get("phva", {})
    frozen_z = phva_cfg.get("frozen_z_ang") if phva_cfg.get("enabled") else None

    if frozen_z is not None:
        z_min = atoms.positions[:, 2].min()
        mask = atoms.positions[:, 2] < z_min + float(frozen_z)
        frozen_idx = list(np.where(mask)[0])
        atoms.set_constraint(FixAtoms(indices=frozen_idx))
        n_active = len(atoms) - len(frozen_idx)
        logger.info(
            f"  phva.enabled=true, frozen_z_ang={frozen_z}  "
            f"({len(frozen_idx)} frozen, {n_active} active)"
        )

    return atoms, frozen_idx


def _run_vibration(atoms, engine, vib_name: str, u: float, cycle: int, logger):
    """Run one PHVA/FHVA cycle, auto-save qpoints.yaml, return (parser, stats)."""
    name = f"{vib_name}_c{cycle}"
    analyzer = VibrationalAnalyzer(atoms=atoms, engine=engine, displacement=u, name=name)
    analyzer.run_analysis()

    # run_analysis() writes qpoints.yaml to parent(name)/
    parent = os.path.dirname(name) if os.path.dirname(name) else "."
    qpath = os.path.join(parent, "qpoints.yaml")

    parser = QPointParser(qpath)
    all_freqs = [b["frequency"] for phon in parser.data["phonon"] for b in phon["band"]]
    min_freq = min(all_freqs)
    n_imag = sum(1 for f in all_freqs if f < -0.1)
    energy = atoms.get_potential_energy()

    logger.info(
        f"  E = {energy:.6f} eV  |  "
        f"min_freq = {min_freq:.4f} THz  (imaginary <-0.1 THz: {n_imag})"
    )
    return parser, min_freq, energy, qpath


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_mode_following(config_path: str = "config.yaml"):
    """Iterative mode-following structural refinement.

    Reads a YAML config, loads the structure, runs an initial relaxation,
    then repeatedly:

    1. Computes vibrational modes (PHVA or FHVA).
    2. If imaginary modes exist: applies a combined mode-following
       perturbation (:class:`~autoflow_srxn.vibrational.vibrational_analyzer.MultiModeFollower`)
       and re-relaxes.
    3. Repeats until the minimum frequency is above ``freq_threshold_thz``
       or ``max_iter`` is reached.

    Parameters
    ----------
    config_path : str
        Path to the YAML config file.  The working directory is changed to
        the config file's directory so that all relative paths resolve
        correctly.

    Returns
    -------
    atoms : ASE Atoms
        Final refined structure (constraints removed).
    """
    config_path = os.path.abspath(config_path)
    config_dir  = os.path.dirname(config_path)
    os.chdir(config_dir)

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Parameters
    vib_cfg  = config["analysis"]["vibrational"]
    mode_ref = vib_cfg.get("mode_refinement", {})
    rel_cfg  = config["engine"].get("relaxation", {})

    u          = vib_cfg.get("displacement_ang", 0.01)
    optimizer  = rel_cfg.get("optimizer", "FIRE")
    fmax_rel   = rel_cfg.get("fmax", 0.01)
    steps_rel  = rel_cfg.get("steps", 300)
    fmax_ref   = mode_ref.get("fmax", fmax_rel)
    steps_ref  = mode_ref.get("steps", steps_rel)
    max_iter   = mode_ref.get("max_iter", 5)
    threshold  = mode_ref.get("freq_threshold_thz", -0.1)
    stag_eps   = mode_ref.get("stagnation_epsilon", 0.05)
    stag_fac   = mode_ref.get("stagnation_factor", 0.5)
    init_alpha = mode_ref.get("perturbation_alpha", 0.3)
    vib_name   = vib_cfg.get("name", "results/vib")

    out_prefix = config["paths"].get("output_prefix", "results/output")
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)

    log_file = f"{out_prefix}.log"
    logger   = setup_logger(log_path=log_file, verbose=True)
    logger.info(f"Config : {os.path.relpath(config_path)}")
    logger.info(f"Output : {out_prefix}.*")

    engine = SimulationEngine(config=config)
    atoms, frozen_idx = _load_structure(config, config_dir, engine, logger)

    # Initial relaxation
    logger.info(
        f"\nInitial relaxation  "
        f"(optimizer={optimizer}, fmax={fmax_rel}, steps={steps_rel})"
    )
    engine.relax(atoms, fmax=fmax_rel, steps=steps_rel, optimizer=optimizer)

    # Cycle 0 — baseline vibrational analysis
    logger.info("\n" + "=" * 60)
    logger.info("CYCLE 0 - initial vibrational analysis")
    logger.info("=" * 60)
    parser, min_freq, energy, qpath = _run_vibration(
        atoms, engine, vib_name, u, cycle=0, logger=logger
    )
    history = [{"cycle": 0, "energy": energy, "min_freq": min_freq, "alpha": 0.0}]

    if min_freq >= threshold:
        logger.info(
            f"  [Done] Already stable "
            f"(min_freq={min_freq:.4f} >= threshold={threshold} THz)"
        )
        max_iter = 0

    # Iterative mode-following
    current_alpha = init_alpha

    for cycle in range(1, max_iter + 1):
        logger.info("\n" + "=" * 60)
        logger.info(f"CYCLE {cycle}/{max_iter}  (alpha={current_alpha:.3f} Ang)")
        logger.info("=" * 60)

        iter_cfg = copy.deepcopy(config)
        (iter_cfg["analysis"]["vibrational"]
                  ["mode_refinement"]["perturbation_alpha"]) = current_alpha

        target_modes = [b for phon in parser.data["phonon"] for b in phon["band"]]

        logger.info("  Mode-following perturbation + relaxation...")
        follower = MultiModeFollower(engine, config=iter_cfg)
        atoms = follower.optimize(
            atoms,
            modes=target_modes,
            fmax=fmax_ref,
            steps=steps_ref,
            optimizer=optimizer,
        )

        # Re-apply FixAtoms defensively (MultiModeFollower does atoms.copy())
        if frozen_idx is not None and not any(
            isinstance(c, FixAtoms) for c in atoms.constraints
        ):
            atoms.set_constraint(FixAtoms(indices=frozen_idx))
            logger.info("  [FixAtoms] Re-applied frozen zone constraint.")

        logger.info(f"  Vibrational analysis (cycle {cycle})...")
        parser, min_freq, energy, qpath = _run_vibration(
            atoms, engine, vib_name, u, cycle=cycle, logger=logger
        )
        history.append(
            {"cycle": cycle, "energy": energy, "min_freq": min_freq,
             "alpha": current_alpha}
        )

        if min_freq >= threshold:
            logger.info(
                f"  [Converged] min_freq={min_freq:.4f} >= threshold={threshold} THz"
            )
            break

        improvement = min_freq - history[-2]["min_freq"]
        if improvement < stag_eps:
            new_alpha = current_alpha * stag_fac
            logger.warning(
                f"  [Stagnation] improvement={improvement:.4f} < eps={stag_eps}. "
                f"alpha: {current_alpha:.3f} -> {new_alpha:.3f}"
            )
            current_alpha = new_alpha

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(
        f"{'Cycle':>5}  {'Energy (eV)':>14}  {'Min Freq (THz)':>14}  {'Alpha':>6}"
    )
    logger.info("-" * 50)
    for h in history:
        logger.info(
            f"{h['cycle']:5d}  {h['energy']:14.6f}  {h['min_freq']:14.4f}  "
            f"{h['alpha']:6.3f}"
        )

    final_freq = history[-1]["min_freq"]
    if final_freq < threshold:
        logger.warning(
            f"\n[Warning] Final min_freq={final_freq:.4f} THz < threshold={threshold}."
        )
        if len(history) - 1 >= max_iter:
            logger.warning("  -> Increase max_iter.")
        if abs(final_freq - history[0]["min_freq"]) < 0.1:
            logger.warning("  -> Little improvement: increase perturbation_alpha.")
    else:
        logger.info(f"\n[Success] Stable structure achieved.")

    # Save outputs
    out_vasp = f"{out_prefix}_final.vasp"
    atoms_out = atoms.copy()
    atoms_out.set_constraint()      # strip FixAtoms before writing
    atoms_out.write(out_vasp)
    logger.info(f"Saved final structure : {os.path.relpath(out_vasp)}")

    final_qpoints = os.path.join(os.path.dirname(vib_name), "qpoints_final.yaml")
    if os.path.exists(qpath):
        shutil.copy2(qpath, final_qpoints)
        logger.info(f"Saved final qpoints   : {os.path.relpath(final_qpoints)}")

    return atoms
