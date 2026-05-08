"""Mode-following structural refinement — molecule and slab.

Handles two structure types transparently:

  Molecule  ``paths.precursor``       — centered in vacuum, Full Hessian.
  Slab      ``paths.input_structure`` — periodic cell kept, FixAtoms applied
                                        to frozen zone, PHVA used.

Usage
-----
    python run_mode_following.py <config.yaml>

    # precursor example (DIPAS molecule):
    python run_mode_following.py precursor/config.yaml

    # slab example (DIPAS / SiO2):
    python run_mode_following.py slab/config.yaml
"""

from __future__ import annotations

import copy
import os
import sys

import numpy as np
import yaml
from ase.constraints import FixAtoms
from ase.io import read

# Resolve package root for direct execution
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from autoflow_srxn.analysis.vibrational_analyzer import MultiModeFollower, VibrationalAnalyzer
from autoflow_srxn.simulation.potentials import SimulationEngine
from autoflow_srxn.simulation.qpoint_handler import QPointParser
from autoflow_srxn.utils.logger_utils import setup_logger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_structure(config: dict, config_dir: str, logger) -> tuple:
    """Return (atoms, is_slab, frozen_idx | None)."""
    paths = config["paths"]

    # Discriminate molecule vs slab by config key
    if "precursor" in paths and "input_structure" not in paths:
        struct_path = paths["precursor"]
        is_molecule = True
    else:
        struct_path = paths["input_structure"]
        is_molecule = False

    if not os.path.isabs(struct_path):
        struct_path = os.path.join(config_dir, struct_path)
    if not os.path.exists(struct_path):
        raise FileNotFoundError(f"Structure not found: {struct_path}")

    atoms = read(struct_path)
    logger.info(f"  Loaded : {os.path.relpath(struct_path)} ({len(atoms)} atoms)")

    frozen_idx = None

    if is_molecule:
        # Isolated molecule: centre in a large vacuum box
        atoms.center(vacuum=10.0)
        cell = atoms.get_cell()
        logger.info(
            f"  Mode   : molecule  (cell {cell[0,0]:.1f} x "
            f"{cell[1,1]:.1f} x {cell[2,2]:.1f} Ang)"
        )
    else:
        # Periodic slab: apply FixAtoms for the frozen zone
        phva_cfg = config["analysis"]["vibrational"].get("phva", {})
        frozen_z = phva_cfg.get("frozen_z_ang") if phva_cfg.get("enabled") else None
        if frozen_z is not None:
            z_min = atoms.positions[:, 2].min()
            mask = atoms.positions[:, 2] < z_min + float(frozen_z)
            frozen_idx = list(np.where(mask)[0])
            atoms.set_constraint(FixAtoms(indices=frozen_idx))
            logger.info(
                f"  Mode   : slab  ({len(frozen_idx)} atoms frozen, "
                f"{len(atoms) - len(frozen_idx)} active, "
                f"frozen_z_ang={frozen_z})"
            )
        else:
            logger.info(f"  Mode   : slab  (no frozen zone)")

    return atoms, is_molecule, frozen_idx


def _run_vibration(atoms, engine, vib_name: str, u: float, cycle: int, logger):
    """Run vibrational analysis for *cycle*, save qpoints.yaml, return (parser, min_freq)."""
    # Each cycle gets its own cache dir to avoid stale data
    name = f"{vib_name}_c{cycle}"
    analyzer = VibrationalAnalyzer(atoms=atoms, engine=engine, displacement=u, name=name)
    # run_analysis() auto-saves qpoints.yaml to parent(name)/qpoints.yaml
    analyzer.run_analysis()

    # Determine where qpoints.yaml was saved
    parent = os.path.dirname(name) if os.path.dirname(name) else "."
    qpath = os.path.join(parent, "qpoints.yaml")

    parser = QPointParser(qpath)
    all_freqs = [b["frequency"] for phon in parser.data["phonon"] for b in phon["band"]]
    min_freq = min(all_freqs)
    n_imag = sum(1 for f in all_freqs if f < -0.1)

    energy = atoms.get_potential_energy()
    logger.info(
        f"  E = {energy:.6f} eV  |  min_freq = {min_freq:.4f} THz  "
        f"(imaginary < -0.1 THz: {n_imag})"
    )
    return parser, min_freq, energy, qpath


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_mode_following(config_path: str = "config.yaml"):
    config_path = os.path.abspath(config_path)
    config_dir  = os.path.dirname(config_path)

    # Work relative to the config file so all relative paths resolve correctly
    os.chdir(config_dir)

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # ── Parameters ────────────────────────────────────────────────────────────
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

    # ── Structure ────────────────────────────────────────────────────────────
    atoms, is_molecule, frozen_idx = _load_structure(config, config_dir, logger)

    # ── Engine ───────────────────────────────────────────────────────────────
    engine = SimulationEngine(config=config)

    # ── Initial relaxation ────────────────────────────────────────────────────
    logger.info(
        f"\nInitial relaxation  (optimizer={optimizer}, "
        f"fmax={fmax_rel}, steps={steps_rel})"
    )
    engine.relax(atoms, fmax=fmax_rel, steps=steps_rel, optimizer=optimizer)

    # ── Cycle 0 — baseline vibrational analysis ───────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("CYCLE 0 — initial vibrational analysis")
    logger.info("=" * 60)

    parser, min_freq, energy, qpath = _run_vibration(
        atoms, engine, vib_name, u, cycle=0, logger=logger
    )
    history = [{"cycle": 0, "energy": energy, "min_freq": min_freq, "alpha": 0.0}]

    if min_freq >= threshold:
        logger.info(
            f"  [Done] Structure already stable "
            f"(min_freq={min_freq:.4f} >= threshold={threshold} THz)"
        )
        max_iter = 0

    # ── Iterative mode-following ──────────────────────────────────────────────
    current_alpha = init_alpha

    for cycle in range(1, max_iter + 1):
        logger.info("\n" + "=" * 60)
        logger.info(f"CYCLE {cycle}/{max_iter}  (alpha={current_alpha:.3f} Ang)")
        logger.info("=" * 60)

        # Build iter config with current alpha
        iter_cfg = copy.deepcopy(config)
        (iter_cfg["analysis"]["vibrational"]
                  ["mode_refinement"]["perturbation_alpha"]) = current_alpha

        target_modes = [b for phon in parser.data["phonon"] for b in phon["band"]]

        # Mode-following perturbation + relaxation
        logger.info(f"  Mode-following perturbation...")
        follower = MultiModeFollower(engine, config=iter_cfg)
        atoms = follower.optimize(
            atoms,
            modes=target_modes,
            fmax=fmax_ref,
            steps=steps_ref,
            optimizer=optimizer,
        )

        # Re-apply FixAtoms if slab (copy() in MultiModeFollower preserves it,
        # but be defensive)
        if frozen_idx is not None and not any(
            isinstance(c, FixAtoms) for c in atoms.constraints
        ):
            atoms.set_constraint(FixAtoms(indices=frozen_idx))
            logger.info("  [FixAtoms] Re-applied frozen zone constraint.")

        # Vibrational analysis of new geometry
        logger.info(f"  Vibrational analysis (cycle {cycle})...")
        parser, min_freq, energy, qpath = _run_vibration(
            atoms, engine, vib_name, u, cycle=cycle, logger=logger
        )
        history.append(
            {"cycle": cycle, "energy": energy, "min_freq": min_freq, "alpha": current_alpha}
        )

        # Convergence check
        if min_freq >= threshold:
            logger.info(
                f"  [Converged] min_freq={min_freq:.4f} >= threshold={threshold} THz"
            )
            break

        # Stagnation detection
        improvement = min_freq - history[-2]["min_freq"]
        if improvement < stag_eps:
            new_alpha = current_alpha * stag_fac
            logger.warning(
                f"  [Stagnation] Improvement {improvement:.4f} THz < eps={stag_eps}. "
                f"alpha: {current_alpha:.3f} -> {new_alpha:.3f}"
            )
            current_alpha = new_alpha

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"{'Cycle':>5}  {'Energy (eV)':>14}  {'Min Freq (THz)':>14}  {'Alpha (Ang)':>10}")
    logger.info("-" * 55)
    for h in history:
        logger.info(
            f"{h['cycle']:5d}  {h['energy']:14.6f}  {h['min_freq']:14.4f}  {h['alpha']:10.3f}"
        )

    final_freq = history[-1]["min_freq"]
    if final_freq < threshold:
        logger.warning(
            f"\n[Warning] Final min_freq={final_freq:.4f} THz still below "
            f"threshold={threshold} THz."
        )
        if len(history) - 1 >= max_iter:
            logger.warning("  -> Consider increasing max_iter.")
        if abs(final_freq - history[0]["min_freq"]) < 0.1:
            logger.warning("  -> Little improvement: increase perturbation_alpha.")
    else:
        logger.info(f"\n[Success] Stable structure achieved.")

    # ── Save final structure ──────────────────────────────────────────────────
    out_vasp = f"{out_prefix}_final.vasp"
    # Remove constraints before writing so VASP file is clean
    atoms_out = atoms.copy()
    atoms_out.set_constraint()
    atoms_out.write(out_vasp)
    logger.info(f"Saved final structure: {os.path.relpath(out_vasp)}")

    # Keep the last qpoints.yaml as the canonical result
    import shutil
    final_qpoints = os.path.join(os.path.dirname(vib_name), "qpoints_final.yaml")
    if os.path.exists(qpath):
        shutil.copy2(qpath, final_qpoints)
        logger.info(f"Saved final qpoints : {os.path.relpath(final_qpoints)}")

    return atoms


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    run_mode_following(os.path.abspath(cfg))
