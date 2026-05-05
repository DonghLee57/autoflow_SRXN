"""
Interface Lattice-Match Search — standalone example script
===========================================================
Uses :class:`~autoflow_srxn.interface.InterfaceWorkflow` for 2D ZSL screening
and slab construction.

Output (written to ``output_dir`` defined in the config)
---------------------------------------------------------
* ``interface_summary.txt``      — plain-text candidate ranking table
* ``candidates.json``            — full candidate list (JSON)
* ``candidates.html``            — interactive Plotly dashboard
* ``interface_<N>.extxyz``       — stacked interface structure for candidate N

Usage
-----
    python run_interface.py              # reads config.yaml in this directory
    python run_interface.py config.yaml  # explicit config path
"""

from __future__ import annotations
import os
import sys
import yaml
import numpy as np
import ase
from ase.io import write as ase_write

from autoflow_srxn.logger_utils import setup_logger
from autoflow_srxn.interface import (
    InterfaceWorkflow,
    save_candidates_json,
    save_candidates_html,
)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    config_dir = os.path.dirname(os.path.abspath(config_path))
    for key in ("sub_path", "film_path"):
        val = cfg.get(key, "")
        if val and not os.path.isabs(val):
            candidate = os.path.join(config_dir, val)
            if os.path.exists(candidate):
                cfg[key] = candidate
    return cfg


# ---------------------------------------------------------------------------
# Candidate post-processing helpers
# ---------------------------------------------------------------------------

def _polar_ok(cand) -> bool:
    """Return True when neither polar-termination warning is present."""
    return not any(
        w in cand.notes
        for w in ("WARN:sub_polar_termination", "WARN:film_polar_termination")
    )


def _large_cell_warn(cand) -> bool:
    return any(w.startswith("WARN:large_cell") for w in cand.notes)


def _mark_recommended(candidates) -> list[bool]:
    """Return True for the first (lowest-vm) candidate per film_miller group."""
    seen: set = set()
    flags = []
    for c in candidates:
        key = c.film_miller
        if key not in seen:
            seen.add(key)
            flags.append(True)
        else:
            flags.append(False)
    return flags


# ---------------------------------------------------------------------------
# Miller index resolution
# ---------------------------------------------------------------------------

def _resolve_millers(
    explicit: list | None,
    max_m: int | None,
    structure,
    mode: str = "distinct",
) -> tuple[list[tuple], str]:
    """Resolve Miller indices from config, returning (miller_list, source_label)."""
    if explicit:
        millers = [tuple(int(x) for x in m) for m in explicit]
        return millers, f"explicit ({len(millers)} faces)"

    if max_m is not None:
        if mode == "distinct":
            from pymatgen.core.surface import get_symmetrically_distinct_miller_indices
            millers = [
                tuple(m)
                for m in get_symmetrically_distinct_miller_indices(structure, max_m)
            ]
            return millers, f"distinct  max_miller={max_m}  ({len(millers)} faces)"
        else:
            from math import gcd
            millers = []
            for h in range(0, max_m + 1):
                for k in range(0, max_m + 1):
                    for l in range(0, max_m + 1):
                        if h == k == l == 0: continue
                        if gcd(gcd(h, k), l) == 1:
                            millers.append((h, k, l))
            return millers, f"raw  max_miller={max_m}  ({len(millers)} faces)"

    default = [(0, 0, 1), (1, 1, 0), (1, 1, 1)]
    return default, "default (3 faces)"


# ---------------------------------------------------------------------------
# Interface stacking
# ---------------------------------------------------------------------------

def _stack_interface(
    sub_slab: ase.Atoms,
    film_slab: ase.Atoms,
    gap_ang: float = 2.5,
    vacuum_ang: float = 15.0,
) -> ase.Atoms:
    """Stack substrate and film slabs into a single interface Atoms object."""
    from ase import Atoms

    sub_cell = np.array(sub_slab.cell)
    film_cell = np.array(film_slab.cell)

    sub_c_z = sub_cell[2, 2]
    film_c_z = film_cell[2, 2]

    sub_frac = sub_slab.get_scaled_positions()
    film_frac = film_slab.get_scaled_positions()

    sub_z = sub_frac[:, 2] * sub_c_z
    film_z = film_frac[:, 2] * film_c_z

    sub_z_min, sub_z_max = sub_z.min(), sub_z.max()
    sub_thickness = sub_z_max - sub_z_min

    film_z_min, film_z_max = film_z.min(), film_z.max()
    film_thickness = film_z_max - film_z_min

    sub_z_shift = -sub_z_min
    sub_pos_new = sub_slab.get_positions().copy()
    sub_pos_new[:, 2] += sub_z_shift

    sub_cell_2d = sub_cell[:2, :2]
    film_cart_xy_new = film_frac[:, :2] @ sub_cell_2d
    film_z_shift = sub_thickness + gap_ang - film_z_min
    film_z_new = film_z + film_z_shift

    film_pos_new = np.column_stack([film_cart_xy_new, film_z_new])

    all_pos = np.vstack([sub_pos_new, film_pos_new])
    all_symbols = list(sub_slab.get_chemical_symbols()) + list(film_slab.get_chemical_symbols())
    all_tags = [0] * len(sub_slab) + [1] * len(film_slab)

    new_cell = sub_cell.copy()
    new_cell[2] = [0.0, 0.0, sub_thickness + gap_ang + film_thickness + vacuum_ang]

    return Atoms(
        symbols=all_symbols,
        positions=all_pos,
        cell=new_cell,
        pbc=[True, True, True],
        tags=all_tags,
    )


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def run_interface_search(config_path: str = "config.yaml") -> None:
    cfg = load_config(config_path)
    out_dir = cfg.get("out_dir", ".")
    os.makedirs(out_dir, exist_ok=True)

    logger = setup_logger(os.path.join(out_dir, "interface_match.log"))

    try:
        from pymatgen.core import Structure
    except ImportError:
        logger.error("pymatgen is required for interface matching.")
        sys.exit(1)

    sub_path, film_path = cfg.get("sub_path", ""), cfg.get("film_path", "")
    for label, path in [("Substrate", sub_path), ("Film", film_path)]:
        if not os.path.exists(path):
            logger.error(f"{label} bulk file not found: {path!r}")
            sys.exit(1)

    logger.info("Loading substrate bulk: %s", os.path.relpath(sub_path))
    sub_struct = Structure.from_file(sub_path)
    logger.info("Loading film bulk:      %s", os.path.relpath(film_path))
    film_struct = Structure.from_file(film_path)

    sub_sg_sym, sub_sg_num = sub_struct.get_space_group_info()
    film_sg_sym, film_sg_num = film_struct.get_space_group_info()
    sub_name = cfg.get("sub_name") or sub_struct.formula.replace(" ", "")
    film_name = cfg.get("film_name") or film_struct.formula.replace(" ", "")

    from autoflow_srxn.interface import POLAR_SG
    sub_polar = sub_sg_num in POLAR_SG
    film_polar = film_sg_num in POLAR_SG

    logger.info("  Substrate: %s  SG#%d %s  polar=%s", sub_name, sub_sg_num, sub_sg_sym, sub_polar)
    logger.info("  Film:      %s  SG#%d %s  polar=%s", film_name, film_sg_num, film_sg_sym, film_polar)

    mode = cfg.get("miller_mode", "distinct")
    sub_millers, sub_src = _resolve_millers(cfg.get("sub_millers"), cfg.get("sub_max_miller"), sub_struct, mode)
    film_millers, film_src = _resolve_millers(cfg.get("film_millers"), cfg.get("film_max_miller"), film_struct, mode)

    logger.info("  Sub  Miller indices [%s]: %s", sub_src, sub_millers)
    logger.info("  Film Miller indices [%s]: %s", film_src, film_millers)

    wf = InterfaceWorkflow(
        sub_structure=sub_struct,
        film_structure=film_struct,
        sub_millers=sub_millers,
        film_millers=film_millers,
        max_det=cfg.get("max_det", 6),
        strain_cutoff=cfg.get("strain_cutoff", 0.05),
        max_atoms=cfg.get("max_atoms", 400),
        min_slab_thickness=cfg.get("min_slab_thickness", 12.0),
        vacuum=cfg.get("vacuum", 15.0),
    )

    logger.info("Screening candidates...")
    candidates = wf.screen()
    if not candidates:
        logger.warning("No candidates found.")
        return

    logger.info("Found %d candidates.", len(candidates))
    recommended_flags = _mark_recommended(candidates)

    summary = wf.summary(candidates, top_n=20)
    summary_path = os.path.join(out_dir, "interface_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write(summary)
    logger.info("Summary written: %s", os.path.relpath(summary_path))

    json_path = os.path.join(out_dir, "candidates.json")
    save_candidates_json(candidates, recommended_flags, sub_name, film_name, json_path)
    logger.info("JSON written:    %s", os.path.relpath(json_path))

    html_path = os.path.join(out_dir, "candidates.html")
    save_candidates_html(candidates, recommended_flags, sub_name, film_name, f"#{sub_sg_num}", f"#{film_sg_num}", sub_polar, film_polar, html_path)
    logger.info("HTML written:    %s", os.path.relpath(html_path))

    raw_k = cfg.get("build_top_k")
    if raw_k is None or (isinstance(raw_k, int) and raw_k < 0):
        build_top_k = len(candidates)
        logger.info("build_top_k is null or negative: Building ALL %d candidates.", build_top_k)
    else:
        build_top_k = int(raw_k)
    interface_gap = float(cfg.get("interface_gap", 2.5))
    vacuum_ang = float(cfg.get("vacuum", 15.0))

    for idx, (cand, is_rec) in enumerate(zip(candidates[:build_top_k], recommended_flags[:build_top_k])):
        logger.info("  [%d/%d] sub%s | film%s  vm=%.4f", idx + 1, build_top_k, cand.sub_miller, cand.film_miller, cand.vm)
        try:
            sub_slab, film_slab = wf.build(cand)
            interface = _stack_interface(sub_slab, film_slab, gap_ang=interface_gap, vacuum_ang=vacuum_ang)
            iface_out = os.path.join(out_dir, f"interface_{idx}.extxyz")
            ase_write(iface_out, interface)
            logger.info("      Saved: %s (%d atoms)", os.path.relpath(iface_out), len(interface))
        except Exception as e:
            logger.error("      Build failed: %s", e)

    logger.info("Done. All output in: %s", os.path.relpath(out_dir))


if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    if not os.path.exists(config_file):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(script_dir, "config.yaml")
    run_interface_search(config_file)
