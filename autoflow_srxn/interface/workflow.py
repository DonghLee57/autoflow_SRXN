"""
autoflow_srxn.interface.workflow
==================================
High-level InterfaceWorkflow class.

Orchestrates coincidence lattice screening and slab construction for
heteroepitaxial interface models.

Requires **pymatgen**.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

try:
    from pymatgen.core import Structure
except ImportError as e:
    raise ImportError(
        "autoflow_srxn.interface.workflow requires pymatgen. "
        f"(Original error: {e})"
    ) from e

from autoflow_srxn.interface.builder import (
    InterfaceCandidate,
    build_symmetric_slab,
    find_coincidences,
    get_slab_atom_count,
    get_surface_lattice_2d,
    miller_polar_inplane,
    polar_axis_for_sg,
    POLAR_SG,
)

logger = logging.getLogger(__name__)


class InterfaceWorkflow:
    """Screen and build symmetric interface models.

    Parameters
    ----------
    sub_structure:
        Bulk pymatgen Structure of the substrate.
    film_structure:
        Bulk pymatgen Structure of the film.
    sub_millers:
        List of Miller index tuples to consider for the substrate.
    film_millers:
        List of Miller index tuples to consider for the film.
    max_det:
        Maximum HNF determinant (supercell size) for each material.
    strain_cutoff:
        Maximum von Mises strain to accept (fraction, e.g. 0.05 = 5 %).
    max_atoms:
        Maximum atom count of the combined interface cell.  Candidates
        exceeding this limit are flagged (not rejected).
    min_slab_thickness:
        Minimum slab thickness in Angstrom for slab construction.
    vacuum:
        Vacuum layer in Angstrom.
    """

    def __init__(
        self,
        sub_structure: Structure,
        film_structure: Structure,
        sub_millers: Sequence[tuple[int, int, int]] | None = None,
        film_millers: Sequence[tuple[int, int, int]] | None = None,
        max_det: int = 6,
        strain_cutoff: float = 0.05,
        max_atoms: int = 300,
        min_slab_thickness: float = 12.0,
        vacuum: float = 15.0,
    ) -> None:
        self.sub = sub_structure
        self.film = film_structure
        self.sub_millers = sub_millers or [(0, 0, 1), (1, 1, 0), (1, 1, 1)]
        self.film_millers = film_millers or [(0, 0, 1), (1, 1, 0), (1, 1, 1)]
        self.max_det = max_det
        self.strain_cutoff = strain_cutoff
        self.max_atoms = max_atoms
        self.min_slab_thickness = min_slab_thickness
        self.vacuum = vacuum

        # Polarity checks
        self._sub_sg: int = sub_structure.get_space_group_info()[1]
        self._film_sg: int = film_structure.get_space_group_info()[1]
        self._sub_polar_axis = polar_axis_for_sg(self._sub_sg)
        self._film_polar_axis = polar_axis_for_sg(self._film_sg)

    # ------------------------------------------------------------------
    def screen(self) -> list[InterfaceCandidate]:
        """Run coincidence lattice screening over all Miller combinations.

        Returns
        -------
        list[InterfaceCandidate]
            Sorted by ascending von Mises strain.
        """
        candidates: list[InterfaceCandidate] = []

        for sub_m in self.sub_millers:
            A_sub = get_surface_lattice_2d(self.sub, sub_m)
            for film_m in self.film_millers:
                A_film = get_surface_lattice_2d(self.film, film_m)
                matches = find_coincidences(
                    A_sub, A_film,
                    max_det=self.max_det,
                    strain_cutoff=self.strain_cutoff,
                )
                for m in matches:
                    notes: list[str] = []

                    # Polarity check
                    if not miller_polar_inplane(sub_m, self._sub_polar_axis):
                        notes.append("WARN:sub_polar_termination")
                    if not miller_polar_inplane(film_m, self._film_polar_axis):
                        notes.append("WARN:film_polar_termination")

                    n_sub = get_slab_atom_count(
                        self.sub, sub_m,
                        min_thickness_ang=self.min_slab_thickness,
                        HNF=m["Na"],
                    )
                    n_film = get_slab_atom_count(
                        self.film, film_m,
                        min_thickness_ang=self.min_slab_thickness,
                        HNF=m["Nb"],
                    )
                    n_total = n_sub + n_film
                    if n_total > self.max_atoms:
                        notes.append(f"WARN:large_cell({n_total}_atoms)")

                    candidates.append(
                        InterfaceCandidate(
                            sub_miller=sub_m,
                            film_miller=film_m,
                            Na=m["Na"],
                            Nb=m["Nb"],
                            eps1=m["eps1"],
                            eps2=m["eps2"],
                            vm=m["vm"],
                            n_atoms=n_total,
                            notes=notes,
                        )
                    )
                    logger.debug(
                        "Candidate sub%s|film%s vm=%.3f n=%d",
                        sub_m, film_m, m["vm"], n_total,
                    )

        candidates.sort(key=lambda c: c.vm)
        logger.info("Screen complete: %d candidates found.", len(candidates))
        return candidates

    # ------------------------------------------------------------------
    def build(self, candidate: InterfaceCandidate) -> tuple:
        """Build substrate and film slab ASE Atoms objects for *candidate*.

        Returns
        -------
        tuple[ase.Atoms, ase.Atoms]
            ``(sub_slab, film_slab)`` — each slab individually relaxed
            and centred; vacuum already applied.
        """
        sub_slab = build_symmetric_slab(
            self.sub,
            candidate.sub_miller,
            min_thickness_ang=self.min_slab_thickness,
            vacuum_ang=self.vacuum,
            HNF=candidate.Na,
        )
        film_slab = build_symmetric_slab(
            self.film,
            candidate.film_miller,
            min_thickness_ang=self.min_slab_thickness,
            vacuum_ang=self.vacuum,
            HNF=candidate.Nb,
        )
        return sub_slab, film_slab

    # ------------------------------------------------------------------
    def summary(self, candidates: list[InterfaceCandidate], top_n: int = 10) -> str:
        """Return a human-readable summary table of the top candidates.

        Parameters
        ----------
        candidates:
            Output of :meth:`screen`.
        top_n:
            Number of candidates to include.
        """
        lines = [
            f"{'#':>3}  {'sub':>9}  {'film':>9}  "
            f"{'vm%':>6}  {'eps1%':>7}  {'eps2%':>7}  "
            f"{'NatSub':>7}  {'NatFil':>7}  {'notes'}",
            "-" * 80,
        ]
        for i, c in enumerate(candidates[:top_n]):
            n_sub = get_slab_atom_count(
                self.sub, c.sub_miller,
                min_thickness_ang=self.min_slab_thickness, HNF=c.Na,
            )
            n_film = get_slab_atom_count(
                self.film, c.film_miller,
                min_thickness_ang=self.min_slab_thickness, HNF=c.Nb,
            )
            notes_str = ", ".join(c.notes) if c.notes else "OK"
            lines.append(
                f"{i + 1:>3}  "
                f"{'({},{},{})'.format(*c.sub_miller):>9}  "
                f"{'({},{},{})'.format(*c.film_miller):>9}  "
                f"{c.vm * 100:>6.2f}  "
                f"{c.eps1 * 100:>7.2f}  "
                f"{c.eps2 * 100:>7.2f}  "
                f"{n_sub:>7}  "
                f"{n_film:>7}  "
                f"{notes_str}"
            )
        return "\n".join(lines)


def run_interface_screening(config_dict: dict) -> None:
    """Run the complete interface screening workflow based on a configuration dictionary.

    Parameters
    ----------
    config_dict:
        A dictionary containing the workflow configuration.
        Keys: sub_path, film_path, sub_name, film_name, miller_mode,
              sub_millers, sub_max_miller, film_millers, film_max_miller,
              max_det, strain_cutoff, max_atoms, min_slab_thickness,
              vacuum, interface_gap, build_top_k, output_dir.
    """
    import os
    import sys
    from ase.io import write as ase_write
    from autoflow_srxn.utils import setup_logger
    from autoflow_srxn.interface import (
        InterfaceWorkflow,
        save_candidates_json,
        save_candidates_html,
        resolve_millers,
        stack_interface,
        mark_recommended_candidates,
        POLAR_SG,
    )
    from pymatgen.core import Structure

    cfg = config_dict
    out_dir = cfg.get("output_dir", ".")
    os.makedirs(out_dir, exist_ok=True)

    logger = setup_logger(os.path.join(out_dir, "interface_match.log"))

    sub_path, film_path = cfg.get("sub_path", ""), cfg.get("film_path", "")
    for label, path in [("Substrate", sub_path), ("Film", film_path)]:
        if not path or not os.path.exists(path):
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

    sub_polar = sub_sg_num in POLAR_SG
    film_polar = film_sg_num in POLAR_SG

    logger.info(
        "  Substrate: %s  SG#%d %s  polar=%s", sub_name, sub_sg_num, sub_sg_sym, sub_polar
    )
    logger.info(
        "  Film:      %s  SG#%d %s  polar=%s", film_name, film_sg_num, film_sg_sym, film_polar
    )

    mode = cfg.get("miller_mode", "distinct")
    sub_millers, sub_src = resolve_millers(
        cfg.get("sub_millers"), cfg.get("sub_max_miller"), sub_struct, mode
    )
    film_millers, film_src = resolve_millers(
        cfg.get("film_millers"), cfg.get("film_max_miller"), film_struct, mode
    )

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
    recommended_flags = mark_recommended_candidates(candidates)

    summary = wf.summary(candidates, top_n=20)
    summary_path = os.path.join(out_dir, "interface_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write(summary)
    logger.info("Summary written: %s", os.path.relpath(summary_path))

    json_path = os.path.join(out_dir, "candidates.json")
    save_candidates_json(candidates, recommended_flags, sub_name, film_name, json_path)
    logger.info("JSON written:    %s", os.path.relpath(json_path))

    html_path = os.path.join(out_dir, "candidates.html")
    save_candidates_html(
        candidates,
        recommended_flags,
        sub_name,
        film_name,
        f"#{sub_sg_num}",
        f"#{film_sg_num}",
        sub_polar,
        film_polar,
        html_path,
    )
    logger.info("HTML written:    %s", os.path.relpath(html_path))

    raw_k = cfg.get("build_top_k")
    if raw_k is None or (isinstance(raw_k, int) and raw_k < 0):
        build_top_k = len(candidates)
        logger.info(
            "build_top_k is null or negative: Building ALL %d candidates.", build_top_k
        )
    else:
        build_top_k = int(raw_k)
    interface_gap = float(cfg.get("interface_gap", 2.5))
    vacuum_ang = float(cfg.get("vacuum", 15.0))

    for idx, (cand, is_rec) in enumerate(
        zip(candidates[:build_top_k], recommended_flags[:build_top_k])
    ):
        logger.info(
            "  [%d/%d] sub%s | film%s  vm=%.4f",
            idx + 1,
            build_top_k,
            cand.sub_miller,
            cand.film_miller,
            cand.vm,
        )
        try:
            sub_slab, film_slab = wf.build(cand)
            interface = stack_interface(
                sub_slab, film_slab, gap_ang=interface_gap, vacuum_ang=vacuum_ang
            )
            iface_out = os.path.join(out_dir, f"interface_{idx}.extxyz")
            ase_write(iface_out, interface)
            logger.info(
                "      Saved: %s (%d atoms)", os.path.relpath(iface_out), len(interface)
            )
        except Exception as e:
            logger.error("      Build failed: %s", e)

    logger.info("Done. All output in: %s", os.path.relpath(out_dir))
