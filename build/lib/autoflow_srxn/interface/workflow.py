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

from autoflow_srxn.interface.lattice_match import (
    find_coincidences,
    miller_polar_inplane,
    polar_axis_for_sg,
    POLAR_SG,
)
from autoflow_srxn.interface.builder import (
    InterfaceCandidate,
    build_symmetric_slab,
    get_slab_atom_count,
    get_surface_lattice_2d,
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
            n_sub = get_slab_atom_count(self.sub, c.sub_miller, HNF=c.Na)
            n_film = get_slab_atom_count(self.film, c.film_miller, HNF=c.Nb)
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
