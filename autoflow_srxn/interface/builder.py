"""
autoflow_srxn.interface.builder
=================================
ASE/pymatgen-based symmetric slab and interface builder.

Requires **pymatgen**.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

try:
    from pymatgen.core import Structure
    from pymatgen.core.surface import SlabGenerator
    from pymatgen.io.ase import AseAtomsAdaptor
except ImportError as e:
    raise ImportError(
        "autoflow_srxn.interface.builder requires pymatgen. "
        f"(Original error: {e})"
    ) from e


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class InterfaceCandidate:
    """Holds a screened coincidence-lattice match between substrate and film.

    Attributes
    ----------
    sub_miller, film_miller:
        Miller indices of the substrate / film terminations.
    Na, Nb:
        2x2 HNF expansion matrices for substrate / film.
    eps1, eps2:
        Principal engineering strains (film w.r.t. substrate).
    vm:
        Von Mises strain metric.
    n_atoms:
        Approximate total atom count of the matched interface.
    notes:
        Arbitrary human-readable tags (polarity warnings, etc.).
    """

    sub_miller: tuple[int, int, int]
    film_miller: tuple[int, int, int]
    Na: np.ndarray
    Nb: np.ndarray
    eps1: float
    eps2: float
    vm: float
    n_atoms: int = 0
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def get_surface_lattice_2d(structure: Structure, miller: Sequence[int]) -> np.ndarray:
    """Return the 2x2 in-plane lattice matrix for a given Miller plane.

    A minimal SlabGenerator is used internally; only the surface lattice
    is extracted (no full slab created).

    Parameters
    ----------
    structure:
        Bulk pymatgen Structure.
    miller:
        Miller indices (hkl).

    Returns
    -------
    np.ndarray
        2x2 matrix whose rows are the in-plane lattice vectors (Angstrom).
    """
    gen = SlabGenerator(
        structure,
        miller_index=list(miller),
        min_slab_size=1,        # we only need the surface cell geometry
        min_vacuum_size=0,
        center_slab=False,
        in_unit_planes=False,
    )
    slabs = gen.get_slabs()
    if not slabs:
        raise ValueError(f"SlabGenerator produced no slab for miller={miller}.")
    slab = slabs[0]
    # The first two lattice vectors span the surface plane
    a = slab.lattice.matrix[0, :2]
    b = slab.lattice.matrix[1, :2]
    return np.array([a, b])


def get_slab_atom_count(
    structure: Structure,
    miller: Sequence[int],
    min_thickness_ang: float = 12.0,
    HNF: np.ndarray | None = None,
) -> int:
    """Estimate the atom count of a slab with the given settings.

    Parameters
    ----------
    structure:
        Bulk pymatgen Structure.
    miller:
        Miller indices (hkl).
    min_thickness_ang:
        Minimum slab thickness in Angstrom.
    HNF:
        Optional 2x2 HNF expansion matrix.  If provided the count is
        multiplied by ``|det(HNF)|``.
    """
    gen = SlabGenerator(
        structure,
        miller_index=list(miller),
        min_slab_size=min_thickness_ang,
        min_vacuum_size=0,
        in_unit_planes=False,
    )
    slabs = gen.get_slabs()
    if not slabs:
        return 0
    n = len(slabs[0])
    if HNF is not None:
        n *= int(round(abs(np.linalg.det(HNF))))
    return n


def build_symmetric_slab(
    structure: Structure,
    miller: Sequence[int],
    min_thickness_ang: float = 12.0,
    vacuum_ang: float = 15.0,
    HNF: np.ndarray | None = None,
) -> "ase.Atoms":  # noqa: F821
    """Build a symmetric slab and return an ASE Atoms object.

    Parameters
    ----------
    structure:
        Bulk pymatgen Structure.
    miller:
        Miller indices (hkl).
    min_thickness_ang:
        Minimum slab thickness in Angstrom.
    vacuum_ang:
        Vacuum layer thickness in Angstrom.
    HNF:
        Optional 2x2 HNF matrix for in-plane supercell expansion.

    Returns
    -------
    ase.Atoms
        Slab with vacuum, centred vertically.
    """
    gen = SlabGenerator(
        structure,
        miller_index=list(miller),
        min_slab_size=min_thickness_ang,
        min_vacuum_size=vacuum_ang,
        center_slab=True,
        in_unit_planes=False,
        symmetrize=True,
    )
    slabs = gen.get_slabs()
    if not slabs:
        raise ValueError(f"SlabGenerator produced no slab for miller={miller}.")

    # Prefer the lowest-energy (first) symmetric slab
    slab_pmg = slabs[0]

    if HNF is not None:
        det = int(round(abs(np.linalg.det(HNF))))
        if det > 1:
            scaling = np.eye(3, dtype=int)
            scaling[:2, :2] = HNF
            slab_pmg = slab_pmg.make_supercell(scaling)

    adaptor = AseAtomsAdaptor()
    atoms = adaptor.get_atoms(slab_pmg)
    atoms.center(vacuum=vacuum_ang / 2, axis=2)
    return atoms
