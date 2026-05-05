"""
autoflow_srxn.interface.builder
=================================
ASE/pymatgen-based symmetric slab and interface builder, including
lattice matching utilities (HNF enumeration and ZSL matching).

Requires **pymatgen**.

References
----------
Zur & McGill, J. Appl. Phys. 55, 378 (1984)  [HNF enumeration]
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence, Iterator
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
# Polar space-group registry (pyroelectric groups, ITA Table 10.2.1)
# ---------------------------------------------------------------------------
POLAR_SG: frozenset[int] = frozenset(
    [1]
    + list(range(3, 5))          # monoclinic: P2, P2_1
    + list(range(6, 10))         # Cs variants: Pm, Pc, Cm, Cc
    + list(range(25, 47))        # orthorhombic polar
    + list(range(99, 111))       # tetragonal C4v
    + [143, 144, 145, 146]       # trigonal C3
    + list(range(156, 162))      # trigonal C3v
    + list(range(168, 187))      # hexagonal C6, C6v
)


def polar_axis_for_sg(sg: int) -> np.ndarray | None:
    """Return the crystallographic polar axis for space group *sg*."""
    if sg not in POLAR_SG:
        return None
    if 3 <= sg <= 9:
        return np.array([0, 1, 0])
    return np.array([0, 0, 1])


def miller_polar_inplane(miller: tuple[int, int, int], polar_axis: np.ndarray | None) -> bool:
    """Return True if the polar axis lies in the surface plane."""
    if polar_axis is None:
        return True
    normal = np.array(miller, dtype=float)
    return abs(np.dot(normal, polar_axis.astype(float))) < 1e-9


# ---------------------------------------------------------------------------
# HNF matrix enumeration and Strain calculation
# ---------------------------------------------------------------------------

def iter_hnf_2d(max_det: int) -> Iterator[np.ndarray]:
    """Yield all 2x2 lower-triangular HNF matrices with determinant in [1, max_det]."""
    for det in range(1, max_det + 1):
        for a in range(1, det + 1):
            if det % a != 0:
                continue
            b = det // a
            for c in range(0, b):
                yield np.array([[a, 0], [c, b]], dtype=int)


def strain_from_F(A_sub: np.ndarray, A_film: np.ndarray) -> tuple[float, float, float]:
    """Compute principal strains and von Mises strain from A_sub @ inv(A_film)."""
    try:
        F = A_sub @ np.linalg.inv(A_film)
    except np.linalg.LinAlgError:
        return 1.0, 1.0, 1.0

    if abs(np.linalg.det(F)) < 1e-12:
        return 1.0, 1.0, 1.0

    C = F.T @ F
    eigvals = np.linalg.eigvalsh(C)
    eigvals = np.clip(eigvals, 0, None)
    sv = np.sqrt(eigvals)
    eps1, eps2 = sorted(sv - 1.0)
    vm = float(np.sqrt(0.5 * (eps1**2 + eps2**2 + (eps1 - eps2) ** 2)))
    return float(eps1), float(eps2), vm


def find_coincidences(
    A_sub: np.ndarray,
    A_film: np.ndarray,
    max_det: int = 8,
    strain_cutoff: float = 0.05,
) -> list[dict]:
    """Find coincidence supercells between substrate and film 2D lattices."""
    results: list[dict] = []
    for Na in iter_hnf_2d(max_det):
        for Nb in iter_hnf_2d(max_det):
            A_Na = Na.astype(float) @ A_sub
            A_Nb = Nb.astype(float) @ A_film
            eps1, eps2, vm = strain_from_F(A_Na, A_Nb)
            if vm > strain_cutoff:
                continue
            det_a = int(round(abs(np.linalg.det(Na))))
            det_b = int(round(abs(np.linalg.det(Nb))))
            area_sub = abs(np.linalg.det(A_Na))
            area_film = abs(np.linalg.det(A_Nb))
            area_ratio = area_sub / area_film if area_film > 1e-12 else float("inf")
            results.append(
                dict(
                    Na=Na.copy(), Nb=Nb.copy(),
                    det_Na=det_a, det_Nb=det_b,
                    eps1=eps1, eps2=eps2, vm=vm,
                    area_ratio=area_ratio,
                )
            )
    results.sort(key=lambda r: r["vm"] + 0.001 * max(r["det_Na"], r["det_Nb"]))
    return results


# ---------------------------------------------------------------------------
# Data container and Builder utilities
# ---------------------------------------------------------------------------

@dataclass
class InterfaceCandidate:
    """Holds a screened coincidence-lattice match between substrate and film."""
    sub_miller: tuple[int, int, int]
    film_miller: tuple[int, int, int]
    Na: np.ndarray
    Nb: np.ndarray
    eps1: float
    eps2: float
    vm: float
    n_atoms: int = 0
    notes: list[str] = field(default_factory=list)


def get_surface_lattice_2d(structure: Structure, miller: Sequence[int]) -> np.ndarray:
    """Return the 2x2 in-plane lattice matrix for a given Miller plane."""
    gen = SlabGenerator(
        structure,
        miller_index=list(miller),
        min_slab_size=1,
        min_vacuum_size=0,
        center_slab=False,
        in_unit_planes=False,
    )
    slabs = gen.get_slabs()
    if not slabs:
        raise ValueError(f"SlabGenerator produced no slab for miller={miller}.")
    slab = slabs[0]
    v1, v2 = slab.lattice.matrix[0], slab.lattice.matrix[1]
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm1 < 1e-8 or norm2 < 1e-8:
        return np.array([v1[:2], v2[:2]])
    cos_gamma = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
    sin_gamma = np.sqrt(1.0 - cos_gamma**2)
    return np.array([
        [norm1, 0.0],
        [norm2 * cos_gamma, norm2 * sin_gamma]
    ])


def get_slab_atom_count(
    structure: Structure,
    miller: Sequence[int],
    min_thickness_ang: float = 12.0,
    HNF: np.ndarray | None = None,
) -> int:
    """Estimate the atom count of a slab with the given settings."""
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
    """Build a symmetric slab and return an ASE Atoms object."""
    gen = SlabGenerator(
        structure,
        miller_index=list(miller),
        min_slab_size=min_thickness_ang,
        min_vacuum_size=vacuum_ang,
        center_slab=True,
        in_unit_planes=False,
    )
    slabs = gen.get_slabs(symmetrize=True)
    if not slabs:
        raise ValueError(f"SlabGenerator produced no slab for miller={miller}.")
    slab_pmg = slabs[0]
    if HNF is not None:
        det = int(round(abs(np.linalg.det(HNF))))
        if det > 1:
            scaling = np.eye(3, dtype=int)
            scaling[:2, :2] = HNF
            slab_pmg = slab_pmg.make_supercell(scaling)
    from pymatgen.io.ase import AseAtomsAdaptor
    adaptor = AseAtomsAdaptor()
    atoms = adaptor.get_atoms(slab_pmg)
    normal = np.cross(atoms.cell[0], atoms.cell[1])
    atoms.rotate(normal, [0, 0, 1], rotate_cell=True)
    atoms.center(vacuum=vacuum_ang / 2, axis=2)
    return atoms
