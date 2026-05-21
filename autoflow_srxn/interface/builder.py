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


def get_surface_lattice_2d(
    structure: Structure,
    miller: Sequence[int],
    min_slab_size: float = 8.0,
    min_vacuum_size: float = 1.0,
) -> np.ndarray:
    """Return the 2x2 in-plane lattice matrix for a given Miller plane.

    Parameters
    ----------
    min_slab_size :
        Minimum slab thickness in Angstroms.  Must be > 0 and consistent
        with the value used in :func:`build_symmetric_slab` so the two
        functions use the same surface primitive cell.
    min_vacuum_size :
        A small positive vacuum is required for pymatgen to build a proper
        slab (as opposed to a periodic bulk cell).  Any value > 0 works;
        the default 1 Å keeps computation fast.
    """
    gen = SlabGenerator(
        structure,
        miller_index=list(miller),
        min_slab_size=min_slab_size,
        min_vacuum_size=min_vacuum_size,
        center_slab=False,
        in_unit_planes=False,
    )
    slabs = gen.get_slabs(symmetrize=False)
    if not slabs:
        raise ValueError(f"SlabGenerator produced no slab for miller={miller}.")
    slab = slabs[0]
    # Convert to ASE and rotate so the surface normal points to z.
    # Only after this rotation are cell[0] and cell[1] truly in-plane (z≈0).
    adaptor = AseAtomsAdaptor()
    atoms = adaptor.get_atoms(slab)
    normal = np.cross(atoms.cell[0], atoms.cell[1])
    atoms.rotate(normal, [0, 0, 1], rotate_cell=True)
    cell = np.array(atoms.cell)
    v1_xy = cell[0, :2]
    v2_xy = cell[1, :2]
    norm1, norm2 = np.linalg.norm(v1_xy), np.linalg.norm(v2_xy)
    if norm1 < 1e-8 or norm2 < 1e-8:
        return np.array([[np.linalg.norm(cell[0]), 0.0], [0.0, np.linalg.norm(cell[1])]])
    cos_gamma = np.clip(np.dot(v1_xy, v2_xy) / (norm1 * norm2), -1.0, 1.0)
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


def stack_interface(
    sub_slab: "ase.Atoms",
    film_slab: "ase.Atoms",
    gap_ang: float = 2.5,
    vacuum_ang: float = 15.0,
) -> "ase.Atoms":
    """Stack substrate and film slabs into a single interface Atoms object.

    This function performs explicit in-plane alignment (v1 || X) to ensure
    that the 2D lattice matching is physical and doesn't introduce spurious
    shears/rotations due to basis vector misalignment.

    Parameters
    ----------
    sub_slab:
        The substrate slab (Atoms).
    film_slab:
        The film slab (Atoms).
    gap_ang:
        The gap between slabs in Angstroms.
    vacuum_ang:
        The vacuum size to add on top.

    Returns
    -------
    ase.Atoms:
        The combined interface structure.
    """
    import numpy as np
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

    # --- Align in-plane lattice vectors before stacking ---
    v1_sub = sub_cell[0]
    v1_film = film_cell[0]
    angle_sub = np.arctan2(v1_sub[1], v1_sub[0])
    angle_film = np.arctan2(v1_film[1], v1_film[0])
    # Rotate film to match sub's v1 direction in XY plane
    film_slab_copy = film_slab.copy()
    film_slab_copy.rotate(np.degrees(angle_sub - angle_film), "z", rotate_cell=True)

    # Refresh film variables after rotation
    film_cell = np.array(film_slab_copy.cell)
    film_frac = film_slab_copy.get_scaled_positions()
    film_z = film_frac[:, 2] * film_cell[2, 2]
    # ------------------------------------------------------------

    sub_cell_2d = sub_cell[:2, :2]
    film_cart_xy_new = film_frac[:, :2] @ sub_cell_2d
    film_z_shift = sub_thickness + gap_ang - film_z.min()
    film_z_new = film_z + film_z_shift

    film_pos_new = np.column_stack([film_cart_xy_new, film_z_new])

    all_pos = np.vstack([sub_pos_new, film_pos_new])
    all_symbols = list(sub_slab.get_chemical_symbols()) + list(
        film_slab_copy.get_chemical_symbols()
    )
    all_tags = [0] * len(sub_slab) + [1] * len(film_slab_copy)

    new_cell = sub_cell.copy()
    new_cell[2] = [0.0, 0.0, sub_thickness + gap_ang + film_thickness + vacuum_ang]

    return Atoms(
        symbols=all_symbols,
        positions=all_pos,
        cell=new_cell,
        pbc=[True, True, True],
        tags=all_tags,
    )


def wrap_interface_for_dft(
    interface: "ase.Atoms",
    vacuum_ang: float = 15.0,
    bottom_margin_ang: float = 0.5,
    sort_atoms: bool = True,
) -> "ase.Atoms":
    """Wrap and prepare an interface structure for DFT submission.

    After :func:`stack_interface` the cell may have:
    * atoms drifting slightly outside [0,1) in fractional coordinates due to
      floating-point arithmetic in the rotation/stacking step;
    * the vacuum sitting above the film with no bottom margin.

    This function corrects these issues so the resulting POSCAR is clean and
    VASP-ready.

    Parameters
    ----------
    interface:
        Output of :func:`stack_interface`.
    vacuum_ang:
        Total vacuum layer thickness to enforce above the top of the film.
    bottom_margin_ang:
        Gap between z=0 and the lowest atom in the slab.
    sort_atoms:
        If True, sort atoms by (atomic number, z-coordinate) for a clean
        POSCAR species block ordering.

    Returns
    -------
    ase.Atoms
        New Atoms object with all positions wrapped to the unit cell,
        slab translated so the lowest atom sits at *bottom_margin_ang*,
        and the c-vector trimmed to slab_height + vacuum_ang.
        The a- and b-vectors are preserved exactly.
    """
    from ase import Atoms

    atoms = interface.copy()

    # 1. Wrap all fractional coordinates to [0, 1)
    atoms.wrap()

    # 2. After wrapping some atoms may have jumped from z≈0 to z≈cell_c.
    #    Re-detect the slab as atoms below the midpoint of the cell.
    cell_c = float(atoms.cell[2, 2])
    frac = atoms.get_scaled_positions()
    z_cart = frac[:, 2] * cell_c

    # Distinguish slab from vacuum: atoms with z_cart in lower half of cell
    # (works because vacuum is always on top after stack_interface)
    z_min = z_cart.min()
    z_max = z_cart.max()

    # 3. Shift slab so lowest atom is at bottom_margin_ang
    shift = bottom_margin_ang - z_min
    pos = atoms.get_positions()
    pos[:, 2] += shift
    atoms.set_positions(pos)
    z_max += shift

    # 4. Recompute cell c to slab_height + vacuum
    new_c = z_max + vacuum_ang
    new_cell = atoms.cell.copy()
    new_cell[2] = np.array([0.0, 0.0, new_c])
    atoms.set_cell(new_cell, scale_atoms=False)

    # 5. Sort by (atomic number, z)
    if sort_atoms:
        from ase.build import sort as ase_sort
        atoms = ase_sort(atoms, tags=atoms.get_atomic_numbers())

    return atoms


def resolve_millers(
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
                        if h == k == l == 0:
                            continue
                        if gcd(gcd(h, k), l) == 1:
                            millers.append((h, k, l))
            return millers, f"raw  max_miller={max_m}  ({len(millers)} faces)"

    default = [(0, 0, 1), (1, 1, 0), (1, 1, 1)]
    return default, "default (3 faces)"


def is_candidate_polar_ok(candidate: InterfaceCandidate) -> bool:
    """Return True when neither polar-termination warning is present."""
    return not any(
        w in candidate.notes
        for w in ("WARN:sub_polar_termination", "WARN:film_polar_termination")
    )


def has_large_cell_warning(candidate: InterfaceCandidate) -> bool:
    """Return True if the candidate has a large supercell warning."""
    return any(w.startswith("WARN:large_cell") for w in candidate.notes)


def mark_recommended_candidates(candidates: list[InterfaceCandidate]) -> list[bool]:
    """Return a list of boolean flags where True indicates a recommended candidate.

    A candidate is recommended if it is the first (lowest-strain) match
    found for a specific film orientation.
    """
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
