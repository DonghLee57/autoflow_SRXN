"""
autoflow_srxn.interface.lattice_match
======================================
Pure-NumPy utilities for 2D coincidence lattice matching.

All public functions in this module are dependency-free (NumPy only),
so they can be tested without pymatgen.

References
----------
Zur & McGill, J. Appl. Phys. 55, 378 (1984)  [HNF enumeration]
"""

from __future__ import annotations

import math
from typing import Iterator

import numpy as np

# ---------------------------------------------------------------------------
# Polar space-group registry (pyroelectric groups, ITA Table 10.2.1)
# ---------------------------------------------------------------------------
# Triclinic  : 1
# Monoclinic : 3-4 (unique axis b, C2 and Cs are non-polar; 4 = P2_1)
# Orthorhombic: 25-46 (polar when no mirror || to unique axis)
# Tetragonal : 99-110
# Trigonal   : 143-146, 156-161
# Hexagonal  : 168-186
# Cubic      : (none)
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
assert len(POLAR_SG) == 70, f"Expected 70 polar SGs, got {len(POLAR_SG)}"


def polar_axis_for_sg(sg: int) -> np.ndarray | None:
    """Return the crystallographic polar axis for space group *sg*.

    Returns ``None`` for non-polar space groups.  The axis is expressed as
    a Miller-index-like unit vector (e.g. ``[0, 0, 1]`` for *c*-axis polar
    materials, ``[0, 1, 0]`` for monoclinic *b*-axis polarity).
    """
    if sg not in POLAR_SG:
        return None
    # Monoclinic (3-9): polar along b
    if 3 <= sg <= 9:
        return np.array([0, 1, 0])
    # Default: polar along c
    return np.array([0, 0, 1])


def miller_polar_inplane(miller: tuple[int, int, int], polar_axis: np.ndarray | None) -> bool:
    """Return ``True`` if the polar axis lies *in the surface plane*.

    Parameters
    ----------
    miller:
        Miller indices of the surface, e.g. ``(0, 0, 1)``.
    polar_axis:
        Output of :func:`polar_axis_for_sg`.  If ``None`` (non-polar
        material) the function always returns ``True`` (safe to use any
        termination).
    """
    if polar_axis is None:
        return True
    # The surface normal is the Miller direction; polar is in-plane when
    # it is perpendicular to the normal.
    normal = np.array(miller, dtype=float)
    return abs(np.dot(normal, polar_axis.astype(float))) < 1e-9


# ---------------------------------------------------------------------------
# HNF matrix enumeration
# ---------------------------------------------------------------------------

def iter_hnf_2d(max_det: int) -> Iterator[np.ndarray]:
    """Yield all 2x2 lower-triangular Hermite Normal Form matrices with
    determinant in [1, max_det].

    Each matrix has the form::

        [[a, 0],
         [c, b]]

    with ``a * b == det``, ``b >= 1``, and ``0 <= c < b``.
    """
    for det in range(1, max_det + 1):
        for a in range(1, det + 1):
            if det % a != 0:
                continue
            b = det // a
            for c in range(0, b):
                yield np.array([[a, 0], [c, b]], dtype=int)


# ---------------------------------------------------------------------------
# Strain calculation
# ---------------------------------------------------------------------------

def strain_from_F(A_sub: np.ndarray, A_film: np.ndarray) -> tuple[float, float, float]:
    """Compute principal strains and von Mises strain from the 2D deformation
    gradient **F = A_sub @ inv(A_film)**.

    Parameters
    ----------
    A_sub, A_film:
        2x2 matrices whose rows are the 2D lattice vectors of the substrate
        and film supercells respectively.

    Returns
    -------
    eps1, eps2, vm:
        Principal engineering strains (dimensionless) and von Mises metric
        ``vm = sqrt(0.5 * (e1^2 + e2^2 + (e1-e2)^2))``.
    """
    try:
        F = A_sub @ np.linalg.inv(A_film)
    except np.linalg.LinAlgError:
        return 1.0, 1.0, 1.0

    if abs(np.linalg.det(F)) < 1e-12:
        return 1.0, 1.0, 1.0

    # Polar decomposition: F = R U  ->  C = F^T F = U^2
    C = F.T @ F
    eigvals = np.linalg.eigvalsh(C)
    eigvals = np.clip(eigvals, 0, None)
    sv = np.sqrt(eigvals)      # singular values (principal stretches)
    eps1, eps2 = sorted(sv - 1.0)

    vm = float(np.sqrt(0.5 * (eps1**2 + eps2**2 + (eps1 - eps2) ** 2)))
    return float(eps1), float(eps2), vm


# ---------------------------------------------------------------------------
# Coincidence lattice search
# ---------------------------------------------------------------------------

def find_coincidences(
    A_sub: np.ndarray,
    A_film: np.ndarray,
    max_det: int = 8,
    strain_cutoff: float = 0.05,
) -> list[dict]:
    """Find coincidence supercells between substrate and film 2D lattices.

    Parameters
    ----------
    A_sub, A_film:
        2x2 matrices (rows = lattice vectors) for the substrate and film
        surface cells respectively.
    max_det:
        Maximum determinant (i.e. maximum supercell size) to consider for
        *each* material independently.
    strain_cutoff:
        Maximum von Mises strain to accept.

    Returns
    -------
    list[dict]
        Sorted (ascending composite key = vm + 0.001 * max(det_Na, det_Nb))
        list of dictionaries with keys:
        ``Na``, ``Nb``, ``det_Na``, ``det_Nb``, ``eps1``, ``eps2``,
        ``vm``, ``area_ratio``.
    """
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
                    Na=Na.copy(),
                    Nb=Nb.copy(),
                    det_Na=det_a,
                    det_Nb=det_b,
                    eps1=eps1,
                    eps2=eps2,
                    vm=vm,
                    area_ratio=area_ratio,
                )
            )

    results.sort(key=lambda r: r["vm"] + 0.001 * max(r["det_Na"], r["det_Nb"]))
    return results
