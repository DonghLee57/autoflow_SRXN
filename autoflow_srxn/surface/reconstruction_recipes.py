"""System-specific surface reconstruction recipes.

This module holds *hardcoded, non-generic* reconstruction implementations that
are tied to particular material systems.  They are intentionally kept **separate
from surface_utils.py** (which contains only crystal-system-agnostic utilities)
so that the core surface module stays extensible.

How to add a new recipe
-----------------------
1. Implement a function with the signature::

       def reconstruct_<system>(atoms, side="top", **kwargs) -> Atoms

2. Register it in ``auto_reconstruct_surface()`` in ``surface_utils.py`` by
   passing the appropriate ``miller`` and material conditions.

Current recipes
---------------
- Si(100) 2×1 buckled dimer (``reconstruct_si100_2x1_buckled``)
- Si surface bond identification and oxidation helpers
- Standard Si(100) slab presets (``build_si100_slab``, ``generate_standard_surfaces``)
"""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.geometry import get_distances
from ase.neighborlist import neighbor_list

from .surface_utils import (
    find_surface_indices,
    generate_vsepr_vectors,
    passivate_surface_coverage_general,
    standardize_vasp_atoms,
)

# ---------------------------------------------------------------------------
# Shared valence map for Si-surface utilities
# ---------------------------------------------------------------------------

SI_VALENCE_MAP: dict = {"Si": 4, "O": 2, "H": 1, "F": 1, "Cl": 1}


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _get_natural_pairing_vector(atoms, idx, neighbor_data=None):
    """Return the lateral pairing axis at surface atom *idx*.

    Used to locate the partner atom of a Si dimer.
    Returns None if fewer than two dangling-bond directions are found.
    """
    vecs = generate_vsepr_vectors(
        atoms, idx, neighbor_data=neighbor_data, num_missing=2
    )
    if len(vecs) == 2:
        d = vecs[0] - vecs[1]
        d[2] = 0.0
        mag = np.linalg.norm(d)
        if mag > 1e-3:
            return d / mag
    return None


def _canonical_dimer_vector(vec):
    """Orient an in-plane dimer vector deterministically."""
    v = np.array(vec, dtype=float)
    if v[0] < -1e-6 or (abs(v[0]) < 1e-6 and v[1] < -1e-6):
        v = -v
    return v


def _iter_perfect_matchings(indices, candidates):
    """Yield perfect matchings from pair candidates for small surface cells."""
    remaining = tuple(indices)
    cand_by_atom = {i: [] for i in remaining}
    for cand in candidates:
        i, j = cand["ids"]
        cand_by_atom.setdefault(i, []).append(cand)
        cand_by_atom.setdefault(j, []).append(cand)

    def rec(rem):
        if not rem:
            yield []
            return
        i = rem[0]
        rem_set = set(rem)
        for cand in cand_by_atom.get(i, []):
            i1, i2 = cand["ids"]
            j = i2 if i1 == i else i1
            if j not in rem_set:
                continue
            next_rem = tuple(k for k in rem if k not in (i, j))
            for rest in rec(next_rem):
                yield [cand] + rest

    yield from rec(remaining)


def _dimer_seed_positions(atoms, dimers, rows, cols, buckle, bond_length, pattern, side):
    """Return trial positions for a proposed dimer matching."""
    trial = atoms.positions.copy()
    pairset = set()
    inv = np.linalg.inv(atoms.cell[:2, :2])
    dxy = np.sqrt(max(0.0, bond_length ** 2 - buckle ** 2))
    zsign = 1.0 if side == "top" else -1.0

    for d in dimers:
        r = rows.index(round((d["mid"][:2] @ inv)[1] * 8, 1))
        c = cols.index(round((d["mid"][:2] @ inv)[0] * 8, 1))

        if pattern == "checkerboard":
            S = (-1) ** (r + c)
        elif pattern == "stripe":
            S = (-1) ** c
        else:
            S = 1

        i1, i2 = d["ids"]
        v = _canonical_dimer_vector(d["vec"])
        if np.dot(v, d["vec"]) < 0:
            i1, i2 = i2, i1

        mid = atoms.positions[i1] + v / 2
        u = v / np.linalg.norm(v)

        trial[i1] = mid + u * (dxy / 2) + np.array([0, 0, zsign * S * buckle / 2])
        trial[i2] = mid - u * (dxy / 2) - np.array([0, 0, zsign * S * buckle / 2])
        pairset.add(tuple(sorted((i1, i2))))

    return trial, pairset


def _filter_to_dominant_direction(candidates, back_bond_angle=None):
    """Filter dimer candidates to a single lateral direction.

    Si(100) has two degenerate nearest-neighbor directions at ~45° and ~135°
    relative to the cell axes.  The correct dimer direction is perpendicular
    to the back-bond projection onto the surface plane; the parallel direction
    is a saddle point that the ML relaxer will undo.

    When *back_bond_angle* (degrees, mod 180) is provided, the direction most
    perpendicular to the back-bond is selected among equally-populated bins.
    Without it, the smallest-angle bin wins the tie — use only as a fallback.
    """
    if not candidates:
        return candidates

    bin_width = 30.0
    n_bins = 6  # covers [0°, 180°)

    bins: list[list] = [[] for _ in range(n_bins)]
    for cand in candidates:
        v = cand["vec"][:2]
        angle = np.degrees(np.arctan2(v[1], v[0])) % 180.0
        bins[int(angle / bin_width) % n_bins].append(cand)

    max_count = max(len(b) for b in bins)
    top_bins = [i for i in range(n_bins) if len(bins[i]) == max_count]

    if back_bond_angle is not None and len(top_bins) > 1:
        # Among equally-populated bins pick the one most perpendicular to back-bonds.
        def _perp(bi):
            center = (bi + 0.5) * bin_width
            diff = abs(center - back_bond_angle)
            return min(diff, 180.0 - diff)
        dominant = max(top_bins, key=_perp)
    else:
        dominant = max(top_bins, key=lambda b: -b)

    dominant_center = (dominant + 0.5) * bin_width

    filtered = []
    for cand in candidates:
        v = cand["vec"][:2]
        angle = np.degrees(np.arctan2(v[1], v[0])) % 180.0
        diff = abs(angle - dominant_center)
        if min(diff, 180.0 - diff) <= bin_width:
            filtered.append(cand)

    return filtered if filtered else candidates


def _score_dimer_matching(atoms, dimers, rows, cols, buckle, bond_length, pattern, side):
    """Score a matching by the closest non-dimer top-layer Si-Si distance."""
    trial, pairset = _dimer_seed_positions(
        atoms, dimers, rows, cols, buckle, bond_length, pattern, side
    )
    top_ids = [i for d in dimers for i in d["ids"]]
    min_nonbond = np.inf

    for pos_i, i in enumerate(top_ids):
        for j in top_ids[pos_i + 1:]:
            if tuple(sorted((i, j))) in pairset:
                continue
            _, dist = get_distances(
                trial[i], trial[j], cell=atoms.cell, pbc=atoms.pbc
            )
            min_nonbond = min(min_nonbond, float(dist[0][0]))

    return min_nonbond


def _select_stable_dimer_matching(
    atoms, idx_list, buckle, bond_length, pattern, side,
    back_bond_angle=None, verbose=False
):
    """Choose Si(100) surface dimers that do not create near-collisions."""
    candidates = []
    for n, i1 in enumerate(idx_list):
        pot = list(idx_list[n + 1:])
        if not pot:
            continue
        D, d = get_distances(
            atoms.positions[i1], atoms.positions[pot],
            cell=atoms.cell, pbc=atoms.pbc,
        )
        for sub, i2 in enumerate(pot):
            if 2.8 < d[0][sub] < 4.2:
                mid = atoms.positions[i1] + D[0][sub] / 2
                candidates.append({"ids": (i1, i2), "vec": D[0][sub], "mid": mid})

    if not candidates or len(idx_list) % 2:
        return []

    # In small cells, ASE's minimum-image convention may return the same
    # direction for both degenerate images (e.g. (2.715,2.715) and
    # (-2.715,-2.715) both map to 45°).  Explicitly add the perpendicular
    # alternative image for each such pair so the filter has both axes to
    # choose from.
    if back_bond_angle is not None:
        target_perp = (back_bond_angle + 90.0) % 180.0
        extra = []
        seen_perp = set()
        for cand in candidates:
            vec = cand["vec"]
            ang = np.degrees(np.arctan2(vec[1], vec[0])) % 180.0
            if min(abs(ang - back_bond_angle), 180.0 - abs(ang - back_bond_angle)) > 30.0:
                continue  # already off back-bond axis, no need to augment
            cid = tuple(sorted(cand["ids"]))
            if cid in seen_perp:
                continue
            seen_perp.add(cid)
            for rv in (np.array([vec[0], -vec[1], vec[2]]),
                       np.array([-vec[0], vec[1], vec[2]])):
                ang_r = np.degrees(np.arctan2(rv[1], rv[0])) % 180.0
                if min(abs(ang_r - target_perp), 180.0 - abs(ang_r - target_perp)) <= 30.0:
                    mid_r = atoms.positions[cand["ids"][0]] + rv / 2
                    extra.append({"ids": cand["ids"], "vec": rv, "mid": mid_r})
                    break
        candidates = candidates + extra

    # Restrict to one lateral direction, choosing the axis perpendicular to
    # back-bonds so the seed geometry cannot relax straight back to ideal.
    candidates = _filter_to_dominant_direction(candidates, back_bond_angle=back_bond_angle)

    inv = np.linalg.inv(atoms.cell[:2, :2])
    rows = sorted(set(round((d["mid"][:2] @ inv)[1] * 8, 1) for d in candidates))
    cols = sorted(set(round((d["mid"][:2] @ inv)[0] * 8, 1) for d in candidates))

    if len(idx_list) <= 16:
        best_score, best = -np.inf, []
        for matching in _iter_perfect_matchings(idx_list, candidates):
            if len(matching) != len(idx_list) // 2:
                continue
            score = _score_dimer_matching(
                atoms, matching, rows, cols, buckle, bond_length, pattern, side
            )
            if score > best_score:
                best_score, best = score, matching
        if verbose and best:
            print(f"  [Si100 Recipe] stable matching min non-dimer spacing score = {best_score:.3f}")
        return best

    # Fallback for larger cells: greedily keep the shortest candidate whose
    # trial structure preserves the largest current non-dimer spacing.
    paired, dimers = set(), []
    for cand in sorted(candidates, key=lambda d: d["vec"][0] ** 2 + d["vec"][1] ** 2):
        i1, i2 = cand["ids"]
        if i1 in paired or i2 in paired:
            continue
        dimers.append(cand)
        paired.update([i1, i2])
    return dimers


# ---------------------------------------------------------------------------
# Si(100) 2x1 buckled-dimer recipe
# ---------------------------------------------------------------------------

def reconstruct_si100_2x1_buckled(
    atoms,
    side: str = "top",
    buckle: float = 0.7,
    bond_length: float = 2.30,
    pattern: str = "checkerboard",
    verbose: bool = False,
):
    """Apply a Si(100) 2×1 buckled-dimer reconstruction as a seed geometry.

    This is a *geometric approximation* intended to provide a physically
    reasonable starting configuration for a subsequent ML-potential relaxation.
    The ML potential will find the true local minimum; the seed avoids
    the optimizer getting trapped in a symmetric (unreconstructed) saddle point.

    Parameters
    ----------
    atoms : ase.Atoms
        Si(100) slab.  Must already be oriented so that the (100) normal is
        along +Z.
    side : {"top", "bottom"}
        Which surface to reconstruct.
    buckle : float
        Vertical buckling amplitude (Å).  Default 0.7 Å matches DFT values.
    bond_length : float
        Target Si–Si dimer bond length (Å).  Default 2.30 Å.
    pattern : {"checkerboard", "stripe", "uniform"}
        Buckling phase pattern across the surface.
    verbose : bool
        Print dimer-finding statistics.

    Returns
    -------
    ase.Atoms
        Atoms object with displaced surface Si atoms.

    Notes
    -----
    This recipe is **only valid for Si(100)**.  Do NOT call it for Si(110),
    Si(111), or any non-Si group-IV surface.  Use ``random_noise`` + ML relax
    for other orientations.
    """
    idx_list = find_surface_indices(atoms, side, species="Si")
    if not len(idx_list):
        return atoms

    # Count only Si-Si bonds so that passivating H atoms don't inflate coordination.
    i_list, j_list, D_list = neighbor_list("ijD", atoms, 2.6)
    si_mask = np.array(atoms.get_chemical_symbols()) == "Si"
    idx_set = set(idx_list.tolist())
    undercoord_idx = np.array([
        i for i in idx_list
        if np.sum(si_mask[j_list[i_list == i]]) < 4
    ])

    # Compute average back-bond XY direction so the dimer filter can choose
    # the perpendicular direction (the correct 2×1 axis).
    zsign = -1.0 if side == "top" else 1.0  # back-bonds go into the bulk
    back_vecs = []
    for ii in undercoord_idx:
        mask = i_list == ii
        for jj, dv in zip(j_list[mask], D_list[mask]):
            if si_mask[jj] and jj not in idx_set and dv[2] * zsign > 0.2:
                v2 = dv[:2]
                norm = np.linalg.norm(v2)
                if norm > 1e-3:
                    v2 = v2 / norm
                    if v2[0] < -1e-6 or (abs(v2[0]) < 1e-6 and v2[1] < -1e-6):
                        v2 = -v2
                    back_vecs.append(v2)
    back_bond_angle = None
    if back_vecs:
        avg = np.mean(back_vecs, axis=0)
        n = np.linalg.norm(avg)
        if n > 1e-6:
            back_bond_angle = float(np.degrees(np.arctan2(avg[1], avg[0])) % 180.0)

    dimers = _select_stable_dimer_matching(
        atoms,
        undercoord_idx,
        buckle=buckle,
        bond_length=bond_length,
        pattern=pattern,
        side=side,
        back_bond_angle=back_bond_angle,
        verbose=verbose,
    )

    if verbose:
        print(f"  [Si100 Recipe] {len(dimers)} dimers found on '{side}' surface.")

    if not dimers:
        return atoms

    # Even dimer counts have two degenerate checkerboard phases that differ
    # across supercells.  Force uniform buckling so the seed is reproducible.
    effective_pattern = "uniform" if len(dimers) % 2 == 0 else pattern

    inv = np.linalg.inv(atoms.cell[:2, :2])
    rows = sorted(set(round((d["mid"][:2] @ inv)[1] * 8, 1) for d in dimers))
    cols = sorted(set(round((d["mid"][:2] @ inv)[0] * 8, 1) for d in dimers))

    trial, _ = _dimer_seed_positions(
        atoms, dimers, rows, cols, buckle, bond_length, effective_pattern, side
    )
    atoms.positions[:] = trial

    atoms.wrap()
    return atoms


# ---------------------------------------------------------------------------
# Si surface bond and oxidation utilities
# ---------------------------------------------------------------------------

def identify_surface_bonds(atoms, cutoff: float = 2.6):
    """Classify top-layer Si bonds into dimer bonds and back-bonds.

    Returns
    -------
    dimers : list of (int, int)
        Intra-surface-layer Si–Si pairs (dimer bonds).
    back_bonds : list of (int, int)
        Surface-to-subsurface Si–Si pairs (back-bonds).
    """
    l1 = find_surface_indices(atoms, "top", threshold=0.8, species="Si")
    zt = np.max(atoms.positions[l1, 2])
    l2 = np.where(
        (atoms.symbols == "Si")
        & (atoms.positions[:, 2] < zt - 0.5)
        & (atoms.positions[:, 2] > zt - 2.5)
    )[0]

    nl_i, nl_j, _ = neighbor_list("ijD", atoms, cutoff)
    dimers, back_bonds, seen = [], [], set()

    for i1 in l1:
        for ni in nl_j[nl_i == i1]:
            if ni == i1 or atoms.symbols[ni] != "Si":
                continue
            b = tuple(sorted((i1, ni)))
            if b in seen:
                continue
            if ni in l1:
                dimers.append(b)
            elif ni in l2:
                back_bonds.append(b)
            seen.add(b)

    return dimers, back_bonds


def _insert_o_bridge(atoms, idx1: int, idx2: int,
                     target_si_o: float = 1.63, target_angle: float = 144.0):
    """Insert one bridging O atom between two Si atoms (geometry-only)."""
    p1, p2 = atoms.positions[idx1].copy(), atoms.positions[idx2].copy()
    if p2[2] > p1[2]:
        idx1, idx2, p1, p2 = idx2, idx1, p2, p1

    v = p1 - p2
    bl = np.linalg.norm(v)
    u = v / bl
    mid = (p1 + p2) / 2.0

    dp = target_si_o * np.sin(np.deg2rad(target_angle / 2))
    da = target_si_o * np.cos(np.deg2rad(target_angle / 2))

    perp = np.cross(u, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(perp) < 1e-3:
        perp = np.cross(u, np.array([0.0, 1.0, 0.0]))
    perp /= np.linalg.norm(perp)

    atoms.positions[idx1] += u * (da - bl / 2) * 0.8
    atoms.positions[idx2] -= u * (da - bl / 2) * 0.2
    atoms += Atoms("O", positions=[mid + perp * dp])
    return atoms


def oxidize_si_surface(slab, dimer_coverage: float = 0.0,
                        backbond_coverage: float = 0.0, verbose: bool = False):
    """Place bridging O atoms on Si dimer or back-bond sites.

    Parameters
    ----------
    slab : ase.Atoms
        Si(100) slab (possibly reconstructed).
    dimer_coverage : float
        Fraction [0, 1] of dimer bonds to oxidize.
    backbond_coverage : float
        Fraction [0, 1] of back-bonds to oxidize.
    """
    dimers, bbs = identify_surface_bonds(slab)
    nd = int(round(len(dimers) * dimer_coverage))
    nb = int(round(len(bbs) * backbond_coverage))

    res = slab.copy()
    oc = {i: 0 for i in range(len(slab))}

    def greedy(at, cands, n):
        curr, s = at.copy(), 0
        avail = list(cands)
        while s < n and avail:
            opos = curr.positions[curr.symbols == "O"]
            best_b, best_s, best_i = None, -1.0, -1
            for ib, (b1, b2) in enumerate(avail):
                if oc[b1] >= 2 or oc[b2] >= 2:
                    continue
                mid = (curr.positions[b1] + curr.positions[b2]) / 2.0
                if not len(opos):
                    score = 100.0 - np.linalg.norm(
                        mid[:2] - np.sum(curr.cell, axis=0)[:2] / 2
                    )
                else:
                    score = float(np.min(get_distances(mid, opos,
                                                       cell=curr.cell, pbc=curr.pbc)[1]))
                if score > best_s:
                    others = np.delete(curr.positions, [b1, b2], axis=0)
                    if np.any(np.linalg.norm(others - mid, axis=1) < 1.5):
                        continue
                    best_s, best_b, best_i = score, (b1, b2), ib

            if best_b is not None:
                curr = _insert_o_bridge(curr, best_b[0], best_b[1])
                oc[best_b[0]] += 1
                oc[best_b[1]] += 1
                s += 1
                avail.pop(best_i)
            else:
                break
        return curr, s

    res, _ = greedy(res, dimers, nd)
    res, _ = greedy(res, bbs, nb)
    return res


# ---------------------------------------------------------------------------
# Standard Si(100) slab presets
# ---------------------------------------------------------------------------

def build_si100_slab(bulk_atoms, size=(4, 4), layers: int = 8,
                     vacuum: float = 10.0):
    """Build a bare Si(100) slab with surface/bulk/bottom tags.

    Tags: 1 = top surface, 4 = bottom surface, 0 = bulk.
    """
    from ase.build import surface as ase_surface

    slab = ase_surface(bulk_atoms, (1, 0, 0), layers=layers, vacuum=vacuum)
    slab = slab * (size[0], size[1], 1)

    z = slab.positions[:, 2]
    zmax, zmin = z.max(), z.min()
    for a in slab:
        if a.position[2] > zmax - 0.5:
            a.tag = 1
        elif a.position[2] < zmin + 0.5:
            a.tag = 4
        else:
            a.tag = 0
    return slab


def get_surface_h_mapping(atoms, cutoff: float = 1.8, side: str = "top") -> dict:
    """Return a dict mapping each surface Si index to its bonded H index.

    Only considers H atoms within *cutoff* of a Si atom on the specified side.
    """
    from ase.geometry import get_distances as _gd

    hi = np.where(atoms.symbols == "H")[0]
    si = np.where(atoms.symbols == "Si")[0]
    if not len(hi):
        return {}

    z = atoms.positions[:, 2]
    z_ref = np.max(z) if side == "top" else np.min(z)
    hi = [i for i in hi if abs(z[i] - z_ref) < 3.0]

    mapping = {}
    for h in hi:
        _, d = _gd(atoms.positions[h], atoms.positions[si],
                   cell=atoms.cell, pbc=atoms.pbc)
        if np.any(d[0] < cutoff):
            mapping[si[np.argmin(d[0])]] = h
    return mapping


def generate_standard_surfaces(bulk_si, verbose: bool = False):
    """Return four standard Si(100) surface configurations.

    S1 : clean 2×1 reconstructed
    S2 : H-passivated (top + bottom)
    S3 : partially oxidized (50% dimer + 50% back-bond)
    S4 : oxidized + H-passivated
    """
    base = build_si100_slab(bulk_si, size=(4, 4), layers=8)

    s1 = base.copy()
    s1 = reconstruct_si100_2x1_buckled(s1, verbose=verbose)
    s1.info["label"] = "S1_Clean_2x1"

    s2 = s1.copy()
    s2 = passivate_surface_coverage_general(s2, 1.0, SI_VALENCE_MAP, side="top")
    s2 = passivate_surface_coverage_general(s2, 1.0, SI_VALENCE_MAP, side="bottom")
    s2.info["label"] = "S2_H_Passivated"

    s3 = s1.copy()
    s3 = oxidize_si_surface(s3, dimer_coverage=0.5, backbond_coverage=0.5)
    s3.info["label"] = "S3_Oxidized"

    s4 = s3.copy()
    s4 = passivate_surface_coverage_general(s4, 1.0, SI_VALENCE_MAP, side="top")
    s4 = passivate_surface_coverage_general(s4, 1.0, SI_VALENCE_MAP, side="bottom")
    s4.info["label"] = "S4_Oxidized_H_Passivated"

    return [s1, s2, s3, s4]
