import math
import numpy as np
from ase import Atoms
from ase.build import make_supercell, surface
from ase.geometry import get_distances
from ..utils.knowledge_engine import chem_kb
from ..metadynamics.knowledge import GlobalKnowledge

def standardize_vasp_atoms(atoms, z_min_offset=0.5):
    """Standardize Atoms object for VASP export:
    1. Sort by atomic number (element).
    2. Align minimum Z-coordinate to z_min_offset.
    3. Wrap all positions into the periodic cell (frac in [0,1)).
    Returns: Sorted, translated, and wrapped Atoms copy.
    """
    sorted_atoms = atoms[atoms.numbers.argsort(kind='stable')]
    z_min = sorted_atoms.positions[:, 2].min()
    sorted_atoms.translate([0, 0, z_min_offset - z_min])
    sorted_atoms.wrap()
    sorted_atoms.calc = atoms.calc
    sorted_atoms.info = atoms.info
    return sorted_atoms

def write_standardized_vasp(filepath, atoms, z_min_offset=0.5):
    """Standardizes the atoms object and saves it to a VASP file."""
    from ase.io import write
    standardized = standardize_vasp_atoms(atoms, z_min_offset=z_min_offset)
    write(filepath, standardized, format="vasp")

def find_surface_indices(atoms, side="top", threshold=1.0, species=None):
    """Find indices of atoms at the top or bottom surface based on Z-coordinates."""
    if species:
        indices = np.where(atoms.symbols == species)[0]
    else:
        indices = np.arange(len(atoms))
    if len(indices) == 0: return []
    z_coords = atoms.positions[indices, 2]
    z_target = np.max(z_coords) if side == "top" else np.min(z_coords)
    mask = np.abs(z_coords - z_target) < threshold
    return indices[mask]

def get_pair_bond_cutoff(sym1, sym2, bond_slack=0.45, max_cutoff=3.1):
    """Return an element-pair covalent cutoff for coordination counting."""
    r1 = chem_kb.get_radius(sym1, "covalent")
    r2 = chem_kb.get_radius(sym2, "covalent")
    return min(max_cutoff, r1 + r2 + bond_slack)

def filter_bonded_neighbor_vectors(atoms, idx, j_list, D_list, bond_slack=0.45, max_cutoff=3.1):
    """Keep only chemically plausible bonded neighbors of one atom."""
    bonded = []
    sym = atoms.symbols[idx]
    for j, vec in zip(j_list, D_list):
        dist = np.linalg.norm(vec)
        if dist <= 0.1:
            continue
        cutoff = get_pair_bond_cutoff(sym, atoms.symbols[j], bond_slack=bond_slack, max_cutoff=max_cutoff)
        if dist < cutoff:
            bonded.append((j, vec))
    return bonded

def calculate_haptic_vbs(atoms, indices):
    """Calculates the Virtual Bonding Site (centroid) for a set of atoms."""
    if not indices: return None
    return np.mean(atoms.positions[indices], axis=0)

def calculate_haptic_normal(atoms, indices):
    """Calculates the normal vector for a haptic ligand plane."""
    if len(indices) < 3: return np.array([0.0, 0.0, 1.0])
    pos = atoms.positions[indices]
    centered = pos - np.mean(pos, axis=0)
    _, _, vh = np.linalg.svd(centered)
    normal = vh[2, :]
    return normal / np.linalg.norm(normal)

def generate_vsepr_vectors(atoms, idx, neighbor_data=None, num_missing=1, cutoff=3.1, bond_slack=0.45):
    """Calculate generic dangling bond vectors using VSEPR approximation."""
    from ase.neighborlist import neighbor_list
    if neighbor_data: i_list, j_list, D_list = neighbor_data
    else: i_list, j_list, D_list = neighbor_list("ijD", atoms, cutoff)
    mask = i_list == idx
    bonded = filter_bonded_neighbor_vectors(
        atoms,
        idx,
        j_list[mask],
        D_list[mask],
        bond_slack=bond_slack,
        max_cutoff=cutoff,
    )
    vectors = np.array([vec for _, vec in bonded])
    if len(vectors) == 0: return [np.array([0.0, 0.0, 1.0])] * num_missing
    norm_vecs = vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]
    sum_vec = np.sum(norm_vecs, axis=0)
    v_target = -sum_vec
    if np.linalg.norm(v_target) < 1e-4: v_target = np.array([0.0, 0.0, 1.0])
    v_target /= np.linalg.norm(v_target)
    if num_missing == 1: return [v_target]
    if num_missing == 2 and len(vectors) == 2:
        w_unit = v_target
        u = norm_vecs[0] - norm_vecs[1]
        u_norm = np.linalg.norm(u)
        if u_norm > 1e-4:
            u_unit = u / u_norm
            p_unit = np.cross(w_unit, u_unit)
            p_unit /= np.linalg.norm(p_unit)
            v1 = w_unit * 0.577 + p_unit * 0.816
            v2 = w_unit * 0.577 - p_unit * 0.816
            return [v1, v2]
    results = []
    theta = np.deg2rad(20.0)
    perp_vec = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(v_target, perp_vec)) > 0.9: perp_vec = np.array([0.0, 1.0, 0.0])
    axis_1 = np.cross(v_target, perp_vec)
    axis_1 /= np.linalg.norm(axis_1)
    axis_2 = np.cross(v_target, axis_1)
    for i in range(num_missing):
        phi = 2 * np.pi * i / num_missing
        v = v_target * np.cos(theta) + (axis_1 * np.cos(phi) + axis_2 * np.sin(phi)) * np.sin(theta)
        results.append(v / np.linalg.norm(v))
    return results

def get_all_dangling_bonds_general(atoms, valence_map, vector_generator=None, cutoff=3.1, side="top", bond_slack=0.45):
    """Identify missing valences for surface atoms."""
    from ase.neighborlist import neighbor_list
    surface_indices = find_surface_indices(atoms, side=side, threshold=2.0)
    i_list, j_list, D_list = neighbor_list("ijD", atoms, cutoff)
    neighbor_data = (i_list, j_list, D_list)
    if vector_generator is None: vector_generator = generate_vsepr_vectors
    all_bonds = []
    for idx in surface_indices:
        sym = atoms.symbols[idx]
        target_val = chem_kb.get_ideal_coordination(sym, config=valence_map if isinstance(valence_map, dict) else None)
        if target_val <= 0: continue
        mask = i_list == idx
        bonded = filter_bonded_neighbor_vectors(
            atoms,
            idx,
            j_list[mask],
            D_list[mask],
            bond_slack=bond_slack,
            max_cutoff=cutoff,
        )
        num_n = len(bonded)
        num_missing = target_val - num_n
        if num_missing > 0:
            bonded_i = np.full(len(bonded), idx, dtype=int)
            bonded_j = np.array([j for j, _ in bonded], dtype=int)
            bonded_D = np.array([vec for _, vec in bonded], dtype=float).reshape((-1, 3))
            bonded_neighbor_data = (bonded_i, bonded_j, bonded_D)
            try: vecs = vector_generator(atoms, idx, neighbor_data=bonded_neighbor_data, num_missing=num_missing)
            except TypeError: vecs = vector_generator(atoms, idx, neighbor_data=bonded_neighbor_data)
            for v in vecs:
                if (side == "top" and v[2] > -0.1) or (side == "bottom" and v[2] < 0.1):
                    all_bonds.append({"parent": idx, "vector": v, "parent_sym": sym})
    return all_bonds

def passivate_surface_coverage_general(atoms, coverage, valence_map, vector_generator=None, element="H", cutoff=3.1, side="top", verbose=False):
    """Uniformly passivate a surface using a greedy max-min distance algorithm."""
    from ase.geometry import get_distances
    candidates = get_all_dangling_bonds_general(atoms, valence_map, vector_generator, cutoff, side)
    if not candidates: return standardize_vasp_atoms(atoms, z_min_offset=0.5)
    n_target = int(round(len(candidates) * coverage))
    if n_target == 0: return atoms
    current_atoms = atoms.copy()
    success = 0
    available = list(candidates)
    r_pass = chem_kb.get_radius(element, "covalent")
    while success < n_target and available:
        pass_indices = [i for i, sym in enumerate(current_atoms.symbols) if sym == element]
        ref_indices = pass_indices + [i for i, sym in enumerate(current_atoms.symbols) if sym == "O"]
        ref_pos = current_atoms.positions[ref_indices] if ref_indices else []
        best_cand_idx = -1
        best_score = -1.0
        best_b_len = 0.0
        for i_c, cand in enumerate(available):
            parent_pos = current_atoms.positions[cand["parent"]]
            r_parent = chem_kb.get_radius(atoms.symbols[cand["parent"]], "covalent")
            b_len = r_parent + r_pass
            if cand["parent_sym"] == "Si" and element == "H": b_len = 1.48
            elif cand["parent_sym"] == "O" and element == "H": b_len = 0.96
            h_pos_candidate = parent_pos + cand["vector"] * b_len
            if len(ref_pos) == 0: score = 100.0
            else:
                dists = get_distances(h_pos_candidate, ref_pos, cell=current_atoms.cell, pbc=current_atoms.pbc)[1]
                score = np.min(dists)
            if score > best_score:
                _, all_dists_list = get_distances(h_pos_candidate, current_atoms.positions, cell=current_atoms.cell, pbc=current_atoms.pbc)
                all_dists = all_dists_list[0]
                mask = np.ones(len(all_dists), dtype=bool)
                mask[cand["parent"]] = False
                if np.any(all_dists[mask] < 0.8): continue
                best_score, best_cand_idx, best_b_len = score, i_c, b_len
        if best_cand_idx != -1:
            cand = available.pop(best_cand_idx)
            h_pos = current_atoms.positions[cand["parent"]] + cand["vector"] * best_b_len
            current_atoms += Atoms(element, positions=[h_pos])
            current_atoms.wrap()
            success += 1
        else: break
    return standardize_vasp_atoms(current_atoms, z_min_offset=0.5)

def identify_protectors(atoms, config, verbose=False):
    """Infers which atoms belong to the protector layer vs the base substrate."""
    tags = atoms.get_tags()
    species = config.get("protector", {}).get("species", [])
    sub_idx, prot_idx = [], []
    for i, atom in enumerate(atoms):
        if atom.symbol in species or tags[i] >= 2: prot_idx.append(i)
        else: sub_idx.append(i)
    return np.array(sub_idx), np.array(prot_idx)

class CavityDetector:
    def __init__(self, slab, substrate_indices, protector_indices, grid_res=0.2, verbose=False):
        self.slab, self.sub_idx, self.prot_idx, self.grid_res, self.verbose = slab, substrate_indices, protector_indices, grid_res, verbose
    def find_void_centers(self, top_clearance=4.0):
        if len(self.prot_idx) == 0:
            z_max = np.max(self.slab.positions[self.sub_idx, 2]) if len(self.sub_idx) else np.max(self.slab.positions[:, 2])
            nx, ny = int(np.ceil(self.slab.cell[0, 0] / 5.0)), int(np.ceil(self.slab.cell[1, 1] / 5.0))
            return [np.array([(i + 0.5) * (self.slab.cell[0, 0] / nx), (j + 0.5) * (self.slab.cell[1, 1] / ny), z_max + top_clearance]) for i in range(nx) for j in range(ny)]
        from ase.data import vdw_radii
        from scipy.ndimage import distance_transform_edt, maximum_filter
        cell = self.slab.get_cell()
        lx, ly = cell[0, 0], cell[1, 1]
        z_sub_top = np.max(self.slab.positions[self.sub_idx, 2])
        z_prot_top = np.max(self.slab.positions[self.prot_idx, 2])
        if z_prot_top <= z_sub_top: return [np.array([lx / 2, ly / 2, z_sub_top + top_clearance])]
        nx, ny = int(np.ceil(lx / self.grid_res)), int(np.ceil(ly / self.grid_res))
        lz = (z_prot_top + top_clearance) - z_sub_top
        nz = int(np.ceil(lz / self.grid_res))
        if nx <= 0 or ny <= 0 or nz <= 0: return []
        grid = np.ones((nx, ny, nz), dtype=bool)
        for idx in self.prot_idx:
            pos = self.slab.positions[idx]
            r = 1.5
            try:
                r = vdw_radii[self.slab.numbers[idx]]
                if np.isnan(r): r = 1.5
            except (IndexError, KeyError): pass
            gx, gy, gz = int((pos[0] % lx) / self.grid_res), int((pos[1] % ly) / self.grid_res), int((pos[2] - z_sub_top) / self.grid_res)
            ir = int(np.ceil((r + 1.8) / self.grid_res))
            x_min, x_max = max(0, gx - ir), min(nx, gx + ir + 1)
            y_min, y_max = max(0, gy - ir), min(ny, gy + ir + 1)
            z_min, z_max = max(0, gz - ir), min(nz, gz + ir + 1)
            grid[x_min:x_max, y_min:y_max, z_min:z_max] = False
        dist = distance_transform_edt(grid) * self.grid_res
        local_max = maximum_filter(dist, size=3) == dist
        local_max[dist < 0.5] = False
        max_coords = np.argwhere(local_max)
        centers, sizes = [], []
        for c in max_coords:
            centers.append(np.array([(c[0] + 0.5) * self.grid_res, (c[1] + 0.5) * self.grid_res, z_sub_top + (c[2] + 0.5) * self.grid_res]))
            sizes.append(dist[c[0], c[1], c[2]])
        centers = [x for _, x in sorted(zip(sizes, centers), key=lambda pair: pair[0], reverse=True)]
        pulled = []
        for c in centers:
            best_z = c[2]
            for z_test in np.arange(c[2], z_sub_top + 1.5, -0.2):
                if any(np.linalg.norm(np.array([c[0], c[1], z_test]) - self.slab.positions[p]) < 2.0 for p in self.prot_idx): break
                best_z = z_test
            pulled.append(np.array([c[0], c[1], best_z]))
        filtered = []
        for c in pulled:
            if not filtered or np.all(np.linalg.norm(np.array(filtered) - c, axis=1) > 2.0): filtered.append(c)
            if len(filtered) >= 5: break
        return filtered

def create_slab_from_bulk(bulk_atoms, miller_indices, thickness, vacuum, target_area=None, supercell_matrix=None, termination=None, top_termination=None, bottom_termination=None, verbose=False):
    """Generates a substrate slab from a bulk structure."""
    s1, s2 = surface(bulk_atoms, miller_indices, layers=1), surface(bulk_atoms, miller_indices, layers=2)
    d_hkl = max(0.1, (np.max(s2.positions[:, 2]) - np.min(s2.positions[:, 2])) - (np.max(s1.positions[:, 2]) - np.min(s1.positions[:, 2])))
    num_layers = int(math.ceil(thickness / d_hkl))
    if termination and not top_termination: top_termination = termination
    if termination and not bottom_termination: bottom_termination = termination
    if any([termination, top_termination, bottom_termination]):
        test_slab = surface(bulk_atoms, miller_indices, layers=num_layers * 2, vacuum=0)
        test_slab.wrap()
        z = test_slab.positions[:, 2]; sort_idx = np.argsort(z); sorted_z = z[sort_idx]
        planes = []
        if len(sorted_z):
            curr = [sort_idx[0]]
            for i in range(1, len(sorted_z)):
                if sorted_z[i] - sorted_z[i-1] < 0.5: curr.append(sort_idx[i])
                else: planes.append(curr); curr = [sort_idx[i]]
            planes.append(curr)
        plane_data = [{"atom_indices": p, "elements": set(test_slab.symbols[p]), "sym_list": sorted(test_slab.symbols[p]), "z": np.mean(test_slab.positions[p, 2])} for p in planes]
        best_p, best_s = None, -1e9
        for i in range(len(plane_data)):
            for j in range(i + 1, len(plane_data)):
                p1, p2 = plane_data[i], plane_data[j]
                score = (2000 if bottom_termination in p1["elements"] else 0) + (2000 if top_termination in p2["elements"] else 0) + (500 if p1["elements"] == p2["elements"] else 0) + (200 if p1["sym_list"] == p2["sym_list"] else 0) - abs(p2["z"] - p1["z"] - thickness) * 20
                if score > best_s: best_s, best_p = score, (p1, p2)
        if best_p: slab = test_slab[(z >= best_p[0]["z"] - 0.1) & (z <= best_p[1]["z"] + 0.1)]
        else: slab = surface(bulk_atoms, miller_indices, layers=num_layers, vacuum=0)
    else: slab = surface(bulk_atoms, miller_indices, layers=num_layers, vacuum=0)
    slab.center(vacuum=vacuum, axis=2)
    if supercell_matrix:
        m = np.eye(3); m[0,0], m[0,1], m[1,0], m[1,1] = supercell_matrix[0][0], supercell_matrix[0][1], supercell_matrix[1][0], supercell_matrix[1][1]
        slab = make_supercell(slab, m)
    elif target_area:
        a1, a2 = slab.cell[0], slab.cell[1]; area_prim = np.linalg.norm(np.cross(a1, a2)); search = int(math.ceil(target_area / area_prim)) + 1
        l1, l2 = np.linalg.norm(a1), np.linalg.norm(a2); bn, bm, bs = 1, 1, -1e9
        for n in range(1, search + 2):
            for m in range(1, search + 2):
                ca = n * m * area_prim
                if ca < target_area * 0.8 or (ca > target_area * 1.5 and ca > 100): continue
                s = (1.0 / (1.0 + abs((n*l1)/(m*l2) - 1.0))) * 10.0 - abs(ca - target_area) / target_area
                if s > bs: bs, bn, bm = s, n, m
        slab = slab * (bn, bm, 1)
    v1xy = np.array([slab.cell[0,0], slab.cell[0,1], 0.0])
    if np.linalg.norm(v1xy) > 1e-4: slab.rotate(-math.atan2(v1xy[1], v1xy[0]) * 180 / math.pi, "z", rotate_cell=True)
    slab.wrap()
    return standardize_vasp_atoms(slab, z_min_offset=0.5)

def apply_surface_reconstruction(atoms, strategy="auto", side="top", verbose=False, miller=None, **kwargs):
    """Applies surface reconstruction.

    Parameters
    ----------
    strategy : str
        "auto" dispatches based on crystal-system heuristics.
        "random_noise" applies a symmetry-breaking random displacement.
    miller : sequence of int or None
        Miller indices of the surface.  Required for system-specific recipes
        (e.g., Si(100) 2x1).  When None, auto falls back to random_noise for
        group-IV surfaces instead of applying the Si(100) recipe blindly.
    """
    if strategy in ["auto", True]: res = auto_reconstruct_surface(atoms, side=side, verbose=verbose, miller=miller, **kwargs)
    elif strategy == "random_noise": res = apply_random_surface_noise(atoms, side=side, verbose=verbose, **kwargs)
    else: res = atoms
    return standardize_vasp_atoms(res, z_min_offset=0.5)

def auto_reconstruct_surface(atoms, side="top", verbose=False, miller=None, **kwargs):
    """Crystal-system-aware reconstruction dispatcher.

    Dispatch logic
    --------------
    * **Group-IV + miller=(1,0,0)** : Si(100) 2x1 buckled-dimer seed from
      ``reconstruction_recipes.reconstruct_si100_2x1_buckled``.
    * **Group-IV, other orientation** : random_noise (ML relax determines the
      correct reconstruction from a broken-symmetry starting point).
    * **Ionic** (|Δχ| > 1.5 eV) : electrostatic rumpling (anion up / cation down).
    * **Metallic** (mean χ < 1.9) : surface contraction + noise.
    * **Other** : random_noise.

    The Si(100) recipe is kept in ``reconstruction_recipes.py`` and imported
    lazily so that system-specific code does not pollute the core utility module.
    """
    idx = find_surface_indices(atoms, side=side, threshold=1.5)
    if not len(idx): return atoms
    chi = np.array([GlobalKnowledge.get_electronegativity(atoms.symbols[i]) for i in idx])
    is_iv = all(n in [6, 14, 32] for n in atoms.numbers[idx])
    is_ionic = (np.max(chi) - np.min(chi)) > 1.5
    is_metal = np.mean(chi) < 1.9
    if is_iv:
        miller_tuple = tuple(int(m) for m in miller) if miller is not None else None
        if miller_tuple == (1, 0, 0):
            from .reconstruction_recipes import reconstruct_si100_2x1_buckled
            if verbose: print(f"  [Reconstruct] Si(100) 2x1 buckled-dimer recipe (from reconstruction_recipes).")
            return reconstruct_si100_2x1_buckled(atoms, side=side, verbose=verbose)
        else:
            if verbose: print(f"  [Reconstruct] Group-IV surface, miller={miller_tuple}: random_noise -> ML relax.")
            return apply_random_surface_noise(atoms, side=side, amplitude=0.15)
    elif is_ionic:
        res, m = atoms.copy(), np.mean(chi)
        for i, j in enumerate(idx): res.positions[j, 2] += (0.2 if chi[i] > m else -0.2) * (1 if side == "top" else -1)
        return apply_random_surface_noise(res, side=side, amplitude=0.05)
    elif is_metal:
        res = atoms.copy(); res.positions[idx, 2] += -0.15 * (1 if side == "top" else -1)
        return apply_random_surface_noise(res, side=side, amplitude=0.1)
    return apply_random_surface_noise(atoms, side=side, amplitude=0.15)

def apply_random_surface_noise(atoms, side="top", amplitude=0.1, verbose=False, **kwargs):
    """General-purpose symmetry breaker."""
    res = atoms.copy(); idx = find_surface_indices(res, side=side, threshold=1.5)
    if len(idx): res.positions[idx] += np.random.normal(0, amplitude, (len(idx), 3))
    res.wrap(); return res

# ---------------------------------------------------------------------------
# Backward-compatibility shim: re-export Si-specific names that may be
# imported directly from surface_utils elsewhere in the codebase.
# New code should import from reconstruction_recipes directly.
# ---------------------------------------------------------------------------
from .reconstruction_recipes import (  # noqa: F401  (re-export)
    SI_VALENCE_MAP,
    reconstruct_si100_2x1_buckled,
    identify_surface_bonds,
    oxidize_si_surface,
    build_si100_slab,
    generate_standard_surfaces,
    get_surface_h_mapping,
)
