import math
import numpy as np
from ase import Atoms
from ase.build import make_supercell, surface
from ase.geometry import get_distances
from .knowledge_engine import chem_kb

def standardize_vasp_atoms(atoms, z_min_offset=0.5):
    """Standardize Atoms object for VASP export:
    1. Sort by atomic number (element).
    2. Align minimum Z-coordinate to z_min_offset.
    Returns: Sorted and translated Atoms copy.
    """
    sorted_atoms = atoms[atoms.numbers.argsort(kind='stable')]
    z_min = sorted_atoms.positions[:, 2].min()
    sorted_atoms.translate([0, 0, z_min_offset - z_min])
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

def check_overlap(atoms, cutoff=1.2, verbose=False):
    """Check for steric overlaps between atoms using a simple distance threshold."""
    from ase.neighborlist import neighbor_list
    i_list, j_list, dists = neighbor_list("ijd", atoms, cutoff)
    if len(i_list) > 0:
        if verbose: print(f"Overlap detected: {len(i_list) // 2} pairs closer than {cutoff}A")
        return True
    return False

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

def generate_vsepr_vectors(atoms, idx, neighbor_data=None, num_missing=1, cutoff=2.6):
    """Calculate generic dangling bond vectors using VSEPR approximation."""
    from ase.neighborlist import neighbor_list
    if neighbor_data: i_list, j_list, D_list = neighbor_data
    else: i_list, j_list, D_list = neighbor_list("ijD", atoms, cutoff)
    mask = i_list == idx
    vectors = D_list[mask]
    dists = np.linalg.norm(vectors, axis=1)
    vectors = vectors[(dists > 0.1) & (dists < cutoff)]
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

def get_all_dangling_bonds_general(atoms, valence_map, vector_generator=None, cutoff=3.1, side="top"):
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
        dists = np.linalg.norm(D_list[mask], axis=1)
        num_n = np.sum((dists > 0.1) & (dists < 2.6))
        num_missing = target_val - num_n
        if num_missing > 0:
            try: vecs = vector_generator(atoms, idx, neighbor_data=neighbor_data, num_missing=num_missing)
            except TypeError: vecs = vector_generator(atoms, idx, neighbor_data=neighbor_data)
            for v in vecs:
                if (side == "top" and v[2] > -0.1) or (side == "bottom" and v[2] < 0.1):
                    all_bonds.append({"parent": idx, "vector": v, "parent_sym": sym})
    return all_bonds

def passivate_surface_coverage_general(atoms, h_coverage, valence_map, vector_generator=None, element="H", cutoff=3.1, side="top", verbose=False):
    """Uniformly passivate a surface using a greedy max-min distance algorithm."""
    from ase.geometry import get_distances
    candidates = get_all_dangling_bonds_general(atoms, valence_map, vector_generator, cutoff, side)
    if not candidates: return standardize_vasp_atoms(atoms, z_min_offset=0.5)
    n_target = int(round(len(candidates) * h_coverage))
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
        for i_c, cand in enumerate(available):
            parent_pos = current_atoms.positions[cand["parent"]]
            r_parent = chem_kb.get_radius(atoms.symbols[cand["parent"]], "covalent")
            b_len = r_parent + r_pass
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
                best_score, best_cand_idx = score, i_c
        if best_cand_idx != -1:
            cand = available.pop(best_cand_idx)
            r_parent = chem_kb.get_radius(atoms.symbols[cand["parent"]], "covalent")
            b_len = r_parent + r_pass
            if cand["parent_sym"] == "Si" and element == "H": b_len = 1.48
            if cand["parent_sym"] == "O" and element == "H": b_len = 0.96
            h_pos = current_atoms.positions[cand["parent"]] + cand["vector"] * b_len
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
            except: pass
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
    if any([termination, top_termination, bottom_termination]) or termination in ["symmetric", "uniform"]:
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

def apply_surface_reconstruction(atoms, strategy="auto", side="top", verbose=False, **kwargs):
    """Applies surface reconstruction."""
    if strategy in ["auto", True]: res = auto_reconstruct_surface(atoms, side=side, verbose=verbose, **kwargs)
    elif strategy == "random_noise": res = apply_random_surface_noise(atoms, side=side, verbose=verbose, **kwargs)
    else: res = atoms
    return standardize_vasp_atoms(res, z_min_offset=0.5)

PAULING_EN = {1: 2.20, 2: 0.0, 3: 0.98, 4: 1.57, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 10: 0.0, 11: 0.93, 12: 1.31, 13: 1.61, 14: 1.90, 15: 2.19, 16: 2.58, 17: 3.16, 18: 0.0, 19: 0.82, 20: 1.0, 21: 1.36, 22: 1.54, 23: 1.63, 24: 1.66, 25: 1.55, 26: 1.83, 27: 1.88, 28: 1.91, 29: 1.90, 30: 1.65, 31: 1.81, 32: 2.01, 33: 2.18, 34: 2.55, 35: 2.96, 36: 3.0, 37: 0.82, 38: 0.95, 39: 1.22, 40: 1.33, 41: 1.60, 42: 2.16, 43: 1.90, 44: 2.20, 45: 2.28, 46: 2.20, 47: 1.93, 48: 1.69, 49: 1.78, 50: 1.96, 51: 2.05, 52: 2.10, 53: 2.66, 54: 2.60, 55: 0.79, 56: 0.89, 57: 1.10, 58: 1.12, 59: 1.13, 60: 1.14, 61: 1.13, 62: 1.17, 63: 1.20, 64: 1.20, 65: 1.22, 66: 1.23, 67: 1.24, 68: 1.24, 69: 1.25, 70: 1.10, 71: 1.27, 72: 1.30, 73: 1.50, 74: 2.36, 75: 1.90, 76: 2.20, 77: 2.20, 78: 2.28, 79: 2.54, 80: 2.00, 81: 1.62, 82: 2.33, 83: 2.02, 84: 2.00, 85: 2.20, 86: 2.20, 87: 0.70, 88: 0.90, 89: 1.10, 90: 1.30, 91: 1.50, 92: 1.38, 93: 1.36, 94: 1.28}

def auto_reconstruct_surface(atoms, side="top", verbose=False, **kwargs):
    """Intelligent Reconstruction Engine."""
    idx = find_surface_indices(atoms, side=side, threshold=1.5)
    if not len(idx): return atoms
    chi = np.array([PAULING_EN.get(n, 2.0) for n in atoms.numbers[idx]])
    is_iv = all(n in [6, 14, 32] for n in atoms.numbers[idx])
    is_ionic = (np.max(chi) - np.min(chi)) > 1.5
    is_metal = np.mean(chi) < 1.9 or all(n not in [6,7,8,9,15,16,17] for n in atoms.numbers)
    if is_iv: return reconstruct_si100_2x1_buckled(atoms, side=side, verbose=verbose)
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

SI_VALENCE_MAP = {"Si": 4, "O": 2, "H": 1, "F": 1, "Cl": 1}

def get_natural_pairing_vector(atoms, idx, neighbor_data=None):
    """Determine the lateral pairing axis."""
    vecs = generate_vsepr_vectors(atoms, idx, neighbor_data=neighbor_data, num_missing=2)
    if len(vecs) == 2:
        d = vecs[0] - vecs[1]; d[2] = 0; mag = np.linalg.norm(d)
        if mag > 1e-3: return d / mag
    return None

def reconstruct_si100_2x1_buckled(atoms, side="top", buckle=0.7, bond_length=2.30, pattern="checkerboard", verbose=False):
    """Advanced Vector-Agnostic 2x1 reconstruction."""
    idx_list = find_surface_indices(atoms, side)
    if not len(idx_list): return atoms
    paired, dimers = set(), []
    from ase.neighborlist import neighbor_list
    i_list, _ = neighbor_list("ij", atoms, 2.6)
    for i1 in idx_list:
        if i1 in paired or np.sum(i_list == i1) >= 4: continue
        pv = get_natural_pairing_vector(atoms, i1)
        if pv is None: continue
        pot = [i for i in idx_list if i != i1 and i not in paired]
        if not pot: continue
        D, d = get_distances(atoms.positions[i1], atoms.positions[pot], cell=atoms.cell, pbc=atoms.pbc)
        for sub, i2 in enumerate(pot):
            if 2.0 < d[0][sub] < 4.2 and abs(np.dot(D[0][sub]/d[0][sub], pv)) > 0.8:
                dimers.append({"ids": (i1, i2), "vec": D[0][sub], "mid": (atoms.positions[i1] + atoms.positions[i1] + D[0][sub])/2})
                paired.update([i1, i2]); break
    if not dimers: return atoms
    inv = np.linalg.inv(atoms.cell[:2, :2])
    rows = sorted(list(set(round((d["mid"][:2] @ inv)[1] * 8, 1) for d in dimers)))
    cols = sorted(list(set(round((d["mid"][:2] @ inv)[0] * 8, 1) for d in dimers)))
    for d in dimers:
        r, c = rows.index(round((d["mid"][:2] @ inv)[1] * 8, 1)), cols.index(round((d["mid"][:2] @ inv)[0] * 8, 1))
        S = (-1)**(r+c) if pattern=="checkerboard" else ((-1)**c if pattern=="stripe" else 1)
        i1, i2 = d["ids"]; v = d["vec"] if (d["vec"][0] > 1e-4 or (abs(d["vec"][0]) < 1e-4 and d["vec"][1] > 1e-4)) else -d["vec"]
        if v[0] < 0: i1, i2 = i2, i1
        mid, u = atoms.positions[i1] + v/2, v/np.linalg.norm(v)
        dxy = np.sqrt(max(0, bond_length**2 - buckle**2))
        atoms.positions[i1] = mid + u*(dxy/2) + [0, 0, S*buckle/2]
        atoms.positions[i2] = mid - u*(dxy/2) - [0, 0, S*buckle/2]
    atoms.wrap(); return atoms

def identify_surface_bonds(atoms, cutoff=2.6):
    """Categorize bonds."""
    l1 = find_surface_indices(atoms, "top", threshold=0.8, species="Si"); zt = np.max(atoms.positions[l1, 2])
    l2 = np.where((atoms.symbols == "Si") & (atoms.positions[:,2] < zt-0.5) & (atoms.positions[:,2] > zt-2.5))[0]
    i, j, _ = neighbor_list("ijD", atoms, cutoff); dims, bbs, seen = [], [], set()
    for i1 in l1:
        for ni in j[i == i1]:
            if ni == i1 or atoms.symbols[ni] != "Si": continue
            b = tuple(sorted((i1, ni)))
            if b in seen: continue
            if ni in l1: dims.append(b)
            elif ni in l2: bbs.append(b)
            seen.add(b)
    return dims, bbs

def oxidize_si_surface(slab, dimer_coverage=0.0, backbond_coverage=0.0, verbose=False):
    """Oxidize."""
    dims, bbs = identify_surface_bonds(slab); nd, nb = int(round(len(dims)*dimer_coverage)), int(round(len(bbs)*backbond_coverage))
    res, oc = slab.copy(), {i: 0 for i in range(len(slab))}
    def greedy(at, cands, n):
        curr, s = at.copy(), 0
        avail = list(cands)
        while s < n and avail:
            opos = curr.positions[curr.symbols == "O"]; bb, bs, bi = None, -1.0, -1
            for ib, (b1, b2) in enumerate(avail):
                if oc[b1] >= 2 or oc[b2] >= 2: continue
                mid = (curr.positions[b1] + curr.positions[b2])/2.0
                score = 100.0 - np.linalg.norm(mid[:2] - np.sum(curr.cell, 0)[:2]/2) if not len(opos) else np.min(get_distances(mid, opos, cell=curr.cell, pbc=curr.pbc)[1])
                if score > bs:
                    if np.any(np.linalg.norm(np.delete(curr.positions, [b1, b2], 0) - mid, 1) < 1.5): continue
                    bs, bb, bi = score, (b1, b2), ib
            if bb:
                curr = insert_o_bridge_pure_geo(curr, bb[0], bb[1]); oc[bb[0]] += 1; oc[bb[1]] += 1; s += 1; avail.pop(bi)
            else: break
        return curr, s
    res, _ = greedy(res, dims, nd); res, _ = greedy(res, bbs, nb); return res

def insert_o_bridge_pure_geo(atoms, idx1, idx2, target_si_o=1.63, target_angle=144.0):
    """Insert O."""
    p1, p2 = atoms.positions[idx1].copy(), atoms.positions[idx2].copy()
    if p2[2] > p1[2]: idx1, idx2, p1, p2 = idx2, idx1, p2, p1
    v = p1 - p2; bl = np.linalg.norm(v); u = v / bl; mid = (p1 + p2)/2.0
    dp, da = target_si_o * np.sin(np.deg2rad(target_angle/2)), target_si_o * np.cos(np.deg2rad(target_angle/2))
    perp = np.cross(u, [0,0,1])
    if np.linalg.norm(perp) < 1e-3: perp = np.cross(u, [0,1,0])
    perp /= np.linalg.norm(perp)
    atoms.positions[idx1] += u * (da - bl/2) * 0.8; atoms.positions[idx2] -= u * (da - bl/2) * 0.2
    atoms += Atoms("O", positions=[mid + perp * dp]); return atoms

def build_si100_slab(bulk_atoms, size=(4, 4), layers=8, vacuum=10.0):
    """Standardized (100) slab generation."""
    slab = surface(bulk_atoms, (1, 0, 0), layers=layers, vacuum=vacuum) * (size[0], size[1], 1)
    z = slab.positions[:, 2]; zmax, zmin = z.max(), z.min()
    for a in slab:
        if a.position[2] > zmax - 0.5: a.tag = 1
        elif a.position[2] < zmin + 0.5: a.tag = 4
        else: a.tag = 0
    return slab

def generate_standard_surfaces(bulk_si, verbose=False):
    """Generate surfaces."""
    base = build_si100_slab(bulk_si, size=(4, 4), layers=8)
    s1 = base.copy(); reconstruct_si100_2x1_buckled(s1, verbose=verbose); s1.info["label"] = "S1_Clean_2x1"
    s2 = s1.copy(); s2 = passivate_surface_coverage_general(s2, 1.0, SI_VALENCE_MAP, side="top"); s2 = passivate_surface_coverage_general(s2, 1.0, SI_VALENCE_MAP, side="bottom"); s2.info["label"] = "S2_H_Passivated"
    s3 = s1.copy(); s3 = oxidize_si_surface(s3, 0.5, 0.5); s3.info["label"] = "S3_Oxidized"
    s4 = s3.copy(); s4 = passivate_surface_coverage_general(s4, 1.0, SI_VALENCE_MAP, side="top"); s4 = passivate_surface_coverage_general(s4, 1.0, SI_VALENCE_MAP, side="bottom"); s4.info["label"] = "S4_Oxidized_H_Passivated"
    return [s1, s2, s3, s4]

def get_surface_h_mapping(atoms, cutoff=1.8, side="top"):
    """Map H."""
    hi, si = np.where(atoms.symbols == "H")[0], np.where(atoms.symbols == "Si")[0]
    if not len(hi): return {}
    hi = [i for i in hi if (atoms.positions[i, 2] > np.max(atoms.positions[:, 2]) - 3.0 if side == "top" else atoms.positions[i, 2] < np.min(atoms.positions[:, 2]) + 3.0)]
    m = {}
    for h in hi:
        _, d = get_distances(atoms.positions[h], atoms.positions[si], cell=atoms.cell, pbc=atoms.pbc)
        if np.any(d[0] < cutoff): m[si[np.argmin(d[0])]] = h
    return m
