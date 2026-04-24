import numpy as np
from ase import Atoms
from ase.build import surface, make_supercell
from ase.io import read
import math
from knowledge_engine import chem_kb

def standardize_vasp_atoms(atoms, z_min_offset=0.5):
    """
    Standardize Atoms object for VASP export:
    1. Sort by atomic number (element).
    2. Align minimum Z-coordinate to z_min_offset.
    Returns: Sorted and translated Atoms copy.
    """
    # 1. Sort by atomic number
    sorted_atoms = atoms[atoms.numbers.argsort()]
    
    # 2. Align Z-min
    z_min = sorted_atoms.positions[:, 2].min()
    sorted_atoms.translate([0, 0, z_min_offset - z_min])
    
    return sorted_atoms

def write_standardized_vasp(filepath, atoms, z_min_offset=0.5):
    """
    Standardizes the atoms object (sort by element, align z_min) and saves it to a VASP file.
    This provides a common routine for VASP structure generation.
    """
    from ase.io import write
    standardized = standardize_vasp_atoms(atoms, z_min_offset=z_min_offset)
    write(filepath, standardized, format='vasp')

def find_surface_indices(atoms, side='top', threshold=1.0, species=None):
    """Find indices of atoms at the top or bottom surface based on Z-coordinates."""
    if species:
        indices = np.where(atoms.symbols == species)[0]
    else:
        indices = np.arange(len(atoms))
    
    if len(indices) == 0: return []
    z_coords = atoms.positions[indices, 2]
    
    z_target = np.max(z_coords) if side == 'top' else np.min(z_coords)
    mask = np.abs(z_coords - z_target) < threshold
    return indices[mask]

def check_overlap(atoms, cutoff=1.2, verbose=False):
    """Check for steric overlaps between atoms using a simple distance threshold."""
    from ase.neighborlist import neighbor_list
    i_list, j_list, dists = neighbor_list('ijd', atoms, cutoff)
    if len(i_list) > 0:
        if verbose:
            print(f"Overlap detected: {len(i_list)//2} pairs closer than {cutoff}A")
        return True
    return False

def generate_vsepr_vectors(atoms, idx, neighbor_data=None, num_missing=1, cutoff=2.6):
    """
    Calculate generic dangling bond vectors using VSEPR approximation.
    Distributes num_missing vectors symmetrically around the inverse sum of neighbors.
    """
    from ase.neighborlist import neighbor_list
    if neighbor_data:
        i_list, j_list, D_list = neighbor_data
    else:
        i_list, j_list, D_list = neighbor_list('ijD', atoms, cutoff)
    
    mask = (i_list == idx)
    vectors = D_list[mask]
    
    # Filter to only covalent neighbors
    dists = np.linalg.norm(vectors, axis=1)
    vectors = vectors[(dists > 0.1) & (dists < cutoff)]
    
    if len(vectors) == 0:
        # Fallback: point along Z-axis if no neighbors found
        return [np.array([0., 0., 1.])] * num_missing
        
    # Normalize neighbor vectors
    norm_vecs = vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]
    sum_vec = np.sum(norm_vecs, axis=0)
    
    # Primary direction: away from the geometric center of existing neighbors
    v_target = -sum_vec
    if np.linalg.norm(v_target) < 1e-4:
        v_target = np.array([0., 0., 1.])
    v_target /= np.linalg.norm(v_target)

    if num_missing == 1:
        return [v_target]

    if num_missing == 2 and len(vectors) == 2:
        # Optimized for Tetrahedral/Square-planar geometries (AX2E2)
        # Mirrors the logic for Si(100) surface dimers
        w_unit = v_target
        u = norm_vecs[0] - norm_vecs[1]
        u_norm = np.linalg.norm(u)
        if u_norm > 1e-4:
            u_unit = u / u_norm
            p_unit = np.cross(w_unit, u_unit)
            p_unit /= np.linalg.norm(p_unit)
            
            # tetrahedral coefficients (cos(54.7), sin(54.7))
            v1 = w_unit * 0.577 + p_unit * 0.816
            v2 = w_unit * 0.577 - p_unit * 0.816
            return [v1, v2]

    # General fallback for num_missing > 1: Conical distribution
    results = []
    # Small cone spread for multiple dangling bonds
    theta = np.deg2rad(20.0) 
    
    # Find perpendicular axes
    perp_vec = np.array([1., 0., 0.])
    if abs(np.dot(v_target, perp_vec)) > 0.9:
        perp_vec = np.array([0., 1., 0.])
    
    axis_1 = np.cross(v_target, perp_vec)
    axis_1 /= np.linalg.norm(axis_1)
    axis_2 = np.cross(v_target, axis_1)
    
    for i in range(num_missing):
        phi = 2 * np.pi * i / num_missing
        v = v_target * np.cos(theta) + (axis_1 * np.cos(phi) + axis_2 * np.sin(phi)) * np.sin(theta)
        results.append(v / np.linalg.norm(v))
        
    return results

def get_all_dangling_bonds_general(atoms, valence_map, vector_generator=None, cutoff=3.1, side='top'):
    """
    Identify missing valences for surface atoms using platform-independent logic.
    valence_map: dict {sym: int} or function(symbol) -> int
    vector_generator: function(atoms, idx, neighbor_data, num_missing) -> list of vectors.
                      If None, generate_vsepr_vectors is used.
    """
    from ase.neighborlist import neighbor_list
    surface_indices = find_surface_indices(atoms, side=side, threshold=2.0)
    
    # Use a generous cutoff for neighbor list to share with vector generator
    i_list, j_list, D_list = neighbor_list('ijD', atoms, cutoff)
    neighbor_data = (i_list, j_list, D_list)
    
    if vector_generator is None:
        vector_generator = generate_vsepr_vectors
        
    all_bonds = []
    for idx in surface_indices:
        sym = atoms.symbols[idx]
        target_val = chem_kb.get_ideal_coordination(sym, config=valence_map if isinstance(valence_map, dict) else None)
        if target_val <= 0: continue

        mask = (i_list == idx)
        dists = np.linalg.norm(D_list[mask], axis=1)
        # Count only true covalent neighbors
        num_n = np.sum((dists > 0.1) & (dists < 2.6))
        num_missing = target_val - num_n
        
        if num_missing > 0:
            # Pass num_missing to the generator
            try:
                vecs = vector_generator(atoms, idx, neighbor_data=neighbor_data, num_missing=num_missing)
            except TypeError:
                # Fallback for generators that don't take num_missing yet
                vecs = vector_generator(atoms, idx, neighbor_data=neighbor_data)
                
            for v in vecs:
                # Basic directional filter: point into vacuum
                if (side == 'top' and v[2] > -0.1) or (side == 'bottom' and v[2] < 0.1):
                    all_bonds.append({'parent': idx, 'vector': v, 'parent_sym': sym})
    return all_bonds

def passivate_surface_coverage_general(atoms, h_coverage, valence_map, vector_generator=None, 
                                       element='H', cutoff=3.1, side='top', verbose=False):
    """Uniformly passivate a surface using a greedy max-min distance algorithm."""
    from ase.geometry import get_distances
    from ase.data import covalent_radii
    
    candidates = get_all_dangling_bonds_general(atoms, valence_map, vector_generator, cutoff, side)
    if not candidates: 
        if verbose: print(f"  [Passivation] No dangling bonds found on {side} surface.")
        return atoms
    
    n_target = int(round(len(candidates) * h_coverage))
    if n_target == 0: return atoms
    
    current_atoms = atoms.copy()
    success = 0
    available = list(candidates)
    
    r_pass = covalent_radii[Atoms(element).numbers[0]]
    
    if verbose: print(f"  [Passivation] Targeting {n_target} {element} sites on {side} surface.")
    
    while success < n_target and available:
        pass_indices = [i for i, sym in enumerate(current_atoms.symbols) if sym == element]
        # Reference positions to maximize distance from (existing passivation + oxygens for steric)
        ref_indices = pass_indices + [i for i, sym in enumerate(current_atoms.symbols) if sym == 'O']
        ref_pos = current_atoms.positions[ref_indices] if ref_indices else []
        
        best_cand_idx = -1
        best_score = -1.0
        
        for i_c, cand in enumerate(available):
            parent_pos = current_atoms.positions[cand['parent']]
            r_parent = covalent_radii[atoms.numbers[cand['parent']]]
            b_len = r_parent + r_pass
            
            h_pos_candidate = parent_pos + cand['vector'] * b_len
            
            if len(ref_pos) == 0:
                score = 100.0
            else:
                dists = get_distances(h_pos_candidate, ref_pos, cell=current_atoms.cell, pbc=current_atoms.pbc)[1]
                score = np.min(dists)
            
            if score > best_score:
                # Overlap check
                _, all_dists_list = get_distances(h_pos_candidate, current_atoms.positions, 
                                                 cell=current_atoms.cell, pbc=current_atoms.pbc)
                all_dists = all_dists_list[0]
                
                mask = np.ones(len(all_dists), dtype=bool)
                mask[cand['parent']] = False
                # Use a combined radius for overlap threshold
                if np.any(all_dists[mask] < 0.8): continue
                
                best_score = score
                best_cand_idx = i_c
        
        if best_cand_idx != -1:
            cand = available.pop(best_cand_idx)
            r_parent = covalent_radii[atoms.numbers[cand['parent']]]
            b_len = r_parent + r_pass
            
            # Empirical refinements if needed (e.g. Si-H, O-H)
            if cand['parent_sym'] == 'Si' and element == 'H': b_len = 1.48
            if cand['parent_sym'] == 'O' and element == 'H': b_len = 0.96
            
            h_pos = current_atoms.positions[cand['parent']] + cand['vector'] * b_len
            current_atoms += Atoms(element, positions=[h_pos])
            current_atoms.wrap()
            success += 1
        else:
            break
            
    if verbose: print(f"  [Passivation] Successfully placed {success}/{n_target} {element} atoms on {side} surface.")
    return current_atoms

def identify_protectors(atoms, config, verbose=False):
    """
    Infers which atoms belong to the protector layer vs the base substrate.
    Enhanced with element-based filtering for robust identification in Case B.
    """
    import numpy as np
    # Priority: inline 'protector' key (legacy) > reaction_search.mechanisms.protector_exchange > {}
    protector_cfg = config.get('protector',
                    config.get('reaction_search', {}).get('mechanisms', {}).get('protector_exchange', {}))
    
    heuristic = protector_cfg.get('heuristic', 'graph')
    inhibitor_elements = protector_cfg.get('inhibitor_elements', [])
    
    if heuristic == 'tag':
        target_tags = protector_cfg.get('target_tags', [4, 5])
        tags = atoms.get_tags()
        p_mask = np.isin(tags, target_tags)
        s_mask = ~p_mask
        return np.where(s_mask)[0], np.where(p_mask)[0]
        
    elif heuristic in ['z_height', 'graph']:
        from scipy.sparse.csgraph import connected_components
        from scipy.sparse import csr_matrix
        from ase.data import covalent_radii
        from ase.geometry import get_distances
        
        n_atoms = len(atoms)
        adj = np.zeros((n_atoms, n_atoms), dtype=int)
        D, d = get_distances(atoms.positions, atoms.positions, cell=atoms.cell, pbc=atoms.pbc)
        
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                cutoff = covalent_radii[atoms.numbers[i]] + covalent_radii[atoms.numbers[j]] + 0.3
                if d[i,j] < cutoff and d[i,j] > 0.1:
                    adj[i,j] = 1
                    adj[j,i] = 1
                    
        graph = csr_matrix(adj)
        n_comp, labels = connected_components(csgraph=graph, directed=False)
        
        comp_sizes = np.bincount(labels)
        substrate_comp = np.argmax(comp_sizes)
        
        s_indices = np.where(labels == substrate_comp)[0]
        p_indices = []
        
        for c in range(n_comp):
            if c == substrate_comp: continue
            
            cluster_indices = np.where(labels == c)[0]
            cluster_symbols = set(atoms.symbols[cluster_indices])
            
            # If inhibitor_elements is provided, only treat it as protector 
            # if it matches the elements. Otherwise it's part of substrate or noise.
            if inhibitor_elements:
                if any(sym in inhibitor_elements for sym in cluster_symbols):
                    p_indices.extend(cluster_indices)
                else:
                    # Treat as substrate if it doesn't match inhibitor profile
                    # (e.g. surface reconstructions or native oxides)
                    pass
            else:
                # Default: all non-substrate clusters are protectors
                p_indices.extend(cluster_indices)
        
        p_indices = np.array(p_indices, dtype=int)
        s_mask = np.ones(n_atoms, dtype=bool)
        s_mask[p_indices] = False
        
        if verbose and len(p_indices) > 0:
            print(f"  [Protector Inference] Identified {len(p_indices)} protector atoms across {n_comp-1} clusters.")
            
        return np.where(s_mask)[0], p_indices
        
    return np.arange(len(atoms)), np.array([], dtype=int)
    

class CavityDetector:
    def __init__(self, slab, substrate_indices, protector_indices, grid_res=0.2, verbose=False):
        self.slab = slab
        self.sub_idx = substrate_indices
        self.prot_idx = protector_indices
        self.grid_res = grid_res
        self.verbose = verbose
        
    def find_void_centers(self, top_clearance=4.0):
        import numpy as np
        if len(self.prot_idx) == 0:
            # If no protectors, generate a grid across the entire cell surface
            z_max = np.max(self.slab.positions[self.sub_idx, 2]) if len(self.sub_idx) else np.max(self.slab.positions[:, 2])
            nx = int(np.ceil(self.slab.cell[0,0] / 5.0)) # ~5A spacing for sites
            ny = int(np.ceil(self.slab.cell[1,1] / 5.0))
            grid_centers = []
            for i in range(nx):
                for j in range(ny):
                    grid_centers.append(np.array([
                        (i + 0.5) * (self.slab.cell[0,0] / nx),
                        (j + 0.5) * (self.slab.cell[1,1] / ny),
                        z_max + top_clearance
                    ]))
            return grid_centers
            
        from ase.data import vdw_radii
        from scipy.ndimage import distance_transform_edt
        from scipy.ndimage import maximum_filter
        
        cell = self.slab.get_cell()
        lx, ly = cell[0,0], cell[1,1]
        
        z_sub_top = np.max(self.slab.positions[self.sub_idx, 2])
        z_prot_top = np.max(self.slab.positions[self.prot_idx, 2])
        
        if z_prot_top <= z_sub_top:
            return [np.array([lx/2, ly/2, z_sub_top + top_clearance])]
            
        nx = int(np.ceil(lx / self.grid_res))
        ny = int(np.ceil(ly / self.grid_res))
        lz = (z_prot_top + top_clearance) - z_sub_top
        nz = int(np.ceil(lz / self.grid_res))
        
        if nx <= 0 or ny <= 0 or nz <= 0:
            return []
            
        grid = np.ones((nx, ny, nz), dtype=bool)
        
        for idx in self.prot_idx:
            pos = self.slab.positions[idx]
            r = 1.5
            try:
                r = vdw_radii[self.slab.numbers[idx]]
                if np.isnan(r): r = 1.5
            except:
                pass
            
            gx = int((pos[0] % lx) / self.grid_res)
            gy = int((pos[1] % ly) / self.grid_res)
            gz = int((pos[2] - z_sub_top) / self.grid_res)
            
            ir = int(np.ceil((r + 1.2) / self.grid_res)) # 1.2A steric buffer
            x_min, x_max = max(0, gx-ir), min(nx, gx+ir+1)
            y_min, y_max = max(0, gy-ir), min(ny, gy+ir+1)
            z_min, z_max = max(0, gz-ir), min(nz, gz+ir+1)
            grid[x_min:x_max, y_min:y_max, z_min:z_max] = False
            
        dist = distance_transform_edt(grid) * self.grid_res
        
        local_max = maximum_filter(dist, size=3) == dist
        local_max[dist < 0.5] = False
        
        max_coords = np.argwhere(local_max)
        centers = []
        sizes = []
        for c in max_coords:
            x = (c[0] + 0.5) * self.grid_res
            y = (c[1] + 0.5) * self.grid_res
            z = z_sub_top + (c[2] + 0.5) * self.grid_res
            centers.append(np.array([x, y, z]))
            sizes.append(dist[c[0], c[1], c[2]])
            
        centers = [x for _, x in sorted(zip(sizes, centers), key=lambda pair: pair[0], reverse=True)]
        
        if self.verbose:
            print(f"  [CavityDetector] Found {len(centers)} potential void centers inside the protector layer.")
            
        # Cluster centers that are too close to reduce redundancy
        filtered_centers = []
        for c in centers:
            if not filtered_centers:
                filtered_centers.append(c)
            else:
                dists = np.linalg.norm(np.array(filtered_centers) - c, axis=1)
                if np.all(dists > 2.0):
                    filtered_centers.append(c)
            if len(filtered_centers) >= 5: # keep top 5 unique cavities
                break
                
        return filtered_centers

def create_slab_from_bulk(bulk_atoms, miller_indices, thickness, vacuum, target_area=None, supercell_matrix=None, 
                           termination=None, top_termination=None, bottom_termination=None, verbose=False):
    """
    Generates a substrate slab from a bulk structure with geometric constraints.
    Supports asymmetric termination control (top vs bottom).
    """
    # 1. Determine layers for thickness
    s1 = surface(bulk_atoms, miller_indices, layers=1)
    s2 = surface(bulk_atoms, miller_indices, layers=2)
    
    z1 = np.max(s1.positions[:, 2]) - np.min(s1.positions[:, 2])
    z2 = np.max(s2.positions[:, 2]) - np.min(s2.positions[:, 2])
    d_hkl = z2 - z1
    
    if d_hkl < 0.1: d_hkl = 2.0 
    num_layers = int(math.ceil(thickness / d_hkl))
    
    # 2. Handle Termination & Slicing (Layer-Wise Engine)
    # If generic 'termination' is provided, treat it as symmetric top/bottom
    if termination and not top_termination: top_termination = termination
    if termination and not bottom_termination: bottom_termination = termination

    # Enable layer discovery if any termination constraint is set
    any_term = any([termination, top_termination, bottom_termination])
    
    if any_term or termination in ["symmetric", "uniform"]:
        if verbose: print(f"  [Substrate Factory] Executing Layer-Wise Discovery (Bottom={bottom_termination}, Top={top_termination})...")
        
        # Create a sufficiently thick base slab
        test_slab = surface(bulk_atoms, miller_indices, layers=num_layers * 2, vacuum=0)
        test_slab.wrap()
        
        # [Step 1] Identify Atomic Planes via Z-Clustering
        z_coords = test_slab.positions[:, 2]
        sorted_indices = np.argsort(z_coords)
        sorted_z = z_coords[sorted_indices]
        
        planes = []
        if len(sorted_z) > 0:
            current_plane = [sorted_indices[0]]
            for i in range(1, len(sorted_z)):
                if sorted_z[i] - sorted_z[i-1] < 0.5:
                    current_plane.append(sorted_indices[i])
                else:
                    planes.append(current_plane)
                    current_plane = [sorted_indices[i]]
            planes.append(current_plane)
            
        # [Step 2] Fingerprint Planes
        plane_data = []
        for p_idx, p_atoms in enumerate(planes):
            syms = sorted(test_slab.symbols[p_atoms])
            elem_set = set(syms)
            z_avg = np.mean(test_slab.positions[p_atoms, 2])
            plane_data.append({
                'idx': p_idx,
                'atom_indices': p_atoms,
                'elements': elem_set,
                'sym_list': syms,
                'z': z_avg
            })
            
        # [Step 3] Combinatorial Search for Matching Planes
        best_pair = None
        best_score = -1e9
        
        for i in range(len(plane_data)):
            for j in range(i + 1, len(plane_data)):
                p_bot, p_top = plane_data[i], plane_data[j]
                
                # Baseline scoring factors
                dist = p_top['z'] - p_bot['z']
                thickness_err = abs(dist - thickness)
                
                score = 0
                
                # Criterion 1: User-defined Termination Match
                bot_match = (not bottom_termination) or (bottom_termination in p_bot['elements'])
                top_match = (not top_termination) or (top_termination in p_top['elements'])
                
                if bottom_termination and bot_match: score += 2000
                if top_termination and top_match: score += 2000
                
                # Criterion 2: Symmetry/Uniformity (if requested or as secondary score)
                species_match = (p_bot['elements'] == p_top['elements'])
                count_match = (p_bot['sym_list'] == p_top['sym_list'])
                if species_match: score += 500
                if count_match: score += 200
                
                # Favor O-termination for unknown oxides
                if 'O' in p_top['elements'] and len(p_top['elements']) == 1:
                    score += 100
                
                # Criterion 3: Thickness Proximity (Penalty)
                score -= thickness_err * 20 
                
                if score > best_score:
                    best_score = score
                    best_pair = (p_bot, p_top)
        
        if best_pair:
            p_bot, p_top = best_pair
            
            # Check for "Impossible Conditions" warning
            actual_bot_match = (not bottom_termination) or (bottom_termination in p_bot['elements'])
            actual_top_match = (not top_termination) or (top_termination in p_top['elements'])
            if (bottom_termination or top_termination) and not (actual_bot_match and actual_top_match):
                print(f"  [Substrate Factory] WARNING: Requested termination ({bottom_termination}/{top_termination}) could not be fully satisfied. Using best match.")
            
            # Extraction: All atoms between these planes (inclusive)
            mask = (z_coords >= p_bot['z'] - 0.1) & (z_coords <= p_top['z'] + 0.1)
            slab = test_slab[mask]
            if verbose:
                print(f"  [Substrate Factory] Selected planes {p_bot['idx']} and {p_top['idx']}.")
                print(f"  [Substrate Factory] Terminal Bottom: {p_bot['elements']}, Top: {p_top['elements']}, Thickness: {p_top['z']-p_bot['z']:.2f} A")
        else:
            if verbose: print("  [Substrate Factory] Warning: No valid layers found. Falling back.")
            slab = surface(bulk_atoms, miller_indices, layers=num_layers, vacuum=0)
    else:
        # Default cut
        slab = surface(bulk_atoms, miller_indices, layers=num_layers, vacuum=0)

    # Add vacuum and center
    slab.center(vacuum=vacuum, axis=2)
    
    # 3. Supercell Expansion
    if supercell_matrix is not None:
        if verbose: print(f"  [Substrate Factory] Applying manual supercell matrix: {supercell_matrix}")
        # Matrix should be 3x3 for make_supercell, but user usually gives 2x2.
        m = np.eye(3)
        m[0,0], m[0,1] = supercell_matrix[0][0], supercell_matrix[0][1]
        m[1,0], m[1,1] = supercell_matrix[1][0], supercell_matrix[1][1]
        slab = make_supercell(slab, m)
    elif target_area is not None:
        a1, a2 = slab.cell[0], slab.cell[1]
        area_prim = np.linalg.norm(np.cross(a1, a2))
        
        # Max repeats while area <= target_area
        max_repeats = int(target_area // area_prim)
        if max_repeats < 1: max_repeats = 1
        
        # Find n, m such that n*m <= max_repeats and final cell is square-ish
        # Target aspect ratio is 1.0. Current primitive aspect ratio is |a1|/|a2|
        l1, l2 = np.linalg.norm(a1), np.linalg.norm(a2)
        
        best_n, best_m = 1, 1
        best_score = -1.0
        
        for n in range(1, max_repeats + 1):
            for m in range(1, max_repeats // n + 1):
                current_area = n * m * area_prim
                ratio = (n * l1) / (m * l2)
                aspect_score = 1.0 / (1.0 + abs(ratio - 1.0)) # 1.0 is perfect, < 1.0 is worse
                
                # Balanced score: Area * AspectScore
                # This penalizes highly elongated cells even if they have slightly more area
                score = current_area * aspect_score
                
                if score > best_score + 1e-4:
                    best_score = score
                    best_n, best_m = n, m
        
        if verbose:
            print(f"  [Substrate Factory] Primitive area: {area_prim:.2f} A^2. Target: {target_area} A^2.")
            print(f"  [Substrate Factory] Selected expansion: {best_n}x{best_m} (Final area: {best_n*best_m*area_prim:.2f} A^2, Score: {best_score:.2f})")
            
        slab = slab * (best_n, best_m, 1)

    # 4. Alignment: Rotate so first lattice vector is along [1,0,0]
    v1 = slab.cell[0]
    # Projected vector on XY plane
    v1_xy = np.array([v1[0], v1[1], 0.0])
    if np.linalg.norm(v1_xy) > 1e-4:
        angle = -math.atan2(v1_xy[1], v1_xy[0])
        slab.rotate(angle * 180 / math.pi, 'z', rotate_cell=True)
    
    # Final centering and wrapping
    slab.wrap()
    
    return slab
