import os
from itertools import combinations

import numpy as np
from ase import Atoms

from .ads_workflow_mgr import AdsorptionWorkflowManager
from ..utils.knowledge_engine import chem_kb

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None


def _bond_length_to_h(symbol: str) -> float:
    """Returns the X-H bond length (A) used for byproduct/passivation placement."""
    if symbol in ("N", "O"):
        return 1.0
    if symbol == "C":
        return 1.1
    return 1.5


def _unique_ligands(ligands: list) -> list:
    """Returns ligands de-duplicated by chemical formula (first occurrence wins)."""
    seen = set()
    result = []
    for l in ligands:
        f = l.get("formula", "Unknown")
        if f not in seen:
            seen.add(f)
            result.append(l)
    return result


def analyze_surface_reactivity(surface, config, prot_idx=[], verbose=False, results_dir=None, stage_type="precursor"):
    import numpy as np
    from ase.neighborlist import neighbor_list

    from .surface_utils import identify_protectors

    max_pair_dist = config.get("reaction_search", {}).get("candidate_filter", {}).get("max_pair_dist", 5.0)

    sub_idx, prot_idx = identify_protectors(surface, config, verbose=False)

    i_list, j_list, D_list = neighbor_list("ijD", surface, cutoff=4.0)
    d_list = np.linalg.norm(D_list, axis=1)

    dangling_sites = []
    exchange_sites = []

    # --- Configuration for Coordination Analysis ---
    chem_cfg = config.get("reaction_search", {}).get("mechanisms", {}).get(stage_type, {}).get("chemisorption", {})
    coord_cfg = chem_cfg.get("coordination_analysis", {})
    bond_slack = coord_cfg.get("bond_slack", 0.2)
    max_nb_dist = coord_cfg.get("max_neighbor_dist", 4.0)
    z_surface_threshold = coord_cfg.get("z_surface_threshold", 3.5)

    z_max = max(surface.positions[:, 2])
    z_sub_max = max(surface.positions[sub_idx, 2]) if len(sub_idx) else z_max

    # Use a NeighborList for efficient per-atom lookups
    from ase.neighborlist import NeighborList
    # Pre-calculate covalent radii for all elements in the surface
    radii = [chem_kb.get_radius(s, "covalent") + bond_slack for s in surface.symbols]
    nl = NeighborList(radii, skin=0.0, self_interaction=False, bothways=True)
    nl.update(surface)

    for idx in range(len(surface)):
        sym = surface.symbols[idx]
        
        # Substrate filtering: Only ignore inner substrate atoms.
        if idx in sub_idx and surface.positions[idx, 2] < z_sub_max - z_surface_threshold:
            continue

        # --- Mechanism 1: Protector Exchange ---
        if idx in prot_idx:
            _protex = config.get("reaction_search", {}).get("mechanisms", {}).get("protector_exchange", {})
            if not _protex.get("enabled", False):
                continue
            reactive_leaves = _protex.get("reactive_leaves", ["H"])
            if sym in reactive_leaves:
                indices, offsets = nl.get_neighbors(idx)
                # For exchange, we expect exactly 1 backbone neighbor within covalent range
                backbone_neighbors = []
                for n_idx, offset in zip(indices, offsets):
                    pos_j = surface.positions[n_idx] + np.dot(offset, surface.cell)
                    dist = np.linalg.norm(pos_j - surface.positions[idx])
                    if dist < 2.0: # Hard cutoff for leaf->backbone
                        backbone_neighbors.append((n_idx, pos_j - surface.positions[idx]))
                
                if len(backbone_neighbors) == 1:
                    db_vec = -backbone_neighbors[0][1]
                    db_vec = db_vec / np.linalg.norm(db_vec)
                    exchange_sites.append({
                        "index": idx,
                        "backbone_idx": backbone_neighbors[0][0],
                        "sym": sym,
                        "pos": surface.positions[backbone_neighbors[0][0]],
                        "leaf_pos": surface.positions[idx],
                        "db_vector": db_vec,
                        "missing_bonds": 1,
                    })
            continue

        # --- Mechanism 2: Substrate Dangling Bonds ---
        indices, offsets = nl.get_neighbors(idx)
        actual_coord = 0
        for n_idx, offset in zip(indices, offsets):
            pos_j = surface.positions[n_idx] + np.dot(offset, surface.cell)
            dist = np.linalg.norm(pos_j - surface.positions[idx])
            bond_cutoff = chem_kb.get_radius(sym, "covalent") + \
                          chem_kb.get_radius(surface.symbols[n_idx], "covalent") + bond_slack
            if dist < bond_cutoff:
                actual_coord += 1
        
        # Resolve ideal coordination: config override → chem_data default
        _ideal_coord = config.get("surface_prep", {}).get("surface_analysis", {}).get("ideal_coordination", {})
        expected = chem_kb.get_ideal_coordination(sym, _ideal_coord if _ideal_coord else None)

        if verbose:
            if surface.positions[idx, 2] > z_sub_max - 1.5:
                 print(f"  [Debug] Atom {idx}({sym}) at z={surface.positions[idx, 2]:.2f}: actual={actual_coord}, expected={expected}")

        if actual_coord < expected:
            from .surface_utils import generate_vsepr_vectors
            num_missing = expected - actual_coord
            # Use the pre-filtered i_list/j_list/D_list for VSEPR to maintain compatibility
            # or regenerate if needed. Here we use the original neighbor_list data for VSEPR logic.
            vecs = generate_vsepr_vectors(surface, idx, neighbor_data=(i_list, j_list, D_list), num_missing=num_missing)
            for db_vec in vecs:
                db_vec = db_vec / np.linalg.norm(db_vec)
                hit = False
                if len(prot_idx) > 0:
                    for p in prot_idx:
                        p_vec = surface.positions[p] - surface.positions[idx]
                        proj = np.dot(p_vec, db_vec)
                        if proj > 0.3: 
                            dist_to_ray = np.linalg.norm(p_vec - proj * db_vec)
                            if dist_to_ray < 1.0: 
                                hit = True
                                break
                if not hit and db_vec[2] > 0.1:
                    dangling_sites.append({
                        "index": idx,
                        "sym": sym,
                        "pos": surface.positions[idx],
                        "db_vector": db_vec,
                        "missing_bonds": expected - actual_coord,
                    })
    
    if verbose:
        print(f"  [Reactivity Analysis] Identified {len(dangling_sites)} dangling sites and {len(exchange_sites)} exchange sites.")

    # --- Proximity Filtering Logic ---
    mechs_cfg = config.get("reaction_search", {}).get("mechanisms", {})
    stage_cfg = mechs_cfg.get(stage_type, {})
    prox_cfg = stage_cfg.get("chemisorption", {}).get("proximity_filter", {})
    
    if prox_cfg.get("enabled", False) and len(prot_idx) > 0:
        from ase.geometry import get_distances
        cutoff = prox_cfg.get("cutoff", 7.0)
        
        inh_pos = surface.positions[prot_idx]
        
        def filter_proximity(sites):
            if not sites: return []
            site_pos = np.array([s["pos"] for s in sites])
            _, dists = get_distances(site_pos, inh_pos, cell=surface.cell, pbc=surface.pbc)
            min_dists = np.min(dists, axis=1)
            filtered = [s for s, d in zip(sites, min_dists) if d < cutoff]
            return filtered

        # Capture unfiltered sites for visualization
        unfiltered_dan = list(dangling_sites)
        unfiltered_exc = list(exchange_sites)

        orig_dan = len(dangling_sites)
        orig_exc = len(exchange_sites)
        dangling_sites = filter_proximity(dangling_sites)
        exchange_sites = filter_proximity(exchange_sites)
        
        if verbose:
            print(f"  [Proximity Filter] Reduced sites based on distance to inhibitor (cutoff={cutoff} A):")
            print(f"    - Dangling: {orig_dan} -> {len(dangling_sites)}")
            print(f"    - Exchange: {orig_exc} -> {len(exchange_sites)}")

        if prox_cfg.get("visualize", False) and results_dir:
            from ..utils.viz_utils import plot_site_proximity
            img_path = os.path.join(results_dir, "site_proximity_map.png")
            all_sites = unfiltered_dan + unfiltered_exc
            filt_sites = dangling_sites + exchange_sites
            plot_site_proximity(surface, all_sites, filt_sites, prot_idx, cutoff, img_path)
            if verbose:
                print(f"  [Proximity Filter] Visualization saved to {os.path.relpath(img_path)}")

    results = {"single": dangling_sites, "unique_single": [], "pairs": [], "exchange": exchange_sites}

    # Analyze Symmetry to reduce pair redundancies AND single-site duplicates
    import spglib

    lattice = surface.get_cell()
    pos = surface.get_scaled_positions()
    nums = surface.get_atomic_numbers()
    symprec = config.get("reaction_search", {}).get("symprec", 0.2)

    equiv_atoms = np.arange(len(surface))
    for prec in [symprec, 0.5]:
        try:
            dataset = spglib.get_symmetry_dataset((lattice, pos, nums), symprec=prec)
            if dataset:
                equiv_atoms = (
                    dataset.equivalent_atoms if hasattr(dataset, "equivalent_atoms") else dataset["equivalent_atoms"]
                )
                if len(np.unique(equiv_atoms)) < len(surface) or prec == 0.5:
                    break
        except Exception:
            pass

    # --- Symmetry-reduced single sites (one representative per symmetry class) ---
    unique_single_by_class = {}
    for s in dangling_sites:
        cls = int(equiv_atoms[s["index"]])
        if cls not in unique_single_by_class:
            unique_single_by_class[cls] = s
    results["unique_single"] = list(unique_single_by_class.values())

    # --- Symmetry-reduced pairs ---
    unique_pairs = {}
    pair_count = 0

    for s1, s2 in combinations(dangling_sites, 2):
        if s1["index"] == s2["index"]:
            continue
        dist = np.linalg.norm(s1["pos"] - s2["pos"])
        if dist <= max_pair_dist:
            pair_count += 1
            # Pair signature: sorted tuple of symmetry classes + rounded distance
            c1 = equiv_atoms[s1["index"]]
            c2 = equiv_atoms[s2["index"]]
            sig = tuple(sorted([c1, c2])) + (round(dist, 1),)

            if sig not in unique_pairs:
                unique_pairs[sig] = (s1, s2)

    results["pairs"] = list(unique_pairs.values())

    if verbose:
        print(
            f"  [Generic Reactivity] Identified {pair_count} potential active site pairs -> "
            f"Symmetry-reduced to {len(results['pairs'])} unique reaction pairs, "
            f"{len(results['unique_single'])} unique single sites."
        )

    return results


def analyze_molecule_ligands(molecule, center_target="Si", verbose=True, config=None):
    """Algorithmically fragments the precursor molecule to identify reactive ligands.
    Uses AdsorptionWorkflowManager implicitly for the heavy lifting.
    """
    symprec = config.get("reaction_search", {}).get("symprec", 0.2) if config else 0.2
    # Create a temporary manager to use its fragmentation logic.
    # Set verbose=False to silence molecule/fragment symmetry logs.
    mgr = AdsorptionWorkflowManager(molecule, symprec=symprec, verbose=False)
    c_idx, ligands = mgr.discover_ligands(molecule, center_target=center_target, verbose=verbose)
    return c_idx, ligands


def build_chemisorption_structures(
    molecule, center_target="Si", surface=None, rot_steps=8, config=None,
    verbose=True, tag=2, results_dir=None, stage_type="precursor"
):
    """Entry point for algorithmic chemisorption generation based on input molecule and surface.
    Identifies valid mechanisms based on available surface sites.
    """
    if verbose:
        print(f"\n--- Starting Algorithmic Chemisorption Routing (tag={tag}) ---")

    if config is None:
        config = {}
    
    # Extract chemisorption verbose flag from config
    mechs_cfg = config.get("reaction_search", {}).get("mechanisms", {})
    stage_cfg = mechs_cfg.get(stage_type, {})
    chem_verbose = stage_cfg.get("chemisorption", {}).get("verbose", False)
    
    from .surface_utils import standardize_vasp_atoms
    surface = standardize_vasp_atoms(surface)
    
    sites = analyze_surface_reactivity(surface, config, verbose=verbose, results_dir=results_dir, stage_type=stage_type)
    c_idx, ligands = analyze_molecule_ligands(molecule, center_target=center_target, verbose=verbose, config=config)

    candidates = []
    failed_candidates = []

    if not ligands:
        if verbose:
            print("  [Warning] No detachable ligands found. Aborting chemisorption.")
        return candidates

    # Get symmetry precision from config
    symprec = config.get("reaction_search", {}).get("symprec", 0.2)
    
    # We instantiate a manager scoped to the current surface for coordinate placement/overlap tests.
    # Silence redundant symmetry logs as they were already printed in the discovery stage.
    mgr = AdsorptionWorkflowManager(surface, config=config, symprec=symprec, verbose=False)

    # Route 1: Single-site adsorption — main fragment binds to one dangling bond,
    # departing ligand is placed above the surface as a gas-phase byproduct.
    # Runs when unique_single sites exist; the symmetry-reduced list avoids
    # generating duplicate structures for equivalent surface atoms.
    if sites.get("unique_single"):
        if verbose:
            print(f"  -> Routing to Generic Single-Site Chemisorption on {len(sites['unique_single'])} Sites...")
        s_cands = _execute_generic_single_site(
            mgr, molecule, c_idx, ligands, sites["unique_single"], rot_steps, tag=tag, failed_candidates=failed_candidates, stage_type=stage_type
        )
        candidates.extend(s_cands)

    # Route 4: Haptic ligand adsorbs directly at surface — whole molecule intact,
    # the eta-n ligand face (allyl/Cp) approaches the dangling bond while the metal
    # centre remains coordinated above.  Only activated when haptic ligands exist.
    if sites.get("unique_single") and any(l["hapticity"] > 1 for l in ligands):
        if verbose:
            n_hap = sum(1 for l in ligands if l["hapticity"] > 1)
            print(f"  -> Routing to Haptic-Ligand Site Adsorption "
                  f"({n_hap} haptic ligand(s), {len(sites['unique_single'])} site(s))...")
        h_cands = _execute_haptic_ligand_site(
            mgr, molecule, c_idx, ligands, sites["unique_single"], rot_steps, tag=tag, failed_candidates=failed_candidates
        )
        candidates.extend(h_cands)

    # Route 2: Dissociative adsorption on active site pairs — both the main
    # fragment and the departing ligand bind to surface dangling-bond sites.
    if sites.get("pairs"):
        if verbose:
            print(f"  -> Routing to Generic Dissociative Chemisorption on {len(sites['pairs'])} Pairs...")
        d_cands = _execute_generic_dissociation(mgr, molecule, c_idx, ligands, sites["pairs"], rot_steps, tag=tag, verbose=chem_verbose, failed_candidates=failed_candidates)
        candidates.extend(d_cands)

    # Route 3: Protector exchange — reactive leaf of an inhibitor layer is replaced.
    if sites.get("exchange"):
        if verbose:
            print(f"  -> Routing to Protector Exchange Chemisorption on {len(sites['exchange'])} Sites...")
        x_cands = _execute_protector_exchange(mgr, molecule, c_idx, ligands, sites["exchange"], rot_steps, tag=tag, verbose=chem_verbose, failed_candidates=failed_candidates, stage_type=stage_type)
        candidates.extend(x_cands)

    if failed_candidates and results_dir:
        import os
        from ase.io import write
        os.makedirs(results_dir, exist_ok=True)
        fail_path = os.path.join(results_dir, "chemisorption_failed.extxyz")
        write(fail_path, failed_candidates)
        if verbose:
            print(f"  [Debug Mode] Saved {len(failed_candidates)} failed chemisorption poses to: {fail_path}")

    if verbose:
        print(f"--- Finished Chemisorption Builder. Total Generated: {len(candidates)} ---")
        if not candidates:
            n_unique_single = len(sites.get("unique_single", []))
            n_pairs         = len(sites.get("pairs", []))
            n_exchange      = len(sites.get("exchange", []))
            if n_unique_single == 0 and n_pairs == 0 and n_exchange == 0:
                print(
                    "  [Chemisorption] WARNING: 0 candidates - no active surface sites found "
                    "(dangling bonds, pairs, or exchange sites). Possible causes:\n"
                    "    - Surface is fully saturated (all expected bonds satisfied).\n"
                    "    - Coordination number thresholds in config may not match this surface termination.\n"
                    "    - Try adjusting 'coord_num_threshold' or 'symprec' in config."
                )
            else:
                print(
                    f"  [Chemisorption] WARNING: 0 candidates despite active sites "
                    f"(unique_single={n_unique_single}, pairs={n_pairs}, exchange={n_exchange}).\n"
                    f"  All poses were rejected -- see per-route warnings above for details.\n"
                    f"  Common causes:\n"
                    f"    - Molecule too large for available surface geometry.\n"
                    f"    - Internal molecular bonds flagged as overlap "
                    f"(check_internal=True bug -- verify check_internal=False is used).\n"
                    f"    - Placement bond length too short (fragment centre placed inside surface atom).\n"
                    f"    - Haptic route (Route 4): whole-molecule steric clash prevented all poses."
                )
        print()

    return candidates


def _min_nonbonded_clearance(combined, n_slab, skip_pairs=None, skip_indices=None):
    """Minimum distance between newly added atoms and the slab, excluding bonded pairs.

    Used to rank valid poses: larger clearance -> better initial geometry for relaxation.
    """
    from ase.geometry import get_distances

    skip_idx_set = frozenset(int(i) for i in (skip_indices or []))
    skip_pair_set = frozenset(
        tuple(sorted((int(a), int(b)))) for a, b in (skip_pairs or [])
    )

    new_idx = [i for i in range(n_slab, len(combined)) if i not in skip_idx_set]
    env_idx = [i for i in range(n_slab) if i not in skip_idx_set]

    if not new_idx or not env_idx:
        return np.inf

    _, dists = get_distances(
        combined.positions[new_idx], combined.positions[env_idx],
        cell=combined.cell, pbc=combined.pbc,
    )

    min_d = np.inf
    for i, ni in enumerate(new_idx):
        for j, ej in enumerate(env_idx):
            if tuple(sorted((ni, ej))) in skip_pair_set:
                continue
            if dists[i, j] < min_d:
                min_d = dists[i, j]
    return float(min_d)


def _execute_generic_single_site(mgr, molecule, c_idx, ligands, sites, rot_steps, tag=2, failed_candidates=None, stage_type="precursor"):
    """Internal subroutine to execute Generic Single Site Addition/Exchange.

    Tries all rot_steps angles per site and keeps the pose with the largest
    minimum non-bonded clearance (best initial geometry for subsequent relaxation).
    """

    candidates = []
    stats = {"overlap": 0, "total_tries": 0}

    chem_cfg = mgr.config.get("reaction_search", {}).get("mechanisms", {}).get(stage_type, {}).get("chemisorption", {})
    byproduct_placement = chem_cfg.get("byproduct_placement", "vacuum")

    for l_info in _unique_ligands(ligands):

        indices_b = l_info["indices"]
        frag_b = molecule[indices_b]
        binding_idx_b = [indices_b.index(idx) for idx in l_info["binding_atoms"]]

        indices_a = list(set(range(len(molecule))) - set(indices_b))
        frag_a = molecule[indices_a]
        binding_idx_a = indices_a.index(c_idx)

        site_iter = (
            _tqdm(sites, desc="[Chem] single sites", unit="site", leave=True, dynamic_ncols=True)
            if _tqdm else sites
        )
        for s in site_iter:
            si_pos = s["pos"]
            h_vec_norm = s["db_vector"]

            # Element-specific bond length (center atom -> surface atom)
            bond_len_a = chem_kb.get_radius(molecule.symbols[c_idx], "covalent") + chem_kb.get_radius(mgr.slab.symbols[s["index"]], "covalent")

            best_pose = None
            best_clearance = -np.inf

            for angle in np.linspace(0, 360, rot_steps, endpoint=False):
                stats["total_tries"] += 1
                p_a = mgr._place_at_dangling_bond(
                    frag_a,
                    binding_idx_a,
                    l_info["bond_vec"],
                    si_pos,
                    h_vec_norm,
                    bond_len_a,
                    rot_angle=angle,
                )

                p_b = mgr._form_byproduct(frag_b, binding_idx_b[0], -l_info["bond_vec"])
                if byproduct_placement == "surface":
                    sub_mask = (mgr.slab.get_tags() < 2) | (mgr.slab.get_tags() == 4)
                    z_ref = mgr.slab.positions[sub_mask, 2].max() if np.any(sub_mask) else np.max(mgr.slab.positions[:, 2])
                    z_clearance = z_ref + 2.5
                    rad = np.radians(angle)
                    xy_offset = np.array([4.0 * np.cos(rad), 4.0 * np.sin(rad), 0.0])
                    p_b.translate([si_pos[0], si_pos[1], z_clearance] + xy_offset - p_b.positions[0])
                else:
                    z_clearance = np.max(mgr.slab.positions[:, 2]) + 4.0
                    p_b.translate([si_pos[0], si_pos[1], z_clearance] - p_b.positions[0])

                combined = mgr.slab.copy()
                for a in p_a:
                    a.tag = tag
                combined += p_a
                
                for a in p_b:
                    a.tag = tag
                combined += p_b

                skip_indices = [s["index"]] + [len(mgr.slab) + i for i in range(len(p_a) + len(p_b))]

                new_start = len(mgr.slab)
                frag_a_indices_local = list(range(new_start, new_start + len(p_a)))
                frag_b_indices_local = list(range(new_start + len(p_a), len(combined)))
                
                skip_pairs_local = [(s["index"], new_start + binding_idx_a)]
                skip_pairs_local += list(combinations(frag_a_indices_local, 2))
                skip_pairs_local += list(combinations(frag_b_indices_local, 2))

                if not mgr.check_overlap(combined, skip_pairs=skip_pairs_local,
                                         verbose=False, check_internal=False):
                    clearance = _min_nonbonded_clearance(
                        combined, len(mgr.slab),
                        skip_pairs=skip_pairs_local,
                    )
                    if clearance > best_clearance:
                        best_clearance = clearance
                        comp_a = "".join(frag_a.symbols)
                        comp_b = "".join(p_b.symbols)
                        if comp_b == "HH":
                            comp_b = "H2"
                        combined.info["mechanism"] = (
                            f"Single-Site Chemisorption: {comp_a} on {s['index']}, byproduct={comp_b}, tag={tag}, rot={angle:.1f}"
                        )
                        combined.info["reaction_type"] = "single_site_chemisorption"
                        combined.info["isolated_byproduct"] = p_b
                        combined.info["index_mapping"] = {
                            "frag_a": indices_a,
                            "frag_b": indices_b
                        }
                        best_pose = combined
                else:
                    stats["overlap"] += 1
                    if failed_candidates is not None:
                        combined_fail = combined.copy()
                        comp_a = "".join(frag_a.symbols)
                        comp_b = "".join(p_b.symbols)
                        if comp_b == "HH":
                            comp_b = "H2"
                        combined_fail.info["mechanism"] = (
                            f"Single-Site Chemisorption FAIL (Overlap): {comp_a} on {s['index']}, byproduct={comp_b}, tag={tag}, rot={angle:.1f}"
                        )
                        combined_fail.info["failed_stage"] = "overlap_clash"
                        failed_candidates.append(combined_fail)

            if best_pose:
                candidates.append(best_pose)

    if not candidates:
        _tried = stats["total_tries"]
        _ov    = stats["overlap"]
        if _tried == 0:
            print(
                f"  [SingleSite] WARNING: 0 candidates - no site/ligand combinations to try "
                f"({len(sites)} site(s), {len(list(_unique_ligands(ligands)))} unique ligand(s))."
            )
        else:
            print(
                f"  [SingleSite] WARNING: 0 candidates generated from {_tried} poses "
                f"across {len(sites)} site(s).\n"
                f"    Rejection breakdown:\n"
                f"      external overlap (fragment vs surface) : {_ov}\n"
                f"      passed overlap but yielded no pose    : {_tried - _ov}\n"
                f"    Note: internal molecular bond pairs are excluded from the overlap check "
                f"(check_internal=False).\n"
                f"    If all rejections are external, the surface sites may be too crowded "
                f"or the placement height too low."
            )

    return candidates


def _execute_haptic_ligand_site(mgr, molecule, c_idx, ligands, sites, rot_steps, tag=2, failed_candidates=None):
    """Route 4: Intact-molecule haptic adsorption.

    The WHOLE molecule is placed with the haptic ligand face (eta-n) toward a surface
    dangling bond.  No bond is broken; the metal centre and remaining ligands stay above.

    Geometry
    --------
    anchor   = centroid of haptic binding atoms (VBS)
    align    = -haptic_normal  (haptic_normal points VBS->metal; negating keeps metal ABOVE)
               _place_at_dangling_bond does f.rotate(align_vec, -db_vector):
                 -haptic_normal -> -db  =>  haptic_normal -> +db  =>  metal stays above VBS
    bond_len = mean(cov_r of haptic C atoms) + cov_r(surface atom)

    Only ligands with hapticity >= 2 are processed.
    """
    candidates = []
    stats = {"overlap": 0, "total_tries": 0}

    haptic_ligands = [l for l in _unique_ligands(ligands) if l["hapticity"] > 1]
    if not haptic_ligands:
        return candidates

    for l_info in haptic_ligands:
        binding_idx   = l_info["binding_atoms"]    # atom indices in the whole molecule
        haptic_normal = l_info["normal_vector"]     # points VBS -> metal

        binding_syms  = [molecule.symbols[i] for i in binding_idx]
        avg_cov_r_lig = np.mean([chem_kb.get_radius(s, "covalent") for s in binding_syms])

        site_iter = (
            _tqdm(sites, desc="[Chem] haptic sites", unit="site", leave=True, dynamic_ncols=True)
            if _tqdm else sites
        )
        for s in site_iter:
            si_pos     = s["pos"]
            h_vec_norm = s["db_vector"]
            bond_len   = avg_cov_r_lig + chem_kb.get_radius(mgr.slab.symbols[s["index"]], "covalent")

            best_pose      = None
            best_clearance = -np.inf

            for angle in np.linspace(0, 360, rot_steps, endpoint=False):
                stats["total_tries"] += 1

                # Pass -haptic_normal: after _place_at_dangling_bond rotates align_vec -> -db,
                # haptic_normal (VBS->metal) ends up aligned with +db (upward) => metal above.
                placed = mgr._place_at_dangling_bond(
                    molecule,
                    binding_idx,
                    l_info["bond_vec"],         # fallback; not used when haptic_normal is set
                    si_pos,
                    h_vec_norm,
                    bond_len,
                    rot_angle=angle,
                    haptic_normal=-haptic_normal,
                )

                combined = mgr.slab.copy()
                for a in placed:
                    a.tag = tag
                combined += placed

                new_start  = len(mgr.slab)
                mol_global = list(range(new_start, new_start + len(molecule)))

                # Skip new haptic-C/surface bonds and all intra-molecule pairs
                skip_pairs  = [(s["index"], new_start + bi) for bi in binding_idx]
                skip_pairs += list(combinations(mol_global, 2))

                if not mgr.check_overlap(combined, skip_pairs=skip_pairs,
                                         verbose=False, check_internal=False):
                    clearance = _min_nonbonded_clearance(
                        combined, len(mgr.slab), skip_pairs=skip_pairs
                    )
                    if clearance > best_clearance:
                        best_clearance = clearance
                        combined.info["mechanism"] = (
                            f"Haptic-Ligand Adsorption: {l_info['formula']}"
                            f"(eta{l_info['hapticity']}) on site {s['index']}, "
                            f"rot={angle:.1f}"
                        )
                        combined.info["reaction_type"] = "haptic_ligand_chemisorption"
                        combined.info["index_mapping"] = {
                            "haptic_ligand_indices": l_info["indices"],
                            "binding_atoms":         binding_idx,
                            "metal_idx":             c_idx,
                        }
                        best_pose = combined
                else:
                    stats["overlap"] += 1
                    if failed_candidates is not None:
                        combined_fail = combined.copy()
                        combined_fail.info["mechanism"] = (
                            f"Haptic-Ligand Adsorption FAIL (Overlap): {l_info['formula']}"
                            f"(eta{l_info['hapticity']}) on site {s['index']}, "
                            f"rot={angle:.1f}"
                        )
                        combined_fail.info["failed_stage"] = "overlap_clash"
                        failed_candidates.append(combined_fail)

            if best_pose:
                candidates.append(best_pose)

    _tried = stats["total_tries"]
    _ov    = stats["overlap"]
    if not candidates:
        n_hap = len(haptic_ligands)
        n_sit = len(sites)
        if _tried == 0:
            print(
                f"  [HapticSite] WARNING: 0 candidates -- no haptic-ligand/site combinations "
                f"({n_sit} site(s), {n_hap} haptic ligand(s))."
            )
        else:
            print(
                f"  [HapticSite] WARNING: 0 candidates from {_tried} poses "
                f"({n_sit} site(s), {n_hap} haptic ligand(s)).\n"
                f"    Rejection: external overlap={_ov}, passed-but-no-pose={_tried - _ov}.\n"
                f"    Common cause: molecule too bulky to land haptic face without steric clash."
            )

    return candidates


def _execute_generic_dissociation(mgr, molecule, c_idx, ligands, pairs, rot_steps, tag=2, verbose=False, failed_candidates=None):
    """Internal subroutine to execute Generic Dissociative Chemisorption on pairs of dangling bonds.

    Algorithmic improvements vs naive first-valid-angle approach:
    - Element-specific bond length (covalent radii sum) for the center->surface bond.
    - Both site permutations (s1->s2 and s2->s1) and all rot_steps angles are evaluated;
      the pose with the largest minimum non-bonded clearance is selected.  This maximises
      the distance budget available for the subsequent MLIP relaxation and reduces the risk
      of energy blow-up from overlapping atoms.
    """

    candidates = []
    stats = {"overlap": 0, "total_tries": 0}

    for l_info in _unique_ligands(ligands):
        indices_b = l_info["indices"]
        frag_b = molecule[indices_b]
        binding_idx_b = [indices_b.index(idx) for idx in l_info["binding_atoms"]]

        indices_a = list(set(range(len(molecule))) - set(indices_b))
        frag_a = molecule[indices_a]
        binding_idx_a = indices_a.index(c_idx)

        pair_iter = (
            _tqdm(pairs, desc="[Chem] dissociation pairs", unit="pair", leave=True, dynamic_ncols=True)
            if _tqdm else pairs
        )
        for s1, s2 in pair_iter:
            best_pose = None
            best_clearance = -np.inf

            for active_1, active_2 in [(s1, s2), (s2, s1)]:
                # Element-specific bond length for frag_a center -> surface site
                bond_len_a = chem_kb.get_radius(molecule.symbols[c_idx], "covalent") + chem_kb.get_radius(mgr.slab.symbols[active_1["index"]], "covalent")

                bond_len_b = 2.1
                if l_info["hapticity"] == 1 and frag_b.symbols[binding_idx_b[0]] == "H":
                    bond_len_b = 1.48
                elif l_info["hapticity"] > 1:
                    bond_len_b = 2.0

                for angle in np.linspace(0, 360, rot_steps, endpoint=False):
                    stats["total_tries"] += 1
                    p_a = mgr._place_at_dangling_bond(
                        frag_a,
                        binding_idx_a,
                        l_info["bond_vec"],
                        active_1["pos"],
                        active_1["db_vector"],
                        bond_len_a,
                        rot_angle=angle,
                    )

                    p_b = mgr._place_at_dangling_bond(
                        frag_b,
                        binding_idx_b,
                        -l_info["bond_vec"],
                        active_2["pos"],
                        active_2["db_vector"],
                        bond_len_b,
                        rot_angle=0,
                        haptic_normal=l_info.get("normal_vector"),
                    )

                    combined = mgr.slab.copy()
                    for a in p_a:
                        a.tag = tag
                    combined += p_a
                    for a in p_b:
                        a.tag = tag
                    combined += p_b

                    new_start = len(mgr.slab)
                    frag_a_indices = list(range(new_start, new_start + len(frag_a)))
                    frag_b_indices = list(range(new_start + len(frag_a), new_start + len(frag_a) + len(frag_b)))

                    # Support haptic ligands: skip overlap for ALL atoms in the binding set
                    skip_pairs = [(active_1["index"], new_start + binding_idx_a)]
                    
                    # For frag_b (ligand)
                    for b_idx in binding_idx_b:
                        skip_pairs.append((active_2["index"], new_start + len(frag_a) + b_idx))
                    skip_pairs += list(combinations(frag_a_indices, 2))
                    skip_pairs += list(combinations(frag_b_indices, 2))

                    if not mgr.check_overlap(combined, skip_pairs=skip_pairs, verbose=verbose, check_internal=False):
                        clearance = _min_nonbonded_clearance(combined, new_start, skip_pairs=skip_pairs)
                        if clearance > best_clearance:
                            best_clearance = clearance
                            formula_a = frag_a.get_chemical_formula()
                            formula_b = frag_b.get_chemical_formula()
                            combined.info["mechanism"] = f"Chemisorption (Dissociation: {formula_a}+{formula_b})"
                            combined.info["reaction_type"] = "chemisorption"
                            combined.info["index_mapping"] = {
                                "frag_a": indices_a,
                                "frag_b": indices_b
                            }
                            best_pose = combined
                    else:
                        stats["overlap"] += 1
                        if failed_candidates is not None:
                            combined_fail = combined.copy()
                            formula_a = frag_a.get_chemical_formula()
                            formula_b = frag_b.get_chemical_formula()
                            combined_fail.info["mechanism"] = (
                                f"Chemisorption Dissociation FAIL (Overlap): {formula_a}+{formula_b} on pair {active_1['index']}-{active_2['index']}, rot={angle:.1f}"
                            )
                            combined_fail.info["failed_stage"] = "overlap_clash"
                            failed_candidates.append(combined_fail)

            if best_pose:
                candidates.append(best_pose)

    _tried = stats["total_tries"]
    _ov    = stats["overlap"]
    if not candidates:
        if _tried == 0:
            print(
                f"  [Dissociation] WARNING: 0 candidates - no site pair/ligand combinations to try "
                f"({len(pairs)} pair(s), {len(list(_unique_ligands(ligands)))} unique ligand(s))."
            )
        else:
            print(
                f"  [Dissociation] WARNING: 0 candidates generated from {_tried} poses "
                f"across {len(pairs)} pair(s).\n"
                f"    Rejection breakdown:\n"
                f"      external overlap (fragment vs surface) : {_ov}\n"
                f"      passed overlap but yielded no pose    : {_tried - _ov}\n"
                f"    Note: internal molecular bond pairs are excluded from the overlap check "
                f"(check_internal=False).\n"
                f"    Tip: a high external overlap count often means the two surface sites are "
                f"too close for the molecule to bridge both simultaneously."
            )
    elif mgr.verbose and _tried > 0:
        print(f"  [Dissociation Stats] Tried {_tried} poses, {_ov} failed overlap check.")

    return candidates


def _execute_protector_exchange(mgr, molecule, c_idx, ligands, exchange_sites, rot_steps, tag=3, verbose=False, failed_candidates=None, stage_type="precursor"):
    """Internal subroutine to execute Ligand Exchange with Protector leaves.

    Tries all rot_steps angles and keeps the pose with the largest minimum
    non-bonded clearance (same best-clearance strategy as _execute_generic_dissociation).
    Bond length for the new center->backbone bond uses covalent radii.
    """
    candidates = []
    stats = {"overlap": 0, "total_tries": 0}
    
    chem_cfg = mgr.config.get("reaction_search", {}).get("mechanisms", {}).get(stage_type, {}).get("chemisorption", {})
    byproduct_placement = chem_cfg.get("byproduct_placement", "vacuum")

    for l_info in _unique_ligands(ligands):
        indices_b = l_info["indices"]
        frag_b = molecule[indices_b]
        binding_idx_b = [indices_b.index(idx) for idx in l_info["binding_atoms"]]

        indices_a = list(set(range(len(molecule))) - set(indices_b))
        frag_a = molecule[indices_a]
        binding_idx_a = indices_a.index(c_idx)

        exchange_iter = (
            _tqdm(exchange_sites, desc="[Chem] exchange sites", unit="site", leave=True, dynamic_ncols=True)
            if _tqdm else exchange_sites
        )
        for s in exchange_iter:
            backbone_pos = s["pos"]
            h_vec_norm = s["db_vector"]  # points AWAY from surface

            # Element-specific bond length (center -> backbone atom)
            bond_len_a = chem_kb.get_radius(molecule.symbols[c_idx], "covalent") + chem_kb.get_radius(mgr.slab.symbols[s["backbone_idx"]], "covalent")

            best_pose = None
            best_clearance = -np.inf

            for angle in np.linspace(0, 360, rot_steps, endpoint=False):
                stats["total_tries"] += 1
                p_a = mgr._place_at_dangling_bond(
                    frag_a,
                    binding_idx_a,
                    l_info["bond_vec"],
                    backbone_pos,
                    h_vec_norm,
                    bond_len_a,
                    rot_angle=angle,
                )

                byproduct = frag_b.copy()
                bp_h_pos = (
                    byproduct.positions[binding_idx_b[0]]
                    + (l_info["bond_vec"] / np.linalg.norm(l_info["bond_vec"])) * _bond_length_to_h(s["sym"])
                )
                byproduct += Atoms(s["sym"], positions=[bp_h_pos])

                combined = mgr.slab.copy()
                del combined[s["index"]]  # remove the exchanged leaf atom

                for a in p_a:
                    a.tag = tag
                combined += p_a
                
                # Position byproduct based on configuration
                if byproduct_placement == "surface":
                    sub_mask = (combined.get_tags() < 2) | (combined.get_tags() == 4)
                    z_ref = combined.positions[sub_mask, 2].max() if np.any(sub_mask) else np.max(combined.positions[:, 2])
                    z_clearance = z_ref + 2.5
                    rad = np.radians(angle)
                    xy_offset = np.array([4.0 * np.cos(rad), 4.0 * np.sin(rad), 0.0])
                    byproduct.translate([backbone_pos[0], backbone_pos[1], z_clearance] + xy_offset - byproduct.positions[0])
                else:
                    z_clearance = np.max(combined.positions[:, 2]) + 4.0
                    byproduct.translate([backbone_pos[0], backbone_pos[1], z_clearance] - byproduct.positions[0])
                    
                for a in byproduct:
                    a.tag = tag
                combined += byproduct

                # After leaf deletion, backbone index may shift by -1
                new_backbone_idx = s["backbone_idx"]
                if s["index"] < s["backbone_idx"]:
                    new_backbone_idx -= 1

                n_slab_trimmed = len(mgr.slab) - 1
                
                # Support haptic ligands: skip overlap for ALL atoms in the binding set
                skip_pairs = [(new_backbone_idx, n_slab_trimmed + binding_idx_a)]
                
                frag_a_indices = list(range(n_slab_trimmed, n_slab_trimmed + len(frag_a)))
                frag_b_indices = list(range(n_slab_trimmed + len(frag_a), len(combined)))
                
                for i in range(len(frag_a_indices)):
                    for j in range(i + 1, len(frag_a_indices)):
                        skip_pairs.append((frag_a_indices[i], frag_a_indices[j]))
                        
                for i in range(len(frag_b_indices)):
                    for j in range(i + 1, len(frag_b_indices)):
                        skip_pairs.append((frag_b_indices[i], frag_b_indices[j]))

                if not mgr.check_overlap(combined, skip_pairs=skip_pairs, verbose=verbose, check_internal=False):
                    clearance = _min_nonbonded_clearance(combined, n_slab_trimmed, skip_pairs=skip_pairs)
                    if clearance > best_clearance:
                        best_clearance = clearance
                        comp_a = "".join(frag_a.symbols)
                        comp_b = "".join(byproduct.symbols)
                        combined.info["mechanism"] = (
                            f"Protector Exchange: {comp_a} on backbone {s['backbone_idx']}, "
                            f"byproduct={comp_b}, tag={tag}, rot={angle:.1f}"
                        )
                        combined.info["reaction_type"] = "protector_exchange"
                        combined.info["isolated_byproduct"] = byproduct
                        combined.info["index_mapping"] = {
                            "frag_a": indices_a,
                            "frag_b": indices_b,
                            "protector_idx": s["index"]
                        }

                        best_pose = combined
                else:
                    stats["overlap"] += 1
                    if failed_candidates is not None:
                        combined_fail = combined.copy()
                        comp_a = "".join(frag_a.symbols)
                        comp_b = "".join(byproduct.symbols)
                        combined_fail.info["mechanism"] = (
                            f"Protector Exchange FAIL (Overlap): {comp_a} on backbone {s['backbone_idx']}, byproduct={comp_b}, tag={tag}, rot={angle:.1f}"
                        )
                        combined_fail.info["failed_stage"] = "overlap_clash"
                        failed_candidates.append(combined_fail)

            if best_pose:
                candidates.append(best_pose)

    return candidates

