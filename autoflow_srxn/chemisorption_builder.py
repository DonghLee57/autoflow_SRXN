import numpy as np
from ase import Atoms

from .ads_workflow_mgr import AdsorptionWorkflowManager
from .knowledge_engine import chem_kb


def analyze_surface_reactivity(surface, config, verbose=True):
    import numpy as np
    from ase.data import covalent_radii
    from ase.neighborlist import neighbor_list

    from .surface_utils import identify_protectors

    max_pair_dist = config.get("reaction_search", {}).get("candidate_filter", {}).get("max_pair_dist", 5.0)

    sub_idx, prot_idx = identify_protectors(surface, config, verbose=False)

    i_list, j_list, D_list = neighbor_list("ijD", surface, cutoff=3.0)
    d_list = np.linalg.norm(D_list, axis=1)

    dangling_sites = []
    exchange_sites = []

    z_max = max(surface.positions[:, 2])
    z_sub_max = max(surface.positions[sub_idx, 2]) if len(sub_idx) else z_max

    for idx in range(len(surface)):
        if idx in sub_idx and surface.positions[idx, 2] < z_sub_max - 2.5:
            continue
        sym = surface.symbols[idx]

        if idx in prot_idx:
            _protex = (
                config.get("reaction_search", {})
                .get("mechanisms", {})
                .get("protector_exchange", config.get("protector", {}))
            )
            reactive_leaves = _protex.get("reactive_leaves", [])
            if sym in reactive_leaves:
                neighbors = []
                for n_i, n_j, dist, vec in zip(i_list, j_list, d_list, D_list):
                    if n_i == idx and dist > 0.1 and dist < 2.0:
                        neighbors.append((n_j, vec))
                if len(neighbors) == 1:
                    db_vec = -neighbors[0][1]
                    db_vec = db_vec / np.linalg.norm(db_vec)
                    exchange_sites.append(
                        {
                            "index": idx,
                            "backbone_idx": neighbors[0][0],
                            "sym": sym,
                            "pos": surface.positions[neighbors[0][0]],
                            "leaf_pos": surface.positions[idx],
                            "db_vector": db_vec,
                            "missing_bonds": 1,
                        }
                    )
            continue

        neighbors = []
        for n_i, n_j, dist in zip(i_list, j_list, d_list):
            if n_i == idx:
                cutoff_val = covalent_radii[surface.numbers[n_i]] + covalent_radii[surface.numbers[n_j]] + 0.3
                if dist < cutoff_val and dist > 0.1:
                    neighbors.append(n_j)

        actual_coord = len(neighbors)
        _ideal_coord = config.get("surface_prep", {}).get("surface_analysis", {}).get("ideal_coordination", {})
        expected = chem_kb.get_ideal_coordination(sym, _ideal_coord)

        if actual_coord < expected:
            from .surface_utils import generate_vsepr_vectors

            vecs = generate_vsepr_vectors(surface, idx, neighbor_data=(i_list, j_list, D_list))
            for db_vec in vecs:
                db_vec = db_vec / np.linalg.norm(db_vec)
                hit = False
                if len(prot_idx) > 0:
                    for p in prot_idx:
                        p_vec = surface.positions[p] - surface.positions[idx]
                        proj = np.dot(p_vec, db_vec)
                        if proj > 0.5:
                            dist_to_ray = np.linalg.norm(p_vec - proj * db_vec)
                            if dist_to_ray < 1.5:
                                hit = True
                                break
                if not hit:
                    dangling_sites.append(
                        {
                            "index": idx,
                            "sym": sym,
                            "pos": surface.positions[idx],
                            "db_vector": db_vec,
                            "missing_bonds": expected - actual_coord,
                        }
                    )
                    break

    if verbose:
        print(
            f"  [Generic Reactivity] Identified {len(dangling_sites)} undercoordinated surface sites and {len(exchange_sites)} protector exchange sites."
        )

    results = {"single": dangling_sites, "pairs": [], "exchange": exchange_sites}

    # Analyze Symmetry to reduce pair redundancies
    import spglib

    lattice = surface.get_cell()
    pos = surface.get_scaled_positions()
    nums = surface.get_atomic_numbers()
    symprec = config.get("reaction_search", {}).get("candidate_filter", {}).get("symprec", 0.2)

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

    from itertools import combinations

    unique_pairs = {}
    pair_count = 0

    for s1, s2 in combinations(dangling_sites, 2):
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
            f"  [Generic Reactivity] Formed {pair_count} active site pairs -> Symmetry reduced to {len(results['pairs'])} unique pairs."
        )

    return results


def analyze_molecule_ligands(molecule, center_target="Si", verbose=True):
    """Algorithmically fragments the precursor molecule to identify reactive ligands.
    Uses AdsorptionWorkflowManager implicitly for the heavy lifting.
    """
    # Create a temporary manager to use its fragmentation logic
    mgr = AdsorptionWorkflowManager(molecule, verbose=verbose)
    c_idx, ligands = mgr.discover_ligands(molecule, center_target=center_target, verbose=verbose)
    return c_idx, ligands


def build_chemisorption_structures(
    molecule, center_target="Si", surface=None, rot_steps=8, config=None, verbose=True, tag=2
):
    """Entry point for algorithmic chemisorption generation based on input molecule and surface.
    Identifies valid mechanisms based on available surface sites.
    """
    if verbose:
        print(f"\n--- Starting Algorithmic Chemisorption Routing (tag={tag}) ---")

    if config is None:
        config = {}
    sites = analyze_surface_reactivity(surface, config, verbose=verbose)
    c_idx, ligands = analyze_molecule_ligands(molecule, center_target=center_target, verbose=verbose)

    candidates = []

    if not ligands:
        if verbose:
            print("  [Warning] No detachable ligands found. Aborting chemisorption.")
        return candidates

    # We instantiate a manager scoped to the current surface for coordinate placement/overlap tests
    mgr = AdsorptionWorkflowManager(surface, config=config, verbose=verbose)

    # Generic Cohesive Dissociation on active site pairs
    if sites.get("pairs"):
        if verbose:
            print(f"  -> Routing to Generic Dissociative Chemisorption on {len(sites['pairs'])} Pairs...")
        d_cands = _execute_generic_dissociation(mgr, molecule, c_idx, ligands, sites["pairs"], rot_steps, tag=tag)
        candidates.extend(d_cands)

    if sites.get("exchange"):
        if verbose:
            print(f"  -> Routing to Protector Exchange Chemisorption on {len(sites['exchange'])} Sites...")
        x_cands = _execute_protector_exchange(mgr, molecule, c_idx, ligands, sites["exchange"], rot_steps, tag=tag)
        candidates.extend(x_cands)

    if verbose:
        print(f"--- Finished Chemisorption Builder. Total Generated: {len(candidates)} ---\n")

    return candidates


def _execute_generic_single_site(mgr, molecule, c_idx, ligands, sites, rot_steps, tag=2):
    """Internal subroutine to execute Generic Single Site Addition/Exchange"""
    candidates = []
    stats = {"overlap": 0, "deduplicated": 0, "total_tries": 0}
    seen_formulas = set()

    for l_info in ligands:
        formula = l_info.get("formula", "Unknown")
        if formula in seen_formulas:
            stats["deduplicated"] += 1
            continue
        seen_formulas.add(formula)

        indices_b = l_info["indices"]
        frag_b = molecule[indices_b]
        binding_idx_b = [indices_b.index(idx) for idx in l_info["binding_atoms"]]

        indices_a = list(set(range(len(molecule))) - set(indices_b))
        frag_a = molecule[indices_a]
        binding_idx_a = indices_a.index(c_idx)

        for s in sites:
            si_pos = s["pos"]
            h_vec_norm = s["db_vector"]

            for angle in np.linspace(0, 360, rot_steps, endpoint=False):
                stats["total_tries"] += 1
                p_a = mgr._place_at_dangling_bond(
                    frag_a,
                    binding_idx_a,
                    l_info["bond_vec"],
                    si_pos,
                    h_vec_norm,
                    2.35,
                    rot_angle=angle,
                )

                p_b = mgr._form_byproduct(frag_b, binding_idx_b[0], -l_info["bond_vec"])
                z_clearance = np.max(mgr.slab.positions[:, 2]) + 4.0
                p_b.translate([si_pos[0], si_pos[1], z_clearance] - p_b.positions[0])
                p_b.center(vacuum=5.0)  # Isolate byproduct in vacuum

                combined = mgr.slab.copy()
                for a in p_a:
                    a.tag = tag
                combined += p_a

                if not mgr.check_overlap(combined, cutoff=1.2, verbose=False):
                    comp_a = "".join(frag_a.symbols)
                    comp_b = "".join(p_b.symbols)
                    if comp_b == "HH":
                        comp_b = "H2"  # Make it more readable

                    combined.info["mechanism"] = (
                        f"Generic Single-Site: {comp_a} on {s['index']}, byproduct={comp_b}, tag={tag}, rot={angle:.1f}"
                    )
                    combined.info["reaction_type"] = "h_exchange"
                    combined.info["isolated_byproduct"] = p_b
                    candidates.append(combined)
                    break
                else:
                    stats["overlap"] += 1

    return candidates


def _execute_generic_dissociation(mgr, molecule, c_idx, ligands, pairs, rot_steps, tag=2):
    """Internal subroutine to execute Generic Dissociative Chemisorption on pairs of dangling bonds"""
    candidates = []
    stats = {"overlap": 0, "deduplicated": 0, "total_tries": 0}
    seen_formulas = set()

    for l_info in ligands:
        formula = l_info.get("formula", "Unknown")
        if formula in seen_formulas:
            stats["deduplicated"] += 1
            continue
        seen_formulas.add(formula)

        indices_b = l_info["indices"]
        frag_b = molecule[indices_b]
        binding_idx_b = [indices_b.index(idx) for idx in l_info["binding_atoms"]]

        indices_a = list(set(range(len(molecule))) - set(indices_b))
        frag_a = molecule[indices_a]
        binding_idx_a = indices_a.index(c_idx)

        for s1, s2 in pairs:
            for active_1, active_2 in [(s1, s2), (s2, s1)]:
                best_pose = None
                for angle in np.linspace(0, 360, rot_steps, endpoint=False):
                    stats["total_tries"] += 1
                    p_a = mgr._place_at_dangling_bond(
                        frag_a,
                        binding_idx_a,
                        l_info["bond_vec"],
                        active_1["pos"],
                        active_1["db_vector"],
                        2.35,
                        rot_angle=angle,
                    )

                    bond_len_b = 2.1
                    if l_info["hapticity"] == 1 and frag_b.symbols[binding_idx_b[0]] == "H":
                        bond_len_b = 1.48
                    elif l_info["hapticity"] > 1:
                        bond_len_b = 1.8

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
                        a.tag = tag  # Same tag for same stage
                    combined += p_b

                    if not mgr.check_overlap(combined, cutoff=1.2, verbose=mgr.verbose):
                        comp_a = "".join(frag_a.symbols)
                        combined.info["mechanism"] = (
                            f"Generic Chemisorption: {comp_a} on {active_1['index']}, {frag_b.symbols[binding_idx_b[0]]} on {active_2['index']}, tag={tag}, rot={angle:.1f}"
                        )
                        combined.info["reaction_type"] = "chemisorption"
                        best_pose = combined
                        break
                    else:
                        stats["overlap"] += 1

                if best_pose:
                    candidates.append(best_pose)
                    break

    if mgr.verbose and stats["total_tries"] > 0:
        print(f"  [Dissociation Stats] Tried {stats['total_tries']} poses, {stats['overlap']} failed overlap check.")

    return candidates


def _execute_protector_exchange(mgr, molecule, c_idx, ligands, exchange_sites, rot_steps, tag=3):
    """Internal subroutine to execute Ligand Exchange with Protector leaves"""
    candidates = []
    stats = {"overlap": 0, "deduplicated": 0, "total_tries": 0}
    seen_formulas = set()

    for l_info in ligands:
        formula = l_info.get("formula", "Unknown")
        if formula in seen_formulas:
            stats["deduplicated"] += 1
            continue
        seen_formulas.add(formula)

        indices_b = l_info["indices"]
        frag_b = molecule[indices_b]
        binding_idx_b = [indices_b.index(idx) for idx in l_info["binding_atoms"]]

        indices_a = list(set(range(len(molecule))) - set(indices_b))
        frag_a = molecule[indices_a]
        binding_idx_a = indices_a.index(c_idx)

        for s in exchange_sites:
            backbone_pos = s["pos"]
            h_vec_norm = s["db_vector"]  # points AWAY from surface

            for angle in np.linspace(0, 360, rot_steps, endpoint=False):
                stats["total_tries"] += 1
                p_a = mgr._place_at_dangling_bond(
                    frag_a,
                    binding_idx_a,
                    l_info["bond_vec"],
                    backbone_pos,
                    h_vec_norm,
                    1.8,
                    rot_angle=angle,
                )  # 1.8A generic bond

                # Create byproduct (ligand from precursor + leaf from protector)
                byproduct = frag_b.copy()
                b_len = 1.0 if s["sym"] in ["N", "O"] else 1.1 if s["sym"] == "C" else 1.5
                bp_h_pos = (
                    byproduct.positions[binding_idx_b[0]]
                    + (l_info["bond_vec"] / np.linalg.norm(l_info["bond_vec"])) * b_len
                )
                byproduct += Atoms(s["sym"], positions=[bp_h_pos])
                byproduct.center(vacuum=5.0)  # put in its own cell (isolated gas)

                # combined is slab without the leaf atom
                combined = mgr.slab.copy()
                del combined[s["index"]]  # delete the leaf atom

                for a in p_a:
                    a.tag = tag
                combined += p_a

                if not mgr.check_overlap(combined, cutoff=1.2, verbose=mgr.verbose):
                    comp_a = "".join(frag_a.symbols)
                    comp_b = "".join(byproduct.symbols)
                    combined.info["mechanism"] = (
                        f"Protector Exchange: {comp_a} on backbone {s['backbone_idx']}, byproduct={comp_b}, tag={tag}, rot={angle:.1f}"
                    )
                    combined.info["reaction_type"] = "protector_exchange"
                    combined.info["isolated_byproduct"] = byproduct
                    candidates.append(combined)
                    break
                else:
                    stats["overlap"] += 1

    return candidates
