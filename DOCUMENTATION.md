# AutoFlow-SRXN Configuration Manual

This document provides a comprehensive guide to all parameters available in the `AutoFlow-SRXN` workflow. 

---

## 1. Global Workflow Control
Settings that control the overall execution behavior of the screening engine.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `restart` | Boolean | `false` | If `true`, forces re-calculation of all pairs. If `false`, skips pairs where `final_results.extxyz` already exists. |

---

## 2. Path Configuration
Defines where to find input structures and where to manage results.

| Parameter | Description |
| :--- | :--- |
| `precursor` | Path to a single precursor structure file (.vasp, .xyz) OR a directory containing multiple precursor files for batch screening. |
| `inhibitor` | Path to a single inhibitor structure file OR a directory containing multiple inhibitor files for batch screening (or `null`). |
| `substrate_bulk` | Path to the bulk crystalline structure (used if `slab_generation` is enabled). |
| `input_structure` | Path to a pre-generated slab file. |
| `output_prefix` | Base directory name for batch output folders (default: `results`). |
| `include_no_inhibitor` | If `true`, includes a 'clean' baseline run for each precursor. |

---

## 3. Heterointerface Generation (`interface`) — Stage 0a

Optional pre-stage that runs **before** `surface_prep`.  Requires `pymatgen`
(`pip install autoflow-srxn[interface]`).

When `interface.enabled: true`, the engine:
1. Loads substrate and film bulk structures.
2. Runs the 2D ZSL lattice-match search across all requested Miller-index combinations.
3. Builds a symmetric `sub | film | sub` sandwich slab for the best candidate(s).
4. Optionally injects the built slab as the working substrate for the reaction search.

### 3.1 Input & Labelling

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `enabled` | Boolean | `false` | Run Stage 0a before `surface_prep`. |
| `sub_path` | String | — | Path to the substrate bulk structure (CIF, POSCAR, …). |
| `film_path` | String | — | Path to the film bulk structure. |
| `sub_name` | String | `null` | Display label for the substrate (auto-derived from file stem if `null`). |
| `film_name` | String | `null` | Display label for the film. |

### 3.2 Lattice-Match Search

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `sub_millers` | List[[h,k,l]] | `[[0,0,1],[1,1,0],[1,0,0]]` | Surface orientations to search for the substrate. |
| `film_millers` | List[[h,k,l]] | `[[0,0,1],[1,1,0],[1,0,0]]` | Surface orientations to search for the film. |
| `max_det` | Integer | `36` | Maximum HNF supercell determinant. Higher = more exhaustive but slower. |
| `strain_cutoff` | Float | `0.10` | Discard coincidences with von Mises strain above this value. |
| `top_k` | Integer | `10` | Candidates kept per (sub_miller, film_miller) pair. `0` = keep all. |
| `max_atoms` | Integer | `500` | Discard candidates whose estimated atom count exceeds this. |

**Von Mises strain** is computed from the deformation gradient
`F = A_sub_super @ inv(A_film_super)` via SVD:

$$\varepsilon_\mathrm{VM} = \sqrt{\tfrac{1}{2}\bigl(\varepsilon_1^2 + \varepsilon_2^2 + (\varepsilon_1-\varepsilon_2)^2\bigr)}$$

Candidates are ranked by `vm + 0.001 * max(det_Na, det_Nb)` — lowest strain first,
with a small supercell-size penalty to break ties.

### 3.3 Slab Construction

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `sub_layers` | Integer | `6` | Substrate layers per side (symmetric slab: both bottom and top). |
| `film_layers` | Integer | `8` | Film layers in the centre. |
| `nu` | Float | `0.25` | Poisson's ratio for out-of-plane c-relaxation of the strained film: `eps_c = -nu/(1-nu) * (eps1+eps2)`. |
| `build_top_k` | Integer | `1` | Number of top candidates to build as ASE slabs. `0` = search only. |

**Tag convention** in built slabs:

| Tag | Region |
| :--- | :--- |
| `0` | Substrate (bottom + top layers) |
| `1` | Epitaxial film (centre) |
| `2` | Inhibitor (assigned by Stage 1) |
| `3` | Precursor (assigned by Stage 2) |

### 3.4 Output & Integration

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `use_as_substrate` | Boolean | `false` | If `true`, the rank-0 built slab is injected as the substrate for Stages 0–2, overriding `surface_prep.slab_generation`. |
| `plot` | Boolean | `true` | Generate `interface_candidates.html` (Plotly interactive scatter-plot). |
| `output_dir` | String | `null` | Write `interface_*.extxyz`, `interface_candidates.html`, and `interface_summary.txt` here. Defaults to the run output directory. |

**Output files written by Stage 0a:**

| File | Content |
| :--- | :--- |
| `interface_summary.txt` | Plain-text table of all candidates |
| `interface_candidates.html` | Interactive Plotly scatter: VM strain vs. supercell size |
| `interface_0.extxyz` … `interface_N.extxyz` | Built slab structures (`build_top_k` files) |

**Polar-axis filter**: If `spglib` is installed, surfaces that expose a polar axis
perpendicular to the slab normal are flagged (`polar_ok = false`).  Such surfaces
would create a macroscopic depolarisation field across the slab.

---

## 4. Surface Preparation (`surface_prep`)
Handles the creation and modification of the substrate surface.

### 4.1 Slab Generation
- **`enabled`**: Boolean. Enable/disable ASE-based slab cutting from bulk. If `false`, loads slab from `paths.input_structure`.
- **`miller`**: List of 3 integers (e.g., `[1, 0, 0]`). Miller indices of the surface plane.
- **`thickness_ang`**: Float (A). Minimum thickness of the slab.
- **`vacuum_ang`**: Float (A). Vacuum padding on both sides.
- **`target_area_ang2`**: Float (A2). Target surface area; the engine selects the largest supercell that does not exceed this value while optimising aspect ratio.
- **`supercell_matrix`**: List of lists (e.g., `[[2,0],[0,2]]`). Explicit supercell matrix. Overrides `target_area_ang2` if set.
- **`top_termination`**: String (Element symbol, e.g., `"O"`). Ensures the top surface ends with the specified element.
- **`bottom_termination`**: String (Element symbol, e.g., `"O"`). Ensures the bottom surface ends with the specified element.

### 4.2 Reconstruction & Passivation
- **`reconstruction`**: Apply automated surface reconstruction (auto/ionic/covalent/metallic).
- **`passivation`**: Saturate dangling bonds (typically on the bottom side using "H").

### 4.3 Slab Relaxation & Equilibration
These stages are enabled via the top-level `workflow` block (see §1.1).
- **`surface_prep.surface_analysis.ideal_coordination`**: Expected coordination for VSEPR bond detection.
- **`surface_prep.surface_analysis.symprec`**: Precision for symmetry detection.

---

## 5. Reaction Search (`reaction_search`)
Explores the configuration space of adsorbates through two sequential stages.

### 5.0 Global Symmetry Precision (`reaction_search.symprec`)

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `symprec` | Float (Å) | `0.2` | Single tolerance applied to **all** symmetry-reduction steps: surface-site deduplication (`AdsorptionWorkflowManager`), dangling-bond equivalence (chemisorption builder), and coordinate deduplication (`get_unique_coordinates`, `generate_and_plot_site_map`). |

```yaml
reaction_search:
  symprec: 0.2   # single knob for all site-grouping in this stage
```

**Tuning guide**:

| Value | Effect |
| :--- | :--- |
| `0.1–0.2` Å | Conservative — preserves most distinct sites (recommended default) |
| `0.5–1.0` Å | Aggressive grouping — useful for large or near-symmetric supercells (e.g. TiN 2×2) |

### 5.1 Stage-Specific Controls
The `mechanisms` block is now split into two independent stages. Each stage defines its own `physisorption` and `chemisorption` settings.

#### `inhibitor` (Stage 1)
- **`enabled`**: Boolean. Whether to perform inhibitor pre-treatment.
- **`center`**: String or Integer. The binding atom in the inhibitor (Element or index).
- **`branching_limit`**: Integer. Number of top-ranked inhibited surfaces to carry over to Stage 2.
- **`physisorption` / `chemisorption`**: Nested blocks to enable/configure mechanisms for this stage.

#### `precursor` (Stage 2)
- **`center`**: String or Integer. The central reactive atom in the precursor.
- **`physisorption` / `chemisorption`**: Nested blocks to enable/configure mechanisms for this stage.

### 5.2 Physisorption Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `physisorption.placement_height` | Float (A) | `3.5` | Initial height above the top surface layer, interpreted according to `height_mode`. |
| `physisorption.n_rot` | Integer | `8` | Number of Fibonacci-sphere orientations sampled per placement site. |
| `physisorption.center` | String / Int | `"com"` | Rotation center: `"com"` (centre of mass), `"closest"` (atom nearest COM), element symbol, or atom index. |
| `physisorption.height_mode` | String | `"clearance"` | Height interpretation: `"clearance"` — lowest atom at `placement_height` A above surface (see §5.4); `"center"` — rotation center at `placement_height` A above surface. |
| `physisorption.gravity_pull.enabled` | Boolean | `false` | If `true`, the molecule descends step-by-step after initial placement until the first Alvarez vdW contact or the surface hard floor is reached. |
| `physisorption.gravity_pull.step_size` | Float (A) | `0.2` | Descent increment per gravity-pull step. Only active when `gravity_pull.enabled: true`. |

**`chemisorption.rot_steps`**: Integer. Rotational sampling for covalent bond alignment.

### 5.3 Proximity-Based Site Filtering (`proximity_filter`)
When inhibitors are present, the search can be focused around the functionalized regions.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `enabled` | Boolean | `false` | If `true`, only sites within `cutoff` distance of ANY inhibitor atom are considered. |
| `cutoff` | Float (A) | `7.0` | Radius around each inhibitor atom to define the "Active Zone". |
| `visualize` | Boolean | `true` | If `true`, generates `site_proximity_map.png` in the results directory showing the filtering logic. |

### 5.4 Physisorption Site Selection & Orientation Sampling

**Site pool** differs between Stage 1 and Stage 2:

| Stage | Slab state | Site selection |
| :--- | :--- | :--- |
| **Stage 1** (Inhibitor) | Clean | Symmetry-reduced surface atom positions + hollow mid-points between pairs within 2-5 A |
| **Stage 2** (Precursor) | Inhibitor-decorated | `CavityDetector` — finds void centres between inhibitor molecules via EDT distance transform |

`CavityDetector` is only activated when the slab contains atoms with `tag ≥ 2` (actual inhibitor atoms are present). Running Stage 1 on a clean slab always uses the surface-atom path.

**Orientation sampling** — Fibonacci sphere:

For each placement site, $n$ = `n_rot` orientational vectors are generated by the golden-angle spiral on the unit sphere:

$$\vec{v}_i = \left(\cos(\varphi i)\sqrt{1 - y_i^2},\; y_i,\; \sin(\varphi i)\sqrt{1 - y_i^2}\right), \quad y_i = 1 - \tfrac{2i}{n-1}, \quad \varphi = \pi(3 - \sqrt{5})$$

The molecule is rotated so $[0,0,1]$ aligns with each $\vec{v}_i$, then placed at the site.

**Height placement** — `height_mode`:

- `"clearance"` (default): After rotation, the molecule is shifted upward so its **lowest atom** is exactly `placement_height` A above the substrate surface top.  This prevents large molecules (where the COM can be 4+ A above the closest surface-facing atom) from partially embedding in the slab.
- `"center"`: The rotation center (COM or specified element) is placed at `placement_height` A. No extra lift is applied.

**Gravity pull** (optional, `gravity_pull.enabled: true`): After height placement, the molecule descends by `step_size` A per step until either the Alvarez vdW overlap criterion (§5.6) or the hard floor (`z_surface + 0.3 A`) is triggered.

**Steric screening**: Each placed candidate is scored by `_get_steric_fitness`. Poses with any atom pair below the Alvarez vdW threshold (`overlap_scale × (r_i + r_j)`) are rejected immediately (score = −∞). Up to 5 rotationally diverse top-scoring poses per site are kept.

### 5.5 Chemisorption Algorithm
Bond placement is purely geometric (no MLIP required):

1. **Ligand discovery**: The precursor is graph-partitioned at `center` to enumerate detachable ligands and their hapticity.
2. **Dangling-bond mapping**: VSEPR vectors are generated for under-coordinated surface atoms. Directional filter (`db_vec[2] > 0.1`) ensures only vacuum-pointing bonds are used. Symmetry-equivalent pairs are deduplicated.
   - **Substrate Filtering & Buckling**: To avoid calling coordination analysis on deep bulk atoms, substrate atoms whose Z-coordinate is below $z_{\mathrm{sub\_max}} - \texttt{z\_surface\_threshold}$ are filtered out.
   - On strongly buckled surfaces (e.g. Si(110) passivated with bulky inhibitors), surface atoms can relax downward or be pulled upward by over 2.0 Å. Set $\texttt{z\_surface\_threshold} = 3.5$ (default) to ensure these buckled, reactive surface atoms are not accidentally skipped.
3. **Element-specific bond length**: The center->surface bond is placed at $r_{cov}(\text{center}) + r_{cov}(\text{surface})$ (ASE covalent radii). This replaces the previous Si-Si hardcode of 2.35 A.
4. **Best-clearance selection**: All `rot_steps` angles × both site permutations are evaluated. The pose with the largest minimum non-bonded clearance (distance to nearest non-bonded neighbour, excluding bond-forming pairs) is kept. This maximises the geometric buffer available for the subsequent MLIP relaxation and reduces energy blow-up risk.

#### Coordination Analysis Configuration
Nested under `chemisorption`:

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `coordination_analysis.bond_slack` | Float (Å) | `0.2` | Slack added to covalent radii sum when checking coordination neighbors. |
| `coordination_analysis.max_neighbor_dist` | Float (Å) | `4.0` | Maximum search radius for initial neighbor identification. |
| `coordination_analysis.z_surface_threshold` | Float (Å) | `3.5` | Depth below $z_{\mathrm{sub\_max}}$ to consider substrate atoms as surface atoms. |


### 5.6 Candidate Filter

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `overlap_scale` | Float | `0.65` | Scaling factor for the Alvarez (2013) vdW overlap criterion (see below). |
| `max_pair_dist` | Float (A) | `5.0` | Maximum distance between two dangling-bond sites to form a dissociative chemisorption pair. |

> **`symprec` has moved** — symmetry tolerance is now a top-level `reaction_search.symprec` key (§5.0), not nested under `candidate_filter`. Configs that still have `candidate_filter.symprec` will silently ignore that key; move it up one level.

#### Overlap criterion — Alvarez (2013) vdW radii

The steric clash test uses **element-pair-specific** thresholds derived from the Alvarez (2013) database (*Dalton Trans.* **42**, 8617-8636, DOI: [10.1039/c3dt50599e](https://doi.org/10.1039/c3dt50599e)):

$$d_\mathrm{threshold}(i,j) = \texttt{overlap\_scale} \times (r_{\mathrm{vdW},i} + r_{\mathrm{vdW},j})$$

Atom pair $(i, j)$ is rejected if their distance falls below $d_\mathrm{threshold}$.  This applies automatically to **all stages** — physisorption placement, gravity-pull descent, and chemisorption geometry checks — without any extra configuration. (See §5.6 for threshold details).

Selected reference radii (A):

| H | C | N | O | Si | P | S | Fe |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1.20 | 1.77 | 1.66 | 1.50 | 2.19 | 1.90 | 1.89 | 2.44 |

**`overlap_scale` guidelines**:

| Value | Behaviour |
| :--- | :--- |
| `0.55` | Very permissive — only hard nuclear overlap rejected |
| `0.65` | **Recommended default** — rejects genuine clashes while accepting all valid physisorption starting geometries |
| `0.75` | Strict — tighter exclusion; may reject valid poses for compact molecules |

> **Note on `cutoff` override**: An explicit flat threshold (e.g., `cutoff=1.4 A`) can be passed directly to `check_overlap()` for cases where element-independent thresholds are needed (e.g., the chemisorption builder uses `cutoff=1.4` for newly formed bond distance checks). The flat `cutoff` takes precedence over the vdW-based calculation for that specific call.

### 5.7 Transition State Search (Stage 2.5)
This stage connects the physisorption and chemisorption states using a hybrid NEB-ARTn pipeline.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `n_images` | Integer | `7` | Number of intermediate images for the NEB path. |
| `fmax` | Float | `0.05` | Force convergence for TS refinement. |
| `mapping_mode` | String | `"geometric"` | `"geometric"` uses Hungarian algorithm for index mapping; `"identity"` assumes fixed order. |
| `verification` | Boolean | `true` | Automatically runs vibrational analysis after TS refinement to confirm exactly one imaginary frequency. |

#### Hybrid TS Search Workflow
1.  **Alignment & Mapping**: The initial and final states are aligned using the **Minimum Image Convention (MIC)** to ensure periodic boundary consistency. This prevents artificial high-energy barriers caused by atoms crossing unit cell boundaries. If atom indices are inconsistent (e.g., from external VASP files), the **Geometric Mapping Engine** reorders them automatically.
2.  **NEB Interpolation**: An initial reaction path is generated via linear interpolation (or IDPP if enabled).
3.  **ARTn Refinement**: The highest energy image is used as a starting point for the Activation Relaxation Technique (ARTn) to find the exact saddle point.
4.  **Verification**: A partial Hessian is computed at the saddle point to verify the existence of a single imaginary mode corresponding to the reaction coordinate.

---

## 1.1 Pipeline Control (`workflow`)

All stage enable-flags live in one unified block so there is no ambiguity:

```yaml
workflow:
  slab_relax:       false   # [REQUIRES POTENTIAL] Relax bare slab before search
  candidate_relax:  true    # [REQUIRES POTENTIAL] Relax each candidate after placement
  md_equilibrate:   false   # [REQUIRES POTENTIAL] NVT-MD after candidate_relax
  post_md_relax:    true    # Re-relax after MD (needs md_equilibrate: true)
```

Relaxation hyper-parameters are consolidated under a shared top-level block:

```yaml
relaxation:
  fmax:         0.05    # eV/Å  — applies to slab_relax, candidate_relax, post_md_relax
  steps:        100
  frozen_z_ang: 5.5     # Fix atoms below z_min + this height (Å)

equilibration:
  temperature_K: 300
  md_steps:      1000
  timestep_fs:   1.0
  damping:       100.0
```

> **Backward compatibility**: configs using the old `surface_prep.slab_relaxation.enabled` and `verification.relaxation.enabled` keys are still parsed.  The `workflow` block takes priority when present.

---

## 6. Verification Pipeline

### 6.1 Verification Logic
Enable/disable is controlled by `workflow.candidate_relax` (geometry opt) and
`workflow.md_equilibrate` (NVT MD).  The optimizer algorithm is set by
`engine.relaxation.optimizer`; hyper-parameters (fmax, steps, frozen_z_ang)
come from the top-level `relaxation` block.

- **`verification.selected_indices`**: List or expression (e.g., `[0, 5, 10]`). Only process specific candidate indices. If `null`, all candidates are verified.

**Explosion safety**: An `ExplosionMonitor` is attached to every optimizer and MD integrator. It halts the calculation if the per-atom energy turns positive, jumps by more than 10 eV/atom, or shifts by an order of magnitude relative to the initial value. The candidate is discarded and the workflow continues rather than consuming the full step budget on a broken geometry.

### 6.2 Adsorption Energy ($E_{ads}$)
Calculated using: $E_{ads} = E_{total} - (E_{gas} + E_{base})$
- **$E_{gas}$**: Optimized energy of the isolated molecule.
- **$E_{base}$**: Potential energy of the surface (potentially inhibited) before adsorption.

---

## 7. Output Management & Directory Structure
AutoFlow-SRXN uses a hierarchical output structure for clear traceability.

### 7.1 Folder Naming
Each batch pair follows the naming convention:
`{output_prefix}/{inhibitor}_pretreated_{precursor}/`
*(Note: If no inhibitor is used, `{inhibitor}` defaults to `clean`.)*

### 7.2 Internal File Structure
Inside each run directory:
- **`workflow.log`**: Detailed execution trace.
- **`interface_summary.txt`**: Stage 0a candidate table (only when `interface.enabled: true`).
- **`interface_candidates.html`**: Stage 0a interactive Plotly report (only when `interface.plot: true`).
- **`interface_<N>.extxyz`**: Stage 0a built slab(s) (N = 0 … build_top_k-1).
- **`site_map.png`**: Top-view map of unique adsorption sites generated before each physisorption search. Sites are auto-classified as top (red diamond), bridge (blue triangle), or hollow (green circle); surface atoms are coloured by (element, Z-layer) sublattice. Generated automatically when `physisorption.enabled: true`; requires `matplotlib` and `scipy`.
- **`stage1_inhibitor/`**: Intermediate candidates for the inhibitor stage.
  - `stage1_inhibitor_candidates.extxyz`: All generated poses.
  - `stage1_inhibitor_relaxed.extxyz`: Verified (relaxed) poses with energy metadata.
- **`stage2_precursor/`**: Intermediate candidates for the precursor stage (same naming pattern).
- **`final_results.extxyz`**: Final verified stable structures, sorted by adsorption energy.
- **`ref_energies.log`**: Gas-phase reference energies for all unique molecules (written once per batch).

---

## 7.3 Surface Adsorption Site Map (`surface.site_map`)

The `autoflow_srxn.surface.site_map` module provides surface-agnostic top-view visualization of adsorption sites. It is called automatically by the workflow (when `physisorption.enabled: true`) and is also available as a standalone API for post-processing or custom scripts.

### Public API

```python
from autoflow_srxn.surface import plot_adsorption_site_map, generate_and_plot_site_map

# --- Option A: plot a pre-computed site list ---
plot_adsorption_site_map(slab, sites, "site_map.png")

# --- Option B: reproduce the exact workflow site set + plot ---
sites = generate_and_plot_site_map(slab, "site_map.png", symprec=0.2)
```

**`plot_adsorption_site_map(slab, sites, output_path, *, ...)`**

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `slab` | — | ASE `Atoms` — substrate slab (adsorbate atoms with `tag ≥ 2` are excluded automatically). |
| `sites` | — | `list` of array-like `(3,)` Cartesian coordinates from `get_unique_coordinates`. |
| `output_path` | — | Save path; extension determines format (`png`, `pdf`, `svg`). |
| `title` | auto | Figure title; auto-generated from composition + cell dimensions if `None`. |
| `site_labels` | auto | Override auto-generated labels (`T0`, `B0`, `H0`, …). |
| `show_delaunay` | `True` | Draw Delaunay triangulation of surface atoms (light blue). |
| `show_cell` | `True` | Draw unit-cell boundary (dotted). |
| `margin_ang` | `1.5` | Extra margin around the view window (Å). |
| `figsize` | `(10, 10)` | Matplotlib figure size. |
| `dpi` | `150` | Output resolution. |

**`generate_and_plot_site_map(slab, output_path, *, symprec=0.2, ...)`**

Convenience wrapper that:
1. Identifies surface atoms (`find_surface_indices`, top side).
2. Generates raw sites: surface atom positions + Delaunay mid-points (bridges) + triangle centroids (hollows).
3. Deduplicates via `AdsorptionWorkflowManager.get_unique_coordinates` (same routine used by the workflow).
4. Calls `plot_adsorption_site_map` and saves the figure.
5. Returns the `list` of unique site coordinates.

**Site classification** (`_classify_site`):

| Criterion | Label |
| :--- | :--- |
| Nearest surface atom < 0.15 Å | `top` |
| Two nearest atoms at similar distance (< 2.5 Å, relative difference < 8 %) | `bridge` |
| All other sites | `hollow` |

---

## 8. Simulation Engine (`engine`)

### 8.1 Backend Selection

| Backend | Description | Interface |
| :--- | :--- | :--- |
| `mace` | MACE-MP equivariant GNN (89 elements, Materials Project training set) | `mace.calculators.mace_mp` |
| `sevennet` | SevenNet E(3)-equivariant GNN, strong for surface catalysis | `sevenn.calculator.SevenNetCalculator` |
| `emt` | Effective Medium Theory (simple metals only). Use for smoke-tests only. | `ase.calculators.emt.EMT` |

### 8.2 Optimizer Options (`engine.relaxation.optimizer`)

| Optimizer | Best for |
| :--- | :--- |
| `BFGS` (default) | General surface optimisations near equilibrium |
| `LBFGS` | Large supercells (> 500 atoms) — reduced memory |
| `FIRE` | Highly strained initial geometries |
| `CG_FIRE` | Two-stage: SciPy CG escape then FIRE fine-tune |
| `GPMin` | Expensive calculators where minimising force calls matters |

### 8.3 Key Engine Parameters
- **`dtype`**: `"float32"` (MD / coarse screening) or `"float64"` (geometry optimisation, vibrations).
- **`d3`**: Enable Grimme D3(BJ) dispersion — recommended for physisorption and weakly-bound precursors.
- **zbl.enabled**: Add ZBL screened-Coulomb repulsion to prevent MLIP instabilities at sub-bonding distances.
- **Haptic Ligand Support**: Specialized `skip_pairs` logic for $\eta^n$ ligands (Allyl, Cp) ensures multiple bonding atoms are excluded from overlap checks.

### 8.4 Advanced Simulation Logic

#### 8.4.1 Geometric Mapping (Hungarian Algorithm)
To support externally calculated structures (e.g., from VASP) where atom orderings might differ, AutoFlow-SRXN employs an optimal bipartite matching algorithm. It minimizes the total RMSD between states under PBC, ensuring a 1:1 atom correspondence even if indices are randomized.

#### 8.4.2 Constraint-Aware Vibrations
When atoms are frozen (e.g., bottom slab layers), the engine automatically pads the Hessian and eigenvectors. This ensures compatibility with TS searchers (like ARTn) that expect full-rank arrays while correctly ignoring the force contributions from fixed atoms.

#### 8.4.3 Physics-Informed Physisorption Alignment
The physisorption engine uses PCA to align the molecule's thin axis with the surface normal (Flat Alignment). Additionally, it calculates the average hydrogen direction relative to the center of mass; if hydrogens point towards the surface, the molecule is automatically flipped 180° (H-up Logic) to maximize physical plausibility.

---

## 9. Metadynamics (`analysis.metadynamics`)

AutoFlow-SRXN ships a **PLUMED-free, ASE-native (well-tempered) metadynamics**
engine that reconstructs a **2-D free-energy surface (FES)** of the precursor
surface reaction. No external installation is needed: the history-dependent
bias is implemented as an ordinary ASE `Calculator` and added to the physical
(MLIP/EMT) calculator with `ase.calculators.mixing.SumCalculator`.

**Source files**

| File | Contents |
|------|----------|
| `autoflow_srxn/metadynamics/collective_variables.py` | CV definitions + analytic gradients + atom-selection helpers + `build_cv` factory |
| `autoflow_srxn/metadynamics/md_bias.py` | `MetadynamicsBias` (bias calculator), `ColvarLogger`, FES reconstruction |
| `autoflow_srxn/metadynamics/workflow.py` | `MetadynamicsWorkflow` — reads config, runs MD, writes outputs & 2-D FES plot |
| `examples/metadynamics/run_metadynamics.py` | CLI runner (`config.yaml` + `structure.vasp`) |

### 9.1 Theory

#### Bias potential

Given collective variables $\mathbf{s}(\mathbf{R}) = (s_1,\dots,s_N)$, a Gaussian
hill is deposited every $\tau$ steps. The accumulated bias after depositing
$K$ hills is

$$
V(\mathbf{s}) \;=\; \sum_{k=1}^{K} h_k \,
\exp\!\left( -\sum_{d=1}^{N} \frac{\bigl(s_d - c_{k,d}\bigr)^2}{2\,\sigma_d^{2}} \right),
$$

where $c_{k,d}$ is the position of hill $k$ in CV $d$, $\sigma_d$ its width and
$h_k$ its height.

#### Bias force

The force added to each atom is the chain rule through the CVs:

$$
\mathbf{F}^{\text{bias}}_{i}
= -\frac{\partial V}{\partial \mathbf{R}_i}
= -\sum_{d=1}^{N}\frac{\partial V}{\partial s_d}\,
\frac{\partial s_d}{\partial \mathbf{R}_i},
\qquad
\frac{\partial V}{\partial s_d}
= -\sum_{k} \frac{h_k\,g_k\,(s_d-c_{k,d})}{\sigma_d^{2}},
$$

with $g_k$ the Gaussian of hill $k$. This is implemented vectorised in
`MetadynamicsBias`:

```python
def _bias_and_dVds(self, s):
    diff = (s[None, :] - self.centers) / self.sigmas[None, :]   # (nhills, ncv)
    g = self.heights * np.exp(-0.5 * np.sum(diff**2, axis=1))   # (nhills,)
    V = float(g.sum())
    dVds = -np.sum((g[:, None] * diff / self.sigmas[None, :]), axis=0)
    return V, dVds

def calculate(self, atoms=None, ...):
    s, grads = self._cv_values_and_grads(atoms)
    V, dVds = self._bias_and_dVds(s)
    forces = np.zeros((len(atoms), 3))
    for d in range(self.ncv):
        forces -= dVds[d] * grads[d]          # F = -dV/ds · ds/dR
    self.results["energy"] = V
    self.results["forces"] = forces
```

#### Well-tempered deposition

To guarantee convergence, the deposited height is damped by the bias already
present at the current point:

$$
h_k \;=\; h_0 \,\exp\!\left( -\frac{V(\mathbf{s}_k)}{k_B \,\Delta T} \right),
\qquad \Delta T = (\gamma - 1)\,T,
$$

where $\gamma$ is the **bias factor** (`bias_factor`). Setting
`bias_factor: null` recovers standard (non-tempered) metadynamics
($h_k = h_0$).

```python
def deposit(self, atoms):
    s, _ = self._cv_values_and_grads(atoms)
    if self.gamma is not None:
        V, _ = self._bias_and_dVds(s)
        h = self.h0 * np.exp(-V / self._kB_dT)   # _kB_dT = (γ-1)·kB·T
    else:
        h = self.h0
    self.centers = np.vstack([self.centers, s])
    self.heights = np.append(self.heights, h)
```

#### Free-energy reconstruction

At convergence the FES is recovered from the bias:

$$
F(\mathbf{s}) =
\begin{cases}
-\,V(\mathbf{s}) & \text{standard} \\[4pt]
-\dfrac{\gamma}{\gamma-1}\,V(\mathbf{s}) & \text{well-tempered}
\end{cases}
$$

When **more than two** CVs are biased, the 2-D FES over the chosen pair
$(a,b)$ is obtained by marginalising the remaining CVs:

$$
F(s_a,s_b) = -k_B T \,\ln \!\!\sum_{\text{others}} \exp\!\left(-\frac{F(\mathbf{s})}{k_B T}\right).
$$

### 9.2 Collective Variables

All CVs expose `value_and_grad(atoms) -> (s, grad)` with `grad` of shape
`(natoms, 3)`. Every gradient is analytic and is unit-tested against central
differences (`unittests/test_metadynamics.py`). Note the ASE convention
`atoms.get_distance(i, j, vector=True)` returns $\mathbf{R}_j-\mathbf{R}_i$.

#### `distance` — raw bond length

$$ s = \lVert \mathbf{R}_j - \mathbf{R}_i \rVert,\qquad
\frac{\partial s}{\partial \mathbf{R}_j} = \hat{\mathbf{u}},\;
\frac{\partial s}{\partial \mathbf{R}_i} = -\hat{\mathbf{u}},\;
\hat{\mathbf{u}}=\frac{\mathbf{R}_j-\mathbf{R}_i}{s}. $$

Use for a single, well-defined bond — typically the **forming** central-atom ↔
substrate bond (so metadynamics can drive it shorter).

#### `coordination` — rational-switching coordination number

$$ s = \sum_{j \in \text{group}} \frac{1 - x_{ij}^{\,n}}{1 - x_{ij}^{\,m}},
\qquad x_{ij} = \frac{r_{ij}}{r_0}. $$

Permutation-invariant and bounded ($s \to 0$ when the group leaves), so it is
the robust choice for the **breaking** bond when several equivalent ligand
atoms exist (e.g. the four Cl of TiCl₄). The $r_{ij}=r_0$ singularity is
handled by the L'Hôpital limit $f \to n/m$:

```python
def _switch(self, r):
    x = r / self.r0
    if abs(x - 1.0) < 1e-6:
        f = self.n / self.m
        dfdx = (self.n - self.m) / (2.0 * self.m)
    else:
        num, den = 1.0 - x**self.n, 1.0 - x**self.m
        f = num / den
        dfdx = ((-self.n*x**(self.n-1))*den - num*(-self.m*x**(self.m-1))) / den**2
    return f, dfdx / self.r0          # df/dr
```

#### `proton_transfer` — antisymmetric stretch

$$ \xi = d(\text{donor–H}) - d(\text{acceptor–H}). $$

$\xi<0$: the proton sits on the surface donor (O–H / N–H); $\xi>0$: it has
moved onto the leaving ligand. Biasing $\xi$ positive therefore **induces the
ligand–H byproduct bond** (HCl, amine-H). Add it as a third CV when the
mechanism is a concerted proton transfer.

### 9.3 Atom Selection

CV endpoints/groups accept several spec forms (resolved in
`collective_variables.resolve_atom` / `resolve_group`):

| Spec | Meaning |
|------|---------|
| `42` or `{index: 42}` | atom index 42 |
| `"Si"` | all Si atoms |
| `"O@substrate"` | O atoms with `tag < 2` (slab) |
| `"N@adsorbate"` | N atoms with `tag ≥ 2` (precursor) |
| `[3, 7, 11]` | explicit group |

The package's builders set these tags automatically, so reactions can be set
up **without hard-coding indices**. For a single-atom endpoint that matches
several atoms, the one nearest the other endpoint is chosen deterministically.

### 9.4 Configuration Reference

```yaml
analysis:
  metadynamics:
    enabled:           true
    temperature_K:     500.0     # Langevin thermostat temperature
    timestep_fs:       1.0
    friction:          0.01
    steps:             20000     # total MD steps
    deposition_stride: 50        # deposit a Gaussian every N steps
    colvar_stride:     50        # COLVAR / trajectory write interval
    height:            0.05      # initial Gaussian height h0 (eV)
    bias_factor:       10        # well-tempered γ (>1); null → standard
    freeze_below_z:    null      # freeze atoms below this z (Å); null → none

    cvs:                         # ≥ 2 CVs
      - name:    forming_bond
        type:    distance
        center:  "Si@adsorbate"
        partner: "O@substrate"
        sigma:   0.10
        grid_min: 1.4           # plot/grid range (optional)
        grid_max: 4.0
      - name:    breaking_bond
        type:    coordination
        center:  "Si@adsorbate"
        group:   "N@adsorbate"
        r0:      2.2
        n:       6
        m:       12
        sigma:   0.10
      # Optional 3rd CV (marginalised out of the 2-D plot):
      # - name: proton_transfer
      #   type: proton_transfer
      #   donor:    "O@substrate"
      #   acceptor: "N@adsorbate"
      #   sigma:    0.10

    plot:
      cvs:  ["forming_bond", "breaking_bond"]   # the two CVs spanning the FES
      bins: 120
```

**Key notes**
- Define **≥ 2** CVs; any extra ones are biased and marginalised out of the 2-D plot.
- `sigma` ≈ 0.3–0.5 × the CV's thermal fluctuation; `bias_factor` 5–20 is typical.
- For a `distance` breaking-bond CV, set `grid_max` and consider `freeze_below_z` so a dissociated fragment cannot drift off and stall recrossing. `coordination` avoids this by construction.

### 9.5 Outputs

Written under `<output_dir>/` (default `metad/`):

| File | Description |
|------|-------------|
| `fes_2d.png` | 2-D free-energy contour plot (axes = the two selected CVs, colour = eV) |
| `fes_2d.npz` | FES grid arrays (`x`, `y`, `fes`, `cv_x`, `cv_y`) for re-plotting |
| `COLVAR` | time series of CV values + bias energy (convergence check) |
| `HILLS` | deposited Gaussians (PLUMED-like format) |
| `metad_traj.extxyz`, `metad_final.vasp` | MD trajectory and final structure |

### 9.6 Usage

**CLI**

```bash
python examples/metadynamics/run_metadynamics.py config_full.yaml structure.vasp results/metad
```

**Python API**

```python
from autoflow_srxn.simulation.potentials import SimulationEngine
from autoflow_srxn.metadynamics import MetadynamicsWorkflow

engine = SimulationEngine(config)                       # full config (engine.*)
wf = MetadynamicsWorkflow(engine, config=config["analysis"]["metadynamics"])
res = wf.run(atoms, output_dir="results/metad")

x, y, fes = res["fes"]                                  # 2-D FES arrays (eV)
print("Approx. barrier:", fes.max(), "eV")
```

The structure passed to `run()` should carry tags marking slab (`tag < 2`) vs
adsorbate (`tag ≥ 2`) so the `Element@region` selectors resolve correctly; the
package's surface/chemisorption builders set these automatically.

### 9.7 Practical Guidance

- **Check convergence** via `COLVAR`: the CVs should diffuse back and forth over the explored range, and the FES should stop deepening as hills accumulate.
- **CV choice is everything.** A poorly chosen CV that misses a slow orthogonal degree of freedom (surface reconstruction, byproduct diffusion) gives a hysteretic, non-reproducible FES.
- For an ALD ligand-exchange reaction, a good 2-D set is *forming bond* (`distance`/`coordination`) × *breaking bond* (`coordination`); add `proton_transfer` as a third dimension when the byproduct forms via concerted H transfer.
- If a quick NEB/scan (Stage 2.5, §5.7) already shows a **barrierless** path, the FES will be downhill and metadynamics is unnecessary — use it for activated reactions where a barrier exists.

### 9.8 Validation

The engine is validated against references with **known answers**, in two tiers
(`unittests/test_metadynamics_validation.py`, plus an EMT example).

**Tier 1 — algorithm correctness (analytic reference).**
- *FES reconstruction / well-tempered scaling* — deterministic (no MD): a single
  Gaussian of height $h$ must produce a well of depth $h$ (standard) or
  $\tfrac{\gamma}{\gamma-1}h$ (well-tempered). Checked exactly.
- *Double-well potential* $V(x,y)=h_0\big((x/a)^2-1\big)^2+\tfrac12 k y^2$ — the
  canonical enhanced-sampling sanity benchmark, with an exactly known barrier.
  `CoordinateCV` over $(x,y)$ recovers the analytic barrier; convergence is
  clean once the walker is bounded (harmonic walls):

  | MD steps | recovered barrier (eV) | analytic |
  |---------:|:----------------------:|:--------:|
  | 15 000   | 0.117                  | 0.200    |
  | 30 000   | 0.176                  | 0.200    |
  | 50 000   | 0.197 ± 0.022          | 0.200    |

**Tier 2 — physical, literature-comparable (EMT).**
`examples/metadynamics/cu_diffusion_metad.py` runs Cu adatom diffusion on
Cu(100) and compares the metadynamics hop barrier against (a) a NEB barrier on
the **same** EMT potential and (b) the published band:

| Method | Cu(100) hop barrier (eV) |
|--------|:------------------------:|
| **NEB (EMT, this work)**     | **0.420** |
| **MetaD (EMT, this work, 50k steps)** | **0.340** |
| EMT (literature)             | ~0.40 |
| DFT (literature, hop)        | ~0.50 |
| Experiment                   | ~0.28–0.40 |

The metaD↔NEB agreement (~0.08 eV at 50k steps, tightening with longer sampling)
on an identical potential is the rigorous internal check; the literature band
confirms the EMT reference itself is physically reasonable. The same protocol —
metaD FES vs NEB on the production MLIP — is the recommended way to validate CV
choice for a real precursor reaction (a good CV reproduces the NEB barrier).
