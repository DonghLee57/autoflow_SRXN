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
| `adsorbate` | Path to a single precursor structure file (.vasp, .xyz). |
| `adsorbates_dir` | Directory containing multiple precursor files for batch screening. |
| `inhibitor` | Path to a single inhibitor structure file. |
| `inhibitors_dir` | Directory containing multiple inhibitor files for batch screening. |
| `substrate_bulk` | Path to the bulk crystalline structure (used if `slab_generation` is enabled). |
| `substrate_slab` | Path to a pre-generated slab file. |
| `output_prefix` | Base directory name for batch output folders (default: `results`). |
| `include_no_inhibitor` | If `true`, includes a 'clean' baseline run for each precursor. |

---

## 3. Surface Preparation (`surface_prep`)
Handles the creation and modification of the substrate surface.

### 3.1 Slab Generation
- **`enabled`**: Boolean. Enable/disable ASE-based slab cutting from bulk.
- **`miller`**: List of 3 integers (e.g., `[1, 0, 0]`). Miller indices of the surface plane.
- **`thickness_ang`**: Float (Å). Minimum thickness of the slab.
- **`vacuum_ang`**: Float (Å). Vacuum padding on both sides.
- **`target_area_ang2`**: Float (Å²). Target surface area; the engine selects the largest supercell that does not exceed this value while optimising aspect ratio.
- **`supercell_matrix`**: List of lists (e.g., `[[2,0],[0,2]]`). Explicit supercell matrix. Overrides `target_area_ang2` if set.
- **`top_termination`**: String (Element symbol, e.g., `"O"`). Ensures the top surface ends with the specified element.
- **`bottom_termination`**: String (Element symbol, e.g., `"O"`). Ensures the bottom surface ends with the specified element.

### 3.2 Reconstruction & Passivation
- **`reconstruction`**: Apply automated surface reconstruction (auto/ionic/covalent/metallic).
- **`passivation`**: Saturate dangling bonds (typically on the bottom side using "H").

### 3.3 Slab Relaxation & Equilibration
- **`slab_relaxation.enabled`**: Geometry optimisation of the clean substrate before adsorption search.
- **`slab_relaxation.frozen_z_ang`**: Fix atoms below this Z-height to simulate the bulk interior during relaxation.
- **`equilibration.enabled`**: NVT MD pre-equilibration of the substrate at `engine.md.temperature_K`.

---

## 4. Reaction Search (`reaction_search`)
Explores the configuration space of adsorbates through two sequential stages.

### 4.1 Stage-Specific Controls
The `mechanisms` block is now split into two independent stages. Each stage defines its own `physisorption` and `chemisorption` settings.

#### `inhibitor` (Stage 1)
- **`enabled`**: Boolean. Whether to perform inhibitor pre-treatment.
- **`center`**: String or Integer. The binding atom in the inhibitor (Element or index).
- **`branching_limit`**: Integer. Number of top-ranked inhibited surfaces to carry over to Stage 2.
- **`physisorption` / `chemisorption`**: Nested blocks to enable/configure mechanisms for this stage.

#### `precursor` (Stage 2)
- **`center`**: String or Integer. The central reactive atom in the precursor.
- **`physisorption` / `chemisorption`**: Nested blocks to enable/configure mechanisms for this stage.

### 4.2 Physisorption Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `physisorption.placement_height` | Float (Å) | `3.5` | Initial height above the top surface layer, interpreted according to `height_mode`. |
| `physisorption.rot_steps` | Integer | `8` | Number of Fibonacci-sphere orientations sampled per placement site. |
| `physisorption.center` | String / Int | `"com"` | Rotation center: `"com"` (centre of mass), `"closest"` (atom nearest COM), element symbol, or atom index. |
| `physisorption.height_mode` | String | `"clearance"` | Height interpretation: `"clearance"` — lowest atom at `placement_height` Å above surface (see §4.3); `"center"` — rotation center at `placement_height` Å above surface. |
| `physisorption.gravity_pull.enabled` | Boolean | `false` | If `true`, the molecule descends step-by-step after initial placement until the first Alvarez vdW contact or the surface hard floor is reached. |
| `physisorption.gravity_pull.step_size` | Float (Å) | `0.2` | Descent increment per gravity-pull step. Only active when `gravity_pull.enabled: true`. |

**`chemisorption.rot_steps`**: Integer. Rotational sampling for covalent bond alignment.

### 4.3 Physisorption Site Selection & Orientation Sampling

**Site pool** differs between Stage 1 and Stage 2:

| Stage | Slab state | Site selection |
| :--- | :--- | :--- |
| **Stage 1** (Inhibitor) | Clean | Symmetry-reduced surface atom positions + hollow mid-points between pairs within 2–5 Å |
| **Stage 2** (Precursor) | Inhibitor-decorated | `CavityDetector` — finds void centres between inhibitor molecules via EDT distance transform |

`CavityDetector` is only activated when the slab contains atoms with `tag ≥ 2` (actual inhibitor atoms are present). Running Stage 1 on a clean slab always uses the surface-atom path.

**Orientation sampling** — Fibonacci sphere:

For each placement site, $n$ = `rot_steps` orientational vectors are generated by the golden-angle spiral on the unit sphere:

$$\vec{v}_i = \left(\cos(\varphi i)\sqrt{1 - y_i^2},\; y_i,\; \sin(\varphi i)\sqrt{1 - y_i^2}\right), \quad y_i = 1 - \tfrac{2i}{n-1}, \quad \varphi = \pi(3 - \sqrt{5})$$

The molecule is rotated so $[0,0,1]$ aligns with each $\vec{v}_i$, then placed at the site.

**Height placement** — `height_mode`:

- `"clearance"` (default): After rotation, the molecule is shifted upward so its **lowest atom** is exactly `placement_height` Å above the substrate surface top.  This prevents large molecules (where the COM can be 4+ Å above the closest surface-facing atom) from partially embedding in the slab.
- `"center"`: The rotation center (COM or specified element) is placed at `placement_height` Å. No extra lift is applied.

**Gravity pull** (optional, `gravity_pull.enabled: true`): After height placement, the molecule descends by `step_size` Å per step until either the Alvarez vdW overlap criterion (§4.5) or the hard floor (`z_surface + 0.3 Å`) is triggered.

**Steric screening**: Each placed candidate is scored by `_get_steric_fitness`. Poses with any atom pair below the Alvarez vdW threshold (`overlap_scale × (r_i + r_j)`) are rejected immediately (score = −∞). Up to 5 rotationally diverse top-scoring poses per site are kept.

### 4.4 Chemisorption Algorithm
Bond placement is purely geometric (no MLIP required):

1. **Ligand discovery**: The precursor is graph-partitioned at `center` to enumerate detachable ligands and their hapticity.
2. **Dangling-bond mapping**: VSEPR vectors are generated for under-coordinated surface atoms. Directional filter (`db_vec[2] > 0.1`) ensures only vacuum-pointing bonds are used. Symmetry-equivalent pairs are deduplicated.
3. **Element-specific bond length**: The center→surface bond is placed at $r_{cov}(\text{center}) + r_{cov}(\text{surface})$ (ASE covalent radii). This replaces the previous Si–Si hardcode of 2.35 Å.
4. **Best-clearance selection**: All `rot_steps` angles × both site permutations are evaluated. The pose with the largest minimum non-bonded clearance (distance to nearest non-bonded neighbour, excluding bond-forming pairs) is kept. This maximises the geometric buffer available for the subsequent MLIP relaxation and reduces energy blow-up risk.

### 4.5 Candidate Filter

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `overlap_scale` | Float | `0.65` | Scaling factor for the Alvarez (2013) vdW overlap criterion (see below). |
| `symprec` | Float (Å) | `0.2` | Symmetry-equivalence tolerance for site deduplication via spglib. |
| `max_pair_dist` | Float (Å) | `5.0` | Maximum distance between two dangling-bond sites to form a dissociative chemisorption pair. |

#### Overlap criterion — Alvarez (2013) vdW radii

The steric clash test uses **element-pair-specific** thresholds derived from the Alvarez (2013) database (*Dalton Trans.* **42**, 8617–8636, DOI: [10.1039/c3dt50599e](https://doi.org/10.1039/c3dt50599e)):

$$d_\mathrm{threshold}(i,j) = \texttt{overlap\_scale} \times (r_{\mathrm{vdW},i} + r_{\mathrm{vdW},j})$$

Atom pair $(i, j)$ is rejected if their distance falls below $d_\mathrm{threshold}$.  This applies automatically to **all stages** — physisorption placement, gravity-pull descent, and chemisorption geometry checks — without any extra configuration.

Selected reference radii (Å):

| H | C | N | O | Si | P | S | Fe |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1.20 | 1.77 | 1.66 | 1.50 | 2.19 | 1.90 | 1.89 | 2.44 |

**`overlap_scale` guidelines**:

| Value | Behaviour |
| :--- | :--- |
| `0.55` | Very permissive — only hard nuclear overlap rejected |
| `0.65` | **Recommended default** — rejects genuine clashes while accepting all valid physisorption starting geometries |
| `0.75` | Strict — tighter exclusion; may reject valid poses for compact molecules |

> **Note on `cutoff` override**: An explicit flat threshold (e.g., `cutoff=1.4 Å`) can be passed directly to `check_overlap()` for cases where element-independent thresholds are needed (e.g., the chemisorption builder uses `cutoff=1.4` for newly formed bond distance checks). The flat `cutoff` takes precedence over the vdW-based calculation for that specific call.

---

## 5. Verification Pipeline (`verification`)
Standardized validation for discovery candidates.

### 5.1 Verification Logic
- **`relaxation`**: Local geometry optimisation (BFGS / LBFGS / FIRE / GPMin). The `optimizer` key in `engine.relaxation` selects the algorithm.
- **`equilibration`**: Optional NVT MD sampling for thermal stability assessment.
- **`selected_indices`**: List or expression (e.g., `[0, 5, 10]`). Only process specific candidate indices. If `null`, all candidates are verified.

**Explosion safety**: An `ExplosionMonitor` is attached to every optimizer and MD integrator. It halts the calculation if the per-atom energy turns positive, jumps by more than 10 eV/atom, or shifts by an order of magnitude relative to the initial value. The candidate is discarded and the workflow continues rather than consuming the full step budget on a broken geometry.

### 5.2 Adsorption Energy ($E_{ads}$)
Calculated using: $E_{ads} = E_{total} - (E_{gas} + E_{base})$
- **$E_{gas}$**: Optimized energy of the isolated molecule.
- **$E_{base}$**: Potential energy of the surface (potentially inhibited) before adsorption.

---

## 6. Output Management & Directory Structure
AutoFlow-SRXN uses a hierarchical output structure for clear traceability.

### 6.1 Folder Naming
Each batch pair follows the naming convention:
`{output_prefix}/{inhibitor}_pretreated_{precursor}/`
*(Note: If no inhibitor is used, `{inhibitor}` defaults to `clean`.)*

### 6.2 Internal File Structure
Inside each run directory:
- **`workflow.log`**: Detailed execution trace.
- **`stage1_inhibitor/`**: Intermediate candidates for the inhibitor stage.
  - `stage1_inhibitor_candidates.extxyz`: All generated poses.
  - `stage1_inhibitor_relaxed.extxyz`: Verified (relaxed) poses with energy metadata.
- **`stage2_precursor/`**: Intermediate candidates for the precursor stage (same naming pattern).
- **`final_results.extxyz`**: Final verified stable structures, sorted by adsorption energy.
- **`ref_energies.log`**: Gas-phase reference energies for all unique molecules (written once per batch).

---

## 7. Simulation Engine (`engine`)

### 7.1 Backend Selection

| Backend | Description | Interface |
| :--- | :--- | :--- |
| `mace` | MACE-MP equivariant GNN (89 elements, Materials Project training set) | `mace.calculators.mace_mp` |
| `sevennet` | SevenNet E(3)-equivariant GNN, strong for surface catalysis | `sevenn.calculator.SevenNetCalculator` |
| `emt` | Effective Medium Theory (simple metals only). Use for smoke-tests only. | `ase.calculators.emt.EMT` |

### 7.2 Optimizer Options (`engine.relaxation.optimizer`)

| Optimizer | Best for |
| :--- | :--- |
| `BFGS` (default) | General surface optimisations near equilibrium |
| `LBFGS` | Large supercells (> 500 atoms) — reduced memory |
| `FIRE` | Highly strained initial geometries |
| `CG_FIRE` | Two-stage: SciPy CG escape then FIRE fine-tune |
| `GPMin` | Expensive calculators where minimising force calls matters |

### 7.3 Key Engine Parameters
- **`dtype`**: `"float32"` (MD / coarse screening) or `"float64"` (geometry optimisation, vibrations).
- **`d3`**: Enable Grimme D3(BJ) dispersion — recommended for physisorption and weakly-bound precursors.
- **`zbl.enabled`**: Add ZBL screened-Coulomb repulsion to prevent MLIP instabilities at sub-bonding distances.
