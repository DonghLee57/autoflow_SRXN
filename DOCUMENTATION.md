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

### 4.2 Shared Mechanism Parameters
- **`physisorption.placement_height`**: Float (Å). Initial molecule height above the surface.
- **`physisorption.rot_steps`**: Integer. Fibonacci-sphere orientations sampled per site.
- **`chemisorption.rot_steps`**: Integer. Rotational sampling for covalent bond alignment.

### 4.3 Physisorption Site Selection Logic
The site pool for physisorption differs between Stage 1 and Stage 2:

| Stage | Slab state | Site selection |
| :--- | :--- | :--- |
| **Stage 1** (Inhibitor) | Clean | Symmetry-reduced surface atom positions + `placement_height` offset |
| **Stage 2** (Precursor) | Inhibitor-decorated | `CavityDetector` — finds void centres between inhibitor molecules via EDT distance transform + gravity pull toward the substrate |

`CavityDetector` is only activated when the slab contains atoms with `tag ≥ 2` (i.e., actual inhibitor atoms are present). Running Stage 1 on a clean slab always uses the surface-atom path.

### 4.4 Chemisorption Algorithm
Bond placement is purely geometric (no MLIP required):

1. **Ligand discovery**: The precursor is graph-partitioned at `center` to enumerate detachable ligands and their hapticity.
2. **Dangling-bond mapping**: VSEPR vectors are generated for under-coordinated surface atoms. Directional filter (`db_vec[2] > 0.1`) ensures only vacuum-pointing bonds are used. Symmetry-equivalent pairs are deduplicated.
3. **Element-specific bond length**: The center→surface bond is placed at $r_{cov}(\text{center}) + r_{cov}(\text{surface})$ (ASE covalent radii). This replaces the previous Si–Si hardcode of 2.35 Å.
4. **Best-clearance selection**: All `rot_steps` angles × both site permutations are evaluated. The pose with the largest minimum non-bonded clearance (distance to nearest non-bonded neighbour, excluding bond-forming pairs) is kept. This maximises the geometric buffer available for the subsequent MLIP relaxation and reduces energy blow-up risk.

### 4.5 Candidate Filter
- **`overlap_cutoff`**: Hard rejection threshold (Å) for any non-bonded atom pair.
- **`symprec`**: Symmetry-equivalence cutoff for site deduplication.
- **`max_pair_dist`**: Maximum distance (Å) between two dangling-bond sites to form a dissociative chemisorption pair.

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
