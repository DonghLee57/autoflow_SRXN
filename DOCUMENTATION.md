# AutoFlow-SRXN Configuration Manual

This document provides a comprehensive guide to all parameters available in the `AutoFlow-SRXN` workflow. 

---

## 1. Global Workflow Control
Settings that control the overall execution behavior of the screening engine.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `restart` | Boolean | `false` | If `true`, forces re-calculation of all pairs. If `false`, skips pairs where `results_final_verified.extxyz` already exists. |

---

## 2. Path Configuration
Defines where to find input structures and where to save results.

| Parameter | Description |
| :--- | :--- |
| `adsorbate` | Path to a single precursor structure file (.vasp, .xyz). |
| `adsorbates_dir` | Directory containing multiple precursor files for batch screening. |
| `inhibitor` | Path to a single inhibitor structure file. |
| `inhibitors_dir` | Directory containing multiple inhibitor files for batch screening. |
| `substrate_bulk` | Path to the bulk crystalline structure (used if `slab_generation` is enabled). |
| `substrate_slab` | Path to a pre-generated slab file. |
| `output_prefix` | Base directory name for output files (default: `discovery`). |
| `include_no_inhibitor` | If `true`, includes a baseline run without any inhibitor for each precursor. |

---

## 3. Surface Preparation (`surface_prep`)
Handles the creation and modification of the substrate surface.

### 3.1 Slab Generation
- **`enabled`**: Boolean. Enable/disable ASE-based slab cutting from bulk.
- **`miller`**: List of 3 integers (e.g., `[1, 0, 0]`). Miller indices of the surface plane.
- **`thickness_ang`**: Float (Å). Minimum thickness of the slab.
- **`vacuum_ang`**: Float (Å). Vacuum padding on both sides.
- **`target_area_ang2`**: Float (Å²). Target surface area; the engine will find the best supercell expansion to match this.
- **`supercell_matrix`**: List of lists (e.g., `[[2,0],[0,2]]`). Explicit supercell expansion matrix. Overrides `target_area_ang2` if set.
- **`top_termination`**: String (Element symbol, e.g., `"O"`). Ensures the top surface ends with the specified element.
- **`bottom_termination`**: String (Element symbol, e.g., `"O"`). Ensures the bottom surface ends with the specified element.

### 3.2 Reconstruction
- **`enabled`**: Boolean. Apply surface reconstruction patterns.
- **`strategy`**: `"auto"`, `"dimerization"`, `"rumpling"`, or `"tilt"`.
- **`side`**: `"top"`, `"bottom"`, or `"both"`.

### 3.3 Passivation
- **`enabled`**: Boolean. Saturate dangling bonds (typically on the bottom side).
- **`coverage`**: Float (0.0 to 1.0). Fractional coverage of the chosen element.
- **`element`**: String (e.g., `"H"`). Passivating element.
- **`side`**: `"bottom"` (recommended) or `"top"`.

### 3.4 Surface Analysis
Settings used by the engine to analyze the local environment of surface atoms.
- **`symprec`**: Float (Å). Precision for symmetry and site equivalence detection.
- **`ideal_coordination`**: Dictionary mapping elements to their expected bulk coordination numbers (e.g., `Si: 4`, `O: 2`). This is used to detect "dangling bonds" for passivation and active site mapping.

### 3.5 Slab Relaxation
Initial geometry optimization of the substrate.
- **`enabled`**: Boolean. Perform optimization.
- **`fmax`**: Float (eV/Å). Force convergence threshold.
- **`steps`**: Integer. Max optimization steps.
- **`frozen_z_ang`**: Float (Å). Fix atoms below this Z-height to simulate the bulk interior.

### 3.6 Slab Equilibration
Thermal pre-equilibration of the substrate using Molecular Dynamics.
- **`enabled`**: Boolean. Perform MD on the clean slab.
- **`frozen_z_ang`**: Float (Å). Fix bottom layers during MD.

---

## 4. Reaction Search (`reaction_search`)
Defines how the engine explores the configuration space of adsorbates on the surface.

### 4.1 Mechanisms
#### Physisorption
- **`placement_height`**: Float (Å). Initial distance between the molecule's COM and the surface.
- **`rot_steps`**: Integer. Number of rotational orientations to sample for each site.

#### Chemisorption
- **`precursor_center`**: String (Element symbol). The atom in the precursor that intends to bond with the surface.
- **`inhibitor_center`**: String (Element symbol). The atom in the inhibitor that intends to bond with the surface.

#### Inhibition (Stage 1 Branching)
- **`branching_limit`**: Integer. The number of top-ranked inhibited surfaces to carry over to Stage 2 (precursor adsorption).

---

## 5. Verification Pipeline (`verification`)
Standardized multi-step validation for discovery candidates.

### 5.1 Stages
1. **Relaxation**: Local geometry optimization using BFGS/LBFGS.
2. **Equilibration (NVT MD)**:
   - **`temperature_K`**: Float (K). Simulation temperature.
   - **`md_steps`**: Integer. Number of Langevin MD steps.
   - **`timestep_fs`**: Float (fs). MD integration time step.
3. **Post-Relax**: (Optional) Final minimization of the MD-sampled structure to ensure energy consistency.

### 5.2 Adsorption Energy ($E_{ads}$)
Calculated using: $E_{ads} = E_{total} - (E_{gas} + E_{base})$
- **$E_{gas}$**: Pre-calculated optimized energy of the molecule in vacuum.
- **$E_{base}$**: Potential energy of the surface substrate before this specific adsorption event.

---

## 6. Simulation Engine (`engine`)
Configures the interatomic potential backend and hardware settings.

### 6.1 Common Parameters
| Parameter | Default | Description |
| :--- | :--- | :--- |
| `backend` | `"mace"` | Selection of the potential: `"mace"`, `"sevennet"`, or `"emt"`. |
| `device` | `"cpu"` | Compute device: `"cpu"` or `"cuda"`. |
| `dtype` | `"float64"` | Numerical precision: `"float32"` or `"float64"` (Recommended). |

### 6.2 MACE Specifics
- **`model`**: `"small"`, `"medium"`, or `"large"`. Refers to the MACE-MP foundation model size.
- **`d3`**: Boolean. Enable/disable Grimme's D3 dispersion correction.

### 6.3 SevenNet Specifics
- **`model`**: Path to a `.pth` checkpoint or a predefined name (e.g., `"7net-omni"`, `"7net-mf-ompa"`, `"7net-0"`).
- **`modal`**: String. Used to select a specific **task** or **fidelity** level in multi-task architectures (like SevenNet-Omni). This ensures results are consistent with the specific DFT settings used for training. Common tasks include:
    - `"mpa"`: PBE+U (Recommended for general bulk and surface studies).
    - `"matpes_r2scan"`: r²SCAN level of theory.
    - `"omol25_low"`: ωB97M-V level of theory for small molecules.
    - `"omat24"`: High-fidelity PBE for high-force configurations.
- **`enable_flash`**: Boolean. Enables FlashAttention-2 (GPU only) to reduce memory usage.
- **`enable_cueq`**: Boolean. Enables cuEquivariance acceleration (GPU only).

### 6.4 EMT (Effective Medium Theory)
- No additional parameters. Use exclusively for rapid workflow testing on simple metals and light elements.

### 6.5 ZBL Short-Range Repulsion
Repulsive potential at very short distances to prevent non-physical atom overlaps during MD or aggressive relaxation.
- **`enabled`**: Boolean. Enable ZBL mixing.
- **`cutoff_inner`**: Float (Å). Distance below which ZBL is fully active.
- **`cutoff_outer`**: Float (Å). Global fallback distance above which ZBL is inactive.

---

## 7. Output Files
- **`workflow.log`**: Detailed execution trace.
- **`energylog.csv`**: Tabulated energy data for all verified candidates.
- **`results_all_poses.extxyz`**: All generated initial candidates.
- **`results_final_verified.extxyz`**: Verified stable structures with energy metadata.
