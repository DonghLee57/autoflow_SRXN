# AutoFlow-SRXN: Instructional Context & Project Guidelines

This document serves as the primary instructional context for Gemini CLI when interacting with the **AutoFlow-SRXN** repository. It defines the project's architecture, operational standards, and scientific foundations.

---

## 1. Project Overview

**AutoFlow-SRXN** is a high-fidelity, fully-automated framework for high-throughput exploration of adsorption and reaction structures at material interfaces. It is primarily designed for **ALD (Atomic Layer Deposition)** and **CVD (Chemical Vapor Deposition)** process modeling.

### Core Technologies
- **Language:** Python 3.8+
- **Physical Engine:** ASE (Atomic Simulation Environment)
- **Machine Learning Interatomic Potentials (MLIPs):** MACE-MP, SevenNet
- **Symmetry & Geometry:** spglib (symmetry), RDKit (molecular conformers), SciPy (Delaunay/optimization)
- **Interface Modeling:** Pymatgen (optional Stage 0a)

### High-Level Architecture
1.  **Stage 0a (Interface):** Heteroepitaxial lattice-match screening using ZSL algorithm.
2.  **Stage 1 (Surface Prep):** Slab generation (asymmetric/symmetric), surface reconstruction (VSEPR-based), and passivation.
3.  **Stage 2 (Reaction Search):** Potential-free candidate generation via symmetry-aware site enumeration and Fibonacci-sphere sampling.
4.  **Stage 3 (Simulation/Verification):** MLIP-driven structural relaxation and NVT-MD equilibration.
5.  **Stage 4 (Analysis):** Vibrational analysis (Partial Hessian - PHVA), iterative mode-following, and thermodynamic property calculation.

---

## 2. Operational Harness

### Building and Running
- **Installation:**
  ```bash
  pip install .        # Standard
  pip install .[mlip]  # With MACE and SevenNet backends
  ```
- **Execution:**
  The workflow is typically driven by a `config.yaml` file.
  ```bash
  # Example: Run adsorption discovery
  python examples/DIPAS_on_Si110/run_adsorption.py
  ```
- **Testing:**
  ```bash
  python -m unittest discover unittests/
  ```

### Key Commands & Workflows
- **Relaxation:** Handled by `SimulationEngine` in `autoflow_srxn/simulation/potentials.py`. Supports BFGS, FIRE, and GPMin.
- **Vibrational Analysis:** Entry point `autoflow_srxn/vibrational/mode_following.py`. Supports PHVA (Partial Hessian) to save computational cost.

---

## 3. Development Conventions

### Coding Style
- **Linter/Formatter:** **Ruff** is used for PEP8 compliance.
- **Docstrings:** **Google Style** docstrings are mandatory for all public functions and classes.
- **Type Hinting:** Mandatory for all new Python functions.

### Scientific Standards
- **Units:** Energy in **eV**, Distance in **Å**, Frequencies in **cm⁻¹** (internal conversion to THz where needed), Temperature in **K**.
- **Physical Integrity:**
  - **ZBL Repulsion:** Must be enabled for aggressive searches or MD to prevent unphysical overlaps.
  - **Overlaps:** Evaluated using **Alvarez (2013) vdW radii**.
  - **Active Set (PHVA):** Typically includes adsorbate + slab atoms within a 6.0 Å radius.

### Directory Structure
- `autoflow_srxn/`: Core package.
  - `surface/`: Slab building and site identification.
  - `simulation/`: MLIP backends and thermodynamics.
  - `vibrational/`: Frequency analysis and imaginary mode refinement.
  - `utils/`: Chemical data and logging.
- `examples/`: Domain-specific workflow scripts.
- `unittests/`: Verification suite.
- `structures/`: Input geometry library (VASP format).

---

## 4. Scientific Domain Expertise (Governing Principles)

### Multi-Vector VSEPR Engine
Used for surface passivation and dangling bond detection. Identifies orientation based on coordination environment:
- $m=1$: Opposite to neighbor vector sum.
- $m=2$: Tetrahedral/Square-planar ($AX_2E_2$) spread.

### Partial Hessian Vibrational Analysis (PHVA)
Approximates the Hessian $H_{ij} \approx 0$ if $i$ or $j \notin \text{Active Set}$, where the Active Set is dynamically defined by proximity to the precursor.

### Iterative Mode-Following
Autonomous refinement of transition states or local minima by perturbing along imaginary mode eigenvectors:
$$ \mathbf{R}_{new} = \mathbf{R}_{old} + \alpha \frac{\mathbf{u}_k}{\|\mathbf{u}_k\|} $$

---

## 5. Metadata

- **Date Created:** 2026-05-12
- **Agent Version:** Gemini CLI v1.0
- **Project Status:** Active Development (Metadynamics module in progress)
