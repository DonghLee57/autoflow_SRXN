# AutoFlow-SRXN: Automated Surface Reaction Workflow

**AutoFlow-SRXN** is a high-fidelity, fully-automated framework designed for high-throughput exploration and generation of adsorption and reaction structures between arbitrary precursors and substrates. It leverages geometric coordination principles, machine learning interatomic potentials (MLIPs), and statistical mechanics to predict thermodynamic stability and reaction kinetics at material interfaces.

---

## 1. Scientific Domain Expertise

### 1.1 Multi-Vector VSEPR Coordination Engine
The framework utilizes a generalized, valence shell electron pair repulsion (VSEPR) based engine to autonomously detect and passivate undercoordinated surface sites across arbitrary materials.

For a surface atom $i$ with $n$ existing covalent neighbors and a target valence $V_{target}$, the engine identifies $m = V_{target} - n$ dangling bonds. The 3D orientation of these vectors is determined algorithmically:
- **Singular Bonds ($m=1$)**: Points exactly opposite to the normalized sum of existing neighbor vectors.
- **Dual Bonds ($m=2$)**: Optimized for tetrahedral/square-planar environments (e.g., Si(100) dimers), spreading vectors according to $AX_2E_2$ VSEPR geometry.
- **Surface Saturation ($m \ge 3$)**: Distributes vectors in a symmetric conical spread around the primary vacuum-pointing axis.

### 1.2 Asymmetric Substrate Factory
The framework automates the generation of complex surface models with precise termination control.
- **Asymmetric Termination**: Supports separate atomic plane constraints for top and bottom surfaces (e.g., Silanol-terminated top vs. Oxygen-terminated bottom).
- **Side-Specific Passivation**: Enables independent passivation coverage for different sides of the slab, critical for modeling realistic asymmetric experimental conditions.
- **Steric-Constraint Expansion**: Autonomously expands the supercell to satisfy a `target_area` constraint, ensuring periodic boundary stability for large adsorbates like DIPAS.

### 1.3 Partial Hessian Vibrational Analysis (PHVA)
To accelerate thermodynamic calculations and kinetic modeling, the framework implements **Partial Hessian Vibrational Analysis**.
For a system with $N_{total}$ atoms, the full Hessian matrix $H \in \mathbb{R}^{3N \times 3N}$ is approximated by a submatrix $H_{active} \in \mathbb{R}^{3N_{active} \times 3N_{active}}$:
$$ H_{ij} \approx 0 \quad \text{if } i \text{ or } j \notin \text{Active Set} $$
The **Active Set** is dynamically defined as the adsorbate plus all substrate atoms within a user-defined cutoff radius $R_{phva}$ (default 3.5 Å), significantly reducing the number of force calls required for frequency extraction.

### 1.4 Thermodynamics & Gibbs Free Energy
The engine integrates vibrational data to calculate finite-temperature thermodynamic properties using the Harmonic approximation.
- **Vibrational Partition Function**: $Z_{vib} = \prod_i \frac{e^{-\beta \hbar \omega_i / 2}}{1 - e^{-\beta \hbar \omega_i}}$
- **Gibbs Free Energy**: $G(T) = E_{pot} + ZPE + \int C_p dT - TS$

### 1.5 Iterative Mode-Following Refinement
To ensure all generated structures represent true local minima, the framework implements an iterative mode-following algorithm. If a relaxed structure contains imaginary vibrational modes (frequency < -0.1 THz), the system is autonomously perturbed along the Cartesian displacement vector $\mathbf{u}_k$ of the most unstable mode:
$$ \mathbf{R}_{new} = \mathbf{R}_{old} + \alpha \frac{\mathbf{u}_k}{\|\mathbf{u}_k\|} $$
where $\mathbf{u}_{k,i} = \mathbf{e}_{k,i} / \sqrt{m_i}$ is derived from the mass-weighted eigenvector $\mathbf{e}_k$. The process repeats until all significant imaginary frequencies are eliminated.

---

## 2. Strategic Objectives
- **High-Throughput Exploration**: Rational search of the potential energy surface (PES) using symmetry-aware site identification.
- **MLIP-Driven Accuracy**: High-fidelity relaxation and frequency calculations using **MACE-MP-0** foundation models.
- **PHVA/FHVA Benchmarking**: Systematic validation of partial Hessian approximations against full Hessian references.
- **Standardized Data Export**: Generation of human-readable `qpoint.yaml` files and `all_relaxed_candidates.extxyz` for visualization.

---

## 3. Architecture Map

### 3.1 Logical Data Flow
```mermaid
graph TD
    A[config.yaml] --> B(Structure Generation Interface)
    subgraph pkg [autoflow_srxn Package]
        B --> P[Surface Utils]
        P -->|Asymmetric Control| PA[Standardized Substrate]
        B --> C[AdsorptionWorkflowManager]
        PA -.-> C
        C --> D[ChemisorptionBuilder]
        D -->|VSEPR Heuristics| SY[Symmetry Engine]
        
        subgraph VIB [Vibrational Engine]
            C --> VA[VibrationalAnalyzer]
            VA -->|Full Hessian| FH[FHVA]
            VA -->|Partial Hessian| PH[PHVA]
            FH & PH --> TC[ThermoCalculator]
            VA -->|Imaginary Modes| MF[MultiModeFollower]
            MF -->|Perturbed Structure| C
        end
    end
    
    subgraph Output
        SY --> G[all_relaxed_candidates.extxyz]
        TC --> Q[qpoints.yaml]
    end
```

### 3.2 Simulation Backend Design

All relaxation and force calculations are routed through `SimulationEngine` (`src/potentials.py`), which selects an ASE-compatible backend based on `engine.potential.backend` in the configuration. The framework follows a **pure ASE (In-process) architecture**, eliminating external binary dependencies (like LAMMPS) for improved portability and stability.

| Backend | Model | Runtime | Interface |
| :--- | :--- | :--- | :--- |
| **MACE** | MACE-MP-0 | In-process Python | `mace.calculators.mace_mp` |
| **SevenNet** | 7net-0 / multifidelity | In-process Python | `sevenn.calculator.SevenNetCalculator` |
| **EMT** | Standard EMT | In-process Python | `ase.calculators.emt.EMT` |

**MACE** is loaded as an ASE calculator and runs entirely within the Python process. It supports both `float32` (for fast MD) and `float64` (for precise vibrations).

**SevenNet** is driven through the `sevenn` ASE interface. It supports optional D3 dispersion corrections via `SevenNetD3Calculator`.

**Configuration:**
```yaml
engine:
  potential:
    backend: "mace"      # "mace" | "sevennet" | "emt"
    device:  "cpu"       # "cpu" | "cuda"
    model:   null        # null -> use default foundation model
    modal:   null        # multi-fidelity modality (SevenNet only)
    d3:      false       # true -> enable D3 (SevenNet only)
    enable_cueq:  false  # enable cuEquivariance (SevenNet only)
    enable_flash: false  # enable FlashAttention (SevenNet only)
```

### 3.3 Directory Structure
- `src/`: Core package logic (surface utils, adsorption managers, vibration analysis).
- `examples/`:
    - `partial_hamiltonian_vibrational_analysis/`: Full PHVA vs FHVA benchmark on SiO2(001) + DIPAS.
    - `mode_following_relaxation/`: Automated stability refinement of the DIPAS precursor.
    - `example_dipas/`: Si(100) surface reaction stage-wise discovery.
- `structures/`: Base crystal and precursor configurations (VASP format).

---

## 4. Operational Harness

### 4.1 Installation
```bash
pip install -e .
pip install ".[mace]" # For MLIP support
```

### 4.2 Running PHVA Benchmark
To run the PHVA vs FHVA free-energy benchmark on a silanol-terminated SiO2 surface:
```bash
cd examples/partial_hamiltonian_vibrational_analysis
python run_phva_benchmark.py
```
This script will:
1. Generate an asymmetric SiO2(001) slab and equilibrate with MD at 500 K.
2. Relax the gas-phase DIPAS molecule in isolation.
3. Search for physisorption sites, relax the top 8 candidates, and select the lowest-energy structure.
4. Run FHVA (full Hessian) on the gas molecule and adsorbed system.
5. Run PHVA (partial Hessian, active set = adsorbate + slab atoms within 3.5 Å) on the adsorbed system.
6. Save individual `qpoint.yaml` files for each system in `vibrations/`.
7. Compute ΔG_rxn(FHVA) and ΔG_rxn(PHVA) via the harmonic-oscillator approximation and write `results/phva_fhva_comparison.yaml`.

---

## 5. Physical Standards

| Property | Standard Unit | Reference |
| :--- | :--- | :--- |
| **Energy** | Electronvolt (eV) | - |
| **Frequency** | Wavenumber (cm⁻¹) | 1 THz $\approx$ 33.356 cm⁻¹ |
| **Distance** | Angstrom (Å) | - |
| **Temperature**| Kelvin (K) | Default: 298.15 K |

**DOIs & References:**
- MACE Potential: [10.48550/arXiv.2206.07697]
- ASE Framework: [10.1088/1361-648X/aa680e]
