# Mode-Following Structural Relaxation: Stability Refinement of DIPAS Molecule

This example demonstrates the automated refinement of molecular structures by following imaginary vibrational modes. The primary objective is to transcend saddle points on the potential energy surface (PES) to reach a true local minimum, ensuring thermodynamic stability for subsequent chemical reaction modeling.

## 1. Scientific Background and Objectives

In computational chemistry, a stationary point on the PES is defined by the condition where the gradient of the potential energy $V$ with respect to atomic coordinates $\mathbf{R}$ vanishes:
$$\nabla_{\mathbf{R}} V(\mathbf{R}) = 0$$

To distinguish between a local minimum and a saddle point, the Hessian matrix $\mathbf{H}$ (the second derivative of energy) is evaluated:
$$H_{ij} = \frac{\partial^2 V}{\partial R_i \partial R_j}$$

The eigenvalues $\lambda$ of the mass-weighted Hessian correspond to the square of the vibrational frequencies $\omega$:
$$\mathbf{H}_m \mathbf{q} = \omega^2 \mathbf{q}$$

An **imaginary frequency** (where $\omega^2 < 0$) indicates that the structure resides at a maximum along that specific normal mode coordinate, signifying a saddle point. The goal of this example is to:
1. Identify significant imaginary modes using Phonopy and ML-IAPs (MACE).
2. Perturb the structure along the identified eigenvectors to break symmetry and escape the saddle point.
3. Perform ultra-tight structural relaxation using a hierarchical optimization scheme.

## 2. Methodology: Dual-Stage Stability Refinement

The refinement workflow follows a "Perturb-and-Relax" cycle:

### A. Sensitivity-Driven Phonon Analysis
We utilize the finite displacement method to construct the Hessian. To differentiate between actual PES curvature and numerical artifacts (noise), we performed a sensitivity study across varying displacement scales ($u$):
- **Displacement Parameter ($u$)**: Tested at $0.01, 0.005, \text{ and } 0.001$ A.

### B. Hierarchical Relaxation Scheme
To ensure the system reaches the deepest part of the local potential well, we implement a two-stage relaxation:
1. **Conjugate Gradient (CG)**: Utilized for rapid initial descent from the high-energy perturbed state.
2. **FIRE (Fast Inertial Relaxation Engine)**: A robust inertia-based optimizer used for final convergence to an ultra-tight threshold.
- **Convergence Criterion**: $f_{max} < 0.001 \text{ eV/A}$.

### C. Mode-Following Perturbation
Stability refinement is achieved by updating the atomic coordinates $\mathbf{R}$ through a controlled displacement along the unstable normal modes:

$$\mathbf{R}_{\text{new}} = \mathbf{R}_{\text{old}} + \alpha \cdot \mathbf{e}_{\text{imag}}$$

The criteria and parameters for this perturbation are defined as follows:
- **Identification Threshold**: Modes are selected for refinement if the frequency $\nu$ is less than $-0.1$ THz.
- **Perturbation Scale ($\alpha$)**: An initial displacement factor of $0.1$ A is applied to move the system away from the saddle point.
- **Directional Vector ($\mathbf{e}_{\text{imag}}$)**: The normalized eigenvector associated with the specific imaginary frequency.

## 3. Simulation Results and Analysis

Recent experimental runs for the DIPAS (Diisopropylaminosilane) molecule using the refined workflow (linear combination of modes) yielded the following convergence behavior:

| Cycle | Energy (eV) | Min Freq (THz) | Alpha (Ang) | Status |
| :--- | :--- | :--- | :--- | :--- |
| 0 (Initial) | -130.453484 | -1.0071 | 0.000 | Unstable |
| 1 | -130.461327 | -0.1372 | 0.500 | Refined |
| 2 | -130.463844 | -0.1258 | 0.500 | Refined |
| 3 | -130.464692 | -0.1050 | 0.250 | Near Target |
| **4 (Final)** | **-130.464943** | **-0.0921** | **0.125** | **Converged** |

- **Effect of Mode Following**: The transition from Cycle 0 (-1.0071 THz) to Cycle 1 (-0.1372 THz) demonstrates the power of the "Perturb-and-Relax" strategy. A single coordinated displacement along unstable modes moved the system out of a significant saddle point region.
- **Initial Relaxation Sensitivity**: The initial frequency is highly dependent on the number of relaxation steps. Increasing `steps` from 200 to 1000 in `config.yaml` can improve the starting frequency to approximately -0.1074 THz even before mode following, highlighting the importance of thorough local optimization.

## 4. Physical Interpretation of Zero-Frequency Modes

During analysis, you will notice exactly **6 modes** with frequencies very close to zero (typically within $\pm 0.05$ THz). This is a physically expected result for an isolated molecular system.

### A. Degrees of Freedom
For a non-linear molecule with $N$ atoms, the $3N$ total degrees of freedom are partitioned as:
1.  **3 Translational Modes**: Global movement of the molecule along the $x, y, z$ axes.
2.  **3 Rotational Modes**: Global rotation of the molecule around the $x, y, z$ axes.
3.  **$3N - 6$ Vibrational Modes**: Internal relative motions of the atoms.

### B. Symmetry and Invariance
The potential energy surface $V(\mathbf{R})$ of an isolated molecule is invariant under global translation and rotation. Consequently, the Hessian matrix possesses 6 eigenvectors with zero eigenvalues, corresponding to these infinitesimal symmetry operations. In solids (periodic systems), rotational symmetry is broken by the lattice, leaving only 3 translational (acoustic) zero modes.

## 5. Usage Instructions

To execute the refinement study:

```powershell
# Run the refinement with a specific displacement
python run_phonon_refinement.py config.yaml 0.001
```

The script will generate:
- `stability_u[u].log`: Detailed iteration history.
- `refined_u[u]_final.vasp`: The optimized stable structure.
- `mode_anims/`: Animation files (`.extxyz`) showing the direction of the followed modes.

## 6. Implementation Credits & References
- **Potential Model**: MACE-MP-0 (Materials Project Foundation Model).
- **Phonon Engine**: Phonopy.
- **Optimizer**: ASE (Atomic Simulation Environment).
- **Logic**: AutoFlow-SRXN Stability Module.

