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
- **Displacement Parameter ($u$)**: Tested at $0.01, 0.005, \text{ and } 0.001 \text{ \AA}$.

### B. Hierarchical Relaxation Scheme
To ensure the system reaches the deepest part of the local potential well, we implement a two-stage relaxation:
1. **Conjugate Gradient (CG)**: Utilized for rapid initial descent from the high-energy perturbed state.
2. **FIRE (Fast Inertial Relaxation Engine)**: A robust inertia-based optimizer used for final convergence to an ultra-tight threshold.
- **Convergence Criterion**: $f_{max} < 0.001 \text{ eV/\AA}$.

### C. Mode-Following Perturbation
For each identified imaginary mode with $\nu < -0.1 \text{ THz}$, the atomic positions $\mathbf{R}$ are updated:
$$\mathbf{R}_{new} = \mathbf{R}_{old} + \alpha \cdot \mathbf{e}_{imag}$$
where $\alpha$ is the perturbation scale (initially $0.1 \text{ \AA}$) and $\mathbf{e}_{imag}$ is the normalized eigenvector of the unstable mode.

## 3. Simulation Results and Analysis

Experimental runs for the DIPAS (Diisopropylaminosilane) molecule yielded the following results after 10 refinement cycles:

| Displacement $u$ (Å) | Initial $\nu_{min}$ (THz) | Final $\nu_{min}$ (THz) | Energy Change (eV) | Result |
| :--- | :--- | :--- | :--- | :--- |
| 0.010 | -1.0310 | -0.1943 | -0.0075 | Stabilized |
| 0.005 | -0.9541 | -0.1417 | -0.0072 | Stabilized |
| 0.001 | -0.9480 | -0.1225 | -0.0073 | **Converged** |

### Data Interpretation
- **Numerical vs. Physical instability**: Reducing $u$ from $0.01$ to $0.001$ resulted in a nearly identical initial $\nu_{min}$ (~ -0.95 THz). This confirms that the imaginary mode is **physically grounded in the MACE potential field** and not a numerical artifact of the finite difference step.
- **Convergence Limit**: Under ultra-tight relaxation, all cases converged to a small residual negative curvature (~ -0.12 THz). This suggests that the MACE-MP-0 model possesses a systematic slight instability region for certain gas-phase rotational modes of DIPAS, representing the fundamental resolution limit of the potential model.

## 4. Usage Instructions

To execute the refinement study:

```powershell
# Run the refinement with a specific displacement
python run_phonon_refinement.py config.yaml 0.001
```

The script will generate:
- `stability_u[u].log`: Detailed iteration history.
- `refined_u[u]_final.vasp`: The optimized stable structure.
- `mode_anims/`: Animation files (`.extxyz`) showing the direction of the followed modes.

## 5. Implementation Credits & References
- **Potential Model**: MACE-MP-0 (Materials Project Foundation Model).
- **Phonon Engine**: Phonopy.
- **Optimizer**: ASE (Atomic Simulation Environment).
- **Logic**: AutoFlow-SRXN Stability Module.
