# Troubleshooting: Si(100) Reconstruction & Haptic Chemisorption

This directory contains the files used to diagnose and resolve issues with haptic precursor chemisorption on dimer-reconstructed $\text{Si}(100)$ surfaces.

---

## 1. Issue Description

When searching for chemisorption pathways of the haptic precursor $\text{AllylCpNi}$ (containing $\eta^5\text{-Cp}$ and $\eta^3\text{-Allyl}$ ligands) on the reconstructed $\text{Si}(100)\text{-(2}\times\text{1)}$ buckled-dimer surface, the search returned zero candidates. Two underlying physical/geometric issues were identified:

### A. Dimer Bond Coordination Resolution
The ideal covalent radius of $\text{Si}$ is $1.11\ \text{Å}$. The coordination cutoff is calculated as:
$$d_{\text{cutoff}} = R_{\text{cov},i} + R_{\text{cov},j} + \delta_{\text{slack}}$$
With the default `bond_slack` ($\delta_{\text{slack}}$) of $0.2\ \text{Å}$, the maximum detected bond length is:
$$d_{\text{cutoff}} = 1.11 + 1.11 + 0.20 = 2.42\ \text{Å}$$
However, the dimer reconstruction shifts adjacent $\text{Si}$ atoms to form a dimer bond with a length of $d(\text{Si-Si})_{\text{dimer}} \approx 2.46\ \text{Å}$. Because $2.46\ \text{Å} > 2.42\ \text{Å}$, the dimer bond was not recognized. The system calculated the coordination number of surface $\text{Si}$ as $2$ instead of $3$, generating two unphysical VSEPR dangling bonds per atom pointing in incorrect spatial directions.

### B. Steric Clashes of Haptic Ligands
With the dangling bonds aligned along the correct $sp^3$-like vector (single dangling bond per dimer atom), the bulky $\eta^5\text{-Cp}$ and $\eta^3\text{-Allyl}$ ligands clashed with neighboring surface atoms. Under the default `overlap_scale: 0.65`, all candidate structures failed the minimum non-bonded distance check:
$$d_{ij} \ge \gamma_{\text{scale}} \times (r_{\text{vdw},i} + r_{\text{vdw},j})$$

---

## 2. Solutions

### A. Parameter Optimization
To restore the correct physical coordination and allow haptic chemisorption:
1. **`bond_slack: 0.45`**: Increases the coordination cutoff to $2.67\ \text{Å}$, successfully detecting the dimer bond and leaving a single physical dangling bond pointing upwards-outwards.
2. **`overlap_scale: 0.60`**: Relaxes the steric overlap threshold to accommodate the bulky haptic ligands.

These parameters have been successfully integrated into `config_mod.yaml`.

### B. Code Fix: Dynamic Config Scoping
During investigation, a config routing regression was identified in `autoflow_srxn/surface/chemisorption_builder.py`. The builder hardcoded the `"precursor"` key when looking up chemisorption options:
```python
chem_cfg = config.get("reaction_search", {}).get("mechanisms", {}).get("precursor", {}).get("chemisorption", {})
```
This caused configuration settings (like `byproduct_placement` and `coordination_analysis`) specified under `mechanisms -> inhibitor` to be ignored. The code was refactored to propagate the current `stage_type` parameter dynamically, resolving the issue.
