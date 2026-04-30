# AutoFlow-SRXN: Specialized Reaction Discovery Roadmap

## 1. Current Status (Completed Milestones)

### A. Core Discovery Engine
*   **Unified Gradient Logic**: Integrated ML potential (MACE) with Metadynamics hills and induction bias into a single PyTorch graph for exact force derivation.
*   **Specialized Discovery Modes**: 
    *   `ADSORPTION`: Target-specific site approach.
    *   `DIFFUSION`: Surface-constrained mobility.
    *   `DESORPTION`: Multi-CV bias for pairing (Cl-Cl) and vertical lifting.
*   **Biased Metropolis Criteria**: Enabled high-energy transition sampling by judging acceptance based on the biased potential energy.

### B. Intelligent Environment Analysis
*   **EnvironmentAnalyzer**: Automated tagging of atoms (Role 0: Non-active, 1: Surface, 2: Diffusion, 3: Desorption, 4: Adsorption site).
*   **Dynamic Role Evolution**: Real-time role reassignment as atoms move (e.g., Diffusion candidate -> Desorption candidate upon pairing).

### C. Persistence & Knowledge Architecture
*   **KnowledgeHub**: JSON-based saving/loading of Metadynamics Hills, Topology DB (Hashes), and Chemical Reaction Network (CRN) edges.
*   **Resumable Workflow**: Verified that the engine can resume exploration from any crash point while maintaining learned exploration history.

---

## 2. Future Development Roadmap (Next Steps)

### Phase 1: Exploration Intelligence (High Priority)
*   **[ ] Product Predictor**: Implement stoichiometry-based automated product hypothesis (e.g., $N_2, TiCl_n$) and dynamic induction bias generation.
*   **[ ] Curiosity-driven Sampling**: Add `CuriosityScore` to nodes in the Reaction Graph to prioritize exploration of unexplored PES regions.
*   **[ ] Global Knowledge Store**: Separate "Locally shared" (project-specific) and "Globally shared" (general chemistry heuristics) knowledge layers.

### Phase 2: Kinetic Validation & Refinement
*   **[ ] Automated TS Search**: Trigger background `ASE NEB` (Nudged Elastic Band) calculations whenever a new transition edge is discovered.
*   **[ ] Transition State Database**: Store formal barriers and reaction rate constants directly in the `reaction_graph.json`.

### Phase 3: Active Learning Loop
*   **[ ] Uncertainty-driven DFT**: Monitor ML potential uncertainty and automatically trigger single-point DFT calculations to fine-tune the model for transition states.
*   **[ ] Kinetic Monte Carlo (KMC) Integration**: Feed the discovered reaction network into a KMC engine for macroscopic reaction rate prediction.

---

## 3. Physical Standards & Verification
*   **Units**: Energy in `eV`, distances in `Å`, forces in `eV/Å`.
*   **Validation**: Each major transition must be verified via `report.md` confirming physical consistency (e.g., bond lengths, coordination numbers).
