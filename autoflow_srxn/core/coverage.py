import numpy as np
from ase import Atoms
from autoflow_srxn.thermo_engine import ThermoCalculator, GasThermo

class CoverageManager:
    """
    Handles thermodynamics of surface coverage and chemical potential.
    """
    def __init__(self, engine):
        self.engine = engine
        self.gas_data = {} # name -> {energy, atoms, thermo_info}

    def register_gas_species(self, name: str, atoms: Atoms):
        """Pre-calculates DFT energy and thermo info for gas species."""
        atoms.calc = self.engine.get_calculator()
        energy = atoms.get_potential_energy()
        thermo_info = GasThermo.from_atoms(atoms)
        self.gas_data[name] = {
            "energy": energy,
            "atoms": atoms.copy(),
            "thermo": thermo_info
        }

    def get_chemical_potential(self, name: str, T: float, P_pa: float) -> float:
        """
        Calculates chemical potential mu(T, P) for a gas species.
        Units: eV
        """
        if name not in self.gas_data:
            return 0.0
            
        data = self.gas_data[name]
        e_dft = data["energy"]
        thermo = data["thermo"]
        
        # Simplified: Sackur-Tetrode + Rotational
        s_trans = GasThermo.calculate_trans_entropy(thermo["mass"], T, P_pa)
        s_rot = GasThermo.calculate_rot_entropy(thermo["moments"], T, thermo["sigma"], thermo["symmetry"])
        
        # Entropy in J/(mol*K) -> convert to eV/K
        from scipy.constants import Avogadro, e
        s_total_ev_k = (s_trans + s_rot) / (Avogadro * e)
        
        # H_corr in J/mol -> eV
        h_corr_ev = GasThermo.calculate_enthalpy_correction(T, thermo["symmetry"]) / (Avogadro * e)
        
        # mu = E_dft + H_corr - T*S
        mu = e_dft + h_corr_ev - T * s_total_ev_k
        return mu

    def calculate_surface_stability(self, surface_energy_ev: float, stoich_dict: dict, T: float, P_dict: dict) -> float:
        """
        Calculates Grand Canonical Potential Omega.
        Omega = G_surf - sum(N_i * mu_i)
        """
        omega = surface_energy_ev
        for species, count in stoich_dict.items():
            if species in P_dict:
                mu = self.get_chemical_potential(species, T, P_dict[species])
                omega -= count * mu
        return omega

    def is_adsorbed(self, atoms: Atoms, precursor_indices: list, bond_cutoff: float = 2.5) -> bool:
        """Checks if the precursor is chemisorbed to the substrate.

        Criteria: at least one cross-bond shorter than *bond_cutoff* Å exists
        between precursor and substrate atoms.

        Args:
            atoms: Combined slab+adsorbate structure.
            precursor_indices: Indices that belong to the adsorbate.
            bond_cutoff: Maximum bond length to classify as chemisorbed (Å).
        """
        from ase.neighborlist import neighbor_list
        precursor_set = set(precursor_indices)
        substrate_set = set(range(len(atoms))) - precursor_set

        i_arr, j_arr, d_arr = neighbor_list('ijd', atoms, bond_cutoff)

        for idx_i, idx_j, dist in zip(i_arr, j_arr, d_arr):
            is_cross_bond = (idx_i in precursor_set and idx_j in substrate_set) or \
                            (idx_j in precursor_set and idx_i in substrate_set)
            if is_cross_bond:
                return True
        return False

    def is_physical(self, atoms: Atoms, prev_energy: float = None) -> bool:
        """
        Validates if the structure and energy are physically sound.
        """
        # 1. Atomic Distance Check
        from ase.neighborlist import neighbor_list
        i, j, d = neighbor_list('ijd', atoms, 1.2) # check very close neighbors
        if len(d) > 0 and np.min(d) < 0.6: # Less than 0.6 A is usually unphysical overlap
            print(f"  [Safety] Unphysical overlap detected: min_dist = {np.min(d):.3f} A")
            return False
            
        # 2. Energy Jump Check
        energy = atoms.get_potential_energy()
        if prev_energy is not None:
            de = abs(energy - prev_energy)
            # If 1 TiCl4 is added, energy should change by ~100-200 eV max
            # If change is > 500 eV, it's likely a potential blow-up
            if de > 500.0:
                print(f"  [Safety] Unphysical energy jump detected: dE = {de:.2f} eV")
                return False
                
        # 3. NAN check
        if np.isnan(energy):
            return False
            
        return True

    def predict_saturation(self, base_energy: float, current_stoich: dict,
                           dose_species: str, T: float, P_dict: dict):
        """Decides if adding another dose molecule is thermodynamically favorable.

        NOT YET IMPLEMENTED — returns None.  Future implementation should compare
        the grand-canonical potential of the current surface coverage against the
        potential after adding one more adsorbate unit.
        """
        raise NotImplementedError("predict_saturation is not yet implemented.")
