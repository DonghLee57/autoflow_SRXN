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

    def is_adsorbed(self, atoms: Atoms, precursor_indices: list) -> bool:
        """
        Checks if the precursor is truly CHEMISORBED to the substrate.
        Criteria: At least one bond shorter than 2.5 A between precursor and substrate.
        """
        from ase.neighborlist import neighbor_list
        substrate_indices = [i for i in range(len(atoms)) if i not in precursor_indices]
        
        # Use a tighter cutoff for true chemisorption (e.g., 2.5 A)
        i, j, d = neighbor_list('ijd', atoms, 2.5)
        
        for idx_i, idx_j, dist in zip(i, j, d):
            # Check if one index is in adsorbate and other in substrate
            is_cross_bond = (idx_i in precursor_indices and idx_j in substrate_indices) or \
                            (idx_j in precursor_indices and idx_i in substrate_indices)
            
            if is_cross_bond:
                # Further check: is it the Ti atom bonding? (Usually atom 0 in TiCl4)
                # This ensures the molecule is not just 'touching' via Cl.
                atom_i = atoms[idx_i]
                atom_j = atoms[idx_j]
                if atom_i.symbol == 'Ti' or atom_j.symbol == 'Ti':
                    print(f"  [Check] Strong Chemisorption bond found: {atom_i.symbol}-{atom_j.symbol} @ {dist:.3f} A")
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
        """
        Decides if adding another dose molecule is favorable.
        """
        mu_dose = self.get_chemical_potential(dose_species, T, P_dict.get(dose_species, 101325.0))
        pass
