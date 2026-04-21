import numpy as np
from typing import List, Optional, Dict, Union
from scipy import constants as const

# Physical constants (Reference: SciPy Constants / CODATA)
kB = const.k            # Boltzmann constant [J/K]
h = const.h             # Planck constant [J·s]
R = const.R             # Gas constant [J/(mol·K)]
Na = const.Avogadro     # Avogadro constant [1/mol]
c = const.c             # Speed of light [m/s]
eV_to_J_mol = const.e * const.Avogadro  # Conversion factor: eV to J/mol
amu_to_kg = const.atomic_mass   # 1 amu in kg
ang_to_m = const.angstrom       # 1 Angstrom in m

def thz_to_cm1(thz: float) -> float:
    """Convert THz to cm^-1."""
    return thz * 1e12 / (c * 100)

def thz_to_joule(thz: float) -> float:
    """Convert THz to Joules."""
    return h * thz * 1e12

class ThermoCalculator:
    """
    Core engine for vibrational thermochemistry using the Harmonic Oscillator approximation.
    """
    def __init__(self, frequencies_thz: List[float]):
        """
        Args:
            frequencies_thz: List of vibrational frequencies in THz. 
        """
        self.raw_freqs = np.array(frequencies_thz)
        self.freqs = np.array([f for f in frequencies_thz if f > 0.05]) # Real modes (>0.05 THz)
        self.imag_freqs = np.array([f for f in frequencies_thz if f < -0.05]) # Imaginary modes
        
    def assess_stability(self) -> Dict[str, Union[str, int]]:
        """
        Assess structural stability based on the number of imaginary frequencies.
        """
        n_imag = len(self.imag_freqs)
        if n_imag == 0:
            status = "Stable Local Minimum"
        elif n_imag == 1:
            status = "1st-Order Saddle Point (Transition State)"
        else:
            status = f"{n_imag}th-Order Saddle Point (Highly Unstable)"
            
        return {
            "status": status,
            "n_imag": n_imag,
            "imag_modes": self.imag_freqs.tolist()
        }
        
    def calculate_zpe(self) -> float:
        """
        Calculates Zero Point Energy (ZPE) in J/mol.
        
        Physics: E_zpe = sum(0.5 * h * nu)
        """
        zpe = 0.5 * np.sum(h * self.freqs * 1e12) * Na
        return zpe

    def calculate_vib_internal_energy(self, T: float) -> float:
        """
        Calculates thermal vibrational internal energy U_vib(T) in J/mol.
        """
        if T < 1e-6:
            return 0.0
        
        hv = h * self.freqs * 1e12
        exp_factor = np.exp(hv / (kB * T))
        u_vib = np.sum(hv / (exp_factor - 1)) * Na
        return u_vib

    def calculate_vib_entropy(self, T: float) -> float:
        """
        Calculates vibrational entropy S_vib(T) in J/(mol·K).
        """
        if T < 1e-6:
            return 0.0
        
        hv = h * self.freqs * 1e12
        x = hv / (kB * T)
        # Avoid log(0) for high frequencies via small epsilon
        s_vib = R * np.sum(x / (np.exp(x) - 1) - np.log(1 - np.exp(-x) + 1e-16))
        return s_vib

    def calculate_vib_free_energy(self, T: float) -> float:
        """
        Calculates vibrational Helmholtz free energy F_vib(T) in J/mol.
        Includes ZPE.
        """
        zpe = self.calculate_zpe()
        u_vib = self.calculate_vib_internal_energy(T)
        s_vib = self.calculate_vib_entropy(T)
        return zpe + u_vib - T * s_vib

class GasThermo:
    """
    Statistical mechanics utilities for ideal gas translational and rotational contributions.
    """
    @staticmethod
    def calculate_enthalpy_correction(T: float, symmetry: str = 'nonlinear') -> float:
        """
        Calculates the (H_trans + H_rot) contribution to the gas-phase enthalpy.
        
        For non-linear: 2.5RT (trans) + 1.5RT (rot) = 4.0RT
        For linear: 2.5RT (trans) + 1.0RT (rot) = 3.5RT
        """
        coeff = 3.5 if symmetry == 'linear' else 4.0
        return coeff * R * T

    @staticmethod
    def calculate_trans_entropy(mass_amu: float, T: float, P: float = 101325.0) -> float:
        """
        Calculates translational entropy using the Sackur-Tetrode equation.
        
        Args:
            mass_amu: Molecular mass in amu.
            T: Temperature in K.
            P: Pressure in Pa (default 1 atm).
        """
        M = mass_amu * 1.660539e-27 # Convert to kg
        prefactor = (2 * np.pi * M * kB * T / (h**2))**1.5
        s_trans = R * (np.log(prefactor * kB * T / P) + 2.5)
        return s_trans

    @staticmethod
    def calculate_rot_entropy(moments_amu_ang2: List[float], T: float, sigma: int, symmetry: str = 'nonlinear') -> float:
        """
        Calculates rotational entropy.
        
        Args:
            moments_amu_ang2: Moments of inertia [I1, I2, I3] in amu*A^2.
            T: Temperature in K.
            sigma: Symmetry number.
            symmetry: 'linear' or 'nonlinear'.
        """
        conv = 1.660539e-27 * (1e-10**2) # amu*A^2 to kg*m^2
        I = [m * conv for m in moments_amu_ang2]
        
        if symmetry == 'linear':
            s_rot = R * (np.log(8 * np.pi**2 * I[0] * kB * T / (sigma * h**2)) + 1)
        else:
            product_I = np.prod(I)
            term1 = np.sqrt(np.pi) / sigma
            term2 = (8 * np.pi**2 * kB * T / (h**2))**1.5
            s_rot = R * (np.log(term1 * term2 * np.sqrt(product_I)) + 1.5)
            
        return s_rot
