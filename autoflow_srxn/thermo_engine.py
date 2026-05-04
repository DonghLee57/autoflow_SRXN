import numpy as np
from scipy import constants as const

try:
    import spglib as _spglib
except ImportError:
    _spglib = None

# Physical constants (Reference: SciPy Constants / CODATA)
kB = const.k  # Boltzmann constant [J/K]
h = const.h  # Planck constant [J·s]
R = const.R  # Gas constant [J/(mol·K)]
Na = const.Avogadro  # Avogadro constant [1/mol]
c = const.c  # Speed of light [m/s]
eV_to_J_mol = const.e * const.Avogadro  # Conversion factor: eV to J/mol
amu_to_kg = const.atomic_mass  # 1 amu in kg
ang_to_m = const.angstrom  # 1 Angstrom in m


def thz_to_cm1(thz: float) -> float:
    """Convert THz to cm^-1."""
    return thz * 1e12 / (c * 100)


def thz_to_joule(thz: float) -> float:
    """Convert THz to Joules."""
    return h * thz * 1e12


class ThermoCalculator:
    """Core engine for vibrational thermochemistry using the Harmonic Oscillator approximation."""

    def __init__(self, frequencies_thz: list[float]):
        """Args:
        frequencies_thz: List of vibrational frequencies in THz.
        """
        self.raw_freqs = np.array(frequencies_thz)
        self.freqs = np.array([f for f in frequencies_thz if f > 0.05])  # Real modes (>0.05 THz)
        self.imag_freqs = np.array([f for f in frequencies_thz if f < -0.05])  # Imaginary modes

    def assess_stability(self) -> dict[str, str | int]:
        """Assess structural stability based on the number of imaginary frequencies."""
        n_imag = len(self.imag_freqs)
        if n_imag == 0:
            status = "Stable Local Minimum"
        elif n_imag == 1:
            status = "1st-Order Saddle Point (Transition State)"
        else:
            status = f"{n_imag}th-Order Saddle Point (Highly Unstable)"

        return {"status": status, "n_imag": n_imag, "imag_modes": self.imag_freqs.tolist()}

    def calculate_zpe(self) -> float:
        """Calculates Zero Point Energy (ZPE) in J/mol.

        Physics: E_zpe = sum(0.5 * h * nu)
        """
        zpe = 0.5 * np.sum(h * self.freqs * 1e12) * Na
        return zpe

    def calculate_vib_internal_energy(self, T: float) -> float:
        """Calculates thermal vibrational internal energy U_vib(T) in J/mol."""
        if T < 1e-6:
            return 0.0

        hv = h * self.freqs * 1e12
        exp_factor = np.exp(hv / (kB * T))
        u_vib = np.sum(hv / (exp_factor - 1)) * Na
        return u_vib

    def calculate_vib_entropy(self, T: float) -> float:
        """Calculates vibrational entropy S_vib(T) in J/(mol·K)."""
        if T < 1e-6:
            return 0.0

        hv = h * self.freqs * 1e12
        x = hv / (kB * T)
        # Avoid log(0) for high frequencies via small epsilon
        s_vib = R * np.sum(x / (np.exp(x) - 1) - np.log(1 - np.exp(-x) + 1e-16))
        return s_vib

    def calculate_vib_free_energy(self, T: float) -> float:
        """Calculates vibrational Helmholtz free energy F_vib(T) in J/mol.
        Includes ZPE.
        """
        zpe = self.calculate_zpe()
        u_vib = self.calculate_vib_internal_energy(T)
        s_vib = self.calculate_vib_entropy(T)
        return zpe + u_vib - T * s_vib


def _is_centrosymmetric(atoms) -> bool:
    """Return True if every atom has an inversion-related partner (used for D∞h detection)."""
    pos = atoms.positions - atoms.get_center_of_mass()
    numbers = atoms.get_atomic_numbers()
    tol = 0.15  # A
    for i, (p, z) in enumerate(zip(pos, numbers)):
        paired = any(j != i and numbers[j] == z and np.linalg.norm(p + pos[j]) < tol for j in range(len(atoms)))
        if not paired:
            return False
    return True


def _compute_sigma_from_atoms(atoms) -> int:
    """Estimate the rotational symmetry number σ via spglib point-group analysis.

    Algorithm
    ---------
    Linear molecules (one principal moment ≈ 0):
        D∞h (H₂, N₂, CO₂ …) -> σ = 2   (inversion-symmetric)
        C∞v (HCl, CO …)      -> σ = 1

    Non-linear molecules:
        Count proper rotation operations (det R = +1) found by spglib.
        This equals the order of the rotational subgroup, which is σ by definition.
        Examples: C₂v -> 2, C₃v -> 3, Td -> 12, Oh -> 24.

    Returns 1 (C₁) on any failure or when spglib is not installed.
    """
    if _spglib is None:
        return 1

    mol = atoms.copy()
    mol.pbc = [False, False, False]
    mol.translate(-mol.get_center_of_mass())

    moments = mol.get_moments_of_inertia()
    # Linear test: smallest moment < 0.1 % of largest
    if min(moments) < 1e-3 * max(moments):
        return 2 if _is_centrosymmetric(mol) else 1

    # Non-linear: embed in a large box so spglib treats it as a crystal point group
    span = float(np.ptp(mol.positions)) if len(mol) > 1 else 1.0
    box = max(span * 3, 20.0)
    mol.translate([box / 2] * 3)
    mol.cell = [box, box, box]

    try:
        sym = _spglib.get_symmetry(
            (mol.cell[:], mol.get_scaled_positions(), mol.get_atomic_numbers()),
            symprec=0.1,
        )
        if sym is None or "rotations" not in sym:
            return 1
        # det(R) = +1 -> proper rotation; det(R) = −1 -> improper (reflection / inversion / Sn)
        n_proper = sum(1 for R in sym["rotations"] if abs(np.linalg.det(R) - 1.0) < 0.1)
        return max(1, n_proper)
    except Exception:
        return 1


class GasThermo:
    """Statistical mechanics utilities for ideal gas translational and rotational contributions."""

    @staticmethod
    def calculate_enthalpy_correction(T: float, symmetry: str = "nonlinear") -> float:
        """Calculates the (H_trans + H_rot) contribution to the gas-phase enthalpy.

        For non-linear: 2.5RT (trans) + 1.5RT (rot) = 4.0RT
        For linear: 2.5RT (trans) + 1.0RT (rot) = 3.5RT
        """
        coeff = 3.5 if symmetry == "linear" else 4.0
        return coeff * R * T

    @staticmethod
    def calculate_trans_entropy(mass_amu: float, T: float, P: float = 101325.0) -> float:
        """Calculates translational entropy using the Sackur-Tetrode equation.

        Args:
            mass_amu: Molecular mass in amu.
            T: Temperature in K.
            P: Pressure in Pa (default 1 atm).
        """
        M = mass_amu * 1.660539e-27  # Convert to kg
        prefactor = (2 * np.pi * M * kB * T / (h**2)) ** 1.5
        s_trans = R * (np.log(prefactor * kB * T / P) + 2.5)
        return s_trans

    @staticmethod
    def calculate_rot_entropy(
        moments_amu_ang2: list[float], T: float, sigma: int, symmetry: str = "nonlinear"
    ) -> float:
        """Calculates rotational entropy.

        Args:
            moments_amu_ang2: Moments of inertia [I1, I2, I3] in amu*A^2.
            T: Temperature in K.
            sigma: Symmetry number.
            symmetry: 'linear' or 'nonlinear'.
        """
        conv = 1.660539e-27 * (1e-10**2)  # amu*A^2 to kg*m^2
        I = [m * conv for m in moments_amu_ang2]

        if symmetry == "linear":
            s_rot = R * (np.log(8 * np.pi**2 * I[0] * kB * T / (sigma * h**2)) + 1)
        else:
            product_I = np.prod(I)
            term1 = np.sqrt(np.pi) / sigma
            term2 = (8 * np.pi**2 * kB * T / (h**2)) ** 1.5
            s_rot = R * (np.log(term1 * term2 * np.sqrt(product_I)) + 1.5)

        return s_rot

    @staticmethod
    def from_atoms(atoms):
        """Derive gas-phase properties from an ASE Atoms object.

        Returns:
        -------
        dict
            mass      : float  — molecular mass (amu)
            moments   : list   — principal moments of inertia (amu·A2)
                                 one element for linear, three for nonlinear
            symmetry  : str    — 'linear' or 'nonlinear'
            sigma     : int    — rotational symmetry number (auto-detected via spglib)
        """
        mass = float(np.sum(atoms.get_masses()))
        moments = [float(m) for m in atoms.get_moments_of_inertia()]

        # Linear test: ASE returns moments sorted ascending; [0] ≈ 0 means linear
        if moments[0] < 1e-4:
            symmetry = "linear"
            moments_filtered = [moments[1]]  # I₂ = I₃ for linear; keep one
        else:
            symmetry = "nonlinear"
            moments_filtered = moments

        sigma = _compute_sigma_from_atoms(atoms)

        return {
            "mass": mass,
            "moments": moments_filtered,
            "symmetry": symmetry,
            "sigma": sigma,
        }


