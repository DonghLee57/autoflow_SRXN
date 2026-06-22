"""
autoflow_srxn.metadynamics.collective_variables
===============================================
Collective variables (CVs) for metadynamics, each with an analytic gradient
so the bias force can be added to the physical force without finite
differences.

Every CV exposes the same interface::

    s, grad = cv.value_and_grad(atoms)   # s: float, grad: (natoms, 3)

and carries plotting/grid metadata (``label``, ``unit``, ``default_sigma``,
``grid_min`` / ``grid_max``).

Atom selection helpers accept several spec forms so CVs can be defined in
config without hard-coding indices:

* ``5``                          -> atom index 5
* ``{"index": 5}``               -> atom index 5
* ``"Si"``                       -> all Si atoms
* ``"O@substrate"`` / ``"O@adsorbate"`` -> O atoms filtered by tag region
  (substrate = tag < 2, adsorbate = tag >= 2, matching the rest of the package)
* ``[0, 3, 7]``                  -> explicit group
"""

from __future__ import annotations

import numpy as np
from ase import Atoms


# ---------------------------------------------------------------------------
# Atom-selection helpers
# ---------------------------------------------------------------------------

def _region_mask(atoms: Atoms, region: str) -> np.ndarray:
    tags = atoms.get_tags()
    if region == "substrate":
        return tags < 2
    if region == "adsorbate":
        return tags >= 2
    raise ValueError(f"Unknown region '{region}' (use 'substrate' or 'adsorbate').")


def resolve_group(spec, atoms: Atoms) -> list[int]:
    """Resolve a selection spec to a list of atom indices."""
    if isinstance(spec, (int, np.integer)):
        return [int(spec)]
    if isinstance(spec, dict) and "index" in spec:
        return [int(spec["index"])]
    if isinstance(spec, (list, tuple)):
        return [int(i) for i in spec]
    if isinstance(spec, str):
        symbols = np.array(atoms.get_chemical_symbols())
        if "@" in spec:
            elem, region = spec.split("@", 1)
            mask = (symbols == elem) & _region_mask(atoms, region)
        else:
            mask = symbols == spec
        idx = np.where(mask)[0].tolist()
        if not idx:
            raise ValueError(f"Selection '{spec}' matched no atoms.")
        return idx
    raise TypeError(f"Unsupported atom spec: {spec!r}")


def resolve_atom(spec, atoms: Atoms, near: int | None = None) -> int:
    """Resolve a spec to a single atom index.

    If the spec matches several atoms, the one closest to atom ``near``
    (when given) is chosen deterministically; otherwise the first match.
    """
    idx = resolve_group(spec, atoms)
    if len(idx) == 1:
        return idx[0]
    if near is not None:
        d = atoms.get_distances(near, idx, mic=True)
        return idx[int(np.argmin(d))]
    return idx[0]


# ---------------------------------------------------------------------------
# CV base class
# ---------------------------------------------------------------------------

class CollectiveVariable:
    label = "cv"
    unit = ""
    default_sigma = 0.1
    grid_min: float | None = None
    grid_max: float | None = None

    def value_and_grad(self, atoms: Atoms):
        raise NotImplementedError

    def value(self, atoms: Atoms) -> float:
        return self.value_and_grad(atoms)[0]


# ---------------------------------------------------------------------------
# Concrete CVs
# ---------------------------------------------------------------------------

class DistanceCV(CollectiveVariable):
    """Raw interatomic distance between atoms ``i`` and ``j`` (Å).

    Use for a single, well-defined bond — e.g. the forming bond between the
    precursor central atom and a specific substrate atom.
    """

    unit = "Å"

    def __init__(self, i: int, j: int, mic: bool = True,
                 sigma: float = 0.1, grid_min=None, grid_max=None,
                 label: str | None = None):
        self.i, self.j, self.mic = int(i), int(j), mic
        self.default_sigma = sigma
        self.grid_min, self.grid_max = grid_min, grid_max
        self.label = label or f"d({i}-{j})"

    def value_and_grad(self, atoms: Atoms):
        d_vec = atoms.get_distance(self.i, self.j, mic=self.mic, vector=True)
        s = float(np.linalg.norm(d_vec))
        grad = np.zeros((len(atoms), 3))
        if s > 1e-9:
            # d_vec = R_j - R_i (ASE convention), so u points i -> j
            u = d_vec / s
            grad[self.i] = -u           # ds/dR_i
            grad[self.j] = u            # ds/dR_j
        return s, grad


class CoordinationCV(CollectiveVariable):
    """Rational-switching coordination number of atom ``i`` to a ``group``.

    s = Σ_j (1 - x^n) / (1 - x^m),  x = r_ij / r0

    Permutation-invariant and naturally bounded (→ 0 when the group leaves),
    so it is the robust choice when several equivalent atoms can react
    (e.g. the 4 Cl of TiCl4, or all surface O around the central atom).
    """

    unit = ""

    def __init__(self, i: int, group: list[int], r0: float,
                 n: int = 6, m: int = 12, mic: bool = True,
                 sigma: float = 0.1, grid_min=None, grid_max=None,
                 label: str | None = None):
        self.i = int(i)
        self.group = [g for g in group if g != self.i]
        self.r0, self.n, self.m, self.mic = float(r0), int(n), int(m), mic
        self.default_sigma = sigma
        self.grid_min, self.grid_max = grid_min, grid_max
        self.label = label or f"CN({i})"

    def _switch(self, r: float):
        """Return f(r) and df/dr for the rational switching function."""
        x = r / self.r0
        if abs(x - 1.0) < 1e-6:                       # L'Hopital at r == r0
            f = self.n / self.m
            dfdx = (self.n - self.m) / (2.0 * self.m)
        else:
            num, den = 1.0 - x**self.n, 1.0 - x**self.m
            f = num / den
            dfdx = ((-self.n * x**(self.n - 1)) * den
                    - num * (-self.m * x**(self.m - 1))) / den**2
        return f, dfdx / self.r0                      # df/dr

    def value_and_grad(self, atoms: Atoms):
        s = 0.0
        grad = np.zeros((len(atoms), 3))
        for j in self.group:
            d_vec = atoms.get_distance(self.i, j, mic=self.mic, vector=True)
            r = float(np.linalg.norm(d_vec))
            if r < 1e-9:
                continue
            f, dfdr = self._switch(r)
            s += f
            # d_vec = R_j - R_i (ASE convention), so u points i -> j
            u = d_vec / r
            grad[self.i] -= dfdr * u
            grad[j] += dfdr * u
        return s, grad


class ProtonTransferCV(CollectiveVariable):
    """Antisymmetric stretch for proton transfer:

        ξ = d(donor-H) - d(acceptor-H)     (Å)

    ξ < 0 : proton sits on the donor (e.g. surface O-H / N-H)
    ξ > 0 : proton has moved to the acceptor (the leaving ligand)

    Biasing ξ toward positive values therefore *induces* the ligand-H bond
    that forms the stable byproduct (HCl, amine-H, ...).
    """

    unit = "Å"

    def __init__(self, donor: int, acceptor: int, proton: int,
                 mic: bool = True, sigma: float = 0.1,
                 grid_min=None, grid_max=None, label: str | None = None):
        self.donor, self.acceptor, self.proton = int(donor), int(acceptor), int(proton)
        self.mic = mic
        self.default_sigma = sigma
        self.grid_min, self.grid_max = grid_min, grid_max
        self.label = label or f"PT(H{proton})"

    def value_and_grad(self, atoms: Atoms):
        h = self.proton
        v_d = atoms.get_distance(self.donor, h, mic=self.mic, vector=True)      # donor -> H
        v_a = atoms.get_distance(self.acceptor, h, mic=self.mic, vector=True)   # acceptor -> H
        r_d, r_a = np.linalg.norm(v_d), np.linalg.norm(v_a)
        s = float(r_d - r_a)
        grad = np.zeros((len(atoms), 3))
        if r_d > 1e-9:
            u_d = v_d / r_d
            # d r_d / dR : r_d = |R_H - R_donor|
            grad[h] += u_d
            grad[self.donor] -= u_d
        if r_a > 1e-9:
            u_a = v_a / r_a
            grad[h] -= u_a
            grad[self.acceptor] += u_a
        return s, grad


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_cv(spec: dict, atoms: Atoms) -> CollectiveVariable:
    """Construct a CV from a config dict.

    Common keys: ``type`` (distance|coordination|proton_transfer),
    ``sigma``, ``grid_min``, ``grid_max``, ``label``.
    """
    cv_type = str(spec.get("type", "distance")).lower()
    common = dict(
        sigma=float(spec.get("sigma", 0.1)),
        grid_min=spec.get("grid_min"),
        grid_max=spec.get("grid_max"),
        label=spec.get("label"),
    )
    mic = bool(spec.get("mic", any(atoms.pbc)))

    if cv_type == "distance":
        i = resolve_atom(spec["center"], atoms)
        j = resolve_atom(spec["partner"], atoms, near=i)
        return DistanceCV(i, j, mic=mic, **common)

    if cv_type == "coordination":
        i = resolve_atom(spec["center"], atoms)
        group = resolve_group(spec["group"], atoms)
        return CoordinationCV(i, group, r0=float(spec["r0"]),
                              n=int(spec.get("n", 6)), m=int(spec.get("m", 12)),
                              mic=mic, **common)

    if cv_type == "proton_transfer":
        donor = resolve_atom(spec["donor"], atoms)
        acceptor = resolve_atom(spec["acceptor"], atoms)
        # proton: explicit, or the H closest to the donor at setup time
        if "proton" in spec:
            proton = resolve_atom(spec["proton"], atoms, near=donor)
        else:
            proton = resolve_atom("H", atoms, near=donor)
        return ProtonTransferCV(donor, acceptor, proton, mic=mic, **common)

    raise ValueError(f"Unknown CV type '{cv_type}'.")
