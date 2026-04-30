import os

import numpy as np
import yaml


class QPointParser:
    """Parses Phonopy qpoints.yaml to extract vibrational frequencies and eigenvectors.
    Focused on identifies unstable (imaginary) modes for structural refinement.
    """

    def __init__(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"  [QPointParser] Could not find: {file_path}")
        self.file_path = file_path
        self.data = self._load_yaml()

    def _load_yaml(self):
        """Loads large YAML files safely."""
        with open(self.file_path) as f:
            # Use CLoader if available for performance
            try:
                from yaml import CLoader as Loader
            except ImportError:
                from yaml import Loader
            return yaml.load(f, Loader=Loader)

    def get_filtered_modes(self, freq_threshold=-0.1, max_modes=None):
        """Extract imaginary modes from the Gamma-point band data.

        Eigenvectors are returned as Cartesian *displacement* vectors (N_atoms, 3),
        i.e. u_{k,α} = e_{k,α} / sqrt(m_k).

        When the file contains the AutoFlow-SRXN ``masses`` extension key, the
        back-conversion is performed automatically.  For plain phonopy files without
        that key the raw eigenvector components (mass-weighted) are returned as-is,
        which is an acceptable approximation for the purpose of mode-following.

        Args:
            freq_threshold (float): Return only modes with frequency < threshold (THz).
            max_modes (int | None): Cap on number of modes returned (most unstable first).

        Returns:
            list[dict]: Each entry has ``'frequency'`` (float, THz) and
                        ``'eigenvector'`` (np.ndarray, shape (N_atoms, 3)).
        """
        if "phonon" not in self.data:
            return []

        # AutoFlow-SRXN extension: atom masses (amu) for e_k → u_k back-conversion.
        # Falls back gracefully if absent (plain phonopy files).
        masses_list = self.data.get("masses", [])

        qpoint = self.data["phonon"][0]
        bands = qpoint.get("band", [])

        modes = []
        for b in bands:
            freq = b.get("frequency")
            if freq is None:
                continue

            if freq < freq_threshold:
                raw_eig = b.get("eigenvector")
                if not raw_eig:
                    continue

                # Detect format: flat list vs nested list
                # Nested: [ [ [ux,ix], [uy,iy], [uz,iz] ], ... ] -> len is num_atoms
                # Flat:   [ [ux,ix], [uy,iy], [uz,iz], ... ]     -> len is 3 * num_atoms
                is_nested = isinstance(raw_eig[0][0], (list, tuple))

                if is_nested:
                    num_atoms = len(raw_eig)
                else:
                    num_atoms = len(raw_eig) // 3

                has_masses = len(masses_list) == num_atoms
                eig_vec = np.zeros((num_atoms, 3))

                for i in range(num_atoms):
                    # Back-convert: u_{k,α} = e_{k,α} / sqrt(m_k)
                    m_sqrt = float(np.sqrt(masses_list[i])) if has_masses else 1.0
                    for j in range(3):
                        if is_nested:
                            e_val = raw_eig[i][j][0]
                        else:
                            e_val = raw_eig[3 * i + j][0]

                        eig_vec[i, j] = e_val / m_sqrt if m_sqrt > 0.0 else e_val

                modes.append({"frequency": freq, "eigenvector": eig_vec})

        # Most unstable (most negative) first
        modes.sort(key=lambda x: x["frequency"])

        if max_modes and max_modes > 0:
            modes = modes[:max_modes]

        return modes
