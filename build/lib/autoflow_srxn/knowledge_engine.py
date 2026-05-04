import json
import os


class KnowledgeBase:
    # Alvarez (2013) van der Waals radii in Angstroms.
    # Source: Dalton Trans. 42, 8617-8636 (2013). DOI: 10.1039/c3dt50599e
    ALVAREZ_VDW_RADII = {
        "H": 1.20, "He": 1.43, "Li": 2.12, "Be": 1.98, "B": 1.91, "C": 1.77,
        "N": 1.66, "O": 1.50, "F": 1.46, "Ne": 1.58, "Na": 2.50, "Mg": 2.51,
        "Al": 2.25, "Si": 2.19, "P": 1.90, "S": 1.89, "Cl": 1.82, "Ar": 1.83,
        "K": 2.73, "Ca": 2.62, "Sc": 2.58, "Ti": 2.46, "V": 2.42, "Cr": 2.45,
        "Mn": 2.45, "Fe": 2.44, "Co": 2.40, "Ni": 2.40, "Cu": 2.38, "Zn": 2.39,
        "Ga": 2.32, "Ge": 2.29, "As": 1.88, "Se": 1.82, "Br": 1.86, "Kr": 2.25,
        "Rb": 3.21, "Sr": 2.84, "Y": 2.75, "Zr": 2.52, "Nb": 2.56, "Mo": 2.45,
        "Ru": 2.46, "Rh": 2.44, "Pd": 2.15, "Ag": 2.53, "Cd": 2.49,
        "In": 2.43, "Sn": 2.42, "Sb": 2.47, "Te": 1.99, "I": 2.04, "Xe": 2.06,
        "Cs": 3.48, "Ba": 3.03, "Hf": 2.63, "Ta": 2.53, "W": 2.57, "Re": 2.49,
        "Os": 2.48, "Ir": 2.41, "Pt": 2.29, "Au": 2.32, "Hg": 2.45,
        "Tl": 2.47, "Pb": 2.60, "Bi": 2.54,
    }

    def __init__(self):
        self.chem_data = {}
        self.zbl_pairs = {}
        
        base_path = os.path.dirname(__file__)
        chem_path = os.path.join(base_path, "chem_data.json")
        zbl_path = os.path.join(base_path, "zbl_pairs.json")

        if os.path.exists(chem_path):
            with open(chem_path, encoding='utf-8-sig') as f:
                self.chem_data = json.load(f)
        
        if os.path.exists(zbl_path):
            with open(zbl_path, encoding='utf-8-sig') as f:
                self.zbl_pairs = json.load(f)

    def get_ideal_coordination(self, symbol, config=None):
        """Returns the standard valency/coordination for an element."""
        if config and isinstance(config, dict) and symbol in config:
            return config[symbol]
        return self.chem_data.get(symbol, {}).get("ideal_coordination", 0)

    def get_radius(self, symbol, rtype="covalent"):
        """Returns covalent or vdW radius.
        rtype="covalent": from chem_data.json
        rtype="vdw": from ALVAREZ_VDW_RADII (falls back to chem_data.json or 1.5)
        """
        if rtype == "vdw":
            return self.ALVAREZ_VDW_RADII.get(symbol, self.chem_data.get(symbol, {}).get("vdw_radius", 1.5))
        
        return self.chem_data.get(symbol, {}).get("covalent_radius", 1.5)

    def get_zbl_cutoff(self, sym1, sym2, fallback=2.5):
        """Returns the ZBL outer switching cutoff for a pair of elements.
        If pair not found, calculates it using covalent radii.
        If any radius is missing, returns fallback (default 2.5).
        """
        pair_key = "-".join(sorted([sym1, sym2]))
        if pair_key in self.zbl_pairs:
            return self.zbl_pairs[pair_key]
        
        # Calculate r_out = max(0.8, min(2.8, round((R_cov_i + R_cov_j) * 1.056, 1)))
        r1 = self.chem_data.get(sym1, {}).get("covalent_radius")
        r2 = self.chem_data.get(sym2, {}).get("covalent_radius")
        
        if r1 is None or r2 is None:
            return fallback
            
        r_out = round((r1 + r2) * 1.056, 1)
        return max(0.8, min(2.8, r_out))


chem_kb = KnowledgeBase()
