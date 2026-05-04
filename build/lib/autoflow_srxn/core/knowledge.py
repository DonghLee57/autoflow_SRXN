import os
import json
import numpy as np

class GlobalKnowledge:
    """
    Universal chemical heuristics shared across all projects.
    """
    # Pauling Electronegativity
    ELECTRONEGATIVITY = {
        'H': 2.20, 'Li': 0.98, 'Be': 1.57, 'B': 2.04, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98,
        'Na': 0.93, 'Mg': 1.31, 'Al': 1.61, 'Si': 1.90, 'P': 2.19, 'S': 2.58, 'Cl': 3.16,
        'K': 0.82, 'Ca': 1.00, 'Sc': 1.36, 'Ti': 1.54, 'V': 1.63, 'Cr': 1.66, 'Mn': 1.55, 'Fe': 1.83,
        'Co': 1.88, 'Ni': 1.91, 'Cu': 1.90, 'Zn': 1.65, 'Ga': 1.81, 'Ge': 2.01, 'As': 2.18, 'Se': 2.55, 'Br': 2.96,
        'Pt': 2.28, 'Au': 2.54
    }
    
    # Covalent Radii (A) - Simplified
    COVALVALENT_RADII = {
        'H': 0.31, 'He': 0.28, 'Li': 1.28, 'Be': 0.96, 'B': 0.84, 'C': 0.73, 'N': 0.71, 'O': 0.66, 'F': 0.57,
        'Na': 1.66, 'Mg': 1.41, 'Al': 1.21, 'Si': 1.11, 'P': 1.07, 'S': 1.05, 'Cl': 1.02,
        'K': 2.03, 'Ca': 1.76, 'Sc': 1.70, 'Ti': 1.60, 'V': 1.53, 'Cr': 1.39, 'Mn': 1.39, 'Fe': 1.32,
        'Co': 1.26, 'Ni': 1.24, 'Cu': 1.32, 'Zn': 1.22
    }

    @staticmethod
    def get_electronegativity(symbol: str) -> float:
        return GlobalKnowledge.ELECTRONEGATIVITY.get(symbol, 2.0)

    @staticmethod
    def get_covalent_radius(symbol: str) -> float:
        # Note: Typo fixed in the key name during rewrite
        return GlobalKnowledge.COVALVALENT_RADII.get(symbol, 1.5)

    @staticmethod
    def is_metal(symbol: str) -> bool:
        # Simplified metal check for common surface science atoms
        en = GlobalKnowledge.get_electronegativity(symbol)
        return en < 1.9 or symbol in ['Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Pt', 'Au', 'Al']

    @staticmethod
    def get_preferred_partners(symbol: str) -> list:
        """
        Returns a list of preferred partner types based on electronegativity.
        Metals prefer non-metals/halogens.
        """
        is_metal = GlobalKnowledge.is_metal(symbol)
        if is_metal:
            return ["Non-metal", "Halogen"]
        else:
            return ["Metal"]

class KnowledgeManager:
    """
    Manages local project data and global heuristics.
    """
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
    def save_local(self, filename: str, data):
        path = os.path.join(self.log_dir, filename)
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
            
    def load_local(self, filename: str):
        path = os.path.join(self.log_dir, filename)
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return None

    def get_binding_preference(self, species_a: str, species_b: str) -> float:
        """
        Calculates a compatibility score between two species.
        Higher score = stronger preference.
        Based on Electronegativity Difference.
        """
        en_a = GlobalKnowledge.get_electronegativity(species_a)
        en_b = GlobalKnowledge.get_electronegativity(species_b)
        
        # Ionic character increases with EN difference
        en_diff = abs(en_a - en_b)
        
        # Heuristic: Metals and Non-metals/Halogens have high compatibility
        metal_a = GlobalKnowledge.is_metal(species_a)
        metal_b = GlobalKnowledge.is_metal(species_b)
        
        score = en_diff
        if metal_a != metal_b:
            score += 1.0 # Bonus for metal-nonmetal pairing
            
        return score
