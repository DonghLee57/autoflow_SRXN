import json
import os
import numpy as np
from ase.data import covalent_radii, vdw_radii

class KnowledgeEngine:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(KnowledgeEngine, cls).__new__(cls)
            cls._instance._load_database()
        return cls._instance
    
    def _load_database(self):
        db_path = os.path.join(os.path.dirname(__file__), 'chem_data.json')
        if os.path.exists(db_path):
            with open(db_path, 'r') as f:
                self.db = json.load(f)
        else:
            self.db = {}
            
    def get_ideal_coordination(self, symbol, config=None):
        """
        Retrieves ideal coordination with hierarchical priority:
        1. User override in config['ideal_coordination']
        2. Internal chem_data.json
        3. Simple valency heuristic (8 - Group Number)
        """
        # 1. Config Override
        if config and 'ideal_coordination' in config:
            val = config['ideal_coordination'].get(symbol)
            if val is not None:
                return val
        
        # 2. Database
        if symbol in self.db:
            return self.db[symbol].get('ideal_coordination', 0)
            
        # 3. Fallback Heuristic
        # Very rough approximation for p-block
        from ase.data import atomic_numbers
        z = atomic_numbers.get(symbol, 0)
        if 5 <= z <= 8: # B, C, N, O
            return 8 - (z - 2 + 10) % 8 # simplistic
        return 0

    def get_covalent_radius(self, symbol):
        from ase.data import atomic_numbers
        z = atomic_numbers.get(symbol, 0)
        if symbol in self.db and 'covalent_radius' in self.db[symbol]:
            return self.db[symbol]['covalent_radius']
        return covalent_radii[z]

    def get_vdw_radius(self, symbol):
        from ase.data import atomic_numbers
        z = atomic_numbers.get(symbol, 0)
        if symbol in self.db and 'vdw_radius' in self.db[symbol]:
            return self.db[symbol]['vdw_radius']
        return vdw_radii[z] if not np.isnan(vdw_radii[z]) else 1.5

# Singleton global instance
chem_kb = KnowledgeEngine()
