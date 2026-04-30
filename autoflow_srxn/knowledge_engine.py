import json
import os


class KnowledgeBase:
    def __init__(self):
        self.data = {}
        # Path to chem_data.json relative to this file
        data_path = os.path.join(os.path.dirname(__file__), "chem_data.json")
        if os.path.exists(data_path):
            with open(data_path) as f:
                self.data = json.load(f)

    def get_ideal_coordination(self, symbol, config=None):
        """Returns the standard valency/coordination for an element."""
        if config and isinstance(config, dict) and symbol in config:
            return config[symbol]
        return self.data.get(symbol, {}).get("ideal_coordination", 0)

    def get_radius(self, symbol, rtype="covalent"):
        """Returns covalent or vdW radius."""
        key = "covalent_radius" if rtype == "covalent" else "vdw_radius"
        return self.data.get(symbol, {}).get(key, 1.5)


chem_kb = KnowledgeBase()
