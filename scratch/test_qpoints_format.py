import os
import sys
import numpy as np
import yaml

# Add src to path
sys.path.append(os.path.abspath('src'))

from vibrational_analyzer import VibrationalAnalyzer
from qpoint_handler import QPointParser
from ase.build import molecule
from ase.calculators.emt import EMT

def test_qpoints_format():
    # 1. Setup a simple system
    atoms = molecule('H2O')
    atoms.calc = EMT()
    
    class MockEngine:
        def get_calculator(self):
            return EMT()
            
    va = VibrationalAnalyzer(atoms, MockEngine())
    va.run_analysis()
    
    # 3. Generate qpoints.yaml
    test_file = 'test_qpoints.yaml'
    va.generate_qpoints_file(test_file)
    
    # 4. Check file content
    with open(test_file, 'r') as f:
        data = yaml.safe_load(f)
        
    band1 = data['phonon'][0]['band'][0]
    eig = band1['eigenvector']
    
    print(f"Number of atoms in YAML: {data['natom']}")
    print(f"Number of elements in eigenvector list: {len(eig)}")
    print(f"First element of eigenvector: {eig[0]}")
    
    # Verify structure
    assert len(eig) == data['natom'], "Eigenvector list should be per-atom"
    assert len(eig[0]) == 3, "Each atom should have 3 components (x, y, z)"
    assert len(eig[0][0]) == 2, "Each component should be a [real, imag] pair"
    
    print("Writing logic: OK (nested format)")
    
    # 5. Test parsing logic
    parser = QPointParser(test_file)
    modes = parser.get_filtered_modes(freq_threshold=100.0) # all modes
    
    print(f"Parsed {len(modes)} modes.")
    print(f"First mode eigenvector shape: {modes[0]['eigenvector'].shape}")
    
    assert modes[0]['eigenvector'].shape == (3, 3), "Parsed eigenvector should be (N, 3)"
    
    print("Parsing logic: OK")
    
    # Cleanup
    os.remove(test_file)

if __name__ == "__main__":
    test_qpoints_format()
