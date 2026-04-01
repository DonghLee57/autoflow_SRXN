import os
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from ase import Atoms
from ase.io import write
import numpy as np

def generate_dipas():
    # 1. DIPAS (Diisopropylaminosilane) SMILES
    # Structure: SiH3-N(CH(CH3)2)2
    smiles = "CC(C)N([SiH3])C(C)C"
    print(f"SMILES for DIPAS: {smiles}")

    # Create RDKit molecule object (no explicit H for 2D)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print("Error: Invalid SMILES")
        return

    # 2. Generate 2D Structure Image (without H, standard convention)
    # RDKit's default drawing omits H on carbons and follows chemical conventions.
    img_path = "dipas_2d.png"
    Draw.MolToFile(mol, img_path, size=(400, 400))
    print(f"2D image saved to: {img_path}")

    # 3. Generate 3D Structure (including H for coordinates)
    mol_3d = Chem.AddHs(mol)
    # Use ETKDG method for conformer generation
    params = AllChem.ETKDG()
    status = AllChem.EmbedMolecule(mol_3d, params)
    
    if status == -1:
        # Fallback if ETKDG fails
        AllChem.EmbedMolecule(mol_3d, AllChem.DistanceGeometryConfig())
    
    # Optimize geometry slightly using UFF
    AllChem.UFFOptimizeMolecule(mol_3d)

    # 4. Save to extxyz using ASE
    conf = mol_3d.GetConformer()
    positions = conf.GetPositions()
    symbols = [atom.GetSymbol() for atom in mol_3d.GetAtoms()]
    
    # Create ASE Atoms object
    dipas_atoms = Atoms(symbols=symbols, positions=positions)
    
    # Save as extxyz
    xyz_path = "dipas_3d.extxyz"
    write(xyz_path, dipas_atoms, format="extxyz")
    print(f"3D coordinates saved to: {xyz_path}")

if __name__ == "__main__":
    generate_dipas()
