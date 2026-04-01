import os
import numpy as np
from ase import Atoms
from ase.io import write
from ase.build import bulk, surface, add_adsorbate
from rdkit import Chem
from rdkit.Chem import AllChem

# Try to import CatKit, handling the MutableMapping fix we applied
try:
    from catkit.gen.adsorption import AdsorptionSites, Builder
    from catkit.gratoms import Gratoms
    CATKIT_AVAILABLE = True
except ImportError:
    CATKIT_AVAILABLE = False
    print("Warning: CatKit could not be imported. Please ensure it is installed correctly.")

from surface_utils import reconstruct_2x1_buckled, passivate_slab

def get_dipas_molecule():
    """Generate DIPAS conformer using RDKit."""
    smiles = "CC(C)N([SiH3])C(C)C"
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.UFFOptimizeMolecule(mol)
    
    conf = mol.GetConformer()
    positions = conf.GetPositions()
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    return Atoms(symbols=symbols, positions=positions)

def get_dissociated_fragments():
    """Generate DIPAS fragments (cleaved Si-H) for chemisorption."""
    smiles_core = "CC(C)N([SiH2])C(C)C"
    mol = Chem.MolFromSmiles(smiles_core)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.UFFOptimizeMolecule(mol)
    
    conf = mol.GetConformer()
    core_atoms = Atoms([a.GetSymbol() for a in mol.GetAtoms()], positions=conf.GetPositions())
    h_atom = Atoms('H', positions=[[0, 0, 0]])
    return core_atoms, h_atom

def generate_si100_slab():
    """Prepare a standard reconstructed Si(100) surface."""
    bulk_si = bulk('Si', 'diamond', a=5.43, cubic=True)
    slab = surface(bulk_si, (1, 0, 0), layers=6, vacuum=10.0)
    slab = slab * (4, 4, 1)
    slab.pbc = [True, True, False]
    reconstruct_2x1_buckled(slab, pattern='checkerboard')
    slab = passivate_slab(slab, side='bottom')
    return slab

def find_optimal_rotation(slab, molecule, site_pos, height=2.3, steps=36):
    """Scan 360 degrees to find the orientation with maximum surface clearance."""
    best_dist = -1.0
    best_atoms = None
    
    # Identify surface atoms for distance calculation
    z_max = slab.positions[:, 2].max()
    surf_mask = slab.positions[:, 2] > z_max - 1.5
    substrate_pos = slab.positions[surf_mask]
    
    # Pre-select Si pivot index
    si_idx = [a.index for a in molecule if a.symbol == 'Si'][0]
    
    for angle in np.linspace(0, 360, steps, endpoint=False):
        m_copy = molecule.copy()
        m_copy.rotate(angle, 'z', center=m_copy.positions[si_idx])
        
        temp_slab = slab.copy()
        # Ensure name unique for this temp test
        add_adsorbate(temp_slab, m_copy, height=height, position=(site_pos[0], site_pos[1]))
        
        # Calculate distance
        n_slab = len(slab)
        ads_pos = temp_slab.positions[n_slab:]
        
        from ase.geometry import get_distances
        _, d = get_distances(ads_pos, substrate_pos, cell=slab.cell, pbc=slab.pbc)
        min_dist = np.min(d)
        
        if min_dist > best_dist:
            best_dist = min_dist
            best_atoms = temp_slab.copy()
            
    return best_atoms, best_dist

def run_catkit_generation():
    if not CATKIT_AVAILABLE:
        return

    print("--- 1. Preparing Surface and Molecule ---")
    slab = generate_si100_slab()
    dipas = get_dipas_molecule()
    core, h_frag = get_dissociated_fragments()

    print("\n--- 2. Identifying Adsorption Sites with CatKit ---")
    gslab = Gratoms(slab)
    z_max = slab.positions[:, 2].max()
    surf_mask = slab.positions[:, 2] > z_max - 0.5
    # Required for CatKit get_surface_atoms()
    gslab.set_array('surface_atoms', surf_mask.astype(int))
    
    ads_sites = AdsorptionSites(gslab)
    unique_sites = ads_sites.get_coordinates()
    print(f"Total unique adsorption sites: {len(unique_sites)}")

    print("\n--- 3. Generating Optimized Physisorption Candidates ---")
    candidates = []
    for i, site_pos in enumerate(unique_sites):
        # Optimization: Scan 360 degrees to avoid Si-C clash
        opt_atoms, clearance = find_optimal_rotation(slab, dipas, site_pos, height=3.0)
        if opt_atoms:
            opt_atoms.info['name'] = f"Phys_Site_{i}_Opt"
            opt_atoms.info['clearance'] = clearance
            candidates.append(opt_atoms)
            print(f"  Site {i:2d}: Optimized Orientation Found. Min Clearance: {clearance:.2f} A")

    print("\n--- 4. Generating Optimized Chemisorption Candidates ---")
    # For chemisorption (dissociative), we optimize the core fragment placement
    for i, site_pos in enumerate(unique_sites[:3]):
        opt_core_atoms, clearance = find_optimal_rotation(slab, core, site_pos, height=2.2)
        if opt_core_atoms:
            # Place dissociated H fragment at a reasonable offset
            h_pos = site_pos + np.array([1.5, 0, 0])
            add_adsorbate(opt_core_atoms, h_frag, height=1.5, position=(h_pos[0], h_pos[1]))
            
            opt_core_atoms.info['name'] = f"Chem_Site_{i}_Opt"
            opt_core_atoms.info['clearance'] = clearance
            candidates.append(opt_core_atoms)
            print(f"  Site {i:2d}: Dissociative Optimized. Core Clearance: {clearance:.2f} A")

    print(f"\n--- 5. Exporting {len(candidates)} optimized candidates ---")
    write('catkit_dipas_candidates_optimized.extxyz', candidates)
    print("Files saved to catkit_dipas_candidates_optimized.extxyz")

if __name__ == "__main__":
    run_catkit_generation()
