import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from data.lipids.lipids import LIPID_SMILES

for name, smiles in LIPID_SMILES.items():
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"FAIL  {name}: RDKit could not parse SMILES")
    else:
        mw = Descriptors.MolWt(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        tpsa = Descriptors.TPSA(mol)
        print(f"OK    {name}")
        print(f"      MW={mw:.1f}  HBD={hbd}  HBA={hba}  TPSA={tpsa:.1f}")
