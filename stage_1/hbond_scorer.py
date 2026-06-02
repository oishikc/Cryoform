# Computes a molar-weighted composite H-bond surface profile for a given
# LNP formulation, then scores candidate protectants against it.

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from data.lipids.lipids import LIPID_SMILES


def get_hbond_profile(smiles):
    """
    Returns H-bond profile dict for a molecule given its SMILES.
    HBD = donor count, HBA = acceptor count, TPSA = polar surface area,
    MW = molecular weight.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")
    return {
        "HBD": rdMolDescriptors.CalcNumHBD(mol),
        "HBA": rdMolDescriptors.CalcNumHBA(mol),
        "TPSA": Descriptors.TPSA(mol),
        "MW": Descriptors.MolWt(mol),
    }


def build_surface_profile(formulation: dict) -> dict:
    """
    Takes a formulation dict: {lipid_name: mole_fraction}
    Mole fractions must sum to 1.0.
    Returns the molar-weighted composite H-bond surface profile.
    """
    total = sum(formulation.values())
    if abs(total - 1.0) > 0.01:
        raise ValueError(f"Mole fractions must sum to 1.0, got {total:.3f}")

    composite = {"HBD": 0.0, "HBA": 0.0, "TPSA": 0.0, "MW": 0.0}

    for lipid_name, mole_frac in formulation.items():
        if lipid_name not in LIPID_SMILES:
            raise ValueError(f"Unknown lipid: {lipid_name}. Add it to lipids.py first.")
        profile = get_hbond_profile(LIPID_SMILES[lipid_name])
        for key in composite:
            composite[key] += mole_frac * profile[key]
    if composite["TPSA"] > 0:
        composite["HBD_density"] = composite["HBD"] / composite["TPSA"]
        composite["HBA_density"] = composite["HBA"] / composite["TPSA"]
    else:
        composite["HBD_density"] = 0.0
        composite["HBA_density"] = 0.0

    return composite


def score_candidate(candidate_smiles: str, surface_profile: dict) -> float:
    """
    Scores a candidate protectant against the LNP surface profile.
    
    The surface has more HBA than HBD (ratio ~2.86:1 for standard LNP).
    A good protectant should be HBD-rich to complement the HBA-heavy surface.
    Score weights HBD contribution 3x more than HBA to reflect this asymmetry,
    then scales by the candidate's HBD:HBA ratio as a complementarity bonus.
    """
    mol = Chem.MolFromSmiles(candidate_smiles)
    if mol is None:
        return 0.0

    cand_hbd = rdMolDescriptors.CalcNumHBD(mol)
    cand_hba = rdMolDescriptors.CalcNumHBA(mol)

    # Surface HBA:HBD ratio — tells us what the surface needs
    surface_hba = surface_profile["HBA"]
    surface_hbd = surface_profile["HBD"]
    surface_ratio = surface_hba / max(surface_hbd, 0.01)  # ~2.86 for standard LNP

    # Candidate complementarity: reward HBD-rich candidates for HBA-rich surfaces
    complementarity = (cand_hbd * surface_ratio) + cand_hba

    # Efficiency: complementarity per heavy atom count (rewards compact molecules)
    mol_no_h = Chem.RemoveHs(mol)
    heavy_atoms = mol_no_h.GetNumAtoms()
    score = complementarity / max(heavy_atoms, 1)

    return round(score, 4)

import pandas as pd

def rank_candidates(candidate_dict: dict, surface_profile: dict) -> pd.DataFrame:
    """
    Takes a dict of {name: SMILES} candidates and a surface profile.
    Returns a DataFrame ranked by H-bond complementarity score.
    """
    rows = []
    for name, smiles in candidate_dict.items():
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        score = score_candidate(smiles, surface_profile)
        rows.append({
            "name":  name,
            "SMILES": smiles,
            "HBD":   rdMolDescriptors.CalcNumHBD(mol),
            "HBA":   rdMolDescriptors.CalcNumHBA(mol),
            "TPSA":  round(Descriptors.TPSA(mol), 1),
            "MW":    round(Descriptors.MolWt(mol), 1),
            "hbond_score": score,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("hbond_score", ascending=False).reset_index(drop=True)
    df.index += 1  # rank starts at 1
    return df

if __name__ == "__main__":
    formulation = {
        "ALC-0315":    0.463,
        "DSPC":        0.094,
        "Cholesterol": 0.427,
        "ALC-0159":    0.015,
    }

    print("=== LNP Surface Profile ===")
    profile = build_surface_profile(formulation)
    for k, v in profile.items():
        print(f"  {k}: {v:.4f}")

    candidates = {
        "Trehalose": "OC[C@H]1O[C@@](CO)(O[C@@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)[C@@H](O)[C@@H]1O",
        "Sucrose":   "OC[C@H]1O[C@](CO)(O[C@@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)[C@@H](O)[C@@H]1O",
        "Mannitol":  "OC[C@@H](O)[C@@H](O)[C@H](O)[C@@H](O)CO",
        "Histidine": "N[C@@H](Cc1cnc[nH]1)C(=O)O",
        "Glycine":   "NCC(=O)O",
    }

    print("\n=== Ranked Candidates ===")
    df = rank_candidates(candidates, profile)
    print(df[["name", "HBD", "HBA", "TPSA", "MW", "hbond_score"]].to_string())
