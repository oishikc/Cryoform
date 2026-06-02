# Trains a Random Forest regression model to predict glass transition
# temperature (Tg) from molecular descriptors computed by RDKit.
# Filters candidate excipients by predicted Tg > 25C threshold.

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.preprocessing import StandardScaler

from data.tg_data import TG_DATA
from data.candidates import CANDIDATE_SMILES
from data.biocompat_filter import load_approved_ingredients, is_approved


def compute_descriptors(smiles: str) -> dict:
    """
    Computes molecular descriptors used as ML features for Tg prediction.
    Returns None if SMILES cannot be parsed.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol_no_h = Chem.RemoveHs(mol)

    return {
        "MW":            Descriptors.MolWt(mol),
        "TPSA":          Descriptors.TPSA(mol),
        "LogP":          Descriptors.MolLogP(mol),
        "HBD":           rdMolDescriptors.CalcNumHBD(mol),
        "HBA":           rdMolDescriptors.CalcNumHBA(mol),
        "RotBonds":      rdMolDescriptors.CalcNumRotatableBonds(mol),
        "Rings":         rdMolDescriptors.CalcNumRings(mol),
        "AromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "HeavyAtoms":    mol_no_h.GetNumAtoms(),
        "FractionCSP3":  rdMolDescriptors.CalcFractionCSP3(mol),
        "BertzCT":       Descriptors.BertzCT(mol),
    }


def build_training_data():
    """
    Builds feature matrix X and target vector y from TG_DATA.
    Excludes crystallizers (unreliable Tg).
    """
    rows = []
    targets = []
    names = []

    for name, smiles, tg, crystallizer in TG_DATA:
        if crystallizer:
            continue  # exclude — Tg is not meaningful for crystallizers
        desc = compute_descriptors(smiles)
        if desc is None:
            continue
        rows.append(desc)
        targets.append(tg)
        names.append(name)

    df = pd.DataFrame(rows)
    return df, np.array(targets), names


def train_model():
    """
    Trains Random Forest regressor on Tg data.
    Returns fitted model, scaler, and feature column names.
    """
    X_df, y, names = build_training_data()
    feature_cols = X_df.columns.tolist()

    scaler = StandardScaler()
    X = scaler.fit_transform(X_df)

    model = RandomForestRegressor(
        n_estimators=200,
        max_features="sqrt",
        random_state=42
    )

    # Leave-one-out CV to evaluate performance on small dataset
    loo = LeaveOneOut()
    scores = cross_val_score(model, X, y, cv=loo, scoring="neg_mean_absolute_error")
    mae = -scores.mean()
    print(f"LOO Cross-validation MAE: {mae:.1f} °C  (n={len(y)})")

    # Fit on full dataset for prediction
    model.fit(X, y)

    return model, scaler, feature_cols


def predict_tg(smiles: str, model, scaler, feature_cols: list) -> float:
    """
    Predicts Tg (°C) for a single molecule given its SMILES.
    Returns None if SMILES is invalid.
    """
    desc = compute_descriptors(smiles)
    if desc is None:
        return None

    X = pd.DataFrame([desc])[feature_cols]
    X_scaled = scaler.transform(X)
    return round(float(model.predict(X_scaled)[0]), 1)


TG_THRESHOLD = 25.0  # °C — minimum Tg for room-temperature stability


if __name__ == "__main__":
    print("=== Training Tg Predictor ===")
    model, scaler, feature_cols = train_model()

    approved = load_approved_ingredients()

    print("\n=== Candidate Tg Predictions ===")
    print(f"{'Name':<16} {'Pred Tg (C)':>12}  {'Biocompat':>10}  {'Tg Pass':>8}")
    print("-" * 54)

    results = []
    for name, smiles in CANDIDATE_SMILES.items():
        tg = predict_tg(smiles, model, scaler, feature_cols)
        approved_status = is_approved(name, approved)
        tg_pass = tg is not None and tg > TG_THRESHOLD
        results.append((name, tg, approved_status, tg_pass))

    # Sort by predicted Tg descending
    results.sort(key=lambda x: x[1] if x[1] is not None else -999, reverse=True)

    passing = []
    for name, tg, biocompat, tg_pass in results:
        tg_str = f"{tg:.1f}" if tg is not None else "N/A"
        bc_str = "PASS" if biocompat else "FAIL"
        tp_str = "PASS" if tg_pass else "FAIL"
        print(f"{name:<16} {tg_str:>12}  {bc_str:>10}  {tp_str:>8}")
        if tg_pass and biocompat:
            passing.append(name)

    print(f"\nCandidates passing both gates: {passing}")
