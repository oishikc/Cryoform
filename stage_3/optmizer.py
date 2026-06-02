# stage_3/optimizer.py
# Composite score optimizer for lyoprotectant formulation design.
# Takes Stage 1 H-bond scores + Stage 2 Tg predictions and finds
# the best binary mixture and ratio for a given LNP formulation.

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import itertools

from data.candidates import CANDIDATE_SMILES
from data.biocompat_filter import load_approved_ingredients, is_approved
from stage_1.hbond_scorer import build_surface_profile, rank_candidates
from stage_2.tg_predictor import train_model, predict_tg, TG_THRESHOLD
from stage_2.gordon_taylor import gordon_taylor_tg


def composite_score(hbond_score: float, tg_mix: float,
                    biocompat: bool,
                    alpha: float = 0.5,
                    beta: float = 0.4,
                    gamma: float = 0.1) -> float:
    """
    Computes composite optimization score for a candidate mixture.

    alpha: weight for H-bond complementarity
    beta:  weight for Tg margin above threshold
    gamma: weight for biocompatibility bonus

    All components normalized to [0, 1] range.
    """
    # H-bond score: normalize by dividing by a reasonable max (3.0)
    hbond_norm = min(hbond_score / 3.0, 1.0)

    # Tg margin: normalize — 200°C above threshold = score of 1.0
    tg_margin = max(tg_mix - TG_THRESHOLD, 0)
    tg_norm = min(tg_margin / 200.0, 1.0)

    # Biocompatibility bonus
    bc_bonus = 1.0 if biocompat else 0.0

    return round(alpha * hbond_norm + beta * tg_norm + gamma * bc_bonus, 4)


def run_optimizer(formulation: dict,
                  alpha: float = 0.5,
                  beta: float = 0.4,
                  gamma: float = 0.1,
                  top_n: int = 3) -> list:
    """
    Full pipeline: takes LNP formulation dict {lipid: mole_fraction},
    runs Stage 1 H-bond scoring, Stage 2 Tg prediction and filtering,
    Gordon-Taylor mixture scan, composite scoring, returns top_n recipes.
    """

    # --- Stage 1: H-bond scoring ---
    surface_profile = build_surface_profile(formulation)
    hbond_df = rank_candidates(CANDIDATE_SMILES, surface_profile)

    # Build name -> hbond_score lookup
    hbond_scores = dict(zip(hbond_df["name"], hbond_df["hbond_score"]))

    # --- Biocompatibility gate ---
    approved = load_approved_ingredients()
    biocompat_pass = {
        name for name in CANDIDATE_SMILES
        if is_approved(name, approved)
    }

    # --- Stage 2: Tg prediction ---
    print("Training Tg model...")
    model, scaler, feature_cols = train_model()

    tg_predictions = {}
    for name, smiles in CANDIDATE_SMILES.items():
        tg = predict_tg(smiles, model, scaler, feature_cols)
        if tg is not None and tg > TG_THRESHOLD and name in biocompat_pass:
            tg_predictions[name] = tg

    print(f"Candidates passing both gates: {list(tg_predictions.keys())}")

    # --- Stage 3: Binary mixture optimization ---
    results = []
    names = list(tg_predictions.keys())

    for name1, name2 in itertools.combinations(names, 2):
        tg1 = tg_predictions[name1]
        tg2 = tg_predictions[name2]
        hs1 = hbond_scores.get(name1, 0)
        hs2 = hbond_scores.get(name2, 0)
        both_biocompat = (name1 in biocompat_pass) and (name2 in biocompat_pass)

        best_score = -1
        best_w1 = None
        best_tg_mix = None
        best_hbond_mix = None

        for w1 in np.arange(0.1, 1.0, 0.1):
            w1 = round(w1, 1)
            w2 = round(1 - w1, 1)

            tg_mix = gordon_taylor_tg(tg1, tg2, w1)
            hbond_mix = w1 * hs1 + w2 * hs2

            score = composite_score(
                hbond_mix, tg_mix, both_biocompat, alpha, beta, gamma
            )

            if score > best_score:
                best_score = score
                best_w1 = w1
                best_tg_mix = tg_mix
                best_hbond_mix = hbond_mix

        results.append({
            "component_1":   name1,
            "component_2":   name2,
            "w1":            best_w1,
            "w2":            round(1 - best_w1, 1),
            "tg_mix":        best_tg_mix,
            "hbond_score":   round(best_hbond_mix, 4),
            "biocompat":     both_biocompat,
            "composite":     best_score,
        })

    results.sort(key=lambda x: x["composite"], reverse=True)
    return results[:top_n]


def print_recipes(recipes: list, formulation: dict):
    print("\n" + "=" * 60)
    print("CRYOFORM — RECOMMENDED LYOPROTECTANT RECIPES")
    print("=" * 60)
    print(f"LNP formulation: {formulation}")
    print()

    for i, r in enumerate(recipes, 1):
        print(f"Rank {i}: {r['component_1']} {int(r['w1']*100)}% / "
              f"{r['component_2']} {int(r['w2']*100)}%")
        print(f"  Predicted Tg_mix : {r['tg_mix']:.1f} °C")
        print(f"  H-bond score     : {r['hbond_score']:.4f}")
        print(f"  Biocompatible    : {'Yes' if r['biocompat'] else 'No'}")
        print(f"  Composite score  : {r['composite']:.4f}")
        print()


if __name__ == "__main__":
    formulation = {
        "ALC-0315":    0.463,
        "DSPC":        0.094,
        "Cholesterol": 0.427,
        "ALC-0159":    0.015,
    }

    recipes = run_optimizer(formulation, alpha=0.5, beta=0.4, gamma=0.1, top_n=3)
    print_recipes(recipes, formulation)
