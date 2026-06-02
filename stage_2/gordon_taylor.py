# Computes mixture glass transition temperature using the Gordon-Taylor equation.
# Used to predict Tg of binary excipient blends across weight fractions.
#
# Gordon-Taylor equation:
#   Tg_mix = (w1 * Tg1 + k * w2 * Tg2) / (w1 + k * w2)
#
# where k is estimated via Couchman-Karasz approximation: k = Tg1 / Tg2
# (both in Kelvin). This avoids needing experimental binary data.

import numpy as np
import itertools


def gordon_taylor_tg(tg1_c: float, tg2_c: float, w1: float) -> float:
    """
    Predicts mixture Tg (°C) for a binary blend.
    tg1_c, tg2_c: pure component Tg values in Celsius
    w1: weight fraction of component 1 (0 to 1)
    """
    tg1_k = tg1_c + 273.15
    tg2_k = tg2_c + 273.15
    w2 = 1.0 - w1

    # Couchman-Karasz k estimate
    k = tg1_k / tg2_k

    tg_mix_k = (w1 * tg1_k + k * w2 * tg2_k) / (w1 + k * w2)
    return round(tg_mix_k - 273.15, 1)


def scan_binary_mixtures(candidates: dict, tg_threshold: float = 40.0) -> list:
    """
    Enumerates all binary pairs from candidates dict {name: predicted_tg}.
    Sweeps weight fractions from 0.1 to 0.9 in steps of 0.1.
    Returns list of dicts for pairs where max predicted Tg_mix > tg_threshold.
    """
    names = list(candidates.keys())
    results = []

    for name1, name2 in itertools.combinations(names, 2):
        tg1 = candidates[name1]
        tg2 = candidates[name2]

        best_tg = -999
        best_w1 = None

        for w1 in np.arange(0.1, 1.0, 0.1):
            tg_mix = gordon_taylor_tg(tg1, tg2, round(w1, 1))
            if tg_mix > best_tg:
                best_tg = tg_mix
                best_w1 = round(w1, 1)

        if best_tg > tg_threshold:
            results.append({
                "component_1":  name1,
                "component_2":  name2,
                "best_w1":      best_w1,
                "best_w2":      round(1 - best_w1, 1),
                "predicted_tg": best_tg,
            })

    results.sort(key=lambda x: x["predicted_tg"], reverse=True)
    return results


if __name__ == "__main__":
    # Test with passing candidates from tg_predictor output
    passing_candidates = {
        "Histidine":    200.1,
        "Arginine":     152.0,
        "Lysine":       141.1,
        "Glycine":      108.8,
        "Trehalose":     93.9,
        "Sucrose":       93.9,
        "Lactose":       91.5,
        "Mannitol":      55.2,
        "Sorbitol":      55.2,
        "Succinic acid": 29.0,
    }

    print("=== Binary Mixture Tg Scan (Gordon-Taylor) ===")
    print(f"Threshold: 40°C | Candidates: {len(passing_candidates)}")
    print()

    mixtures = scan_binary_mixtures(passing_candidates, tg_threshold=40.0)

    print(f"{'Pair':<30} {'Best ratio (w1:w2)':>20}  {'Pred Tg_mix (C)':>16}")
    print("-" * 70)
    for m in mixtures[:10]:  # top 10
        pair = f"{m['component_1']} / {m['component_2']}"
        ratio = f"{m['best_w1']:.1f} : {m['best_w2']:.1f}"
        print(f"{pair:<30} {ratio:>20}  {m['predicted_tg']:>16.1f}")
