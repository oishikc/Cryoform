# CryoForm

**Computational lyoprotectant formulation design for lipid nanoparticles.**

---

## What is this?

CryoForm is an internal R&D tool that helps formulation scientists design lyoprotectant mixtures for lipid nanoparticle (LNP) drug products. It runs a three-stage computational pipeline — hydrogen bond scoring, glass transition temperature prediction, and mixture optimization — and outputs ranked excipient recipes with scores, predicted stability, and uncertainty estimates.

---

## The Problem

Lipid nanoparticles (LNPs) are the delivery vehicle behind mRNA vaccines and a growing class of RNA therapeutics. To store and ship them, manufacturers typically freeze-dry (lyophilize) the product — a process that requires adding protective excipients called **lyoprotectants** that shield the LNP from damage during freezing and drying.

Choosing the right lyoprotectant formulation is not trivial:

- The excipient has to physically interact well with the LNP surface (hydrogen bond complementarity)
- It has to form a stable amorphous glass at storage temperature — measured by its **glass transition temperature (Tg)**
- It has to be biocompatible and approved for the relevant route of administration
- Binary mixtures (two excipients blended together) often outperform single-component systems, but the number of combinations and ratios to test is large

Currently, this is done primarily by trial and error on the bench. That is slow, expensive, and depends heavily on individual expertise.

---

## The Solution

CryoForm replaces the trial-and-error step with a computational pre-screening pipeline. Before a scientist runs a single experiment, CryoForm:

1. Analyzes the LNP surface chemistry and identifies what kind of excipient would interact with it best
2. Predicts the glass transition temperature of each candidate excipient using a machine learning model trained on literature data
3. Scans all binary mixtures of passing candidates across weight fractions using the Gordon-Taylor equation
4. Scores and ranks the mixtures on a composite metric combining H-bond fit, thermal stability, and biocompatibility

The output is a short, ranked list of formulation recipes — each with a predicted Tg, H-bond score, uncertainty estimate, and biocompatibility status — that a scientist can prioritize for wet lab validation.

---

## How It Works

The pipeline has three sequential stages:

### Stage 1 — H-Bond Surface Scoring (`stage_1/hbond_scorer.py`)

LNP membranes are hydrogen bond acceptor (HBA)-rich. A good lyoprotectant needs to be HBD-rich to complement this. CryoForm computes a molar-weighted H-bond surface profile for the input LNP formulation, then scores each candidate excipient against it. HBD contribution is weighted 3× more than HBA to reflect the asymmetry of the LNP surface.

### Stage 2 — Tg Prediction (`stage_2/tg_predictor.py`, `stage_2/gordon_taylor.py`)

A Random Forest regression model is trained on literature glass transition temperature data for ~14 non-crystallizing excipients (crystallizers like Mannitol and Glycine are excluded from training — their Tg values are not meaningful for glassy-state stability). RDKit computes 11 molecular descriptors per compound (MW, TPSA, LogP, HBD, HBA, rotatable bonds, rings, aromatic rings, heavy atom count, fraction Csp3, Bertz complexity). Leave-one-out cross-validation is used to evaluate model performance on the small dataset.

For binary mixtures, the Gordon-Taylor equation predicts the mixture Tg across weight fractions:

```
Tg_mix = (w1·Tg1 + k·w2·Tg2) / (w1 + k·w2)
```

where `k` is estimated via the Couchman-Karasz approximation (`k = Tg1/Tg2`, both in Kelvin). Candidates with predicted Tg below 25 °C are filtered out.

### Stage 3 — Mixture Optimization (`stage_3/optimizer.py`)

All binary combinations of passing candidates are scanned across weight fractions (0.1 to 0.9 in steps of 0.1). Each combination is scored on a composite metric:

```
score = α·(H-bond fit) + β·(Tg margin above threshold) + γ·(biocompatibility bonus)
```

Default weights: α = 0.5, β = 0.4, γ = 0.1. These are configurable in the UI.

### Biocompatibility Gate (`data/biocompat_filter.py`)

Candidates are cross-referenced against the FDA Inactive Ingredient Database (IID) filtered to intravenous, injection, subcutaneous, and intradermal routes. Compounds not on this list (or the internal whitelist) are flagged as failing the biocompatibility gate and excluded from ranked output.

---

## Features

- **Pipeline tab** — run the full three-stage optimizer for a given LNP formulation; view ranked recipes with Tg curves, uncertainty bands, and moisture plasticization analysis
- **Molecule Explorer** — look up any candidate in the library; see predicted Tg with confidence level, nearest training compounds by descriptor distance, and moisture sensitivity
- **Model Transparency** — RF feature importances, LOO-CV predicted vs actual scatter, residuals, and full training data table
- **Custom Candidate** — paste any SMILES string and run it through the full pipeline; evaluate novel excipients before synthesis
- **Data Management** — upload in-house experimental Tg measurements to extend the training set; retrain the model; download the current library as CSV

---

## Installation

**Requirements:** Python 3.9+, RDKit

```bash
git clone https://github.com/your-username/cryoform.git
cd cryoform
pip install -r requirements.txt
streamlit run app.py
```

**requirements.txt:**
```
streamlit
rdkit
scikit-learn
pandas
numpy
plotly
joblib
```

> RDKit is best installed via conda if you run into issues:
> `conda install -c conda-forge rdkit`

---

## Project Structure

```
cryoform/
│
├── app.py                  # Streamlit UI — entry point
├── main.py
│
├── data/
│   ├── candidates.py       # Curated excipient library with SMILES
│   ├── tg_data.py          # Literature Tg training dataset
│   ├── biocompat_filter.py # FDA IID biocompatibility gate
│   ├── IIR_OCOMM.csv       # FDA Inactive Ingredient Database
│   └── lipids/
│       └── lipids.py       # LNP component SMILES library
│
├── stage_1/
│   ├── hbond_scorer.py     # H-bond surface profile and candidate scoring
│   └── verify_lipids.py    # RDKit validation for lipid SMILES
│
├── stage_2/
│   ├── tg_predictor.py     # RF model training and Tg prediction
│   └── gordon_taylor.py    # Binary mixture Tg via Gordon-Taylor equation
│
└── stage_3/
    └── optimizer.py        # Composite scorer and full pipeline runner
```

---

## Science Notes

**Why Gordon-Taylor?** The GT equation is the standard model for predicting binary mixture Tg and requires only pure-component Tg values. The Couchman-Karasz `k` approximation (`k = Tg1/Tg2`) assumes equal heat capacity jumps at Tg, which is a reasonable first-pass assumption for sugar/amino acid pairs but can diverge for structurally dissimilar systems. For production use, fitting `k` experimentally or using tabulated ΔCp values would improve prediction accuracy.

**Why LOO-CV?** The training dataset contains ~14 non-crystallizer compounds. Leave-one-out cross-validation is the standard approach for evaluating model performance on datasets this small — it maximizes the number of training examples used in each fold.

**Moisture plasticization** — Tg drops significantly with residual moisture. A 1% increase in water content typically depresses Tg by ~10 °C for most sugars. The tool outputs moisture-adjusted Tg curves so scientists can account for this when designing freeze-drying cycles.

---

## Limitations

- The RF model is trained on ~14 compounds. Predictions for excipients structurally dissimilar to the training set (sugars and amino acids) should be treated with caution — check the uncertainty estimate and nearest-neighbor distance in the Molecule Explorer tab.
- The Gordon-Taylor `k` approximation is not experimentally fitted. For final formulation decisions, DSC validation of predicted Tg values is recommended.
- Ternary mixtures (sugar + amino acid + buffer) are not yet supported. Binary optimization only.
- Moisture plasticization curves use a simplified linear approximation, not a fitted plasticization model.

---

## Intended Use

CryoForm is an internal R&D pre-screening tool. Its output is a ranked shortlist for experimental prioritization — not a replacement for wet lab validation. Predicted Tg values, uncertainty estimates, and composite scores should be used to guide which formulations to test first, not as final specifications.

---

## Data Sources

- ATHAS database — polymer and small molecule Tg values
- Hancock & Zografi (1997) *Pharm Res* 14:422 — sugar and polyol Tg
- Franks (1990) *Cryo-Letters* 11:93 — disaccharide amorphous state data
- Carpenter et al. (1997) *Pharm Res* 14:969 — amino acid Tg
- Shamblin et al. (1999) *J Phys Chem B* 103:4113 — mixture reference data
- FDA Inactive Ingredient Database (IID) — biocompatibility approval data
- PubChem — canonical SMILES for all excipients and lipid components

---

## License

Internal use only.
