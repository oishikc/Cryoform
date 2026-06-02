# app.py  — CryoForm v2  (upgraded)
# Run with: streamlit run app.py

import sys, os, io, contextlib, json, textwrap, datetime
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from stage_3.optimizer import run_optimizer, composite_score
from stage_1.hbond_scorer import build_surface_profile, rank_candidates, score_candidate
from stage_2.tg_predictor import train_model, predict_tg, TG_THRESHOLD, compute_descriptors, build_training_data
from stage_2.gordon_taylor import gordon_taylor_tg
from data.candidates import CANDIDATE_SMILES
from data.biocompat_filter import load_approved_ingredients, is_approved
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CryoForm",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background-color: #0A0A0F !important;
    color: #FFFFFF !important;
}

#MainMenu, footer, header { visibility: hidden; }
section[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"]  { display: none !important; }

/* ── Drawer ── */
#cf-overlay {
    position: fixed; inset: 0;
    background: rgba(0,0,0,0.45);
    z-index: 998; opacity: 0; pointer-events: none;
    transition: opacity 0.35s ease;
}
#cf-overlay.open { opacity: 1; pointer-events: all; }
#cf-drawer {
    position: fixed; top: 0; left: 0; bottom: 0;
    width: 320px; background: #111118;
    border-right: 1px solid #1E1E2E;
    z-index: 999; transform: translateX(-100%);
    transition: transform 0.35s cubic-bezier(0.4,0,0.2,1);
    overflow-y: auto; padding: 1.5rem 1.25rem 2rem;
    box-sizing: border-box;
}
#cf-drawer.open { transform: translateX(0); }
#cf-drawer::-webkit-scrollbar { width: 4px; }
#cf-drawer::-webkit-scrollbar-thumb { background: #1E1E2E; border-radius: 4px; }
#cf-toggle {
    position: fixed; top: 1.1rem; left: 1.1rem;
    z-index: 1000; background: #111118;
    border: 1px solid #1E1E2E; border-radius: 10px;
    width: 40px; height: 40px;
    display: flex; align-items: center; justify-content: center;
    cursor: pointer;
    transition: border-color 0.2s, box-shadow 0.2s, background 0.2s;
}
#cf-toggle:hover { border-color:#0066FF; box-shadow:0 0 16px rgba(0,102,255,0.2); background:#16161f; }
#cf-toggle svg { transition: transform 0.35s ease; }
#cf-toggle.open svg { transform: rotate(90deg); }

/* Drawer typography */
.dr-logo { font-size:1.05rem;font-weight:800;color:#fff;letter-spacing:-0.01em;margin-bottom:0.2rem;margin-top:0.25rem; }
.dr-sub  { font-size:0.72rem;color:#444;margin-bottom:1.5rem; }
.dr-section { font-size:0.68rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:#0066FF;margin:1.25rem 0 0.6rem; }
.dr-note    { font-size:0.72rem;color:#444;margin-top:-0.4rem;margin-bottom:0.75rem; }
.dr-divider { border:none;border-top:1px solid #1E1E2E;margin:1.25rem 0; }
.dr-total-ok  { font-size:0.78rem;font-weight:600;color:#00C864;margin:0.25rem 0 0.75rem; }
.dr-total-err { font-size:0.78rem;font-weight:600;color:#FF4444;margin:0.25rem 0 0.75rem; }
.stSlider label,.stSelectbox label { color:#CCC!important;font-size:0.82rem!important;font-weight:600!important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap:0.25rem;background:transparent!important;border-bottom:1px solid #1E1E2E!important;padding-bottom:0!important;margin-bottom:2rem; }
.stTabs [data-baseweb="tab"] { background:transparent!important;border:1px solid transparent!important;border-bottom:none!important;border-radius:8px 8px 0 0!important;color:#666!important;font-family:'Plus Jakarta Sans',sans-serif!important;font-weight:600!important;font-size:0.88rem!important;padding:0.6rem 1.4rem!important;transition:color 0.15s,border-color 0.15s!important; }
.stTabs [data-baseweb="tab"]:hover { color:#CCC!important;border-color:#1E1E2E!important; }
.stTabs [aria-selected="true"] { background:#111118!important;border-color:#1E1E2E!important;border-bottom-color:#111118!important;color:#fff!important; }
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"] { display:none!important; }

/* Hero */
.hero { padding:3.5rem 0 2.5rem;border-bottom:1px solid #1E1E2E;margin-bottom:3rem; }
.hero-badge { display:inline-block;background:rgba(0,102,255,0.15);color:#4D94FF;border:1px solid rgba(0,102,255,0.3);font-size:0.7rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;padding:0.3rem 0.9rem;border-radius:999px;margin-bottom:1.25rem; }
.hero-title { font-size:3.5rem;font-weight:800;letter-spacing:-0.03em;color:#fff;line-height:1.05;margin:0 0 1rem; }
.hero-accent { color:#0066FF; }
.hero-sub { font-size:1.05rem;color:#888;max-width:580px;line-height:1.7; }

/* Section */
.section-label { font-size:0.68rem;font-weight:700;letter-spacing:0.15em;text-transform:uppercase;color:#0066FF;margin-bottom:0.5rem; }
.section-title { font-size:1.75rem;font-weight:800;letter-spacing:-0.02em;color:#fff;margin-bottom:1.75rem; }

/* Step cards */
.steps-grid { display:grid;grid-template-columns:repeat(3,1fr);gap:1.25rem;margin-bottom:0.5rem; }
.step-card { background:#111118;border:1px solid #1E1E2E;border-radius:16px;padding:1.5rem;transition:border-color 0.2s,box-shadow 0.2s; }
.step-card:hover { border-color:#0066FF;box-shadow:0 0 24px rgba(0,102,255,0.08); }
.step-num { background:#0066FF;color:#fff;font-weight:800;font-size:0.8rem;width:2rem;height:2rem;border-radius:50%;display:flex;align-items:center;justify-content:center;margin-bottom:1rem; }
.step-title { font-size:0.95rem;font-weight:700;color:#fff;margin-bottom:0.4rem; }
.step-desc  { font-size:0.82rem;color:#666;line-height:1.6; }

/* Metric cards */
.metric-card { background:#111118;border:1px solid #1E1E2E;border-radius:14px;padding:1.25rem 1.5rem;transition:border-color 0.2s; }
.metric-card:hover { border-color:#0066FF; }
.metric-value { font-size:2.25rem;font-weight:800;color:#0066FF;letter-spacing:-0.03em;line-height:1; }
.metric-label { font-size:0.78rem;color:#666;margin-top:0.4rem;font-weight:500;text-transform:uppercase;letter-spacing:0.06em; }

/* Recipe cards */
.recipe-card { background:#111118;border:1px solid #1E1E2E;border-radius:16px;padding:1.75rem 2rem;margin-bottom:1rem;transition:border-color 0.2s,box-shadow 0.2s; }
.recipe-card:hover { border-color:#0066FF;box-shadow:0 0 32px rgba(0,102,255,0.1); }
.recipe-card.rank1 { border-color:rgba(0,102,255,0.4); }
.recipe-rank  { font-size:0.68rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:#0066FF;margin-bottom:0.5rem; }
.recipe-title { font-size:1.3rem;font-weight:700;color:#fff;letter-spacing:-0.01em;margin-bottom:1rem; }
.recipe-meta  { display:flex;gap:2.5rem;flex-wrap:wrap; }
.recipe-meta-item { font-size:0.82rem;color:#666; }
.recipe-meta-item span { font-weight:700;color:#fff; }

/* Pills */
.pill-pass { display:inline-block;background:rgba(0,200,100,0.12);color:#00C864;border:1px solid rgba(0,200,100,0.25);font-size:0.68rem;font-weight:700;padding:0.2rem 0.65rem;border-radius:999px;margin-left:0.75rem;vertical-align:middle; }
.pill-warn { display:inline-block;background:rgba(255,170,0,0.12);color:#FFAA00;border:1px solid rgba(255,170,0,0.25);font-size:0.68rem;font-weight:700;padding:0.2rem 0.65rem;border-radius:999px;margin-left:0.75rem;vertical-align:middle; }
.pill-fail { display:inline-block;background:rgba(255,68,68,0.12);color:#FF4444;border:1px solid rgba(255,68,68,0.25);font-size:0.68rem;font-weight:700;padding:0.2rem 0.65rem;border-radius:999px;margin-left:0.75rem;vertical-align:middle; }

/* Divider */
.divider { border:none;border-top:1px solid #1E1E2E;margin:3rem 0; }

/* Idle state */
.idle-block { text-align:center;padding:5rem 0;color:#444; }
.idle-icon  { font-size:3rem;margin-bottom:1rem; }
.idle-title { font-size:1.1rem;font-weight:600;color:#888;margin-bottom:0.4rem; }
.idle-sub   { font-size:0.85rem;color:#555; }

/* Chart containers */
.chart-wrap { background:#111118;border:1px solid #1E1E2E;border-radius:16px;padding:1.25rem; }

/* Explorer cards */
.ex-card { background:#111118;border:1px solid #1E1E2E;border-radius:14px;padding:1.25rem 1.5rem;margin-bottom:1rem; }
.ex-card-label { font-size:0.68rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:#0066FF;margin-bottom:0.85rem; }
.ex-desc-table { width:100%;border-collapse:collapse; }
.ex-desc-table td { font-size:0.875rem;padding:0.42rem 0;border-bottom:1px solid #1E1E2E;color:#D1D5DB; }
.ex-desc-table td:first-child { color:#555;width:58%; }
.ex-desc-table tr:last-child td { border-bottom:none; }
.tg-badge { font-size:2rem;font-weight:800;letter-spacing:-0.03em;line-height:1; }
.tg-conf  { font-size:0.78rem;color:#555;margin-top:0.4rem; }
.hb-score-val { font-size:2rem;font-weight:800;letter-spacing:-0.03em;line-height:1; }
.hb-bar-track { background:#1E1E2E;border-radius:6px;height:10px;width:100%;overflow:hidden;margin-top:0.75rem; }

/* Uncertainty band indicator */
.unc-row { display:flex;align-items:center;gap:0.75rem;margin-top:0.6rem; }
.unc-label { font-size:0.75rem;color:#555;min-width:5rem; }
.unc-bar-outer { flex:1;background:#1E1E2E;border-radius:4px;height:8px;position:relative;overflow:hidden; }
.unc-bar-inner { height:100%;border-radius:4px; }
.unc-val { font-size:0.75rem;color:#888;min-width:3.5rem;text-align:right; }

/* Model panel */
.model-stat { background:#111118;border:1px solid #1E1E2E;border-radius:10px;padding:1rem 1.25rem;margin-bottom:0.75rem; }
.model-stat-label { font-size:0.68rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#0066FF;margin-bottom:0.3rem; }
.model-stat-val { font-size:1.4rem;font-weight:800;color:#fff;letter-spacing:-0.02em; }

/* Custom candidate */
.custom-badge { display:inline-block;background:rgba(255,170,0,0.12);color:#FFAA00;border:1px solid rgba(255,170,0,0.25);font-size:0.68rem;font-weight:700;padding:0.2rem 0.75rem;border-radius:999px;margin-bottom:0.75rem; }

/* Moisture table */
.moisture-table { width:100%;border-collapse:collapse;font-size:0.85rem; }
.moisture-table th { color:#555;font-weight:600;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.08em;padding:0.5rem 0;border-bottom:1px solid #1E1E2E;text-align:left; }
.moisture-table td { padding:0.45rem 0;border-bottom:1px solid #111;color:#CCC; }
.moisture-table tr:last-child td { border-bottom:none; }

/* Report section */
.report-box { background:#111118;border:1px solid #1E1E2E;border-radius:14px;padding:1.5rem 2rem;font-family:monospace;font-size:0.82rem;color:#AAA;white-space:pre-wrap;line-height:1.7; }

/* Footer */
.footer { text-align:center;padding:2rem 0 1rem;font-size:0.78rem;color:#333; }
.main-content { transition:margin-left 0.35s cubic-bezier(0.4,0,0.2,1); }

/* Stacked info card */
.info-grid { display:grid;grid-template-columns:1fr 1fr;gap:0.75rem;margin-bottom:0.75rem; }
.info-cell { background:#0D0D14;border:1px solid #1E1E2E;border-radius:10px;padding:0.75rem 1rem; }
.info-cell-label { font-size:0.65rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#444;margin-bottom:0.2rem; }
.info-cell-val   { font-size:1rem;font-weight:700;color:#fff; }

/* Upload area */
.stFileUploader > div { border:1px dashed #1E1E2E !important;border-radius:10px!important;background:#111118!important; }
</style>
""", unsafe_allow_html=True)

# ── Drawer HTML + JS ───────────────────────────────────────────────────────────
st.markdown("""
<button id="cf-toggle" onclick="toggleDrawer()" title="Toggle settings panel">
  <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
    <rect y="2"  width="18" height="2" rx="1" fill="#FFF"/>
    <rect y="8"  width="18" height="2" rx="1" fill="#FFF"/>
    <rect y="14" width="18" height="2" rx="1" fill="#FFF"/>
  </svg>
</button>
<div id="cf-overlay" onclick="closeDrawer()"></div>
<div id="cf-drawer">
  <div class="dr-logo">❄️ CryoForm</div>
  <div class="dr-sub">Lyoprotectant Design Pipeline v2</div>
  <hr class="dr-divider">
</div>
<script>
function openDrawer()  { ['cf-drawer','cf-overlay','cf-toggle'].forEach(id=>document.getElementById(id).classList.add('open')); }
function closeDrawer() { ['cf-drawer','cf-overlay','cf-toggle'].forEach(id=>document.getElementById(id).classList.remove('open')); }
function toggleDrawer(){ document.getElementById('cf-drawer').classList.contains('open')?closeDrawer():openDrawer(); }
</script>
""", unsafe_allow_html=True)

# ── Base plot layout ────────────────────────────────────────────────────────────
BASE_LAYOUT = dict(
    plot_bgcolor="#111118", paper_bgcolor="#111118",
    font=dict(family="Plus Jakarta Sans", color="#888", size=11),
    margin=dict(l=10, r=20, t=40, b=20), height=360,
)

def plot_layout(**overrides):
    """Merge BASE_LAYOUT with per-chart overrides safely."""
    return {**BASE_LAYOUT, **overrides}

# ── Cached resources ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_model():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        model, scaler, feature_cols = train_model()
    return model, scaler, feature_cols

@st.cache_resource(show_spinner=False)
def get_approved():
    return load_approved_ingredients()

@st.cache_data(show_spinner=False)
def get_training_data():
    return build_training_data()

# ── Drawer controls ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
div[data-testid="stVerticalBlock"]:has(> div[data-testid="stVerticalBlock"]
    > div[data-testid="element-container"] #drawer-anchor) {
    position:fixed; top:80px; left:0; width:318px;
    z-index:1001; padding:0 1.25rem;
    box-sizing:border-box; max-height:calc(100vh - 100px); overflow-y:auto;
}
</style>
""", unsafe_allow_html=True)
st.markdown('<span id="drawer-anchor" style="display:none"></span>', unsafe_allow_html=True)

st.markdown('<div class="dr-section">LNP Composition</div>', unsafe_allow_html=True)
st.markdown('<div class="dr-note">Molar fractions — must sum to 100%</div>', unsafe_allow_html=True)

lipid_choice = st.selectbox("Ionizable Lipid", ["ALC-0315", "SM-102", "DLin-MC3-DMA"])
f_ion  = st.slider("Ionizable Lipid mol%", 30, 60, 46)
f_dspc = st.slider("DSPC mol%",             5, 20,  9)
f_chol = st.slider("Cholesterol mol%",      30, 55, 43)
f_peg  = st.slider("PEG-Lipid mol%",         1,  5,  2)

total = f_ion + f_dspc + f_chol + f_peg
if total != 100:
    st.markdown(f'<div class="dr-total-err">⚠ Total: {total}% — must be 100%</div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="dr-total-ok">✓ Total: 100%</div>', unsafe_allow_html=True)

st.markdown('<hr class="dr-divider">', unsafe_allow_html=True)
st.markdown('<div class="dr-section">Optimizer Weights</div>', unsafe_allow_html=True)
alpha = st.slider("α — H-bond",    0.0, 1.0, 0.5, 0.05)
beta  = st.slider("β — Tg margin", 0.0, 1.0, 0.4, 0.05)
gamma = st.slider("γ — Biocompat", 0.0, 1.0, 0.1, 0.05)

st.markdown('<hr class="dr-divider">', unsafe_allow_html=True)
st.markdown('<div class="dr-section">Tg Threshold</div>', unsafe_allow_html=True)
tg_threshold_override = st.slider("Min Tg (°C)", 0, 80, int(TG_THRESHOLD), 5)

st.markdown('<hr class="dr-divider">', unsafe_allow_html=True)
st.markdown('<div class="dr-section">Top N Recipes</div>', unsafe_allow_html=True)
top_n = st.slider("Recipes to show", 1, 10, 5)

st.markdown('<hr class="dr-divider">', unsafe_allow_html=True)
run_btn = st.button("Run CryoForm ▶", use_container_width=True, type="primary")

st.markdown('<div style="height:3.5rem"></div>', unsafe_allow_html=True)

# ── Hero ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-badge">Internal R&D · ML-Powered · v2</div>
  <h1 class="hero-title">Cryo<span class="hero-accent">Form</span></h1>
  <p class="hero-sub">
    Physics-informed ML pipeline for designing optimal lyoprotectant formulations
    for mRNA-LNP therapeutics. H-bond scoring · Tg prediction with uncertainty ·
    Gordon-Taylor optimization · moisture plasticization · PDF report export.
  </p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ────────────────────────────────────────────────────────────────────────
tab_pipeline, tab_explorer, tab_model, tab_custom, tab_data = st.tabs([
    "Pipeline", "Molecule Explorer", "Model Transparency", "Custom Candidate", "Data Management"
])

# ════════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════════

def predict_tg_with_uncertainty(smiles, model, scaler, feature_cols):
    """Return (mean_tg, std_tg) from RF tree variance."""
    desc = compute_descriptors(smiles)
    if desc is None:
        return None, None
    X = pd.DataFrame([desc])[feature_cols]
    X_scaled = scaler.transform(X)
    tree_preds = np.array([tree.predict(X_scaled)[0] for tree in model.estimators_])
    return round(float(tree_preds.mean()), 1), round(float(tree_preds.std()), 1)

def tg_moisture_curve(tg_dry, water_tg=-137.0, points=20):
    """
    Gordon-Taylor plasticization: Tg(mix) vs % residual moisture.
    k_water ~ 6.5 (typical for sugars, Roos 1993).
    """
    k = 6.5
    moistures = np.linspace(0, 10, points)
    tgs = []
    for pct in moistures:
        w_water = pct / 100.0
        w_excip = 1.0 - w_water
        tg1k = (tg_dry + 273.15)
        tg2k = (water_tg + 273.15)
        tg_mix_k = (w_excip * tg1k + k * w_water * tg2k) / (w_excip + k * w_water)
        tgs.append(round(tg_mix_k - 273.15, 1))
    return moistures.tolist(), tgs

def find_nearest_training(smiles, X_train_df, names, feature_cols, n=3):
    """Return n nearest training compounds by Euclidean descriptor distance."""
    desc = compute_descriptors(smiles)
    if desc is None:
        return []
    scaler_local = StandardScaler()
    X_train_scaled = scaler_local.fit_transform(X_train_df)
    X_query = scaler_local.transform(pd.DataFrame([desc])[feature_cols])
    nbrs = NearestNeighbors(n_neighbors=min(n, len(X_train_df))).fit(X_train_scaled)
    dists, idxs = nbrs.kneighbors(X_query)
    return [(names[i], round(dists[0][j], 3)) for j, i in enumerate(idxs[0])]

def generate_report_text(recipes, formulation, surface_profile, tg_predictions,
                          alpha, beta, gamma, top_n, tg_threshold, model_mae):
    """Generate plain-text report for export."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "=" * 64,
        "  CRYOFORM — LYOPROTECTANT FORMULATION REPORT",
        f"  Generated: {now}",
        "=" * 64,
        "",
        "LNP FORMULATION",
        "-" * 40,
    ]
    for lipid, frac in formulation.items():
        lines.append(f"  {lipid:<20} {frac*100:.1f} mol%")
    lines += [
        "",
        "LNP SURFACE PROFILE",
        "-" * 40,
        f"  HBD:         {surface_profile['HBD']:.3f}",
        f"  HBA:         {surface_profile['HBA']:.3f}",
        f"  TPSA:        {surface_profile['TPSA']:.2f} Å²",
        f"  HBD density: {surface_profile['HBD_density']:.4f}",
        f"  HBA density: {surface_profile['HBA_density']:.4f}",
        "",
        "OPTIMIZER SETTINGS",
        "-" * 40,
        f"  α (H-bond weight):    {alpha}",
        f"  β (Tg margin weight): {beta}",
        f"  γ (Biocompat bonus):  {gamma}",
        f"  Tg threshold:         {tg_threshold} °C",
        f"  RF model LOO-MAE:     {model_mae:.1f} °C",
        "",
        f"CANDIDATES PASSING GATES ({len(tg_predictions)})",
        "-" * 40,
    ]
    for name, tg in sorted(tg_predictions.items(), key=lambda x: -x[1]):
        lines.append(f"  {name:<20} Tg = {tg:.1f} °C")
    lines += ["", f"TOP {top_n} RECIPES", "-" * 40, ""]
    for i, r in enumerate(recipes, 1):
        lines += [
            f"  Rank {i}: {r['component_1']} {int(r['w1']*100)}% / {r['component_2']} {int(r['w2']*100)}%",
            f"    Predicted Tg_mix : {r['tg_mix']:.1f} °C",
            f"    H-bond score     : {r['hbond_score']:.4f}",
            f"    Biocompatible    : {'Yes' if r['biocompat'] else 'No'}",
            f"    Composite score  : {r['composite']:.4f}",
            "",
        ]
    lines += [
        "NOTES",
        "-" * 40,
        "  Tg predictions from Random Forest (n=14 non-crystallizers).",
        "  Treat as relative ranking, not absolute values.",
        "  Gordon-Taylor k estimated via Couchman-Karasz (Tg1/Tg2 in K).",
        "  Moisture plasticization modelled with k_water=6.5 (Roos 1993).",
        "  For research use only.",
        "=" * 64,
    ]
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — PIPELINE
# ════════════════════════════════════════════════════════════════════════════════
with tab_pipeline:

    st.markdown('<div class="section-label">System Architecture</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">How It Works</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="steps-grid">
      <div class="step-card"><div class="step-num">1</div><div class="step-title">H-Bond Scoring</div><div class="step-desc">RDKit maps donor/acceptor density on the LNP surface weighted by molar composition. Candidates scored by complementarity.</div></div>
      <div class="step-card"><div class="step-num">2</div><div class="step-title">Tg Prediction + Uncertainty</div><div class="step-desc">Random Forest predicts Tg with tree-variance uncertainty bounds. Candidates below threshold are eliminated.</div></div>
      <div class="step-card"><div class="step-num">3</div><div class="step-title">Formulation Optimizer</div><div class="step-desc">Gordon-Taylor binary scan across weight fractions. Composite score ranks recipes. Moisture plasticization curve included.</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    if run_btn:
        st.session_state["pipeline_ran"] = True
        st.session_state["pipeline_fracs"] = (f_ion, f_dspc, f_chol, f_peg)
        st.session_state["pipeline_weights"] = (alpha, beta, gamma)

    pipeline_ready = st.session_state.get("pipeline_ran", False)

    if not pipeline_ready:
        st.markdown("""
        <div class="idle-block">
          <div class="idle-icon">❄️</div>
          <div class="idle-title">Set your LNP composition in the panel</div>
          <div class="idle-sub">Press <strong>☰</strong> (top-left) then <strong>Run CryoForm ▶</strong></div>
        </div>
        """, unsafe_allow_html=True)

    else:
        if total != 100:
            st.error("Molar fractions must sum to 100%. Adjust sliders in the drawer.")
        else:
            formulation = {
                "ALC-0315":    f_ion  / 100,
                "DSPC":        f_dspc / 100,
                "Cholesterol": f_chol / 100,
                "ALC-0159":    f_peg  / 100,
            }

            with st.spinner("Running CryoForm pipeline..."):
                model, scaler, feature_cols = get_model()
                approved = get_approved()
                surface_profile = build_surface_profile(formulation)
                hbond_df = rank_candidates(CANDIDATE_SMILES, surface_profile)
                hbond_scores_map = dict(zip(hbond_df["name"], hbond_df["hbond_score"]))

                tg_predictions = {}
                tg_uncertainty = {}
                for name, smiles in CANDIDATE_SMILES.items():
                    tg_mean, tg_std = predict_tg_with_uncertainty(smiles, model, scaler, feature_cols)
                    if tg_mean is not None and tg_mean > tg_threshold_override and is_approved(name, approved):
                        tg_predictions[name] = tg_mean
                        tg_uncertainty[name] = tg_std

                with contextlib.redirect_stdout(io.StringIO()):
                    recipes = run_optimizer(formulation, alpha=alpha, beta=beta, gamma=gamma, top_n=top_n)

            # Compute LOO MAE for report
            X_df, y_train, train_names = get_training_data()
            from sklearn.model_selection import LeaveOneOut, cross_val_score
            loo_scores = cross_val_score(model, scaler.transform(X_df), y_train,
                                         cv=LeaveOneOut(), scoring="neg_mean_absolute_error")
            model_mae = -loo_scores.mean()

            # ── Metrics ─────────────────────────────────────────────────────
            st.markdown('<div class="section-label">Pipeline Summary</div>', unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            metrics = [
                (str(len(CANDIDATE_SMILES)), "Candidates screened"),
                (str(len(tg_predictions)),   "Passed both gates"),
                (f"{recipes[0]['tg_mix']:.0f}°C", "Top recipe Tg_mix"),
                (f"{recipes[0]['composite']:.3f}", "Top composite score"),
            ]
            for col, (val, label) in zip([m1, m2, m3, m4], metrics):
                with col:
                    st.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

            st.markdown('<hr class="divider">', unsafe_allow_html=True)

            # ── Recipes ──────────────────────────────────────────────────────
            st.markdown('<div class="section-label">Optimizer Output</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Recommended Recipes</div>', unsafe_allow_html=True)

            for i, r in enumerate(recipes, 1):
                pill = '<span class="pill-pass">FDA IID ✓</span>' if r["biocompat"] else '<span class="pill-warn">Check biocompat</span>'
                rank_cls = "rank1" if i == 1 else ""

                # uncertainty for both components
                u1 = tg_uncertainty.get(r["component_1"], None)
                u2 = tg_uncertainty.get(r["component_2"], None)
                unc_str = ""
                if u1 is not None and u2 is not None:
                    unc_str = f'<div class="recipe-meta-item" style="color:#444;font-size:0.78rem;">Tg uncertainty: <span style="color:#666;">±{u1:.1f}°C / ±{u2:.1f}°C</span></div>'

                st.markdown(f"""
                <div class="recipe-card {rank_cls}">
                  <div class="recipe-rank">Rank {i}</div>
                  <div class="recipe-title">
                    {r['component_1']} {int(r['w1']*100)}% &nbsp;/&nbsp; {r['component_2']} {int(r['w2']*100)}%
                    {pill}
                  </div>
                  <div class="recipe-meta">
                    <div class="recipe-meta-item">Predicted Tg_mix &nbsp;<span>{r['tg_mix']:.1f}°C</span></div>
                    <div class="recipe-meta-item">H-bond score &nbsp;<span>{r['hbond_score']:.4f}</span></div>
                    <div class="recipe-meta-item">Composite score &nbsp;<span>{r['composite']:.4f}</span></div>
                    {unc_str}
                  </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('<hr class="divider">', unsafe_allow_html=True)

            # ── Charts ───────────────────────────────────────────────────────
            st.markdown('<div class="section-label">Analytics</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Data & Visualization</div>', unsafe_allow_html=True)

            c1, c2 = st.columns(2)

            with c1:
                bar_colors = ["#0066FF" if n in tg_predictions else "#1E1E2E" for n in hbond_df["name"]]
                fig1 = go.Figure(go.Bar(
                    x=hbond_df["hbond_score"], y=hbond_df["name"],
                    orientation="h",
                    marker=dict(color=bar_colors, line=dict(width=0)),
                    hovertemplate="<b>%{y}</b><br>Score: %{x:.4f}<extra></extra>",
                ))
                fig1.update_layout(**plot_layout(
                    height=420,
                    title=dict(text="H-Bond Complementarity Scores", font=dict(color="#fff", size=13)),
                    xaxis=dict(title="Score", showgrid=True, gridcolor="#1E1E2E", color="#888"),
                    yaxis=dict(autorange="reversed", color="#fff"),
                ))
                st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
                st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})
                st.markdown('</div>', unsafe_allow_html=True)

            with c2:
                all_names, all_tg, all_std, all_pass = [], [], [], []
                for name, smiles in CANDIDATE_SMILES.items():
                    tg_m, tg_s = predict_tg_with_uncertainty(smiles, model, scaler, feature_cols)
                    if tg_m is not None:
                        all_names.append(name)
                        all_tg.append(tg_m)
                        all_std.append(tg_s if tg_s else 0)
                        all_pass.append("Pass" if name in tg_predictions else "Fail")

                tg_df = pd.DataFrame({"name": all_names, "tg": all_tg, "std": all_std, "status": all_pass})
                tg_df["hbond"] = tg_df["name"].map(hbond_scores_map).fillna(0)

                fig2 = go.Figure()
                for status, color in [("Pass", "#0066FF"), ("Fail", "#333344")]:
                    sub = tg_df[tg_df["status"] == status]
                    fig2.add_trace(go.Scatter(
                        x=sub["hbond"], y=sub["tg"],
                        error_y=dict(type="data", array=sub["std"].tolist(), color="#0066FF" if status=="Pass" else "#333", thickness=1),
                        mode="markers+text",
                        text=sub["name"],
                        textposition="top center",
                        textfont=dict(size=9, color="#888"),
                        marker=dict(size=10, color=color, line=dict(width=1, color="#0066FF" if status=="Pass" else "#333")),
                        name=status,
                        hovertemplate="<b>%{text}</b><br>H-bond: %{x:.4f}<br>Tg: %{y:.1f}°C<extra></extra>",
                    ))
                fig2.add_hline(y=tg_threshold_override, line_dash="dash", line_color="#FF4444",
                               annotation_text=f"Tg threshold ({tg_threshold_override}°C)",
                               annotation_font=dict(color="#FF4444", size=10))
                fig2.update_layout(**plot_layout(
                    height=420,
                    title=dict(text="Predicted Tg vs H-bond Score (with uncertainty)", font=dict(color="#fff", size=13)),
                    showlegend=True,
                    legend=dict(font=dict(size=10, color="#888"), bgcolor="rgba(0,0,0,0)"),
                    xaxis=dict(title="H-bond score", showgrid=True, gridcolor="#1E1E2E", color="#888"),
                    yaxis=dict(title="Predicted Tg (°C)", showgrid=True, gridcolor="#1E1E2E", color="#888"),
                ))
                st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
                st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
                st.markdown('</div>', unsafe_allow_html=True)

            # ── Gordon-Taylor top recipe ──────────────────────────────────────
            st.markdown("#### Gordon–Taylor Curve — Top Recipe")
            r = recipes[0]
            tg1 = tg_predictions.get(r["component_1"], 100)
            tg2 = tg_predictions.get(r["component_2"], 100)
            w1_range = np.arange(0.0, 1.01, 0.01)
            tg_curve = [gordon_taylor_tg(tg1, tg2, w) for w in w1_range]

            # uncertainty envelope using per-component std
            std1 = tg_uncertainty.get(r["component_1"], 0)
            std2 = tg_uncertainty.get(r["component_2"], 0)
            tg_upper = [gordon_taylor_tg(tg1 + std1, tg2 + std2, w) for w in w1_range]
            tg_lower = [gordon_taylor_tg(tg1 - std1, tg2 - std2, w) for w in w1_range]

            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=np.concatenate([w1_range*100, w1_range[::-1]*100]),
                y=tg_upper + tg_lower[::-1],
                fill="toself", fillcolor="rgba(0,102,255,0.08)",
                line=dict(color="rgba(0,0,0,0)"), showlegend=True, name="Uncertainty band",
                hoverinfo="skip",
            ))
            fig3.add_trace(go.Scatter(
                x=w1_range*100, y=tg_curve, mode="lines",
                line=dict(color="#0066FF", width=2.5), name="Tg_mix (predicted)",
                hovertemplate="w₁=%{x:.0f}%  Tg=%{y:.1f}°C<extra></extra>",
            ))
            fig3.add_hline(y=tg_threshold_override, line_dash="dash", line_color="#FF4444",
                           annotation_text=f"Stability threshold ({tg_threshold_override}°C)",
                           annotation_font=dict(color="#FF4444", size=10))
            fig3.add_vline(x=r["w1"]*100, line_dash="dot", line_color="#0066FF",
                           annotation_text=f"Optimal ({int(r['w1']*100)}%)",
                           annotation_font=dict(color="#4D94FF", size=10))
            fig3.update_layout(**plot_layout(
                height=320,
                xaxis=dict(title=f"Weight fraction {r['component_1']} (%)", showgrid=True, gridcolor="#1E1E2E", color="#888"),
                yaxis=dict(title="Predicted Tg_mix (°C)", showgrid=True, gridcolor="#1E1E2E", color="#888"),
                legend=dict(font=dict(size=10, color="#888"), bgcolor="rgba(0,0,0,0)"),
            ))
            st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
            st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

            # ── Moisture Plasticization ───────────────────────────────────────
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown('<div class="section-label">Stability Analysis</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Moisture Plasticization</div>', unsafe_allow_html=True)
            st.markdown(
                "<p style='color:#666;font-size:0.88rem;max-width:600px;line-height:1.7;margin-top:-1rem;margin-bottom:1.5rem;'>"
                "Tg drops with residual moisture. This curve shows how the predicted dry-state Tg of the top recipe "
                "degrades as water content increases. Primary drying target should keep Tg(mix) ≥ 20°C above shelf temperature. "
                "k_water = 6.5 (Roos 1993 sugar model)."
                "</p>", unsafe_allow_html=True,
            )

            moisture_cols = st.columns(2)
            for ci, comp_name in enumerate([r["component_1"], r["component_2"]]):
                tg_comp = tg_predictions.get(comp_name, 80)
                moist_x, moist_y = tg_moisture_curve(tg_comp)
                fig_m = go.Figure()
                fig_m.add_trace(go.Scatter(
                    x=moist_x, y=moist_y, mode="lines",
                    line=dict(color="#0066FF", width=2.2),
                    fill="tozeroy", fillcolor="rgba(0,102,255,0.05)",
                    hovertemplate="Moisture: %{x:.1f}%<br>Tg: %{y:.1f}°C<extra></extra>",
                ))
                fig_m.add_hline(y=tg_threshold_override, line_dash="dash", line_color="#FF4444",
                                annotation_text="Stability threshold", annotation_font=dict(color="#FF4444", size=9))
                # find collapse point
                collapse_pct = next((moist_x[i] for i, v in enumerate(moist_y) if v < tg_threshold_override), None)
                if collapse_pct:
                    fig_m.add_vline(x=collapse_pct, line_dash="dot", line_color="#FFAA00",
                                    annotation_text=f"Collapse >{collapse_pct:.1f}%",
                                    annotation_font=dict(color="#FFAA00", size=9))
                fig_m.update_layout(**plot_layout(
                    height=280,
                    title=dict(text=f"{comp_name} — Tg vs Moisture", font=dict(color="#fff", size=12)),
                    xaxis=dict(title="Residual moisture (%)", showgrid=True, gridcolor="#1E1E2E", color="#888"),
                    yaxis=dict(title="Tg (°C)", showgrid=True, gridcolor="#1E1E2E", color="#888"),
                ))
                with moisture_cols[ci]:
                    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
                    st.plotly_chart(fig_m, use_container_width=True, config={"displayModeBar": False})
                    st.markdown('</div>', unsafe_allow_html=True)

                    # moisture table
                    table_rows = "".join(
                        f"<tr><td>{moist_x[i]:.1f}%</td><td style='color:{'#FF4444' if moist_y[i]<tg_threshold_override else '#CCC'}'>{moist_y[i]:.1f}°C</td></tr>"
                        for i in range(0, len(moist_x), 4)
                    )
                    st.markdown(f"""
                    <table class="moisture-table">
                      <tr><th>Moisture</th><th>Predicted Tg</th></tr>
                      {table_rows}
                    </table>
                    """, unsafe_allow_html=True)

            # ── Export report ─────────────────────────────────────────────────
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown('<div class="section-label">Export</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Report</div>', unsafe_allow_html=True)

            report_text = generate_report_text(
                recipes, formulation, surface_profile, tg_predictions,
                alpha, beta, gamma, top_n, tg_threshold_override, model_mae
            )
            st.markdown(f'<div class="report-box">{report_text}</div>', unsafe_allow_html=True)
            st.download_button(
                "⬇ Download Report (.txt)",
                data=report_text,
                file_name=f"cryoform_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
            )

            # CSV export of all recipes
            recipes_df = pd.DataFrame(recipes)
            st.download_button(
                "⬇ Download Recipes (.csv)",
                data=recipes_df.to_csv(index=False),
                file_name=f"cryoform_recipes_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
            )

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown("<div class='footer'>CryoForm v2 · Internal R&D · For research use only</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — MOLECULE EXPLORER
# ════════════════════════════════════════════════════════════════════════════════
with tab_explorer:

    st.markdown('<div class="section-label">Excipient Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Molecule Explorer</div>', unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#666;font-size:0.92rem;margin-top:-1rem;margin-bottom:2rem;max-width:580px;line-height:1.7;'>"
        "Inspect molecular descriptors, predicted Tg with uncertainty, H-bond complementarity, "
        "nearest training compounds, and binary mixture behaviour."
        "</p>", unsafe_allow_html=True,
    )

    selected = st.selectbox("Select excipient", sorted(CANDIDATE_SMILES.keys()), key="explorer_select")
    sel_smiles = CANDIDATE_SMILES[selected]

    @st.cache_data(show_spinner=False)
    def _get_descriptors(smiles):
        mol = Chem.MolFromSmiles(smiles)
        return {
            "MW (Da)":         round(Descriptors.MolWt(mol), 2),
            "TPSA (Å²)":       round(Descriptors.TPSA(mol), 2),
            "LogP":            round(Descriptors.MolLogP(mol), 3),
            "HBD":             rdMolDescriptors.CalcNumHBD(mol),
            "HBA":             rdMolDescriptors.CalcNumHBA(mol),
            "Rotatable bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "Rings":           rdMolDescriptors.CalcNumRings(mol),
            "Fraction Csp³":   round(rdMolDescriptors.CalcFractionCSP3(mol), 3),
            "BertzCT":         round(Descriptors.BertzCT(mol), 1),
        }

    @st.cache_data(show_spinner=False)
    def _get_tg_uncertainty(smiles):
        m, s, fc = get_model()
        return predict_tg_with_uncertainty(smiles, m, s, fc)

    @st.cache_data(show_spinner=False)
    def _get_hbond_score(smiles, name, f_ion, f_dspc, f_chol, f_peg):
        form = {"ALC-0315": f_ion/100, "DSPC": f_dspc/100, "Cholesterol": f_chol/100, "ALC-0159": f_peg/100}
        profile = build_surface_profile(form)
        result_df = rank_candidates({name: smiles}, profile)
        return float(result_df["hbond_score"].iloc[0])

    @st.cache_data(show_spinner=False)
    def _get_nearest(smiles, feature_cols_key):
        m, s, fc = get_model()
        X_df, y, names = get_training_data()
        return find_nearest_training(smiles, X_df, names, fc, n=3)

    @st.cache_data(show_spinner=False)
    def _build_gt_traces(sel_name, sel_smiles, all_smiles, f_ion, f_dspc, f_chol, f_peg):
        m, s, fc = get_model()
        approved = get_approved()
        tg_sel, _ = predict_tg_with_uncertainty(sel_smiles, m, s, fc)
        if tg_sel is None:
            return []
        form = {"ALC-0315": f_ion/100, "DSPC": f_dspc/100, "Cholesterol": f_chol/100, "ALC-0159": f_peg/100}
        profile = build_surface_profile(form)
        hbond_full = rank_candidates(all_smiles, profile)
        hbond_map = dict(zip(hbond_full["name"], hbond_full["hbond_score"]))
        w_range = np.linspace(0, 1, 100)
        traces = []
        for partner_name, partner_smiles in all_smiles.items():
            if partner_name == sel_name:
                continue
            if not is_approved(partner_name, approved):
                continue
            tg_partner, _ = predict_tg_with_uncertainty(partner_smiles, m, s, fc)
            if tg_partner is None or tg_partner <= TG_THRESHOLD:
                continue
            tg_mix = [gordon_taylor_tg(tg_sel, tg_partner, w) for w in w_range]
            traces.append({
                "name": partner_name, "x": w_range.tolist(), "y": tg_mix,
                "tg_partner": round(tg_partner, 1),
                "hbond": round(hbond_map.get(partner_name, 0.0), 4),
            })
        traces.sort(key=lambda t: max(t["y"]), reverse=True)
        return traces

    with st.spinner("Computing molecular properties..."):
        descriptors  = _get_descriptors(sel_smiles)
        pred_tg, pred_std = _get_tg_uncertainty(sel_smiles)
        hbond_score  = _get_hbond_score(sel_smiles, selected, f_ion, f_dspc, f_chol, f_peg)
        gt_traces    = _build_gt_traces(selected, sel_smiles, CANDIDATE_SMILES, f_ion, f_dspc, f_chol, f_peg)
        _, _, fc_key = get_model()
        nearest_comps = _get_nearest(sel_smiles, str(fc_key))

    col_left, col_right = st.columns([1.1, 0.9], gap="large")

    with col_left:
        rows_html = "".join(f"<tr><td>{k}</td><td><strong>{v}</strong></td></tr>" for k, v in descriptors.items())
        st.markdown(f"""
        <div class="ex-card">
            <div class="ex-card-label">Molecular Descriptors</div>
            <table class="ex-desc-table">{rows_html}</table>
        </div>
        """, unsafe_allow_html=True)

        # Nearest training compounds
        nbr_rows = "".join(
            f"<tr><td>{name}</td><td style='color:#888'>{dist:.3f}</td></tr>"
            for name, dist in nearest_comps
        )
        st.markdown(f"""
        <div class="ex-card">
            <div class="ex-card-label">Nearest Training Compounds</div>
            <p style='font-size:0.75rem;color:#444;margin-bottom:0.6rem;'>Euclidean distance in descriptor space — lower = more similar to training data → higher prediction confidence.</p>
            <table class="ex-desc-table">
              <tr><td style='color:#555'>Compound</td><td style='color:#555'>Distance</td></tr>
              {nbr_rows}
            </table>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        tg_display = f"{pred_tg:+.1f} °C" if pred_tg is not None else "N/A"
        tg_colour  = "#0066FF" if (pred_tg is not None and pred_tg > tg_threshold_override) else "#FF4444"
        pass_pill  = '<span class="pill-pass">Passes threshold</span>' if (pred_tg is not None and pred_tg > tg_threshold_override) else '<span class="pill-warn">Below threshold</span>'

        std_pct = min(int((pred_std or 0) / max(abs(pred_tg or 1), 1) * 100), 100)
        conf_color = "#00C864" if std_pct < 15 else ("#FFAA00" if std_pct < 35 else "#FF4444")
        conf_label = "High" if std_pct < 15 else ("Medium" if std_pct < 35 else "Low")

        st.markdown(f"""
        <div class="ex-card">
            <div class="ex-card-label">Predicted Glass Transition Temperature</div>
            <div class="tg-badge" style="color:{tg_colour};">{tg_display}</div>
            {pass_pill}
            <div class="unc-row" style="margin-top:0.9rem;">
              <span class="unc-label">±{pred_std:.1f}°C std</span>
              <div class="unc-bar-outer">
                <div class="unc-bar-inner" style="width:{std_pct}%;background:{conf_color};"></div>
              </div>
              <span class="unc-val" style="color:{conf_color};">{conf_label} conf.</span>
            </div>
            <p class="tg-conf">Random Forest · {len(list(CANDIDATE_SMILES))} candidates · uncertainty from tree variance.</p>
        </div>
        """, unsafe_allow_html=True)

        clamped   = max(0.0, min(1.0, hbond_score))
        pct       = int(clamped * 100)
        hb_colour = "#0066FF" if clamped >= 0.6 else ("#F59E0B" if clamped >= 0.35 else "#EF4444")
        st.markdown(f"""
        <div class="ex-card">
            <div class="ex-card-label">H-Bond Complementarity vs Current LNP</div>
            <div class="hb-score-val" style="color:{hb_colour};">{clamped:.4f}</div>
            <div class="hb-bar-track"><div style="background:{hb_colour};width:{pct}%;height:100%;border-radius:6px;"></div></div>
            <p class="tg-conf" style="margin-top:0.5rem;">Recomputed from current drawer LNP composition.</p>
        </div>
        """, unsafe_allow_html=True)

        # Moisture plasticization mini-card
        if pred_tg is not None:
            moist_x, moist_y = tg_moisture_curve(pred_tg)
            collapse_pct_m = next((moist_x[i] for i, v in enumerate(moist_y) if v < tg_threshold_override), None)
            collapse_str = f"{collapse_pct_m:.1f}%" if collapse_pct_m else ">10%"
            st.markdown(f"""
            <div class="ex-card">
                <div class="ex-card-label">Moisture Sensitivity</div>
                <div class="info-grid">
                  <div class="info-cell"><div class="info-cell-label">Dry-state Tg</div><div class="info-cell-val">{pred_tg:.1f}°C</div></div>
                  <div class="info-cell"><div class="info-cell-label">Collapse moisture</div><div class="info-cell-val">{collapse_str}</div></div>
                </div>
                <p class="tg-conf">Tg drops below stability threshold above collapse moisture. k_water = 6.5.</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Gordon-Taylor multi-partner chart
    st.markdown(
        f'<div class="section-title" style="font-size:1.3rem;margin-bottom:1rem;">'
        f'Gordon–Taylor Curves — <span style="color:#0066FF;">{selected}</span> vs Passing Candidates'
        f'</div>', unsafe_allow_html=True,
    )

    if not gt_traces:
        st.markdown('<div class="idle-block"><div class="idle-icon">📊</div><div class="idle-title">No passing partners found</div></div>', unsafe_allow_html=True)
    else:
        n = len(gt_traces)
        colors = [f"hsl({int(200 + i*100/max(n-1,1))}, 75%, {int(55+i*15/max(n-1,1))}%)" for i in range(n)]
        fig_gt = go.Figure()
        for i, tr in enumerate(gt_traces):
            fig_gt.add_trace(go.Scatter(
                x=tr["x"], y=tr["y"], mode="lines",
                name=f"{tr['name']}  (Tg {tr['tg_partner']:+.0f}°C · H-bond {tr['hbond']:.3f})",
                line=dict(color=colors[i], width=1.8),
                hovertemplate=f"<b>{selected} / {tr['name']}</b><br>w₁(sel) = %{{x:.2f}}<br>Tg(mix) = %{{y:.1f}} °C<extra></extra>",
            ))
        fig_gt.add_hline(y=tg_threshold_override, line_dash="dash", line_color="#FF4444",
                         annotation_text=f"Stability threshold ({tg_threshold_override}°C)",
                         annotation_font=dict(color="#FF4444", size=10))
        fig_gt.update_layout(**plot_layout(
            height=420,
            xaxis=dict(title=f"w₁  ({selected} weight fraction)", title_font=dict(color="#666", size=11), tickfont=dict(color="#666"), showgrid=True, gridcolor="#1E1E2E", zeroline=False, range=[0, 1]),
            yaxis=dict(title="Tg(mix)  (°C)", title_font=dict(color="#666", size=11), tickfont=dict(color="#666"), showgrid=True, gridcolor="#1E1E2E", zeroline=False),
            legend=dict(bgcolor="#111118", bordercolor="#1E1E2E", borderwidth=1, font=dict(color="#888", size=10), orientation="v", x=1.01, y=1, xanchor="left"),
            hoverlabel=dict(bgcolor="#1E1E2E", bordercolor="#0066FF", font_color="#fff"),
        ))
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.plotly_chart(fig_gt, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("<div class='footer'>CryoForm v2 · Internal R&D · For research use only</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL TRANSPARENCY
# ════════════════════════════════════════════════════════════════════════════════
with tab_model:

    st.markdown('<div class="section-label">Model Internals</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Model Transparency</div>', unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#666;font-size:0.92rem;margin-top:-1rem;margin-bottom:2rem;max-width:600px;line-height:1.7;'>"
        "Feature importances, LOO-CV residuals, and training data overview. "
        "Use this panel to understand where the model is reliable and where predictions should be treated with caution."
        "</p>", unsafe_allow_html=True,
    )

    with st.spinner("Loading model diagnostics..."):
        model, scaler, feature_cols = get_model()
        X_df, y_train, train_names = get_training_data()

        # Feature importances
        importances = model.feature_importances_
        fi_df = pd.DataFrame({"feature": feature_cols, "importance": importances})
        fi_df = fi_df.sort_values("importance", ascending=True)

        # LOO-CV residuals
        from sklearn.model_selection import LeaveOneOut
        loo = LeaveOneOut()
        X_scaled = scaler.transform(X_df)
        loo_preds = []
        for train_idx, test_idx in loo.split(X_scaled):
            from sklearn.ensemble import RandomForestRegressor as RFR
            m_loo = RFR(n_estimators=200, max_features="sqrt", random_state=42)
            m_loo.fit(X_scaled[train_idx], y_train[train_idx])
            loo_preds.append(m_loo.predict(X_scaled[test_idx])[0])
        loo_preds = np.array(loo_preds)
        residuals = loo_preds - y_train
        mae_val = np.mean(np.abs(residuals))
        r2_val  = 1 - np.sum(residuals**2) / np.sum((y_train - y_train.mean())**2)

    mt_c1, mt_c2 = st.columns(2)

    with mt_c1:
        # Summary stats
        st.markdown(f"""
        <div class="model-stat"><div class="model-stat-label">LOO-CV MAE</div><div class="model-stat-val">{mae_val:.1f} °C</div></div>
        <div class="model-stat"><div class="model-stat-label">LOO-CV R²</div><div class="model-stat-val">{r2_val:.3f}</div></div>
        <div class="model-stat"><div class="model-stat-label">Training compounds</div><div class="model-stat-val">{len(train_names)}</div></div>
        <div class="model-stat"><div class="model-stat-label">RF estimators</div><div class="model-stat-val">200</div></div>
        """, unsafe_allow_html=True)

        # Feature importance chart
        fig_fi = go.Figure(go.Bar(
            x=fi_df["importance"], y=fi_df["feature"],
            orientation="h",
            marker=dict(
                color=fi_df["importance"],
                colorscale=[[0, "#1E1E2E"], [1, "#0066FF"]],
                line=dict(width=0)
            ),
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        ))
        fig_fi.update_layout(**plot_layout(
            height=380,
            title=dict(text="Feature Importances (Random Forest)", font=dict(color="#fff", size=13)),
            xaxis=dict(title="Mean decrease in impurity", showgrid=True, gridcolor="#1E1E2E", color="#888"),
            yaxis=dict(color="#CCC"),
        ))
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.plotly_chart(fig_fi, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with mt_c2:
        # LOO-CV: predicted vs actual
        fig_loo = go.Figure()
        fig_loo.add_trace(go.Scatter(
            x=y_train, y=loo_preds, mode="markers+text",
            text=train_names,
            textposition="top center",
            textfont=dict(size=8, color="#666"),
            marker=dict(size=10, color="#0066FF", line=dict(width=1, color="#1E1E2E")),
            hovertemplate="<b>%{text}</b><br>Actual: %{x:.1f}°C<br>Predicted: %{y:.1f}°C<extra></extra>",
        ))
        # perfect prediction line
        tg_min, tg_max = y_train.min() - 20, y_train.max() + 20
        fig_loo.add_trace(go.Scatter(
            x=[tg_min, tg_max], y=[tg_min, tg_max],
            mode="lines", line=dict(color="#FF4444", dash="dash", width=1),
            name="Perfect prediction", showlegend=True,
        ))
        fig_loo.update_layout(**plot_layout(
            height=380,
            title=dict(text="LOO-CV: Predicted vs Actual Tg", font=dict(color="#fff", size=13)),
            xaxis=dict(title="Actual Tg (°C)", showgrid=True, gridcolor="#1E1E2E", color="#888"),
            yaxis=dict(title="Predicted Tg (°C)", showgrid=True, gridcolor="#1E1E2E", color="#888"),
            showlegend=True,
            legend=dict(font=dict(size=10, color="#888"), bgcolor="rgba(0,0,0,0)"),
        ))
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.plotly_chart(fig_loo, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

        # Residuals chart
        residual_colors = ["#FF4444" if abs(r) > 30 else "#0066FF" for r in residuals]
        fig_res = go.Figure(go.Bar(
            x=train_names, y=residuals,
            marker=dict(color=residual_colors, line=dict(width=0)),
            hovertemplate="%{x}<br>Residual: %{y:.1f}°C<extra></extra>",
        ))
        fig_res.add_hline(y=0, line_color="#888", line_width=1)
        fig_res.add_hline(y=mae_val, line_dash="dot", line_color="#FFAA00", annotation_text=f"+MAE ({mae_val:.1f}°C)", annotation_font=dict(color="#FFAA00", size=9))
        fig_res.add_hline(y=-mae_val, line_dash="dot", line_color="#FFAA00")
        fig_res.update_layout(**plot_layout(
            height=280,
            title=dict(text="LOO-CV Residuals (Predicted − Actual)", font=dict(color="#fff", size=13)),
            xaxis=dict(tickangle=-45, color="#888", tickfont=dict(size=9)),
            yaxis=dict(title="Residual (°C)", showgrid=True, gridcolor="#1E1E2E", color="#888"),
        ))
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.plotly_chart(fig_res, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Training data table
    st.markdown('<div class="section-label">Training Data</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Tg Dataset</div>', unsafe_allow_html=True)

    train_display = pd.DataFrame({
        "Compound":      train_names,
        "Actual Tg (°C)":   y_train,
        "Predicted Tg (°C)": loo_preds.round(1),
        "Residual (°C)":    residuals.round(1),
        "|Error| (°C)":     np.abs(residuals).round(1),
    })
    train_display = train_display.sort_values("|Error| (°C)", ascending=False)
    st.dataframe(train_display, use_container_width=True, hide_index=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("<div class='footer'>CryoForm v2 · Internal R&D · For research use only</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — CUSTOM CANDIDATE
# ════════════════════════════════════════════════════════════════════════════════
with tab_custom:

    st.markdown('<div class="section-label">Novel Excipient Screening</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Custom Candidate</div>', unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#666;font-size:0.92rem;margin-top:-1rem;margin-bottom:2rem;max-width:600px;line-height:1.7;'>"
        "Paste any SMILES string to screen a novel excipient through the full pipeline: "
        "descriptor computation, Tg prediction with uncertainty, H-bond scoring, nearest training neighbours, "
        "and Gordon-Taylor mixture curves against all library candidates."
        "</p>", unsafe_allow_html=True,
    )

    custom_name   = st.text_input("Candidate name", placeholder="e.g. Raffinose")
    custom_smiles = st.text_input("SMILES string",   placeholder="e.g. OC[C@H]1O...")

    run_custom = st.button("Screen Candidate ▶", type="primary")

    if run_custom and custom_smiles.strip():
        mol = Chem.MolFromSmiles(custom_smiles.strip())
        if mol is None:
            st.error("Could not parse SMILES. Please check the input.")
        else:
            label = custom_name.strip() if custom_name.strip() else "Custom"
            st.markdown(f'<div class="custom-badge">🧪 Screening: {label}</div>', unsafe_allow_html=True)

            with st.spinner("Running full pipeline on custom candidate..."):
                model, scaler, feature_cols = get_model()
                approved = get_approved()
                X_df, y_train, train_names = get_training_data()

                # Descriptors
                cust_desc = compute_descriptors(custom_smiles)
                cust_tg, cust_std = predict_tg_with_uncertainty(custom_smiles, model, scaler, feature_cols)

                form = {"ALC-0315": f_ion/100, "DSPC": f_dspc/100, "Cholesterol": f_chol/100, "ALC-0159": f_peg/100}
                if total == 100:
                    surface_profile_c = build_surface_profile(form)
                    cust_hbond = score_candidate(custom_smiles, surface_profile_c)
                else:
                    cust_hbond = None

                cust_nearest = find_nearest_training(custom_smiles, X_df, train_names, feature_cols, n=3)
                cust_biocompat = is_approved(label, approved)

            cc1, cc2 = st.columns(2)

            with cc1:
                desc_rows = "".join(f"<tr><td>{k}</td><td><strong>{v}</strong></td></tr>" for k, v in (cust_desc or {}).items())
                st.markdown(f"""
                <div class="ex-card">
                    <div class="ex-card-label">Molecular Descriptors</div>
                    <table class="ex-desc-table">{desc_rows}</table>
                </div>
                """, unsafe_allow_html=True)

                nbr_rows2 = "".join(f"<tr><td>{n}</td><td style='color:#888'>{d:.3f}</td></tr>" for n, d in cust_nearest)
                st.markdown(f"""
                <div class="ex-card">
                    <div class="ex-card-label">Nearest Training Compounds</div>
                    <p style='font-size:0.75rem;color:#444;margin-bottom:0.5rem;'>Lower distance = more similar to known training data.</p>
                    <table class="ex-desc-table">
                      <tr><td style='color:#555'>Compound</td><td style='color:#555'>Distance</td></tr>
                      {nbr_rows2}
                    </table>
                </div>
                """, unsafe_allow_html=True)

            with cc2:
                tg_c   = f"{cust_tg:+.1f} °C" if cust_tg is not None else "N/A"
                tg_col = "#0066FF" if (cust_tg is not None and cust_tg > tg_threshold_override) else "#FF4444"
                tg_pass_pill = '<span class="pill-pass">Passes threshold</span>' if (cust_tg is not None and cust_tg > tg_threshold_override) else '<span class="pill-fail">Below threshold</span>'

                std_pct2 = min(int((cust_std or 0) / max(abs(cust_tg or 1), 1) * 100), 100)
                conf_c2  = "#00C864" if std_pct2 < 15 else ("#FFAA00" if std_pct2 < 35 else "#FF4444")
                conf_l2  = "High" if std_pct2 < 15 else ("Medium" if std_pct2 < 35 else "Low")
                bc_pill  = '<span class="pill-pass">FDA IID found</span>' if cust_biocompat else '<span class="pill-warn">Not in IID — verify</span>'

                st.markdown(f"""
                <div class="ex-card">
                    <div class="ex-card-label">Predicted Tg</div>
                    <div class="tg-badge" style="color:{tg_col};">{tg_c}</div>
                    {tg_pass_pill}
                    <div class="unc-row" style="margin-top:0.9rem;">
                      <span class="unc-label">±{cust_std:.1f}°C std</span>
                      <div class="unc-bar-outer"><div class="unc-bar-inner" style="width:{std_pct2}%;background:{conf_c2};"></div></div>
                      <span class="unc-val" style="color:{conf_c2};">{conf_l2} conf.</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if cust_hbond is not None:
                    cl = max(0.0, min(1.0, cust_hbond))
                    hb_col2 = "#0066FF" if cl >= 0.6 else ("#F59E0B" if cl >= 0.35 else "#EF4444")
                    st.markdown(f"""
                    <div class="ex-card">
                        <div class="ex-card-label">H-Bond Complementarity</div>
                        <div class="hb-score-val" style="color:{hb_col2};">{cl:.4f}</div>
                        <div class="hb-bar-track"><div style="background:{hb_col2};width:{int(cl*100)}%;height:100%;border-radius:6px;"></div></div>
                        {bc_pill}
                    </div>
                    """, unsafe_allow_html=True)

                if cust_tg is not None:
                    moist_xc, moist_yc = tg_moisture_curve(cust_tg)
                    coll_c = next((moist_xc[i] for i, v in enumerate(moist_yc) if v < tg_threshold_override), None)
                    coll_str = f"{coll_c:.1f}%" if coll_c else ">10%"
                    st.markdown(f"""
                    <div class="ex-card">
                        <div class="ex-card-label">Moisture Sensitivity</div>
                        <div class="info-grid">
                          <div class="info-cell"><div class="info-cell-label">Dry Tg</div><div class="info-cell-val">{cust_tg:.1f}°C</div></div>
                          <div class="info-cell"><div class="info-cell-label">Collapse moisture</div><div class="info-cell-val">{coll_str}</div></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # Gordon-Taylor curves: custom vs library
            if cust_tg is not None and cust_tg > tg_threshold_override and total == 100:
                st.markdown('<hr class="divider">', unsafe_allow_html=True)
                st.markdown(f'<div class="section-title" style="font-size:1.3rem;">'
                            f'Gordon–Taylor: <span style="color:#0066FF;">{label}</span> vs Library</div>', unsafe_allow_html=True)

                with st.spinner("Computing mixture curves..."):
                    model2, scaler2, fc2 = get_model()
                    w_range2 = np.linspace(0, 1, 100)
                    gt_custom = []
                    for lib_name, lib_smiles in CANDIDATE_SMILES.items():
                        tg_lib, _ = predict_tg_with_uncertainty(lib_smiles, model2, scaler2, fc2)
                        if tg_lib is None or tg_lib <= tg_threshold_override:
                            continue
                        tg_mix2 = [gordon_taylor_tg(cust_tg, tg_lib, w) for w in w_range2]
                        gt_custom.append({"name": lib_name, "x": w_range2.tolist(), "y": tg_mix2, "tg_lib": tg_lib})
                    gt_custom.sort(key=lambda t: max(t["y"]), reverse=True)

                if gt_custom:
                    nc = len(gt_custom)
                    colsc = [f"hsl({int(200+i*100/max(nc-1,1))},75%,{int(55+i*15/max(nc-1,1))}%)" for i in range(nc)]
                    fig_cgt = go.Figure()
                    for i, tr in enumerate(gt_custom):
                        fig_cgt.add_trace(go.Scatter(
                            x=tr["x"], y=tr["y"], mode="lines",
                            name=f"{tr['name']} (Tg {tr['tg_lib']:.0f}°C)",
                            line=dict(color=colsc[i], width=1.8),
                            hovertemplate=f"<b>{label} / {tr['name']}</b><br>w₁={label}: %{{x:.2f}}<br>Tg: %{{y:.1f}}°C<extra></extra>",
                        ))
                    fig_cgt.add_hline(y=tg_threshold_override, line_dash="dash", line_color="#FF4444",
                                      annotation_text=f"Threshold ({tg_threshold_override}°C)",
                                      annotation_font=dict(color="#FF4444", size=10))
                    fig_cgt.update_layout(**plot_layout(
                        height=400,
                        xaxis=dict(title=f"w₁ ({label} weight fraction)", showgrid=True, gridcolor="#1E1E2E", color="#888", range=[0,1]),
                        yaxis=dict(title="Tg_mix (°C)", showgrid=True, gridcolor="#1E1E2E", color="#888"),
                        legend=dict(bgcolor="#111118", bordercolor="#1E1E2E", borderwidth=1, font=dict(color="#888", size=10), x=1.01, y=1, xanchor="left"),
                    ))
                    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
                    st.plotly_chart(fig_cgt, use_container_width=True, config={"displayModeBar": False})
                    st.markdown('</div>', unsafe_allow_html=True)

    elif run_custom and not custom_smiles.strip():
        st.warning("Please enter a SMILES string.")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("<div class='footer'>CryoForm v2 · Internal R&D · For research use only</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 5 — DATA MANAGEMENT
# ════════════════════════════════════════════════════════════════════════════════
with tab_data:

    st.markdown('<div class="section-label">Training Data</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Data Management</div>', unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#666;font-size:0.92rem;margin-top:-1rem;margin-bottom:2rem;max-width:620px;line-height:1.7;'>"
        "Upload in-house experimental Tg measurements to expand the training set. "
        "New data is validated, merged with the base dataset, and the model is retrained automatically. "
        "Upload a CSV with columns: <code style='color:#888'>name, smiles, tg_c, crystallizer</code>."
        "</p>", unsafe_allow_html=True,
    )

    dc1, dc2 = st.columns([1.2, 0.8], gap="large")

    with dc1:
        st.markdown("#### Upload Experimental Data")
        uploaded = st.file_uploader("CSV file (name, smiles, tg_c, crystallizer)", type=["csv"])

        if uploaded is not None:
            try:
                new_df = pd.read_csv(uploaded)
                new_df.columns = [c.strip().lower() for c in new_df.columns]
                required_cols = {"name", "smiles", "tg_c", "crystallizer"}
                missing = required_cols - set(new_df.columns)
                if missing:
                    st.error(f"Missing columns: {missing}")
                else:
                    # Validate SMILES
                    new_df["smiles_valid"] = new_df["smiles"].apply(lambda s: Chem.MolFromSmiles(str(s)) is not None)
                    invalid = new_df[~new_df["smiles_valid"]]
                    valid_new = new_df[new_df["smiles_valid"]].copy()

                    if not invalid.empty:
                        st.warning(f"{len(invalid)} rows have invalid SMILES and will be skipped: {invalid['name'].tolist()}")

                    st.success(f"✓ {len(valid_new)} valid new entries found")
                    st.dataframe(valid_new[["name", "tg_c", "crystallizer", "smiles_valid"]], use_container_width=True, hide_index=True)

                    if st.button("Add to training set & retrain ▶", type="primary"):
                        from data.tg_data import TG_DATA

                        existing_names = {row[0].lower() for row in TG_DATA}
                        added = []
                        for _, row in valid_new.iterrows():
                            if row["name"].lower() not in existing_names:
                                added.append(row["name"])

                        # Build merged dataset in memory and retrain
                        extra_records = [
                            (row["name"], row["smiles"], float(row["tg_c"]), bool(row["crystallizer"]))
                            for _, row in valid_new.iterrows()
                            if row["name"].lower() not in existing_names
                        ]

                        if not extra_records:
                            st.info("All entries already exist in the training set (by name).")
                        else:
                            combined = TG_DATA + extra_records

                            # Temporarily patch tg_data and retrain
                            import data.tg_data as tg_mod
                            original_data = tg_mod.TG_DATA[:]
                            tg_mod.TG_DATA = combined

                            # Clear cache so model retrains
                            get_model.clear()
                            get_training_data.clear()

                            with st.spinner(f"Retraining on {len(combined)} entries..."):
                                new_model, new_scaler, new_fc = get_model()
                                new_X, new_y, new_names = get_training_data()

                            from sklearn.model_selection import LeaveOneOut, cross_val_score
                            new_loo = cross_val_score(new_model, new_scaler.transform(new_X), new_y,
                                                       cv=LeaveOneOut(), scoring="neg_mean_absolute_error")
                            new_mae = -new_loo.mean()

                            st.success(f"✓ Added {len(extra_records)} new entries. Retrained on {len(combined)} total. New LOO-MAE: {new_mae:.1f}°C")
                            st.balloons()

            except Exception as e:
                st.error(f"Could not parse file: {e}")

    with dc2:
        st.markdown("#### Current Training Set")
        with st.spinner("Loading..."):
            X_df2, y2, names2 = get_training_data()
        current_df = pd.DataFrame({
            "Compound": names2,
            "Tg (°C)": y2,
        }).sort_values("Tg (°C)", ascending=False)
        st.dataframe(current_df, use_container_width=True, hide_index=True)
        st.markdown(f"<p style='font-size:0.78rem;color:#444;margin-top:0.5rem;'>{len(names2)} non-crystallizer compounds in active training set.</p>", unsafe_allow_html=True)

        # Download current training set as CSV
        st.download_button(
            "⬇ Download training set (.csv)",
            data=current_df.to_csv(index=False),
            file_name="cryoform_training_data.csv",
            mime="text/csv",
        )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Candidate library management
    st.markdown('<div class="section-label">Candidate Library</div>', unsafe_allow_html=True)
    st.markdown("#### Current Candidate Library")

    with st.spinner("Computing library descriptors..."):
        model3, scaler3, fc3 = get_model()
        approved3 = get_approved()
        lib_rows = []
        for name, smiles in CANDIDATE_SMILES.items():
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            tg_l, tg_s_l = predict_tg_with_uncertainty(smiles, model3, scaler3, fc3)
            lib_rows.append({
                "Name": name,
                "Predicted Tg (°C)": tg_l,
                "Uncertainty ±°C": tg_s_l,
                "FDA IID": "✓" if is_approved(name, approved3) else "—",
                "Tg Pass": "✓" if (tg_l is not None and tg_l > tg_threshold_override) else "✗",
                "MW": round(Descriptors.MolWt(mol), 1),
                "HBD": rdMolDescriptors.CalcNumHBD(mol),
                "HBA": rdMolDescriptors.CalcNumHBA(mol),
            })
    lib_df = pd.DataFrame(lib_rows).sort_values("Predicted Tg (°C)", ascending=False)
    st.dataframe(lib_df, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇ Download library (.csv)",
        data=lib_df.to_csv(index=False),
        file_name="cryoform_candidate_library.csv",
        mime="text/csv",
    )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("<div class='footer'>CryoForm v2 · Internal R&D · For research use only</div>", unsafe_allow_html=True)