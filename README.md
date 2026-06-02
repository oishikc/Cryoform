\documentclass[11pt, a4paper]{article}

% ── Packages ──────────────────────────────────────────────────────────────────
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}

\usepackage[margin=2.5cm]{geometry}
\usepackage{xcolor}
\usepackage{titlesec}
\usepackage{titling}
\usepackage{parskip}
\usepackage{enumitem}
\usepackage{listings}
\usepackage{mdframed}
\usepackage{booktabs}
\usepackage{array}
\usepackage{microtype}
\usepackage{fancyhdr}
\usepackage{amsmath}
\usepackage{hyperref}

% ── Colours ───────────────────────────────────────────────────────────────────
\definecolor{cryoblue}{HTML}{1A3A5C}
\definecolor{cryoacc}{HTML}{2E86AB}
\definecolor{cryogray}{HTML}{F4F6F8}
\definecolor{cryoborder}{HTML}{CBD5E1}
\definecolor{codebg}{HTML}{F1F5F9}
\definecolor{codefg}{HTML}{1E293B}
\definecolor{commentfg}{HTML}{64748B}
\definecolor{keywordfg}{HTML}{0369A1}
\definecolor{warnyellow}{HTML}{FEF3C7}
\definecolor{warnorange}{HTML}{D97706}

% ── Typography ────────────────────────────────────────────────────────────────
\usepackage{helvet}
\renewcommand{\familydefault}{\sfdefault}

% ── Section styling ───────────────────────────────────────────────────────────
\titleformat{\section}
  {\large\bfseries\color{cryoblue}}
  {}
  {0em}
  {}
  [\vspace{-0.4em}\textcolor{cryoacc}{\rule{\linewidth}{1.5pt}}]

\titleformat{\subsection}
  {\normalsize\bfseries\color{cryoblue}}
  {}
  {0em}
  {}

\titlespacing{\section}{0pt}{1.8em}{0.8em}
\titlespacing{\subsection}{0pt}{1.2em}{0.4em}

% ── Header / Footer ───────────────────────────────────────────────────────────
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small\color{commentfg}\textbf{CryoForm} \textbar\ Internal R\&D Tool}
\fancyhead[R]{\small\color{commentfg}Lyoprotectant Formulation Design}
\fancyfoot[C]{\small\color{commentfg}\thepage}
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\headrule}{\hbox to\headwidth{\color{cryoborder}\leaders\hrule height \headrulewidth\hfill}}

% ── Code listings ─────────────────────────────────────────────────────────────
\lstdefinestyle{cryocode}{
  backgroundcolor=\color{codebg},
  basicstyle=\ttfamily\small\color{codefg},
  keywordstyle=\color{keywordfg}\bfseries,
  commentstyle=\color{commentfg}\itshape,
  stringstyle=\color{cryoacc},
  breaklines=true,
  frame=single,
  framerule=0pt,
  rulecolor=\color{cryoborder},
  xleftmargin=1em,
  xrightmargin=1em,
  aboveskip=0.8em,
  belowskip=0.8em,
  showstringspaces=false,
  numbers=none,
}
\lstset{style=cryocode}

% ── Callout box ───────────────────────────────────────────────────────────────
\newmdenv[
  backgroundcolor=warnyellow,
  linecolor=warnorange,
  linewidth=3pt,
  topline=false,
  bottomline=false,
  rightline=false,
  innerleftmargin=12pt,
  innerrightmargin=12pt,
  innertopmargin=8pt,
  innerbottommargin=8pt,
  skipabove=1em,
  skipbelow=1em,
]{warningbox}

\newmdenv[
  backgroundcolor=cryogray,
  linecolor=cryoacc,
  linewidth=3pt,
  topline=false,
  bottomline=false,
  rightline=false,
  innerleftmargin=12pt,
  innerrightmargin=12pt,
  innertopmargin=8pt,
  innerbottommargin=8pt,
  skipabove=1em,
  skipbelow=1em,
]{notebox}

% ── Hyperlinks ────────────────────────────────────────────────────────────────
\hypersetup{
  colorlinks=true,
  linkcolor=cryoacc,
  urlcolor=cryoacc,
  pdftitle={CryoForm — README},
  pdfauthor={Internal R\&D},
}

% ══════════════════════════════════════════════════════════════════════════════
\begin{document}

% ── Title block ───────────────────────────────────────────────────────────────
\begin{center}
  {\fontsize{28}{34}\selectfont\bfseries\color{cryoblue} CryoForm}\\[0.4em]
  {\large\color{cryoacc} Computational Lyoprotectant Formulation Design}\\[0.6em]
  {\small\color{commentfg} Internal R\&D Tool \textbar\ Lipid Nanoparticle Programme}
  \vspace{0.6em}
  \textcolor{cryoborder}{\rule{\linewidth}{1pt}}
\end{center}

\vspace{0.5em}

% ── What is this ──────────────────────────────────────────────────────────────
\section{What Is This?}

CryoForm is an internal R\&D tool that helps formulation scientists design lyoprotectant mixtures for lipid nanoparticle (LNP) drug products. It runs a three-stage computational pipeline — hydrogen bond scoring, glass transition temperature prediction, and mixture optimisation — and outputs ranked excipient recipes with scores, predicted stability, and uncertainty estimates.

% ── The Problem ───────────────────────────────────────────────────────────────
\section{The Problem}

Lipid nanoparticles (LNPs) are the delivery vehicle behind mRNA vaccines and a growing class of RNA therapeutics. To store and ship them, manufacturers typically freeze-dry (lyophilise) the product. This requires adding protective excipients called \textbf{lyoprotectants} that shield the LNP from damage during freezing and drying.

Choosing the right lyoprotectant formulation is not trivial:

\begin{itemize}[leftmargin=1.5em, itemsep=0.3em]
  \item The excipient must physically interact well with the LNP surface through \textbf{hydrogen bond complementarity}
  \item It must form a stable amorphous glass at storage temperature, characterised by its \textbf{glass transition temperature} ($T_g$)
  \item It must be biocompatible and approved for the relevant route of administration
  \item Binary mixtures (two excipients blended together) often outperform single-component systems, but the number of combinations and ratios to screen is large
\end{itemize}

Currently this is done primarily by trial and error on the bench — slow, expensive, and dependent on individual expertise.

% ── The Solution ──────────────────────────────────────────────────────────────
\section{The Solution}

CryoForm replaces the trial-and-error step with a computational pre-screening pipeline. Before a scientist runs a single experiment, CryoForm:

\begin{enumerate}[leftmargin=1.5em, itemsep=0.3em]
  \item Analyses the LNP surface chemistry and identifies what kind of excipient would interact with it best
  \item Predicts the glass transition temperature of each candidate excipient using a machine learning model trained on literature data
  \item Scans all binary mixtures of passing candidates across weight fractions using the Gordon--Taylor equation
  \item Scores and ranks the mixtures on a composite metric combining H-bond fit, thermal stability, and biocompatibility
\end{enumerate}

The output is a short ranked list of formulation recipes — each with a predicted $T_g$, H-bond score, uncertainty estimate, and biocompatibility status — that a scientist can prioritise for wet lab validation.

% ── How It Works ──────────────────────────────────────────────────────────────
\section{How It Works}

The pipeline runs in three sequential stages.

\subsection{Stage 1 — H-Bond Surface Scoring \quad {\normalfont\small\texttt{stage\_1/hbond\_scorer.py}}}

LNP membranes are hydrogen bond acceptor (HBA)-rich. A good lyoprotectant needs to be HBD-rich to complement this. CryoForm computes a molar-weighted H-bond surface profile for the input LNP formulation, then scores each candidate against it. HBD contribution is weighted $3\times$ more than HBA to reflect the asymmetry of the LNP surface:

\[
\text{score} = \frac{(\text{HBD}_\text{cand} \times r_\text{surface}) + \text{HBA}_\text{cand}}{\text{heavy atoms}}
\]

where $r_\text{surface} = \text{HBA}_\text{surface} / \text{HBD}_\text{surface}$ is the surface acceptor-to-donor ratio ($\approx 2.86$ for a standard LNP).

\subsection{Stage 2 — $T_g$ Prediction \quad {\normalfont\small\texttt{stage\_2/tg\_predictor.py}, \texttt{gordon\_taylor.py}}}

A Random Forest regression model is trained on literature $T_g$ data for $\sim$14 non-crystallising excipients. Crystallisers (Mannitol, Glycine) are excluded from training — their $T_g$ values are not meaningful for glassy-state stability. RDKit computes 11 molecular descriptors per compound:

\begin{center}
\begin{tabular}{lll}
  \toprule
  MW & TPSA & LogP \\
  HBD & HBA & Rotatable bonds \\
  Rings & Aromatic rings & Heavy atom count \\
  Fraction C$_\text{sp3}$ & Bertz complexity & \\
  \bottomrule
\end{tabular}
\end{center}

Leave-one-out cross-validation (LOO-CV) is used to evaluate performance on the small dataset. Candidates with predicted $T_g < 25\,^\circ\text{C}$ are filtered out.

For binary mixtures, the \textbf{Gordon--Taylor equation} predicts the mixture $T_g$ across weight fractions:

\[
T_{g,\text{mix}} = \frac{w_1 T_{g1} + k\, w_2 T_{g2}}{w_1 + k\, w_2}
\]

where $k$ is estimated via the Couchman--Karasz approximation ($k = T_{g1}/T_{g2}$, both in Kelvin).

\subsection{Stage 3 — Mixture Optimisation \quad {\normalfont\small\texttt{stage\_3/optimizer.py}}}

All binary combinations of passing candidates are scanned across weight fractions ($w_1 = 0.1$ to $0.9$ in steps of $0.1$). Each combination is scored on a composite metric:

\[
\text{composite} = \alpha \cdot \hat{h} + \beta \cdot \hat{T}_g + \gamma \cdot b
\]

where $\hat{h}$ is the normalised H-bond score, $\hat{T}_g$ is the normalised $T_g$ margin above threshold, and $b$ is the biocompatibility bonus. Default weights: $\alpha = 0.5$, $\beta = 0.4$, $\gamma = 0.1$ (configurable in the UI).

\subsection{Biocompatibility Gate \quad {\normalfont\small\texttt{data/biocompat\_filter.py}}}

Candidates are cross-referenced against the FDA Inactive Ingredient Database (IID) filtered to intravenous, injection, subcutaneous, and intradermal routes. Compounds not on this list (or the internal whitelist) are flagged and excluded from ranked output.

% ── Features ──────────────────────────────────────────────────────────────────
\section{Features}

\begin{itemize}[leftmargin=1.5em, itemsep=0.4em]
  \item \textbf{Pipeline tab} — run the full three-stage optimiser for a given LNP formulation; view ranked recipes with $T_g$ curves, uncertainty bands, and moisture plasticisation analysis
  \item \textbf{Molecule Explorer} — look up any candidate; see predicted $T_g$ with confidence level, nearest training compounds by descriptor distance, and moisture sensitivity
  \item \textbf{Model Transparency} — RF feature importances, LOO-CV predicted vs.\ actual scatter, residuals, and full training data table
  \item \textbf{Custom Candidate} — paste any SMILES string and run it through the full pipeline; evaluate novel excipients before synthesis
  \item \textbf{Data Management} — upload in-house experimental $T_g$ measurements to extend the training set; retrain the model; download the current library as CSV
\end{itemize}

% ── Installation ──────────────────────────────────────────────────────────────
\section{Installation}

\textbf{Requirements:} Python 3.9\textsuperscript{+}, RDKit

\begin{lstlisting}[language=bash]
git clone https://github.com/your-username/cryoform.git
cd cryoform
pip install -r requirements.txt
streamlit run app.py
\end{lstlisting}

\textbf{requirements.txt:}
\begin{lstlisting}
streamlit
rdkit
scikit-learn
pandas
numpy
plotly
joblib
\end{lstlisting}

\begin{notebox}
RDKit is best installed via conda if you run into issues:\\
\texttt{conda install -c conda-forge rdkit}
\end{notebox}

% ── Project Structure ─────────────────────────────────────────────────────────
\section{Project Structure}

\begin{lstlisting}
cryoform/
|
|-- app.py                  # Streamlit UI -- entry point
|-- main.py
|
|-- data/
|   |-- candidates.py       # Curated excipient library with SMILES
|   |-- tg_data.py          # Literature Tg training dataset
|   |-- biocompat_filter.py # FDA IID biocompatibility gate
|   |-- IIR_OCOMM.csv       # FDA Inactive Ingredient Database
|   `-- lipids/
|       `-- lipids.py       # LNP component SMILES library
|
|-- stage_1/
|   |-- hbond_scorer.py     # H-bond surface profile and candidate scoring
|   `-- verify_lipids.py    # RDKit validation for lipid SMILES
|
|-- stage_2/
|   |-- tg_predictor.py     # RF model training and Tg prediction
|   `-- gordon_taylor.py    # Binary mixture Tg via Gordon-Taylor equation
|
`-- stage_3/
    `-- optimizer.py        # Composite scorer and full pipeline runner
\end{lstlisting}

% ── Science Notes ─────────────────────────────────────────────────────────────
\section{Science Notes}

\textbf{Why Gordon--Taylor?} The GT equation is the standard model for predicting binary mixture $T_g$ and requires only pure-component $T_g$ values. The Couchman--Karasz $k$ approximation ($k = T_{g1}/T_{g2}$) assumes equal heat capacity jumps at $T_g$, which is reasonable for sugar/amino acid pairs but can diverge for structurally dissimilar systems. For production use, fitting $k$ experimentally or using tabulated $\Delta C_p$ values would improve prediction accuracy.

\textbf{Why LOO-CV?} The training dataset contains $\sim$14 non-crystalliser compounds. Leave-one-out cross-validation is the standard approach for evaluating model performance on datasets this small — it maximises the number of training examples used in each fold.

\textbf{Moisture plasticisation} — $T_g$ drops significantly with residual moisture. A 1\% increase in water content typically depresses $T_g$ by $\sim$10\,°C for most sugars. The tool outputs moisture-adjusted $T_g$ curves so scientists can account for this when designing freeze-drying cycles.

% ── Limitations ───────────────────────────────────────────────────────────────
\section{Limitations}

\begin{warningbox}
CryoForm is a pre-screening tool. Its output should guide which formulations to test first, not serve as final specifications.
\end{warningbox}

\begin{itemize}[leftmargin=1.5em, itemsep=0.3em]
  \item The RF model is trained on $\sim$14 compounds. Predictions for excipients structurally dissimilar to the training set should be treated with caution — check the uncertainty estimate and nearest-neighbour distance in the Molecule Explorer tab.
  \item The Gordon--Taylor $k$ approximation is not experimentally fitted. For final formulation decisions, DSC validation of predicted $T_g$ values is recommended.
  \item Ternary mixtures (sugar + amino acid + buffer) are not yet supported. Binary optimisation only.
  \item Moisture plasticisation curves use a simplified linear approximation, not a fitted plasticisation model.
\end{itemize}

% ── Data Sources ──────────────────────────────────────────────────────────────
\section{Data Sources}

\begin{itemize}[leftmargin=1.5em, itemsep=0.3em]
  \item ATHAS database — polymer and small molecule $T_g$ values
  \item Hancock \& Zografi (1997) \textit{Pharm Res} \textbf{14}:422 — sugar and polyol $T_g$
  \item Franks (1990) \textit{Cryo-Letters} \textbf{11}:93 — disaccharide amorphous state data
  \item Carpenter et al.\ (1997) \textit{Pharm Res} \textbf{14}:969 — amino acid $T_g$
  \item Shamblin et al.\ (1999) \textit{J Phys Chem B} \textbf{103}:4113 — mixture reference data
  \item FDA Inactive Ingredient Database (IID) — biocompatibility approval data
  \item PubChem — canonical SMILES for all excipients and lipid components
\end{itemize}

% ── Footer rule ───────────────────────────────────────────────────────────────
\vfill
\textcolor{cryoborder}{\rule{\linewidth}{0.5pt}}
\begin{center}
  {\small\color{commentfg} CryoForm \textbar\ Internal use only \textbar\ Not for distribution}
\end{center}

\end{document}
