# CyberSift – Filtering Network Noise, Finding Threats
University AI course project (AI2002, Path A: software, ML + Streamlit). Graded on a rubric, with a live viva where students must explain and modify the code.

## Problem
Security analysts cannot read millions of network connections. We train a Random Forest binary classifier on CIC-IDS2017 flow CSVs (78 numeric flow features + Label) to score unlabelled flows (0 = benign, 1 = attack), rank the riskiest first, and explain why.

## Non-negotiable rules
- Core model: sklearn RandomForestClassifier, binary target is_attack. Baselines allowed: rules, LogisticRegression, DecisionTree.
- Streamlit app is mandatory: input -> model -> understandable result. It must use the SAME saved pipeline and the SAME prepare_features() as training. Never hardcode predictions.
- No leakage: drop duplicates BEFORE splitting; imputer lives inside the sklearn Pipeline (fit on train only); test set only for final evaluation. random_state=42 everywhere.
- Report precision, recall, F1, PR-AUC, confusion matrix, per-attack-type detection, error analysis. Never rely on accuracy alone.
- Raw data (data/raw, data/processed) is never committed. Only data/sample/ (small) and models/ (<50 MB) are committed.
- App needs input validation and friendly errors (missing columns, empty file, non-numeric, inf/NaN).

## Code style (students must be able to explain and edit live)
- Simple, short functions, plain-English comments, no clever abstractions, no extra frameworks.
- All paths and constants in src/cybersift/config.py.
- Never print full DataFrames; print shapes, value_counts, or head(3).
- After each task: run it, then reply with <=6 lines: what changed, how to run, what to check.

## Layout
src/cybersift/{config,data,rules,predict,explain,ui_tabs}.py
scripts/{prepare_data,make_sample,eda,train_models,evaluate}.py
app.py (Streamlit entry), tests/, models/, reports/{figures,tables}, data/{raw,processed,sample}, docs/