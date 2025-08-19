# Responsible-AI Evidence Pack

[![CI – Responsible AI Monitoring](https://github.com/22Ifeoma22/Responsible-AI-Evidence-Pack/actions/workflows/ci.yml/badge.svg)](https://github.com/22Ifeoma22/Responsible-AI-Evidence-Pack/actions)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

A hands-on, reproducible demo of **Responsible AI** auditing for a tabular model (OpenML Adult).  
It **trains** a model, runs a **fairness audit**, generates **explainability** visuals, and writes a **governance audit trail** aligned to **NIST AI RMF 1.0** (Map–Measure–Manage–Monitor) and **ISO/IEC 42001** (AI Management System).

---

##  What you get

- **Automated pipeline:** train → metrics → fairness (by one or more attributes) → explainability (PFI + SHAP optional) → audit JSON  
- **Visuals:** fairness gaps & top feature impact ready for README/slide decks  
- **Governance pack:** model card, data card, risk register, NIST/ISO mapping  
- **CI ready:** GitHub Actions to run the audit and upload artifacts on every push/PR

---
##  Results

![Fairness Gaps](artifacts/fairness_gaps.png)
![Top Features (PFI)](artifacts/feature_importance.png)
[Interactive explainer](artifacts/explain_top_features.html)


## Results

Fairness gaps (lower is better; 0 = parity):  
![Fairness Gaps](artifacts/fairness_gaps.png)

Top feature impact (PFI):  
![Top Features](artifacts/feature_importance.png)

Interactive explanation:  
[explainer HTML](artifacts/explain_top_features.html)

Reference screenshot (for GitHub preview):  
<img width="1600" height="960" alt="Fairness_gaps" src="https://github.com/user-attachments/assets/4f5b7adf-ecb4-4ffe-8e2b-7b7efbc788e1" />

##  Quickstart

```bash
# 1) Environment
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2) Run the basic demo (already in this repo)
python scripts/train_and_audit.py --bias-attr sex --explain shap --outdir artifacts --seed 42

# 3) (Recommended) Run the advanced audit (multi-attribute + SHAP scaffold)
python scripts/advanced_train_and_audit.py --bias-attrs sex race age --shap --outdir artifacts --seed 42
<<<<<<< HEAD

=======
<img width="1600" height="960" alt="fairness_gaps" src="https://github.com/user-attachments/assets/4f5b7adf-ecb4-4ffe-8e2b-7b7efbc788e1" />
>>>>>>> main
