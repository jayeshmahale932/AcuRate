# AcuRate

Predictive Analytics Platform for Personalized Loan Interest Rate Estimation

AcuRate is a reproducible machine learning project and Streamlit web application that estimates the interest rate a borrower is likely to be offered, using credit, income, and loan application attributes derived from the (public) Lending Club dataset (2007–2018 Q4). The goal is to support exploratory analysis and scenario testing for educational and prototyping purposes.

> Disclaimer: This project is for academic / educational use only and must not be used for production lending decisions or financial advice.

---
## Table of Contents
1. Project Highlights
2. Live App (Local Run)
3. Architecture & Components
4. Data Sources & Processing Workflow
5. Feature Engineering Summary
6. Models & Evaluation
7. Reproducibility & Environment
8. Quick Start
9. Usage Guide (App Walkthrough)
10. Repository Structure
11. Roadmap & Possible Enhancements
12. Contributing
13. License & Attribution
14. Contact

---
## 1. Project Highlights
- Multi‑model regression (Linear Regression, Decision Tree, Random Forest, Gradient Boosting Regressor)
- Unified feature engineering pipeline (scaling + derived ratios + one‑hot encoding)
- Interactive Streamlit UI with tabbed workflow (Loan Details / Credit Profile / Predict & Analyze)
- Real‑time interest rate prediction with contextual financial metrics (monthly payment, total interest, repayment)
- Lightweight interpretability via profile assessment heuristics (credit strength & loan characteristics)
- Reusable serialized artifacts (`joblib` models + `MinMaxScaler`)
- Clear separation of concerns: notebooks for experimentation, `src/app.py` for deployment interface

---
## 2. Live App (Local Run)
The app runs locally via Streamlit:
```
streamlit run src/app.py
```
It will start a development server (default: http://localhost:8501 ). No external services are required.

---
## 3. Architecture & Components

High‑level flow:
1. Raw Lending Club CSVs ingested into `data/raw/`
2. Notebook-driven preprocessing & feature engineering → processed CSVs in `data/processed/`
3. Model training notebook fits multiple regressors → serialized to `saved_models/`
4. Streamlit app:
   - Collects user inputs
   - Constructs full feature vector (including derived + one‑hot columns)
   - Applies stored scaler consistently
   - Loads selected model & predicts normalized interest rate → inverse transform to original scale
   - Computes secondary financial metrics & qualitative assessment

Core components:
- Data Layer: CSV files (no database dependency)
- Modeling Layer: Scikit‑learn estimators + scaler
- Presentation Layer: Streamlit UI (`src/app.py`)

---
## 4. Data Sources & Processing Workflow
**Dataset:** Lending Club accepted & rejected loan applications (2007–2018 Q4). Only a subset of engineered variables is used for prediction. Sensitive / personally identifiable information is excluded.  
**Dataset Link:** https://www.kaggle.com/datasets/wordsforthewise/lending-club

Processing stages (see notebooks):
1. Data Cleaning (`notebooks/data_preprocessing.ipynb`)
   - Type coercion, missing value handling, outlier filtering
2. Feature Engineering (`notebooks/feature_engineering.ipynb`)
   - Creating ratios (e.g., income-to-loan)
   - Normalization (MinMax scaling for numeric predictors + target during storage phase)
   - One‑hot encoding of categorical dimensions (loan purpose, verification status)
3. Modeling (`notebooks/model_training.ipynb`)
   - Train/test split
   - Baseline & ensemble models trained
   - Metrics captured (R², MSE)
   - Artifacts persisted with `joblib`

Processed artifacts reside under `data/processed/` for transparency and reproducibility.

---
## 5. Feature Engineering Summary
Core numeric features (scaled):
`loan_amnt`, `annual_inc`, `dti`, `delinq_2yrs`, `inq_last_6mths`, `open_acc`, `pub_rec`, `revol_bal`, `revol_util`, `total_acc`, `open_rv_12m`, `open_rv_24m`, `inc_loan_ratio`, `fico_score` (target internally: `int_rate`).

Categorical (one‑hot encoded):
- Verification Status: `verification_status_Source Verified`, `verification_status_Verified`
- Purpose (subset): `purpose_credit_card`, `purpose_debt_consolidation`, `purpose_home_improvement`, `purpose_house`, `purpose_major_purchase`, `purpose_medical`, `purpose_moving`, `purpose_other`, `purpose_small_business`, `purpose_vacation`

Ordinal / Encoded:
- `term` (36→0, 60→1)
- `grade` (A–G mapped to 0–6)
- `initial_list_status` (Whole/Fractional → 1/0)

Derived:
- `inc_loan_ratio` = `annual_inc` / `loan_amnt`

---
## 6. Models & Evaluation
Implemented regressors:
- Linear Regression (baseline interpretability)
- Decision Tree Regressor
- Random Forest Regressor (default recommended)
- Gradient Boosting Regressor

Evaluation Metrics (see `notebooks/model_training.ipynb`):
- R² Score – proportion of variance explained
- Mean Squared Error (MSE) – average squared prediction error

Artifacts:
```
saved_models/
├── linear_model.joblib
├── dtree_model.joblib
├── rf_model.joblib
├── gbr_model.joblib
└── minmax_scaler.joblib
```

---
## 7. Reproducibility & Environment
All dependencies are version‑pinned in `requirements.txt`.

Recommended Python: 3.10+ (project states 3.8+; verify lower versions locally).

Create isolated environment (venv example – PowerShell):
```
python -m venv venv
./venv/Scripts/Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

---
## 8. Quick Start
1. Clone repository
2. Create & activate virtual environment
3. Install dependencies
4. (Optional) Review notebooks for lineage
5. Launch app:
```
streamlit run src/app.py
```
6. Open browser: http://localhost:8501 (if not auto-opened)

---
## 9. Usage Guide (App Walkthrough)
Sidebar:
- Select a prediction model (default: Random Forest)
- View model description & project info

Tabs:
1. Loan Details – Loan amount, term, purpose, income, verification, listing status
2. Credit Profile – FICO, DTI, revolving utilization & balance, delinquencies, inquiries, accounts
3. Predict & Analyze – Trigger prediction; view interest rate, monthly payment (approx.), total interest, repayment, qualitative assessment

Notes:
- Financial calculations are simplified and for demonstration only
- No confidence intervals or risk-adjusted pricing yet

---
## 10. Repository Structure
```
AcuRate/
├── data/
│   ├── raw/                       # Original Lending Club CSVs
│   └── processed/                 # Cleaned & engineered datasets
├── notebooks/                     # Experiment & pipeline notebooks
├── saved_models/                  # Serialized ML models + scaler
├── src/
│   └── app.py                     # Streamlit application
├── requirements.txt               # Reproducible dependency lock
└── README.md
```

---
## 11. Roadmap & Possible Enhancements
- SHAP / feature importance visualization in UI
- Prediction logging + drift detection (opt‑in)
- Hyperparameter tuning (Grid / Random / Optuna)
- Uncertainty quantification (quantile regression / interval forests)
- Dockerization & CI pipeline
- Unit tests for feature construction & scaling integrity
- More robust amortization & APR computation
- Model performance dashboard page

---
## 12. Contributing
Contributions (issues / PRs) are welcome.

Suggested workflow:
1. Fork repository
2. Create feature branch: `git checkout -b feat/<short-name>`
3. Commit with clear messages
4. Open Pull Request describing: motivation, changes, validation steps

For model updates: include evaluation metrics & methodology notes.

Please avoid adding large raw datasets outside existing structure.

---
## 13. License & Attribution
Academic / educational use. Lending Club dataset © original provider; ensure compliance with their usage terms. If you reuse this project, attribution is appreciated.

---
## 14. Contact
Open an issue for questions or improvement ideas. Provide anonymized example inputs if reporting prediction concerns.

---
Built with Streamlit, Scikit‑learn, Pandas, and curiosity.