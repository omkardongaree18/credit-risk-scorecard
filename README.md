
# Credit Risk Scorecard (Probability of Default Model)

End-to-end **application credit scorecard** built on the Home Credit Default Risk dataset, following the same workflow banks and NBFCs use: WoE/IV binning, logistic regression, score points, score bands, stability check (PSI) and a business cut-off strategy.

## Business Problem
A lender receives loan applications and must decide **who to approve** and **at what risk**. Approving everyone gives a high default rate; rejecting too many loses revenue.

**Goal:** build an explainable model that scores each applicant by default risk, and choose a score cut-off that lowers the bad rate while keeping approval rate healthy.

## Dataset
[Home Credit Default Risk (Kaggle)](https://www.kaggle.com/c/home-credit-default-risk/data) - `application_train.csv`
- ~307K loan applications, 120+ variables
- Target: `TARGET` (1 = defaulted / payment difficulty)
- Default rate ~8% (imbalanced)

> Download `application_train.csv` and place it in the `data/` folder.

## Approach
| Step | Notebook | What was done |
|---|---|---|
| 1 | `01_eda_data_prep.ipynb` | EDA, cleaning (placeholder values), risk ratio features (credit-to-income, annuity-to-income, etc.), drop >60% missing columns, stratified 70/30 split |
| 2 | `02_woe_iv_binning.ipynb` | Quantile binning on train only, WoE/IV for every variable, IV filter (0.02-0.5), correlation filter (>0.7), VIF filter (>5) |
| 3 | `03_model.ipynb` | Logistic regression on WoE variables with backward elimination (p<0.05, positive sign), AUC/Gini/KS, calibration, XGBoost benchmark + SHAP |
| 4 | `04_scorecard_strategy.ipynb` | Score scaling (600 pts @ 50:1 odds, PDO 20), scorecard table, score bands, PSI, cut-off analysis |

## Results
> Fill these in after running the notebooks.

| Model | Train Gini | Test Gini | Test KS |
|---|---|---|---|
| Logistic Scorecard | 0.xx | 0.xx | 0.xx |
| XGBoost (benchmark) | 0.xx | 0.xx | 0.xx |

- **PSI (train vs test):** 0.xxx (stable if < 0.10)
- **Chosen cut-off:** xxx points
- **Business impact:** bad rate reduced from **x.x%** to **y.y%** at **zz%** approval rate

### Score band table (test set)
Paste the output of Section 4 in notebook 04 here.

### Cut-off trade-off
Add the "Approval rate vs bad rate" plot from notebook 04:

<img width="749" height="409" alt="image" src="https://github.com/user-attachments/assets/e80b2573-8cd2-49b4-8075-288cf13cf7d7" />


## Key Learnings
- WoE binning handles missing values and outliers without imputation, and keeps every variable on the same log-odds scale.
- The scorecard ranks risk slightly below XGBoost but is fully explainable: each variable bin maps to fixed points, which is what regulators and credit policy teams need.
- Bad rate rises monotonically across score bands, showing the score rank-orders risk properly.
- PSI shows the score distribution is stable between train and test.

## Limitations & Next Steps
- Uses only application data; adding bureau and previous-application tables should improve Gini.
- No out-of-time validation (dataset has no application dates).
- Cut-off ignores loan amount and profit per customer; a next step is expected-loss and profit-based cut-offs.
- Add reject inference and monthly score monitoring.

## How to Run
```bash
git clone
cd credit-scorecard
pip install -r requirements.txt
# put application_train.csv in data/
cd notebooks
jupyter notebook
```
Run the notebooks in order: 01 -> 02 -> 03 -> 04.

## Repo Structure
```
credit-scorecard/
├── README.md
├── requirements.txt
├── data/                  (put application_train.csv here)
├── notebooks/
│   ├── 01_eda_data_prep.ipynb
│   ├── 02_woe_iv_binning.ipynb
│   ├── 03_model.ipynb
│   └── 04_scorecard_strategy.ipynb


```

## Tech Stack
Python, pandas, NumPy, scikit-learn, statsmodels, XGBoost, SHAP, matplotlib
