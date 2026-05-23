# Diabetes Prediction — GBDT Classification Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)

Kaggle diabetes prediction competition entry using gradient boosting (LightGBM, XGBoost, CatBoost) with Bayesian hyperparameter optimization and ensemble methods.

## Key Results

| Metric | Value |
|--------|-------|
| Best Model | LightGBM + Optuna tuning |
| Best Private AUC | **0.69396** |
| Best Public AUC | 0.69530 |
| Validation | 5-fold stratified cross-validation |

## Models

| Model | Description |
|-------|-------------|
| **LightGBM** | Gradient boosting with leaf-wise tree growth |
| **XGBoost** | Gradient boosting with level-wise tree growth |
| **CatBoost** | Gradient boosting with native categorical support |
| **Ensemble** | Weighted voting + tail ridge regression |

## Pipeline

```
Raw Data → Clinical Feature Engineering → 5-Fold CV Training (Optuna Tuned)
  → Ensemble (Weighted Vote + Ridge) → Submission
```

## Project Structure

```
Main_Programme/
├── main.py                  Pipeline orchestration
├── main_gbdt.py             GBDT training (LGBM/XGB/CatBoost)
├── main_rf.py               Random Forest baseline
├── ensemble.py              Weighted voting ensemble
└── ensemble_tail_ridge.py   Tail ridge regression ensemble
```

## Course Context

Built for DSAI4203 Machine Learning (PolyU, Spring 2026). .

## Author

Tam Sai Ho — BSc FinTech & AI, Year 3, PolyU
