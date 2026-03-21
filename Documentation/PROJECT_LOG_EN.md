# Diabetes Kaggle Project Log (English)

## 1) Scope of this log
This file documents:
- model/code changes over time,
- experiment runs and CV metrics,
- Kaggle leaderboard outcomes shared by the team,
- final model selection rationale.

## 2) Code change history (GBDT track)

### Phase A: Initial setup
- Verified Helen's files and removed duplicate reference file.
- Created `outputs_test` folder for submissions and logs.
- Standardized execution path to `Content_from_Helen/DSAI4203_Project/DSAI4203_Project`.

### Phase B: GBDT baseline and tuning
- Used `main_gbdt.py` for LightGBM/XGBoost CV training and submission generation.
- Added CLI parameter controls for:
  - `num_leaves`, `max_depth`, `min_child_samples`
  - `subsample`, `colsample_bytree`
  - `reg_alpha`, `reg_lambda`

### Phase C: Ensemble and expansion
- Added CatBoost model option (`--model_type cat`).
- Added rank-averaging based blending workflow for submissions.
- Added higher-value engineered features:
  - `pulse_pressure`, `map`
  - `ldl_hdl_ratio`, `chol_hdl_ratio`
  - `age_sq`, `bmi_sq`, `bp_age`, `activity_sleep_ratio`
- Added Optuna search mode:
  - `--optuna_trials`
  - `--optuna_timeout_sec`
  - best parameters auto-saved to `*_optuna_best.json`.

## 3) CV experiment log (from `outputs_test/gbdt_experiment_log.csv`)

| Run | Model | LR | Trees | Train AUC | Val AUC | Gap | Val LogLoss |
|---|---|---:|---:|---:|---:|---:|---:|
| lgbm_v1 | lgbm | 0.05 | 500 | 0.737172 | 0.725179 | 0.011992 | 0.584230 |
| xgb_v1 | xgb | 0.05 | 500 | 0.744976 | 0.724524 | 0.020452 | 0.584730 |
| xgb_v2_lr003_t800 | xgb | 0.03 | 800 | 0.744310 | 0.724623 | 0.019688 | 0.584687 |
| xgb_v3_lr008_t300 | xgb | 0.08 | 300 | 0.743703 | 0.723886 | 0.019817 | 0.585174 |
| lgbm_v2_lr003_t800 | lgbm | 0.03 | 800 | 0.736564 | 0.725083 | 0.011481 | 0.584317 |
| lgbm_v3_lr004_t700 | lgbm | 0.04 | 700 | 0.738949 | 0.725413 | 0.013535 | 0.584044 |
| lgbm_v4_lr006_t400 | lgbm | 0.06 | 400 | 0.736613 | 0.725143 | 0.011470 | 0.584252 |
| lgbm_v5_lr004_t650 | lgbm | 0.04 | 650 | 0.737784 | 0.725255 | 0.012529 | 0.584166 |
| lgbm_v6_lr004_t750 | lgbm | 0.04 | 750 | 0.740079 | 0.725540 | 0.014539 | 0.583943 |
| lgbm_tune1 | lgbm | 0.04 | 750 | 0.739167 | 0.725754 | 0.013413 | 0.583741 |
| lgbm_tune2 | lgbm | 0.04 | 850 | 0.741198 | 0.726085 | 0.015113 | 0.583477 |
| lgbm_tune3 | lgbm | 0.04 | 750 | 0.747298 | 0.726226 | 0.021073 | 0.583375 |
| lgbm_tune4 | lgbm | 0.035 | 900 | 0.748226 | 0.726221 | 0.022005 | 0.583359 |
| lgbm_tune5 | lgbm | 0.045 | 700 | 0.756227 | 0.726676 | 0.029551 | 0.583031 |
| lgbm_tune6 | lgbm | 0.03 | 1000 | 0.739003 | 0.725502 | 0.013501 | 0.583925 |
| cat_v1 | cat | 0.04 | 700 | 0.724702 | 0.721379 | 0.003322 | 0.586888 |
| cat_v2 | cat | 0.03 | 900 | 0.727169 | 0.722278 | 0.004891 | 0.586276 |

## 4) Kaggle leaderboard outcomes (shared in team updates)

| Submission | Private | Public | Note |
|---|---:|---:|---|
| xgb_v1 | 0.69180 | 0.69530 | first XGB baseline |
| xgb_v2_lr003_t800 | 0.69230 | 0.69541 | improved vs xgb_v1 |
| xgb_v3_lr008_t300 | 0.69146 | 0.69488 | dropped |
| lgbm_v2_lr003_t800 | 0.69348 | 0.69693 | better LGBM trend |
| lgbm_v3_lr004_t700 | 0.69371 | 0.69705 | improved |
| lgbm_v4_lr006_t400 | 0.69345 | 0.69688 | slight drop |
| lgbm_v5_lr004_t650 | 0.69369 | 0.69703 | stable |
| lgbm_v6_lr004_t750 | 0.69375 | 0.69708 | best single model |
| blend_lgbm_tune2_v6_45_55 | 0.69396 | 0.69727 | best private blend |
| blend_lgbm_tune2_v6_55_45 | 0.69396 | 0.69727 | same as above |
| rank_blend_3models_45_40_15 | 0.69388 | 0.69731 | best public blend |
| rank_blend_4models_40_35_15_10 | 0.69384 | 0.69720 | lower than 3-model rank |

## 5) Final recommendation (modeling frozen)
- **Private-first**: `blend_lgbm_tune2_v6_45_55.csv` (**Private 0.69396** / **Public 0.69727**).
- **Full-train OOF + Ridge** (`ensemble.py --method ridge` → `ensemble_ridge.csv`): **Private 0.69394** / **Public 0.69724** (essentially tied; stacking narrative).
- **Public-highlight**: `rank_blend_3models_45_40_15.csv` (**Public 0.69731**, Private slightly below best blend).

### Extra ensemble runs (Kaggle)

| Submission | Private | Public | Note |
|---|---:|---:|---|
| ensemble_ridge.csv | 0.69394 | 0.69724 | Ridge weights on full OOF |
| ensemble_auc_weighted.csv | 0.69356 | 0.69701 | Softmax weights from Val_AUC only |
| ensemble_tail_ridge.csv | 0.69346 | 0.69698 | Tail-only Ridge (alpha=10) |
| ensemble_tail_ridge_a1.csv | 0.69332 | 0.69689 | alpha=1 |
| ensemble_tail_ridge_a01.csv | 0.69246 | 0.69608 | alpha=0.1 |

## 6) Further score chasing (paused — focus on report & video)
- Optuna / external target encoding / extra blends did not clearly beat the Private best above; iteration stopped to avoid over-tuning the leaderboard.

## 6b) Log / artifact map (traceability for individual work)

| Kind | Location |
|------|----------|
| Per-run CV append | `outputs_test/gbdt_experiment_log.csv` |
| Project narrative (this file + ZH) | `PROJECT_LOG_*.md` |
| Optuna best params | `outputs_test/*_optuna_best.json` |
| OOF for stacking | `outputs_test/oof_<run_name>.csv` (with `--save_oof`) |
| Ensemble metadata | `ensemble_output/*_meta.json` from `ensemble.py` |

## 8) Individual contribution: how we tuned models and selected the final one (Occam’s razor)

**Role**: GBDT track (LightGBM / XGBoost / CatBoost), feature & CLI extensions, `ensemble.py` / `ensemble_tail_ridge.py`, experiment logging.

**Iteration order (summary)**  
1. Fixed pipeline: 5-fold stratified CV; log **Val AUC** and **Train–Val gap** (overfitting).  
2. Coarse tuning: `learning_rate`, `n_estimators`; then finer knobs (`num_leaves`, `subsample`, `reg_*`, …).  
3. After strong single models, move to **blending** (probability mix, rank average).  
4. With **OOF** saved: `ensemble.py --method ridge` learns linear weights on **full** OOF; without OOF we compared **Val_AUC–weighted** blending as a baseline.  
5. **Negative control**: `ensemble_tail_ridge.py` fits Ridge only on a tail subset (`id ≥ cutoff`) — **Private dropped** vs best blend / full OOF Ridge → document as failed ablation, **not** the final model.

**Selection policy**  
1. Official metric: **ROC-AUC**; prioritize **Private** (plus CV) for generalization.  
2. When scores are **very close** (e.g. 0.69396 vs 0.69394), apply **Occam’s razor**: prefer the **simpler, more interpretable** deliverable.  
3. Hence **two-model probability blend** (minimal structure) and **full-OOF Ridge** (standard stacking) are both defensible; **tail-only Ridge** is more complex and **did not** improve Private → **rejected**.

**One-sentence summary (report-ready)**  
The competition uses ROC-AUC; we judged generalization by **Private AUC and CV**, and when scores were tied we followed **Occam’s razor** by favoring a simpler final model (e.g. two-LGBM blend) and **did not adopt** tail-Ridge variants that failed to improve Private score.

## 7) Optional: external reference target encoding (supported in `main_gbdt.py`)

- Download any competition-allowed BRFSS-style CSV into `data/` using one of the filenames in `_EXTERNAL_CSV_CANDIDATES` (e.g. `original.csv`).
- Target column aliases (`Diabetes_binary`, `Outcome`, …) are mapped to `diagnosed_diabetes`.
- Adds `{col}_ext_te` only for columns with **≤ 50** unique values in the reference table (avoids broken groupby on continuous features).
- **Verify competition rules** before using external data.
- Disable with `--no_external_te`.
