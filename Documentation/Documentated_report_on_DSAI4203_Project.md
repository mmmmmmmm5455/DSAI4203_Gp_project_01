# Documentated report on DSAI4203_Project (Individual draft — Boosting track)

> **Chinese version (中文版)**: `Documentated_report_on_DSAI4203_Project_ZH.md`

> **Document status**: This draft covers **only the author’s GBDT / boosting work, ensembling, and experiment log**. Teammates’ **Baseline (Logistic Regression)** and **Bagging (Random Forest)** will be supplied in a separate file and merged later.  
> **Video**: You may record boosting first; time reserved for teammates is outlined in §6.  
> **Data source**: The Kaggle scores below are taken from exported leaderboard records; **verify against Kaggle screenshots before final submission**.

**Competition**: Kaggle — *Playground Series* (diabetes binary classification)  
**Metric**: ROC-AUC (**Private** primary, **Public** secondary)  
**Table convention (this document)**: **Private \| Public** (consistent with the usual pattern where Private is lower than Public on this leaderboard)

---

## 1. Summary & Objectives (individual section — draft)

### 1.1 Background

This project participates in a Kaggle diabetes prediction challenge. The task is **binary classification on tabular data**, scored by **ROC-AUC**. Unlike optimizing training accuracy alone, the challenge stresses **generalization** and **overfitting**: models may disagree between local cross-validation (CV) and the leaderboard (Public / Private), so experiments must be logged and compared systematically.

### 1.2 Scope of this author (Boosting / GBDT)

- **Gradient boosted decision trees (GBDT)**: **LightGBM** and **XGBoost** as the main line; **CatBoost** as an optional comparison (see §3 for main text vs appendix).
- **Validation**: **Stratified K-fold CV** (e.g. 5-fold), tracking **Train AUC, Val AUC, and Train–Val gap** to diagnose overfitting.
- **Methodological improvements**: feature engineering, single-model tuning, **multi-model probability blending**, **out-of-fold (OOF) second-level stacking with a Ridge meta-learner**; plus documentation of **rejected** or **weaker** variants (e.g. tail-only Ridge), in line with the course’s “learning journey” requirement.

### 1.3 Learning objectives (mapping to the rubric)

| Course objective | How this author’s work addresses it |
|------------------|-------------------------------------|
| Practice a competition-style ML workflow | Reproducible scripts (e.g. `main_gbdt.py`, `ensemble.py`), experiment logs (`gbdt_experiment_log.csv`) |
| Understand generalization and overfitting | Gap, CV vs leaderboard, Private-focused decisions, negative results (tail-Ridge, some blends worse than a simple blend) |
| Develop appropriate solutions | GBDT as a strong tabular baseline; ensembling and second-level learning after stable single models |

### 1.4 Rest of the team (merge later)

- **Baseline (Logistic Regression)** and **Random Forest (Bagging)**: *to be merged from teammates’ document into the full report (§X TBD).*  
- This file **does not** repeat those two tracks; the merged report may add a side-by-side comparison if required.

---

## 2. Improvements in Methodologies — Part 2c (Boosting — required section)

### 2.1 Why boosting (GBDT)

- On tabular data, tree models capture **nonlinearity** and **feature interactions**; gradient boosting reduces bias by adding weak learners sequentially.  
- Compared with a single shallow tree, **boosting** can balance **bias–variance** when **learning rate, subsampling, and regularization (L1/L2, `min_child_samples`, etc.)** are controlled—this must be checked with both **CV gap** and **Private leaderboard** scores.

### 2.2 End-to-end workflow (suitable for a flowchart)

1. **Data loading and feature engineering** (domain/statistical intuition + nonlinear transforms, e.g. blood-pressure derivatives, lipid ratios, age/BMI squares and interactions — follow the actual `main_gbdt.py` implementation).  
2. **Single-model training**: LGBM / XGB (and optional CatBoost) + 5-fold OOF predictions.  
3. **Hyperparameter tuning**: learning rate, number of trees, `num_leaves`, column/row sampling, `reg_alpha` / `reg_lambda`; optionally **Optuna** (§3).  
4. **Ensembling**:  
   - **Weighted probability blending** (linear mix of strong models);  
   - **Rank-based blending** (average ranks; more robust to distribution shift);  
   - **OOF + Ridge**: learn linear weights on OOF predictions (compare to simple blending).  
5. **Decision rule**: when **Private scores are extremely close**, prefer **simpler, more interpretable, lower-maintenance** solutions under **Occam’s razor** (e.g. a two-LGBM blend vs a full OOF pipeline for Ridge).

### 2.3 Main experimental results (Kaggle leaderboard summary)

The tables below are compiled from your submitted runs; **cross-check every row against the Kaggle UI before finalizing**.

#### A. Personal Logistic experiments (Jerry version — reference only, not the boosting main line)

| Description | Private | Public |
|-------------|--------:|-------:|
| C = 1 (`baseline_submission_C1.csv`) | 0.67943 | 0.68492 |
| C = 0.1 (`baseline_submission_C0p1.csv`) | 0.67944 | 0.68492 |
| C = 0.3 (`baseline_submission_C03.csv`) | 0.67943 | 0.68492 |
| `baseline_submission.csv` (note “3”) | 0.67964 | 0.68515 |
| `baseline_submission.csv` (earlier run) | 0.67973 | 0.68515 |

> **Writing tip**: C barely changes LB in this range—consistent with a **linear model ceiling**; keep the **main report focused on GBDT** and move this table to an appendix or one sentence.

#### B. Single models: XGBoost / LightGBM (excerpt)

| Run / file | Private | Public |
|------------|--------:|-------:|
| `xgb_v1` | 0.69180 | 0.69530 |
| `xgb_v2_lr003_t800` | 0.69230 | 0.69541 |
| `xgb_v3_lr008_t300` | 0.69146 | 0.69488 |
| `lgbm_v1` | 0.69360 | 0.69700 |
| `lgbm_v2_lr003_t800` | 0.69348 | 0.69693 |
| `lgbm_v3_lr004_t700` | 0.69371 | 0.69705 |
| `lgbm_v4_lr006_t400` | 0.69345 | 0.69688 |
| `lgbm_v5_lr004_t650` | 0.69369 | 0.69703 |
| `lgbm_v6_lr004_t750` | 0.69375 | 0.69708 |
| `lgbm_tune1` … `lgbm_tune6` (as run) | 0.69338–0.69371 | 0.69618–0.69701 |
| `lgbm_tune2` | **0.69376** | **0.69705** |
| `cat_feat_v1_gbdt_submission` | 0.69152 | 0.69571 |

**Analysis angles**:  
- Among GBDT variants, **LGBM is generally stronger than early XGB** (tie to local CV gap if consistent).  
- The **tune** runs show that **more complexity is not always better**—some tunes do not beat **tune2 / v6** on Private.

#### C. Ensembling and second-level learning (core)

| Submission | Private | Public | Notes |
|------------|--------:|-------:|-------|
| `blend_lgbm_tune2_v6_45_55` | **0.69396** | **0.69727** | Ties 55/45; **best Private band** |
| `blend_lgbm_tune2_v6_55_45` | **0.69396** | **0.69727** | Same |
| `ensemble_ridge.csv` | 0.69394 | 0.69724 | Full-OOF Ridge; **very close** to best blend |
| `ensemble_auc_weighted.csv` | 0.69356 | 0.69701 | Val-AUC softmax weights; weaker |
| `rank_blend_3models_45_40_15` | 0.69388 | **0.69731** | Best Public; slightly lower Private |
| `rank_blend_4models_40_35_15_10` | 0.69384 | 0.69720 | Four models did not beat three |
| `blend3_lgbm_tune2_v6_xgbv2_40_40_20` | 0.69386 | 0.69713 | Three-way blend |
| `blend3_lgbm_tune2_v6_xgbv2_45_45_10` | 0.69392 | 0.69721 | |

**Suggested closing sentence**: With **Private** as the main goal, the **two-LGBM probability blend (45/55)** and **full-data OOF Ridge** are effectively tied at the top; when scores are that close, **Occam’s razor** supports presenting the **simpler two-model blend** as the primary narrative.

#### D. Negative experiments (document “learning and trade-offs”)

| Submission | Private | Public | Interpretation (draft) |
|------------|--------:|-------:|-------------------------|
| `ensemble_tail_ridge.csv` | 0.69346 | 0.69698 | Tail-only Ridge **did not beat** full OOF / best blend |
| `ensemble_tail_ridge_a1.csv` | 0.69332 | 0.69689 | Same |
| `ensemble_tail_ridge_a01.csv` | 0.69246 | 0.69608 | Smaller alpha worse here |
| `lgbm_optuna30_gbdt_submission` | 0.69336 | 0.69598 | **Optuna** search did not beat the best manual region |
| `lgbm_tune2_orig_te_gbdt_submission` | 0.69340 | 0.69688 | External-TE variant did not beat best blend (**cite rules and data source**) |

> **Writing tip**: Negative results are not “failures”—they show that **ideas must be validated on Private**, supporting the generalization theme.

#### E. Other

| Submission | Private | Public | Notes |
|--------------|--------:|-------:|-------|
| `oof_cat_feat_v1.csv` | *Error* | — | Upload failed; one sentence in main text + error detail in appendix |

---

## 3. Optuna, external TE, CatBoost — main text vs appendix / references

| Topic | Placement | Rationale |
|-------|-----------|-----------|
| **Optuna** | **One paragraph in main text + appendix table** | State that **automated search was tried** and **did not beat the best manual config** (align with `lgbm_optuna30`); put trials/search space in appendix or cite `*_optuna_best.json`. |
| **External target encoding (TE)** | **Brief compliance note in main text + appendix** | Main text: **public data only**, consistent with competition rules on external data; technical detail and `lgbm_tune2_orig_te` comparison in appendix. |
| **CatBoost** | **Appendix-first** (or one sentence in main text) | If page-limited: main text says “CatBoost / feature variant was compared”; full CV and `cat_feat_v1` LB in appendix. |
| **XGB vs LGBM** | **Main text** | Core to “methodological improvement” and already supported by tables. |

If the report must stay **≤12 pages**: keep **workflow + main tables + conclusions** in the body; move the rest to **Appendix A: full submission table** and **Appendix B: Optuna / TE settings**.

---

## 4. Individual contribution statement (align with teammates at finalization)

- Implementation and experiments: **GBDT script extensions**, **ensemble scripts**, **experiment logging**, **Kaggle submission iterations**.  
- Methodological narrative: **Private + CV for generalization**, **Occam’s razor**, **negative experiments** in the report.  
- *Teammates’ LR baseline / RF: add a division-of-labor table in the merged document.*

---

## 5. Mapping to `PROJECT_LOG_EN.md` / `PROJECT_LOG_ZH.md` and code

- Cross-reference **`PROJECT_LOG_EN.md`** or **`PROJECT_LOG_ZH.md`** for the project overview and final recommendations (keep numbers consistent).  
- Example artifact paths: `outputs_test/gbdt_experiment_log.csv`, `ensemble_output/`, `oof_*.csv` (if using Ridge).

---

## 6. Video (≤5 minutes) — boosting only for now; time reserved for teammates

| Time | Content | Notes |
|------|---------|-------|
| 0:00–0:30 | Task and metric | ROC-AUC; Public vs Private |
| 0:30–1:00 | **Reserved — teammate: Baseline (Logistic)** | One slide or “see full written report” |
| 1:00–1:30 | **Reserved — teammate: RF (Bagging)** | Same |
| 1:30–4:00 | **Author: Boosting line** | GBDT pipeline, CV, key tables, best blend vs Ridge, negative experiments |
| 4:00–4:45 | Conclusion and compliance | No private sharing; final model choice |
| 4:45–5:00 | Buffer / Q&A | |

> If teammate segments are not ready: use **two “Reserved for teammate” slides** and say verbally that details are in the full written report.

---

## 7. Merge checklist (for future add-on documents)

- [ ] Teammate baseline: method, CV, best LB, file name  
- [ ] Teammate RF: method, CV, best LB, file name  
- [ ] Official team name and **single final submission** with Private/Public  
- [ ] Unified references: scikit-learn, LightGBM, XGBoost, Kaggle competition page  

---

**Filename**: `Documentated_report_on_DSAI4203_Project.md`  
**Last updated**: Draft creation date — update version and date after merging teammates’ sections.
