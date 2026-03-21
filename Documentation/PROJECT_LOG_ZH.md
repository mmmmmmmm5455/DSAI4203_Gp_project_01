# 糖尿病 Kaggle 專案紀錄（中文）

## 1）文件目的
此文件用於完整記錄：
- 模型與程式碼的調整歷程，
- 每次實驗的 CV 指標，
- 團隊回報的 Kaggle 成績，
- 最終模型選擇依據（可直接放入作業附件）。

## 2）程式與流程變更紀錄

### A. 初期整理
- 檢查 Helen 提供檔案，移除重複參考檔。
- 建立 `outputs_test` 作為統一輸出資料夾。
- 固定在正確目錄執行 `main_gbdt.py`。

### B. GBDT 調參階段
- 以 `main_gbdt.py` 進行 LGBM/XGB 的 5-fold CV 與 submission 生成。
- 增加可調參數（CLI）：
  - `num_leaves`, `max_depth`, `min_child_samples`
  - `subsample`, `colsample_bytree`
  - `reg_alpha`, `reg_lambda`

### C. 擴充與融合
- 新增 CatBoost 支援（`--model_type cat`）。
- 新增 rank averaging 融合流程。
- 新增高品質特徵工程：
  - `pulse_pressure`, `map`
  - `ldl_hdl_ratio`, `chol_hdl_ratio`
  - `age_sq`, `bmi_sq`, `bp_age`, `activity_sleep_ratio`
- 新增 Optuna 搜索入口：
  - `--optuna_trials`
  - `--optuna_timeout_sec`
  - 自動輸出最佳參數檔 `*_optuna_best.json`。

## 3）CV 實驗紀錄（來源：`outputs_test/gbdt_experiment_log.csv`）

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

## 4）Kaggle 成績紀錄（依團隊回報整理）

| Submission | Private | Public | 備註 |
|---|---:|---:|---|
| xgb_v1 | 0.69180 | 0.69530 | XGB 初始基線 |
| xgb_v2_lr003_t800 | 0.69230 | 0.69541 | 比 xgb_v1 略好 |
| xgb_v3_lr008_t300 | 0.69146 | 0.69488 | 表現下降 |
| lgbm_v2_lr003_t800 | 0.69348 | 0.69693 | LGBM 明顯較佳 |
| lgbm_v3_lr004_t700 | 0.69371 | 0.69705 | 再提升 |
| lgbm_v4_lr006_t400 | 0.69345 | 0.69688 | 略降 |
| lgbm_v5_lr004_t650 | 0.69369 | 0.69703 | 穩定 |
| lgbm_v6_lr004_t750 | 0.69375 | 0.69708 | 單模型最佳 |
| blend_lgbm_tune2_v6_45_55 | 0.69396 | 0.69727 | 私榜最佳融合 |
| blend_lgbm_tune2_v6_55_45 | 0.69396 | 0.69727 | 同上 |
| rank_blend_3models_45_40_15 | 0.69388 | 0.69731 | 公榜最佳 |
| rank_blend_4models_40_35_15_10 | 0.69384 | 0.69720 | 略低於3模型 |

## 5）目前最終建議（建模已封版）
- **若以 Private 為主**：`blend_lgbm_tune2_v6_45_55.csv`（Private **0.69396** / Public **0.69727**）。
- **全資料 OOF + Ridge 融合**（`ensemble.py --method ridge`）：Private **0.69394** / Public **0.69724**（與上者幾乎同階，可作報告中的「堆疊學習權重」版本）。
- **若展示 Public**：`rank_blend_3models_45_40_15.csv`（Public **0.69731**，Private 略低於最佳 blend）。

### 融合實驗補充（Kaggle）

| Submission | Private | Public | 說明 |
|---|---:|---:|---|
| ensemble_ridge.csv | 0.69394 | 0.69724 | OOF 上 Ridge 學權重 |
| ensemble_auc_weighted.csv | 0.69356 | 0.69701 | 僅用 Val_AUC 軟最大加權 |
| ensemble_tail_ridge.csv | 0.69346 | 0.69698 | 僅 tail 訓練 Ridge（alpha=10） |
| ensemble_tail_ridge_a1.csv | 0.69332 | 0.69689 | alpha=1 |
| ensemble_tail_ridge_a01.csv | 0.69246 | 0.69608 | alpha=0.1 |

## 6）衝擊更高分的後續（已暫停，改寫報告與影片）
- 曾嘗試 Optuna、外部目標編碼、更多融合；**未顯著優於上述 Private 最佳** 則不再迭代，避免過度調 leaderboard。

## 6b）日誌與產出檔對照（個人負責部分可追溯）

| 類型 | 路徑 / 檔案 |
|------|-------------|
| 每次 GBDT 跑完自動追加 | `outputs_test/gbdt_experiment_log.csv`（Train/Val AUC、Gap、LogLoss） |
| 專案總覽（本檔 + `PROJECT_LOG_EN.md`） | 流程變更、CV 表、Kaggle 表、規則與外部資料說明 |
| Optuna 最佳參數 | `outputs_test/*_optuna_best.json` |
| OOF（供 Ridge） | `outputs_test/oof_<run_name>.csv`（`--save_oof` 時產生） |
| 融合權重後設 | `ensemble_output/*_meta.json`（`ensemble.py`） |

## 8）個人負責：如何調整模型、如何選「最好」的一個（含 Occam’s razor）

**分工**：GBDT 主線（LightGBM / XGBoost / CatBoost）、特徵與 CLI 擴充、融合腳本（`ensemble.py`、`ensemble_tail_ridge.py`）、實驗日誌維護。

**調整順序（摘要）**  
1. **固定流程**：5-fold Stratified CV → 記錄 Val AUC 與 Train–Val **Gap**（過擬合診斷）。  
2. **先粗調**：`learning_rate`、`n_estimators`；再細調 `num_leaves`、`subsample`、`reg_*` 等。  
3. **單模型最佳區間** 出現後，改做 **多模型融合**（機率 blend、rank average）。  
4. **有 OOF 時**：`ensemble.py --method ridge` 在**全訓練 OOF** 上學線性權重；**無 OOF 時**曾用 Val_AUC 加權作對照。  
5. **對照實驗（未採用）**：`ensemble_tail_ridge.py` 僅在 `id≥cutoff` 的 tail 上擬合 Ridge——**Private 明顯低於**最佳 blend / 全 OOF Ridge，故寫入報告作「負向結果」，不當 final。

**選擇最終模型的規則**  
1. 競賽官方指標為 **ROC-AUC**；**Private AUC** 作為泛化主參考（輔以 CV）。  
2. 當 **Private 極接近**（例如 0.69396 vs 0.69394）時，依 **Occam’s razor（奧卡姆剃刀）**：在表現相近時**偏好較簡單、可解釋、參數較少**的方案。  
3. 因此：**雙模型機率 blend**（僅兩個 submission 加權）與 **全 OOF Ridge**（需 OOF + 線性元學習）皆合理；**tail-only Ridge** 結構更針對 leaderboard 分段、且分數未提升，**不採用**。

**一句話（可放報告）**  
本競賽以 ROC-AUC 評分；我們以 **Private AUC 與 CV** 判斷泛化，並在分數接近時依 **Occam’s razor** 選擇較簡潔的最終方案（例如雙 LightGBM 融合），**不採用**未提升 Private 的 tail-Ridge 變體。

## 7）可選：外部參考表的目標編碼（`main_gbdt.py` 已支援）

- 將競賽 **Data** 頁允許的 BRFSS 類資料下載到 `data/`，檔名可為：`original.csv`、`diabetes_dataset.csv`、`diabetes_health_indicators.csv` 等（見程式內 `_EXTERNAL_CSV_CANDIDATES`）。
- 目標欄會自動嘗試對齊：`Diabetes_binary` / `Outcome` 等 → `diagnosed_diabetes`。
- 僅對**在參考表中基數 ≤ 50** 的欄位新增 `{col}_ext_te`，避免對連續數值誤做 groupby。
- **使用前請自行確認 Kaggle 競賽規則**是否允許該外部資料。
- 若不想使用：加參數 `--no_external_te`。
