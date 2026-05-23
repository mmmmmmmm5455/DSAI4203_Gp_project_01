# DSAI4203_Project 書面報告（個人草稿 · Boosting 主線）

> **English version**: `Documentated_report_on_DSAI4203_Project.md`

> **文件狀態**：本稿僅含 **本人負責之 GBDT／Boosting、融合與實驗紀錄**。組員之 **Baseline（Logistic Regression）**、**Bagging（Random Forest）** 將以另檔補上後再合併。  
> **影片**：可先只錄 boosting；預留組員段落之時間配額見 §6。  
> **數據來源**：下列 Kaggle 分數為匯出之 leaderboard 紀錄；**定稿前請與 Kaggle 截圖逐筆核對**。

**競賽**：Kaggle — *Playground Series*（糖尿病二元分類）  
**評分指標**：ROC-AUC（**Private** 為主、**Public** 為輔）  
**表格欄位約定（本文件）**：**Private \| Public**（與常見「Private 較低、Public 較高」之顯示一致）

---

## 1. Summary & Objectives（個人章節 · 草稿）

### 1.1 背景摘要

本專題參與 Kaggle 糖尿病預測競賽，任務為**表格資料上的二元分類**，以 **ROC-AUC** 評分。相較於僅優化訓練集準確率，本競賽更強調 **泛化（generalization）** 與 **過擬合（overfitting）**：模型在本地交叉驗證（CV）與 Leaderboard（Public / Private）上可能不一致，故須以系統化實驗紀錄與對照。

### 1.2 本人負責範圍（Boosting / GBDT）

- **梯度提升決策樹（GBDT）**：以 **LightGBM**、**XGBoost** 為主線；**CatBoost** 作為可選對照（正文／附錄配置見 §3）。  
- **驗證策略**：**Stratified K-fold CV**（例如 5-fold），同步追蹤 **Train AUC、Val AUC、Train–Val Gap** 以診斷過擬合。  
- **方法論改進**：特徵工程、單模型調參、**多模型機率融合（blending）**、**基於 OOF 的二階堆疊（Ridge meta-learner）**；並記錄 **未採用** 或 **效果較差** 的變體（例如 tail-only Ridge），呼應課程「學習歷程」之要求。

### 1.3 學習目標（對齊 rubric）

| 課程目標 | 本人工作如何對應 |
|----------|------------------|
| 練習競賽式 ML 流程 | 可重現腳本（如 `main_gbdt.py`、`ensemble.py`）、實驗日誌（`gbdt_experiment_log.csv`） |
| 理解泛化與過擬合 | Gap、CV vs Leaderboard、以 Private 為導向、負向結果（tail-Ridge、部分融合未優於簡單 blend） |
| 發展適當解法 | GBDT 為強表格基線；單模型穩定後再做融合與二階學習 |

### 1.4 團隊其餘部分（預留合併）

- **Baseline（Logistic Regression）**、**Random Forest（Bagging）**：*待組員文件併入總報告（§X 待定）*。  
- 本文件 **不重複** 該兩條線之細節；合併稿中可視課程要求增加橫向對照表。

---

## 2. Improvements in Methodologies — Part 2c（Boosting · 必寫）

### 2.1 為何採用 Boosting（GBDT）

- 於表格資料上，樹模型能捕捉 **非線性** 與 **特徵交互**；梯度提升透過序列化弱學習器降低偏差。  
- 相較單棵淺樹，在控制 **學習率、子採樣、正則（L1/L2、`min_child_samples` 等）** 下，**Boosting** 較易在 **偏差–方差** 間取得平衡——須同時以 **CV Gap** 與 **Private Leaderboard** 驗證。

### 2.2 端到端流程（可繪流程圖）

1. **資料讀入與特徵工程**（醫學／統計直覺 + 非線性變換，例如血壓衍生量、血脂比、年齡／BMI 平方與交互項等——以實際 `main_gbdt.py` 為準）。  
2. **單模型訓練**：LGBM／XGB（及可選 CatBoost）+ 5-fold OOF 預測。  
3. **超參數調整**：學習率、樹數、`num_leaves`、列／行採樣、`reg_alpha`／`reg_lambda`；可輔以 **Optuna**（§3）。  
4. **融合**：  
   - **機率加權 blend**（強模型線性混合）；  
   - **Rank-based blend**（對排名平均，對分佈偏移較穩健）；  
   - **OOF + Ridge**：在 OOF 預測上學習線性權重（與簡單 blend 對照）。  
5. **取捨原則**：當 **Private 分數極接近** 時，依 **Occam’s razor（奧卡姆剃刀）** 偏好 **較簡單、可解釋、維護成本較低** 的方案（例如雙 LGBM blend 對照完整 OOF Ridge 管線）。

### 2.3 主要實驗結果（Kaggle Leaderboard 整理）

下表依你實際提交紀錄整理；**定稿前請與 Kaggle 後台逐筆核對**。

#### A. 個人 Logistic 實驗（Jerry version · 對照用，非 boosting 主線）

| 說明 | Private | Public |
|------|--------:|-------:|
| C = 1（`baseline_submission_C1.csv`） | 0.67943 | 0.68492 |
| C = 0.1（`baseline_submission_C0p1.csv`） | 0.67944 | 0.68492 |
| C = 0.3（`baseline_submission_C03.csv`） | 0.67943 | 0.68492 |
| `baseline_submission.csv`（註記「3」） | 0.67964 | 0.68515 |
| `baseline_submission.csv`（較早） | 0.67973 | 0.68515 |

> **寫作提示**：此區間內調整 C 對 LB 影響極小，呼應 **線性模型天花板**；**正文仍以 GBDT 為主**，上表可放附錄或一句對照。

#### B. 單模型：XGBoost／LightGBM（節選）

| Run／檔案 | Private | Public |
|-----------|--------:|-------:|
| `xgb_v1` | 0.69180 | 0.69530 |
| `xgb_v2_lr003_t800` | 0.69230 | 0.69541 |
| `xgb_v3_lr008_t300` | 0.69146 | 0.69488 |
| `lgbm_v1` | 0.69360 | 0.69700 |
| `lgbm_v2_lr003_t800` | 0.69348 | 0.69693 |
| `lgbm_v3_lr004_t700` | 0.69371 | 0.69705 |
| `lgbm_v4_lr006_t400` | 0.69345 | 0.69688 |
| `lgbm_v5_lr004_t650` | 0.69369 | 0.69703 |
| `lgbm_v6_lr004_t750` | 0.69375 | 0.69708 |
| `lgbm_tune1` … `lgbm_tune6`（依實際跑法） | 0.69338–0.69371 | 0.69618–0.69701 |
| `lgbm_tune2` | **0.69376** | **0.69705** |
| `cat_feat_v1_gbdt_submission` | 0.69152 | 0.69571 |

**可寫的分析要點**：  
- 同為 GBDT，**LGBM 整體優於初期 XGB**（若與本地 CV Gap 一致可並述）。  
- **tune 系列**顯示「並非越複雜越好」——部分 tune 在 Private 上未優於 **tune2／v6**。

#### C. 融合與二階學習（核心）

| 提交檔 | Private | Public | 備註 |
|--------|--------:|-------:|------|
| `blend_lgbm_tune2_v6_45_55` | **0.69396** | **0.69727** | 與 55/45 同分，**私榜最佳區間** |
| `blend_lgbm_tune2_v6_55_45` | **0.69396** | **0.69727** | 同上 |
| `ensemble_ridge.csv` | 0.69394 | 0.69724 | 全 OOF Ridge，**極接近**最佳 blend |
| `ensemble_auc_weighted.csv` | 0.69356 | 0.69701 | 僅 Val AUC 軟加權，略遜 |
| `rank_blend_3models_45_40_15` | 0.69388 | **0.69731** | Public 最佳，Private 略低 |
| `rank_blend_4models_40_35_15_10` | 0.69384 | 0.69720 | 四模型未勝過三模型 |
| `blend3_lgbm_tune2_v6_xgbv2_40_40_20` | 0.69386 | 0.69713 | 三模型 blend |
| `blend3_lgbm_tune2_v6_xgbv2_45_45_10` | 0.69392 | 0.69721 | |

**建議結論句**：以 **Private** 為主目標時，**雙 LGBM 機率 blend（45/55）** 與 **全資料 OOF Ridge** 同屬最強水準；兩者極接近時，可依 **Occam’s razor** 以較簡潔的 **雙模型 blend** 作為主要敘事。

#### D. 負向實驗（必寫「學習與取捨」）

| 提交檔 | Private | Public | 解讀（草稿） |
|--------|--------:|-------:|--------------|
| `ensemble_tail_ridge.csv` | 0.69346 | 0.69698 | 僅 tail 擬合 Ridge，**未優於**全 OOF／最佳 blend |
| `ensemble_tail_ridge_a1.csv` | 0.69332 | 0.69689 | 同上 |
| `ensemble_tail_ridge_a01.csv` | 0.69246 | 0.69608 | 此處較小 alpha 更差 |
| `lgbm_optuna30_gbdt_submission` | 0.69336 | 0.69598 | **Optuna** 搜尋未超越最佳手動區間 |
| `lgbm_tune2_orig_te_gbdt_submission` | 0.69340 | 0.69688 | 含外部 TE 之變體未勝過最佳 blend（**須聲明規則與資料來源**） |

> **寫作提示**：負向結果不是失敗，而是說明 **多數想法須經 Private 驗證**，並呼應泛化主題。

#### E. 其他

| 提交檔 | Private | Public | 備註 |
|--------|--------:|-------:|------|
| `oof_cat_feat_v1.csv` | *Error* | — | 上傳失敗；正文一句 + 附錄錯誤說明 |

---

## 3. Optuna、外部 TE、CatBoost — 正文 vs 附錄／參考文獻

| 主題 | 建議位置 | 理由 |
|------|----------|------|
| **Optuna** | **正文一段 + 附錄表** | 說明「曾作自動調參」且 **未優於手動最佳**（與 `lgbm_optuna30` 呼應）；trials／搜尋空間放附錄或引用 `*_optuna_best.json`。 |
| **外部 Target Encoding（TE）** | **正文簡述合規性 + 附錄** | 正文強調「僅公開資料、符合競賽外部資料規則」；技術細節與 `lgbm_tune2_orig_te` 對照放附錄。 |
| **CatBoost** | **以附錄為主**（或正文一句） | 頁數有限時：正文寫「曾對照 CatBoost／特徵版本」；完整 CV 與 `cat_feat_v1` LB 放附錄。 |
| **XGB vs LGBM 主線** | **正文** | 與「方法論改進」最相關，且已有連續對照表。 |

若全文須 **≤12 頁**：正文保留 **流程 + 主表 + 取捨結論**；其餘以 **附錄 A：完整提交表**、**附錄 B：Optuna／TE 設定** 處理。

---

## 4. 個人貢獻聲明（定稿時與組員對齊）

- 實作與實驗：**GBDT 主程式擴充**、**融合腳本**、**實驗日誌維護**、**Kaggle 提交迭代**。  
- 方法論敘事：**以 Private + CV 判斷泛化**、**Occam’s razor**、**負向實驗** 寫入報告。  
- *組員之 LR baseline／RF：於合併稿補充分工表。*

---

## 5. 與 `PROJECT_LOG_ZH.md`／`PROJECT_LOG_EN.md` 及程式碼的對應

- 總覽與最終建議請與 **`PROJECT_LOG_ZH.md`** 或 **`PROJECT_LOG_EN.md`** 互相引用（避免兩份數字不一致）。  
- 產出示例路徑：`outputs_test/gbdt_experiment_log.csv`、`ensemble_output/`、`oof_*.csv`（若使用 Ridge）。

---

## 6. 影片（≤5 分鐘）建議分鏡 — 目前僅 boosting，預留組員

| 時長 | 內容 | 說明 |
|------|------|------|
| 0:00–0:30 | 任務與指標 | ROC-AUC；Public vs Private |
| 0:30–1:00 | **【預留】組員：Baseline（Logistic）** | 一頁投影片或「詳見書面報告」 |
| 1:00–1:30 | **【預留】組員：RF（Bagging）** | 同上 |
| 1:30–4:00 | **本人：Boosting 主線** | GBDT 流程、CV、重點表、最佳 blend vs Ridge、負向實驗 |
| 4:00–4:45 | 結論與合規 | 無私下分享；最終模型選擇理由 |
| 4:45–5:00 | 緩衝／Q&A | |

> 若組員片段尚未準備：可使用 **兩張「Reserved for teammate」投影片** 佔位，口頭說明細節見完整書面報告。

---

## 7. 合併清單（供日後新增文件使用）

- [ ] 組員 Baseline：方法、CV、最佳 LB、檔名  
- [ ] 組員 RF：方法、CV、最佳 LB、檔名  
- [ ] 團隊正式隊名、**最終繳交之單一 submission** 與 Private／Public  
- [ ] 統一參考文獻：scikit-learn、LightGBM、XGBoost、Kaggle 競賽頁  

---

**檔名**：`Documentated_report_on_DSAI4203_Project_ZH.md`  
**最後更新**：草稿建立日 — 合併組員內容後請更新版本號與日期。
