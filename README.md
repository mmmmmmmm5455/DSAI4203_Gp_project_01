[README_中文_GBDT操作與變更紀錄.md](https://github.com/user-attachments/files/26162395/README_._GBDT.md)
# Kaggle 糖尿病預測：GBDT 說明與變更紀錄（中文）

## 1) XGB 與 LGBM 是什麼？

### XGB（XGBoost）
- 全名是 **Extreme Gradient Boosting**，屬於 GBDT 家族。
- 核心想法：一棵樹一棵樹地學，後面的樹專門修正前面模型犯的錯。
- 優點：通常效果穩定、在表格資料比賽常見、可調參數多。
- 缺點：資料大時訓練可能較慢，參數太多時容易亂調。

### LGBM（LightGBM）
- 也是 GBDT 家族，由 Microsoft 開發。
- 優點：訓練通常比 XGBoost 更快、記憶體效率高、大資料集常很有優勢。
- 缺點：若參數不當，可能更容易過擬合（需看 CV 與 LB）。

### 你這個專案的重點
- 兩者都適合這次 Kaggle 表格型資料。
- 你目前流程是：**先用 LGBM/XGB 建 baseline，再小步調參，比較 CV 與 Kaggle 分數**。

---

## 2) 這次我幫你做了哪些「非程式碼邏輯」變更

> 依照你的要求：不改模型程式邏輯，只做檔案整理與執行。

1. 檢查 Helen 提供的檔案是否重複。  
2. 發現 `main.py`（Helen）與你原本根目錄 `main_for_reference.py` 完全重複。  
3. 已刪除一份重複檔：`d:\Ai thing\DSAI4203\main_for_reference.py`。  
4. 建立輸出資料夾：`outputs_test`（位於 Helen 專案目錄）。  
5. 協助用正確路徑跑 `main_gbdt.py`，並產生 submission 檔與實驗 log。  

---

## 3) 你目前已完成的實驗（已記錄）

實驗 log 檔案：`outputs_test/gbdt_experiment_log.csv`

已跑 run（依時間）：
- `lgbm_v1`：lr=0.05, trees=500, Val_AUC=0.725179, Gap=0.011992
- `xgb_v1`：lr=0.05, trees=500, Val_AUC=0.724524, Gap=0.020452
- `xgb_v2_lr003_t800`：lr=0.03, trees=800, Val_AUC=0.724623, Gap=0.019688
- `xgb_v3_lr008_t300`：lr=0.08, trees=300, Val_AUC=0.723886, Gap=0.019817
- `lgbm_v2_lr003_t800`：lr=0.03, trees=800, Val_AUC=0.725083, Gap=0.011481

你提供的 Kaggle 成績：
- `xgb_v1`：Public = **0.69530**，Private = **0.69180**

---

## 4) 如何解讀目前結果

- **本地 CV（Val_AUC）**：目前最佳接近 `lgbm_v1`（0.725179）。
- **Gap（Train_AUC - Val_AUC）**：XGB 的 Gap 約 0.02，LGBM 約 0.011，代表目前 LGBM 較不容易過擬合。
- 結論：下一輪可優先在 LGBM 附近微調，再用 XGB 做補充比較。

---

## 5) 檔案說明（你最常用）

- `main_gbdt.py`：GBDT 主程式（LGBM / XGB）
- `outputs_test/gbdt_experiment_log.csv`：每次實驗自動紀錄
- `outputs_test/*_gbdt_submission.csv`：上傳 Kaggle 的提交檔

---

## 6) 常用指令（PowerShell）

先進入正確目錄：

```powershell
cd "d:\Ai thing\DSAI4203\Content_from_Helen\DSAI4203_Project\DSAI4203_Project"
```

跑 LGBM：

```powershell
python main_gbdt.py --model_type lgbm --run_name lgbm_v1 --out_dir outputs_test --lr 0.05 --trees 500
```

跑 XGB：

```powershell
python main_gbdt.py --model_type xgb --run_name xgb_v1 --out_dir outputs_test --lr 0.05 --trees 500
```

---

## 7) 交作業時可用的一句話總結

「我們以 GBDT（XGBoost / LightGBM）為主，使用 5-fold CV 控制過擬合，並以固定流程記錄每次參數調整對 Val_AUC 與 Kaggle Leaderboard 的影響，最後選擇泛化能力較佳的模型作為最終提交。」
