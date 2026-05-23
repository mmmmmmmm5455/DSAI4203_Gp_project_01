import argparse
import json
import os
from datetime import datetime
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder

import lightgbm as lgb
import xgboost as xgb

TARGET_COL = "diagnosed_diabetes"
ID_COL = "id"

import warnings

warnings.filterwarnings("ignore", message="X does not have valid feature names")


@dataclass
class GBDTConfig:
    model_type: str
    learning_rate: float = 0.05
    n_estimators: int = 500
    num_leaves: int = 31
    max_depth: int = 6
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1


def read_data(data_dir: Path):
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    sample_df = pd.read_csv(data_dir / "sample_submission.csv")
    return train_df, test_df, sample_df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tg_hdl_ratio"] = df["triglycerides"] / (df["hdl_cholesterol"] + 1e-6)
    df["bmi_age"] = df["bmi"] * df["age"]
    for col in ("triglycerides", "cholesterol_total"):
        df[f"{col}_log"] = np.log1p(df[col])
    return df


def build_preprocessor(train_df, feature_cols):
    cat_cols = [c for c in feature_cols if is_string_dtype(train_df[c])]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sca", MaxAbsScaler())]), num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols),
        ]
    )


def build_gbdt_model(cfg: GBDTConfig):
    if cfg.model_type == "lgbm":
        return lgb.LGBMClassifier(
            learning_rate=cfg.learning_rate,
            n_estimators=cfg.n_estimators,
            num_leaves=cfg.num_leaves,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            reg_alpha=cfg.reg_alpha,
            reg_lambda=cfg.reg_lambda,
            random_state=42,
            n_jobs=-1,
            importance_type='gain',
            verbosity=-1
        )
    elif cfg.model_type == "xgb":
        return xgb.XGBClassifier(
            learning_rate=cfg.learning_rate,
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            reg_alpha=cfg.reg_alpha,
            reg_lambda=cfg.reg_lambda,
            random_state=42,
            n_jobs=-1,
            tree_method="hist"
        )
    raise ValueError(f"Unknown GBDT model: {cfg.model_type}")


def update_gbdt_log(log_path, run_name, cfg, train_auc, val_auc, val_loss):
    log_file = Path(log_path)
    new_entry = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Run_Name": run_name,
        "Model": cfg.model_type,
        "Learning_Rate": cfg.learning_rate,
        "Trees": cfg.n_estimators,
        "Train_AUC": round(train_auc, 6),
        "Val_AUC": round(val_auc, 6),
        "Gap": round(train_auc - val_auc, 6),
        "Val_LogLoss": round(val_loss, 6)
    }
    df_new = pd.DataFrame([new_entry])
    if log_file.exists():
        df_old = pd.read_csv(log_file)
        df_final = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_final = df_new
    df_final.to_csv(log_file, index=False)
    print(f"--> GBDT Experiment log updated at: {log_path}")


def fit_and_predict_gbdt(train_df, test_df, feature_cols, sample_df, cfg, cv_folds, seed, run_name, out_dir):
    X = train_df[feature_cols]
    y = train_df[TARGET_COL].values.astype(int)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    oof_pred = np.zeros(len(X))
    train_scores = []

    print(f"\nStarting {cfg.model_type.upper()} with 5-Fold CV...")

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        preprocessor = build_preprocessor(train_df, feature_cols)
        model = build_gbdt_model(cfg)
        pipe = Pipeline([("preprocess", preprocessor), ("model", model)])

        pipe.fit(X_tr, y_tr)

        tr_auc = roc_auc_score(y_tr, pipe.predict_proba(X_tr)[:, 1])
        va_auc = roc_auc_score(y_va, pipe.predict_proba(X_va)[:, 1])

        train_scores.append(tr_auc)
        oof_pred[va_idx] = pipe.predict_proba(X_va)[:, 1]

        print(f"Fold {fold}: Train AUC = {tr_auc:.4f}, Val AUC = {va_auc:.4f}")

    avg_train_auc = np.mean(train_scores)
    final_val_auc = roc_auc_score(y, oof_pred)
    final_val_loss = log_loss(y, oof_pred)

    update_gbdt_log(out_dir / "gbdt_experiment_log.csv", run_name, cfg, avg_train_auc, final_val_auc, final_val_loss)

    print(f"Fitting final {cfg.model_type} on full data for submission...")
    pipe.fit(X, y)
    sub = sample_df.copy()
    sub[TARGET_COL] = pipe.predict_proba(test_df[feature_cols])[:, 1]
    sub_path = out_dir / f"{run_name}_gbdt_submission.csv"
    sub.to_csv(sub_path, index=False)
    print(f"Final Submission Saved: {sub_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="outputs_gbdt")
    parser.add_argument("--run_name", type=str, default="lgbm_v1")
    parser.add_argument("--model_type", type=str, default="lgbm", choices=["lgbm", "xgb"])
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--trees", type=int, default=500)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    data_dir = Path("data")
    train_df, test_df, sample_df = read_data(data_dir)
    train_df = feature_engineering(train_df)
    test_df = feature_engineering(test_df)

    cfg = GBDTConfig(
        model_type=args.model_type,
        learning_rate=args.lr,
        n_estimators=args.trees
    )

    fit_and_predict_gbdt(
        train_df, test_df,
        [c for c in train_df.columns if c not in (ID_COL, TARGET_COL)],
        sample_df, cfg, 5, 42, args.run_name, out_dir
    )


if __name__ == "__main__":
    main()
