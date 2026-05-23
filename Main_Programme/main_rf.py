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
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder

TARGET_COL = "diagnosed_diabetes"
ID_COL = "id"


@dataclass
class ModelConfig:
    model_type: str
    C: float | None = None
    class_weight_balanced: bool = False
    max_iter: int = 3000
    alpha: float | None = None
    l2_tol: float = 1e-3
    n_estimators: int = 100
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: str | int | float = "sqrt"
    class_weight_rf: str | None = None


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


def build_model(cfg: ModelConfig):
    if cfg.model_type == "logreg":
        return LogisticRegression(solver="saga", C=cfg.C, max_iter=cfg.max_iter, random_state=42,
                                  class_weight="balanced" if cfg.class_weight_balanced else None)
    if cfg.model_type == "random_forest":
        return RandomForestClassifier(n_estimators=cfg.n_estimators, max_depth=cfg.max_depth,
                                      min_samples_split=cfg.min_samples_split, min_samples_leaf=cfg.min_samples_leaf,
                                      random_state=42, n_jobs=-1)
    raise ValueError(f"Unknown model: {cfg.model_type}")


# 【核心功能】：更新实验日志
def update_experiment_log(log_path, run_name, cfg, train_auc, val_auc, val_loss):
    log_file = Path(log_path)
    new_entry = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Run_Name": run_name,
        "Model": cfg.model_type,
        "Params": f"C={cfg.C}, depth={cfg.max_depth}, trees={cfg.n_estimators}",
        "Train_AUC": round(train_auc, 6),
        "Val_AUC": round(val_auc, 6),
        "AUC_Gap": round(train_auc - val_auc, 6),  # 体现过拟合
        "Val_LogLoss": round(val_loss, 6)
    }
    df_new = pd.DataFrame([new_entry])
    if log_file.exists():
        df_old = pd.read_csv(log_file)
        df_final = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_final = df_new
    df_final.to_csv(log_file, index=False)
    print(f"--> Experiment log updated at: {log_path}")


def fit_and_predict(train_df, test_df, feature_cols, sample_df, cfg, cv_folds, seed, run_name, out_dir,
                    make_submission):
    X = train_df[feature_cols]
    y = train_df[TARGET_COL].values.astype(int)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    oof_pred = np.zeros(len(X))
    train_scores = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        pipe = Pipeline([("preprocess", build_preprocessor(train_df, feature_cols)), ("model", build_model(cfg))])
        pipe.fit(X_tr, y_tr)

        # 记录训练集分数用于观察过拟合
        train_proba = pipe.predict_proba(X_tr)[:, 1]
        train_scores.append(roc_auc_score(y_tr, train_proba))

        val_proba = pipe.predict_proba(X_va)[:, 1]
        oof_pred[va_idx] = val_proba
        print(f"Fold {fold} AUC: {roc_auc_score(y_va, val_proba):.4f}")

    avg_train_auc = np.mean(train_scores)
    final_val_auc = roc_auc_score(y, oof_pred)
    final_val_loss = log_loss(y, oof_pred)

    # 保存日志
    update_experiment_log(out_dir / "experiment_log.csv", run_name, cfg, avg_train_auc, final_val_auc, final_val_loss)

    if make_submission:
        pipe.fit(X, y)  # 全量训练提交
        sub = sample_df.copy()
        sub[TARGET_COL] = pipe.predict_proba(test_df[feature_cols])[:, 1]
        sub.to_csv(out_dir / f"{run_name}_submission.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="exp_v1")
    parser.add_argument("--model_type", type=str, default="random_forest")
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--C", type=float, default=1.0)
    args = parser.parse_args()

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    train_df, test_df, sample_df = read_data(Path("data"))
    train_df = feature_engineering(train_df)
    test_df = feature_engineering(test_df)

    cfg = ModelConfig(model_type=args.model_type, C=args.C, max_depth=args.max_depth, n_estimators=args.n_estimators)

    fit_and_predict(train_df, test_df, get_feature_columns(train_df), sample_df, cfg, 5, 42, args.run_name, out_dir,
                    True)


def get_feature_columns(df):
    return [c for c in df.columns if c not in (ID_COL, TARGET_COL)]


if __name__ == "__main__":
    main()