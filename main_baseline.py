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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder

TARGET_COL = "diagnosed_diabetes"
ID_COL = "id"


@dataclass
class BaselineConfig:
    model_type: str = "logistic_regression"
    C: float = 1.0
    max_iter: int = 3000


def read_data(data_dir: Path):
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    sample_df = pd.read_csv(data_dir / "sample_submission.csv")
    return train_df, test_df, sample_df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ("triglycerides", "cholesterol_total"):
        df[f"{col}_log"] = np.log1p(df[col])
    return df


def build_preprocessor(train_df, feature_cols):
    cat_cols = [c for c in feature_cols if is_string_dtype(train_df[c])]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sca", MaxAbsScaler())
            ]), num_cols),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]), cat_cols),
        ]
    )


def update_baseline_log(log_path, run_name, train_auc, val_auc):
    log_file = Path(log_path)
    new_entry = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Run_Name": run_name,
        "Model": "Logistic_Regression_Baseline",
        "Train_AUC": round(train_auc, 6),
        "Val_AUC": round(val_auc, 6),
        "Complexity": "Linear (Baseline)"
    }
    df_new = pd.DataFrame([new_entry])
    if log_file.exists():
        df_old = pd.read_csv(log_file)
        df_final = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_final = df_new
    df_final.to_csv(log_file, index=False)


def run_baseline():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="baseline_v1")
    args = parser.parse_args()

    out_dir = Path("outputs_baseline")
    out_dir.mkdir(exist_ok=True)

    train_df, test_df, sample_df = read_data(Path("data"))
    train_df = feature_engineering(train_df)
    test_df = feature_engineering(test_df)

    feature_cols = [c for c in train_df.columns if c not in (ID_COL, TARGET_COL)]
    X = train_df[feature_cols]
    y = train_df[TARGET_COL].values.astype(int)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_pred = np.zeros(len(X))
    train_aucs = []

    print(f"--- Running Baseline: Logistic Regression ---")
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        model = LogisticRegression(solver="saga", C=1.0, max_iter=3000, random_state=42)
        pipe = Pipeline([
            ("preprocess", build_preprocessor(train_df, feature_cols)),
            ("model", model)
        ])

        pipe.fit(X_tr, y_tr)

        train_aucs.append(roc_auc_score(y_tr, pipe.predict_proba(X_tr)[:, 1]))
        val_proba = pipe.predict_proba(X_va)[:, 1]
        oof_pred[va_idx] = val_proba

        print(f"Fold {fold} Val AUC: {roc_auc_score(y_va, val_proba):.4f}")

    avg_train = np.mean(train_aucs)
    avg_val = roc_auc_score(y, oof_pred)
    print(f"\nFinal Baseline Results:")
    print(f"Average Train AUC: {avg_train:.4f}")
    print(f"Overall Val AUC: {avg_val:.4f}")

    update_baseline_log(out_dir / "baseline_log.csv", args.run_name, avg_train, avg_val)

    pipe.fit(X, y)
    sub = sample_df.copy()
    sub[TARGET_COL] = pipe.predict_proba(test_df[feature_cols])[:, 1]
    sub.to_csv(out_dir / "baseline_submission.csv", index=False)
    print(f"Baseline submission saved to outputs_baseline/")


if __name__ == "__main__":
    run_baseline()
