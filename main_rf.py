import argparse
import os
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder

TARGET_COL = "diagnosed_diabetes"
ID_COL = "id"


@dataclass
class ModelConfig:
    n_estimators: int = 200
    max_depth: int = 10
    min_samples_leaf: int = 20
    max_features: str = "sqrt"
    class_weight: str = "balanced"


def read_data(data_dir: Path):
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    sample_df = pd.read_csv(data_dir / "sample_submission.csv")
    return train_df, test_df, sample_df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tg_hdl_ratio"] = df["triglycerides"] / (df["hdl_cholesterol"] + 1e-6)
    df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]
    df["bmi_age"] = df["bmi"] * df["age"]
    for col in ("triglycerides", "cholesterol_total", "bmi"):
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


def update_experiment_log(log_path, run_name, cfg, train_auc, val_auc):
    log_file = Path(log_path)
    new_entry = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Run_Name": run_name,
        "Model": "RandomForest_Final",
        "Max_Depth": cfg.max_depth,
        "Train_AUC": round(train_auc, 6),
        "Val_AUC": round(val_auc, 6),
        "Gap": round(train_auc - val_auc, 6)
    }
    df_new = pd.DataFrame([new_entry])
    if log_file.exists():
        df_old = pd.read_csv(log_file)
        df_final = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_final = df_new
    df_final.to_csv(log_file, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="rf_final")
    args = parser.parse_args()

    out_dir = Path("outputs_rf")
    out_dir.mkdir(exist_ok=True)

    train_df, test_df, sample_df = read_data(Path("data"))
    train_df = feature_engineering(train_df)
    test_df = feature_engineering(test_df)

    feature_cols = [c for c in train_df.columns if c not in (ID_COL, TARGET_COL)]
    X = train_df[feature_cols]
    y = train_df[TARGET_COL].values.astype(int)

    cfg = ModelConfig()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_pred = np.zeros(len(X))
    train_scores = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        model = RandomForestClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            min_samples_leaf=cfg.min_samples_leaf,
            max_features=cfg.max_features,
            class_weight=cfg.class_weight,
            random_state=42,
            n_jobs=-1
        )

        pipe = Pipeline([("preprocess", build_preprocessor(train_df, feature_cols)), ("model", model)])
        pipe.fit(X_tr, y_tr)

        train_scores.append(roc_auc_score(y_tr, pipe.predict_proba(X_tr)[:, 1]))
        val_proba = pipe.predict_proba(X_va)[:, 1]
        oof_pred[va_idx] = val_proba
        print(f"Fold {fold} Val AUC: {roc_auc_score(y_va, val_proba):.4f}")

    avg_train_auc = np.mean(train_scores)
    final_val_auc = roc_auc_score(y, oof_pred)

    print(f"\nResults:")
    print(f"Train AUC: {avg_train_auc:.4f}, Val AUC: {final_val_auc:.4f}")

    update_experiment_log(out_dir / "rf_log.csv", args.run_name, cfg, avg_train_auc, final_val_auc)

    pipe.fit(X, y)
    sub = sample_df.copy()
    sub[TARGET_COL] = pipe.predict_proba(test_df[feature_cols])[:, 1]
    sub.to_csv(out_dir / "rf_submission.csv", index=False)


if __name__ == "__main__":
    main()
