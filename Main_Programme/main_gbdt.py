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

# 导入 GBDT 模型
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
try:
    import optuna
except ImportError:
    optuna = None

TARGET_COL = "diagnosed_diabetes"
ID_COL = "id"

# Optional external BRFSS-style CSVs (place under data/). Used only for global target-encoding
# of low-cardinality columns. Check competition rules before using extra data.
_EXTERNAL_CSV_CANDIDATES = (
    "original.csv",
    "diabetes_dataset.csv",
    "diabetes_health_indicators.csv",
    "diabetes_012_health_indicators_BRFSS2015.csv",
    "diabetes_binary_health_indicators_BRFSS2015.csv",
)

import warnings

# 忽略掉这个特定的特征名字警告
warnings.filterwarnings("ignore", message="X does not have valid feature names")


@dataclass
class GBDTConfig:
    model_type: str  # 'lgbm' or 'xgb' or 'cat'
    learning_rate: float = 0.05
    n_estimators: int = 500
    num_leaves: int = 31  # LGBM 特有
    max_depth: int = 6  # XGB 特有
    min_child_samples: int = 20  # LGBM 特有
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1  # L1 正则
    reg_lambda: float = 0.1  # L2 正则


def read_data(data_dir: Path):
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    sample_df = pd.read_csv(data_dir / "sample_submission.csv")
    return train_df, test_df, sample_df


def _normalize_external_target_column(orig: pd.DataFrame) -> pd.DataFrame:
    """Map common Kaggle/BRFSS target names to diagnosed_diabetes."""
    orig = orig.copy()
    lower_map = {c.lower(): c for c in orig.columns}
    aliases = {
        "diabetes_binary": TARGET_COL,
        "diabetesbinary": TARGET_COL,
        "outcome": TARGET_COL,
        "diabetes": TARGET_COL,
        "diabetes_012": TARGET_COL,
    }
    for alias, new_name in aliases.items():
        if alias in lower_map:
            old = lower_map[alias]
            if old != new_name:
                orig = orig.rename(columns={old: new_name})
            break
    return orig


def load_optional_original(data_dir: Path) -> pd.DataFrame | None:
    for name in _EXTERNAL_CSV_CANDIDATES:
        path = data_dir / name
        if not path.exists():
            continue
        orig = pd.read_csv(path)
        orig = _normalize_external_target_column(orig)
        if TARGET_COL not in orig.columns:
            print(
                f"[external_te] Skipping {path.name}: no `{TARGET_COL}` "
                f"(or Diabetes_binary / Outcome) column after rename."
            )
            continue
        print(f"[external_te] Loaded reference table: {path.name} ({len(orig)} rows)")
        return orig
    return None


def add_external_target_encoding(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    orig_df: pd.DataFrame,
    max_categories: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each column present in train, test, and orig: if cardinality in orig
    is low enough, add ``{col}_ext_te`` = mean(target | col) from orig only.
    Unknown levels -> global mean in orig (reduces leakage vs using competition train).
    """
    train_df = train_df.copy()
    test_df = test_df.copy()
    orig_df = orig_df.copy()
    global_mean = float(orig_df[TARGET_COL].mean())

    skip = {ID_COL, TARGET_COL}
    for col in train_df.columns:
        if col in skip or col not in orig_df.columns:
            continue
        nuniq = int(orig_df[col].nunique(dropna=False))
        if nuniq > max_categories:
            continue
        mean_map = orig_df.groupby(col, dropna=False)[TARGET_COL].mean()
        new_col = f"{col}_ext_te"
        train_df[new_col] = train_df[col].map(mean_map).astype(float)
        test_df[new_col] = test_df[col].map(mean_map).astype(float)
        train_df[new_col] = train_df[new_col].fillna(global_mean)
        test_df[new_col] = test_df[new_col].fillna(global_mean)

    added = [c for c in train_df.columns if c.endswith("_ext_te")]
    if added:
        print(f"[external_te] Added {len(added)} features: {added[:8]}{'...' if len(added) > 8 else ''}")
    else:
        print("[external_te] No overlapping low-cardinality columns; nothing added.")
    return train_df, test_df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Existing features
    df["tg_hdl_ratio"] = df["triglycerides"] / (df["hdl_cholesterol"] + 1e-6)
    df["bmi_age"] = df["bmi"] * df["age"]
    for col in ("triglycerides", "cholesterol_total"):
        df[f"{col}_log"] = np.log1p(df[col])

    # Added higher-value features (interaction, ratio, polynomial)
    df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]
    df["map"] = df["diastolic_bp"] + (df["pulse_pressure"] / 3.0)
    df["ldl_hdl_ratio"] = df["ldl_cholesterol"] / (df["hdl_cholesterol"] + 1e-6)
    df["chol_hdl_ratio"] = df["cholesterol_total"] / (df["hdl_cholesterol"] + 1e-6)
    df["age_sq"] = df["age"] ** 2
    df["bmi_sq"] = df["bmi"] ** 2
    df["bp_age"] = df["systolic_bp"] * df["age"]
    df["activity_sleep_ratio"] = df["physical_activity_minutes_per_week"] / (
        df["sleep_hours_per_day"] * 7.0 + 1e-6
    )

    # Additional engineered features matching current dataset columns
    df["sbp_dbp_ratio"] = df["systolic_bp"] / (df["diastolic_bp"] + 1e-6)
    df["pp_hr_ratio"] = (df["systolic_bp"] - df["diastolic_bp"]) / (df["heart_rate"] + 1e-6)
    df["bp_hr_interaction"] = df["systolic_bp"] * df["heart_rate"]
    df["ldl_tg_ratio"] = df["ldl_cholesterol"] / (df["triglycerides"] + 1e-6)
    df["non_hdl_chol"] = df["cholesterol_total"] - df["hdl_cholesterol"]
    df["waist_bmi_interaction"] = df["waist_to_hip_ratio"] * df["bmi"]
    df["age_bmi_sq"] = df["age"] * (df["bmi"] ** 2)
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
            max_depth=cfg.max_depth,
            min_child_samples=cfg.min_child_samples,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            reg_alpha=cfg.reg_alpha,
            reg_lambda=cfg.reg_lambda,
            random_state=42,
            n_jobs=-1,
            importance_type='gain',  # 在报告中展示特征增益
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
            tree_method="hist"  # 提速
        )
    elif cfg.model_type == "cat":
        return CatBoostClassifier(
            learning_rate=cfg.learning_rate,
            n_estimators=cfg.n_estimators,
            depth=cfg.max_depth,
            l2_leaf_reg=cfg.reg_lambda,
            random_seed=42,
            loss_function="Logloss",
            eval_metric="AUC",
            verbose=0,
        )
    raise ValueError(f"Unknown GBDT model: {cfg.model_type}")


def run_cv_metrics(train_df, feature_cols, cfg, cv_folds, seed, verbose=True):
    X = train_df[feature_cols]
    y = train_df[TARGET_COL].values.astype(int)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    oof_pred = np.zeros(len(X))
    train_scores = []
    fold_val_aucs = []

    if verbose:
        print(f"\nStarting {cfg.model_type.upper()} with {cv_folds}-Fold CV...")

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        preprocessor = build_preprocessor(train_df, feature_cols)
        model = build_gbdt_model(cfg)
        pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
        pipe.fit(X_tr, y_tr)

        tr_auc = roc_auc_score(y_tr, pipe.predict_proba(X_tr)[:, 1])
        va_pred = pipe.predict_proba(X_va)[:, 1]
        va_auc = roc_auc_score(y_va, va_pred)

        train_scores.append(tr_auc)
        fold_val_aucs.append(va_auc)
        oof_pred[va_idx] = va_pred

        if verbose:
            print(f"Fold {fold}: Train AUC = {tr_auc:.4f}, Val AUC = {va_auc:.4f}")

    avg_train_auc = float(np.mean(train_scores))
    final_val_auc = float(roc_auc_score(y, oof_pred))
    final_val_loss = float(log_loss(y, oof_pred))
    return avg_train_auc, final_val_auc, final_val_loss, fold_val_aucs, oof_pred


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


def fit_and_predict_gbdt(
    train_df,
    test_df,
    feature_cols,
    sample_df,
    cfg,
    cv_folds,
    seed,
    run_name,
    out_dir,
    save_oof: bool = False,
):
    avg_train_auc, final_val_auc, final_val_loss, _, oof_pred = run_cv_metrics(
        train_df=train_df,
        feature_cols=feature_cols,
        cfg=cfg,
        cv_folds=cv_folds,
        seed=seed,
        verbose=True,
    )

    if save_oof:
        oof_path = out_dir / f"oof_{run_name}.csv"
        pd.DataFrame({ID_COL: train_df[ID_COL].values, "oof": oof_pred}).to_csv(
            oof_path, index=False
        )
        print(f"Saved OOF predictions: {oof_path}")

    # 记录到专用的 GBDT 日志
    update_gbdt_log(out_dir / "gbdt_experiment_log.csv", run_name, cfg, avg_train_auc, final_val_auc, final_val_loss)

    # 全量训练并生成提交
    print(f"Fitting final {cfg.model_type} on full data for submission...")
    X = train_df[feature_cols]
    y = train_df[TARGET_COL].values.astype(int)
    preprocessor = build_preprocessor(train_df, feature_cols)
    model = build_gbdt_model(cfg)
    pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
    pipe.fit(X, y)
    sub = sample_df.copy()
    sub[TARGET_COL] = pipe.predict_proba(test_df[feature_cols])[:, 1]
    sub_path = out_dir / f"{run_name}_gbdt_submission.csv"
    sub.to_csv(sub_path, index=False)
    print(f"Final Submission Saved: {sub_path}")


def run_optuna_search(train_df, feature_cols, cfg, cv_folds, seed, n_trials, timeout_sec):
    if optuna is None:
        raise ImportError("Optuna is not installed. Run: python -m pip install optuna")

    base_cfg = asdict(cfg)

    def objective(trial):
        trial_cfg = GBDTConfig(**base_cfg)
        trial_cfg.learning_rate = trial.suggest_float("learning_rate", 0.01, 0.08)
        trial_cfg.n_estimators = trial.suggest_int("n_estimators", 400, 1200)
        trial_cfg.max_depth = trial.suggest_int("max_depth", 4, 10)
        trial_cfg.subsample = trial.suggest_float("subsample", 0.6, 1.0)
        trial_cfg.colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0)
        trial_cfg.reg_alpha = trial.suggest_float("reg_alpha", 0.0, 2.0)
        trial_cfg.reg_lambda = trial.suggest_float("reg_lambda", 0.0, 5.0)

        if trial_cfg.model_type == "lgbm":
            trial_cfg.num_leaves = trial.suggest_int("num_leaves", 20, 96)
            trial_cfg.min_child_samples = trial.suggest_int("min_child_samples", 10, 80)
        elif trial_cfg.model_type == "cat":
            trial_cfg.max_depth = trial.suggest_int("max_depth", 4, 9)

        _, val_auc, _, _, _ = run_cv_metrics(
            train_df=train_df,
            feature_cols=feature_cols,
            cfg=trial_cfg,
            cv_folds=cv_folds,
            seed=seed,
            verbose=False,
        )
        return val_auc

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout_sec if timeout_sec > 0 else None)
    return study


def main():
    parser = argparse.ArgumentParser()
    # 关键参数：输出到另一个文件夹
    parser.add_argument("--out_dir", type=str, default="outputs_gbdt")
    parser.add_argument("--run_name", type=str, default="lgbm_v1")
    parser.add_argument("--model_type", type=str, default="lgbm", choices=["lgbm", "xgb", "cat"])
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--trees", type=int, default=500)
    parser.add_argument("--num_leaves", type=int, default=31)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--min_child_samples", type=int, default=20)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample_bytree", type=float, default=0.8)
    parser.add_argument("--reg_alpha", type=float, default=0.1)
    parser.add_argument("--reg_lambda", type=float, default=0.1)
    parser.add_argument("--optuna_trials", type=int, default=0)
    parser.add_argument("--optuna_timeout_sec", type=int, default=0)
    parser.add_argument(
        "--save_oof",
        action="store_true",
        help="Save out-of-fold train predictions to oof_<run_name>.csv (for Ridge stacking).",
    )
    parser.add_argument(
        "--no_external_te",
        action="store_true",
        help="Disable optional target-encoding from data/*.csv reference tables.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    # 读取数据（假设数据在 data 文件夹）
    data_dir = Path("data")
    train_df, test_df, sample_df = read_data(data_dir)
    if not args.no_external_te:
        orig_df = load_optional_original(data_dir)
        if orig_df is not None:
            train_df, test_df = add_external_target_encoding(train_df, test_df, orig_df)
    train_df = feature_engineering(train_df)
    test_df = feature_engineering(test_df)

    cfg = GBDTConfig(
        model_type=args.model_type,
        learning_rate=args.lr,
        n_estimators=args.trees,
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        min_child_samples=args.min_child_samples,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
    )

    if args.optuna_trials > 0:
        print(f"Running Optuna search: trials={args.optuna_trials}, model={cfg.model_type}")
        study = run_optuna_search(
            train_df=train_df,
            feature_cols=[c for c in train_df.columns if c not in (ID_COL, TARGET_COL)],
            cfg=cfg,
            cv_folds=5,
            seed=42,
            n_trials=args.optuna_trials,
            timeout_sec=args.optuna_timeout_sec,
        )
        best_params = study.best_params
        print("Optuna best score (Val AUC):", study.best_value)
        print("Optuna best params:", best_params)

        # Apply best params back to config
        for k, v in best_params.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

        (out_dir / f"{args.run_name}_optuna_best.json").write_text(
            json.dumps(
                {
                    "best_value": study.best_value,
                    "best_params": best_params,
                    "n_trials": args.optuna_trials,
                    "model_type": cfg.model_type,
                },
                indent=2,
            )
        )

    fit_and_predict_gbdt(
        train_df,
        test_df,
        [c for c in train_df.columns if c not in (ID_COL, TARGET_COL)],
        sample_df,
        cfg,
        5,
        42,
        args.run_name,
        out_dir,
        save_oof=args.save_oof,
    )


if __name__ == "__main__":
    main()
