import argparse
import json
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
    # LogisticRegression params
    C: float | None = None
    class_weight_balanced: bool = False
    max_iter: int = 3000
    # SGDClassifier params
    alpha: float | None = None
    l2_tol: float = 1e-3
    # RandomForest params
    n_estimators: int = 100
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: str | int | float = "sqrt"
    class_weight_rf: str | None = None


def read_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    sample_path = data_dir / "sample_submission.csv"
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    sample_df = pd.read_csv(sample_path)
    if TARGET_COL not in train_df.columns:
        raise ValueError(f"train.csv must contain target column `{TARGET_COL}`")
    if TARGET_COL in test_df.columns:
        raise ValueError(f"test.csv must NOT contain target column `{TARGET_COL}`")
    if ID_COL not in train_df.columns or ID_COL not in test_df.columns:
        raise ValueError("Both train.csv and test.csv must contain `id` column")
    if ID_COL not in sample_df.columns or TARGET_COL not in sample_df.columns:
        raise ValueError("sample_submission.csv must contain `id` and target column")
    return train_df, test_df, sample_df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    feature_cols = [c for c in df.columns if c not in (ID_COL, TARGET_COL)]
    if not feature_cols:
        raise ValueError("No feature columns found.")
    return feature_cols


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]
    df["map"] = df["diastolic_bp"] + (df["pulse_pressure"] / 3.0)
    df["tg_hdl_ratio"] = df["triglycerides"] / (df["hdl_cholesterol"] + 1e-6)
    df["cholesterol_ratio"] = df["cholesterol_total"] / (df["hdl_cholesterol"] + 1e-6)
    df["bmi_age"] = df["bmi"] * df["age"]
    df["activity_sleep_ratio"] = df["physical_activity_minutes_per_week"] / (
            df["sleep_hours_per_day"] * 7.0 + 1e-6
    )
    for col in ("triglycerides", "cholesterol_total"):
        df[f"{col}_log"] = np.log1p(df[col])
    return df


def build_preprocessor(train_df: pd.DataFrame, feature_cols: list[str]) -> ColumnTransformer:
    cat_cols = [c for c in feature_cols if is_string_dtype(train_df[c])]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MaxAbsScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            # 注意：为了方便提取特征名，这里最好设为False，或者保持稀疏但在特征重要性处理时注意维度
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    return preprocessor


def build_model(cfg: ModelConfig):
    if cfg.model_type == "logreg":
        if cfg.C is None:
            raise ValueError("LogisticRegression requires cfg.C")
        return LogisticRegression(
            solver="saga",
            C=cfg.C,
            max_iter=cfg.max_iter,
            random_state=42,
            class_weight="balanced" if cfg.class_weight_balanced else None,
        )
    if cfg.model_type == "sgd_logloss":
        if cfg.alpha is None:
            raise ValueError("SGDClassifier requires cfg.alpha")
        return SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=cfg.alpha,
            max_iter=cfg.max_iter,
            tol=cfg.l2_tol,
            random_state=42,
        )
    if cfg.model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            min_samples_split=cfg.min_samples_split,
            min_samples_leaf=cfg.min_samples_leaf,
            max_features=cfg.max_features,
            class_weight=cfg.class_weight_rf,
            random_state=42,
            n_jobs=-1,
            oob_score=False,
            verbose=0
        )
    raise ValueError(f"Unknown model_type: {cfg.model_type}")


def make_pipeline(preprocessor: ColumnTransformer, cfg: ModelConfig) -> Pipeline:
    return Pipeline(steps=[("preprocess", preprocessor), ("model", build_model(cfg))])


def fit_and_predict(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: list[str],
        sample_df: pd.DataFrame,
        cfg: ModelConfig,
        cv_folds: int,
        seed: int,
        run_name: str,
        out_dir: Path,
        make_submission: bool = True,
):
    X = train_df[feature_cols]
    y = train_df[TARGET_COL].values.astype(int)

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    oof_pred = np.zeros(len(X), dtype=float)
    test_pred_acc = np.zeros(len(test_df), dtype=float)
    fold_scores: list[float] = []

    # 【新增】：用于存储每一折的特征重要性
    fold_importances = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_val = X.iloc[tr_idx], X.iloc[va_idx]
        y_train, y_val = y[tr_idx], y[va_idx]

        preprocessor = build_preprocessor(train_df, feature_cols)
        pipe = make_pipeline(preprocessor, cfg)
        pipe.fit(X_train, y_train)

        val_proba = pipe.predict_proba(X_val)[:, 1]
        oof_pred[va_idx] = val_proba
        fold_auc = roc_auc_score(y_val, val_proba)
        fold_scores.append(float(fold_auc))
        print(f"Fold {fold}/{cv_folds} ROC-AUC: {fold_auc:.6f}")

        # 【新增】：提取特征重要性核心逻辑
        try:
            # 1. 从预处理器中获取转换后的特征名称（特别是 OneHot 编码后展开的特征）
            feature_names = pipe.named_steps["preprocess"].get_feature_names_out()
            model = pipe.named_steps["model"]

            # 2. 根据模型类型提取重要性或权重绝对值
            if hasattr(model, "feature_importances_"):
                # 针对树模型 (如 Random Forest)
                importances = model.feature_importances_
            elif hasattr(model, "coef_"):
                # 针对线性模型 (如 Logistic Regression, SGD), 取绝对值表示重要程度
                importances = np.abs(model.coef_[0])
            else:
                importances = np.zeros(len(feature_names))

            # 3. 存入 DataFrame
            imp_df = pd.DataFrame({
                "feature": feature_names,
                "importance": importances,
                "fold": fold
            })
            fold_importances.append(imp_df)
        except Exception as e:
            print(f"Warning: Could not extract feature importances for fold {fold}. Error: {e}")

        if make_submission:
            test_pred_acc += pipe.predict_proba(test_df[feature_cols])[:, 1] / cv_folds

    valid_metrics = {
        "roc_auc": float(roc_auc_score(y, oof_pred)),
        "log_loss": float(log_loss(y, np.clip(oof_pred, 1e-15, 1 - 1e-15))),
    }
    metrics_out = {
        "run_name": run_name,
        "config": asdict(cfg),
        "cv_folds": cv_folds,
        "seed": seed,
        "fold_roc_auc": fold_scores,
        "metrics": valid_metrics,
    }
    out_dir.mkdir(parents=True, exist_ok=True)

    # 写入评估指标
    (out_dir / f"{run_name}_metrics.json").write_text(json.dumps(metrics_out, indent=2))

    # 【新增】：计算并保存平均特征重要性
    if fold_importances:
        all_imp_df = pd.concat(fold_importances, axis=0)
        # 按特征名称分组，计算交叉验证的平均重要性
        mean_imp_df = all_imp_df.groupby("feature")["importance"].mean().reset_index()
        # 按重要性降序排列
        mean_imp_df = mean_imp_df.sort_values(by="importance", ascending=False)

        # 写入 CSV
        imp_path = out_dir / f"{run_name}_feature_importances.csv"
        mean_imp_df.to_csv(imp_path, index=False)
        print(f"Saved Feature Importances: {imp_path}")

    sub_path = None
    if make_submission:
        submission = sample_df.copy()
        submission[TARGET_COL] = test_pred_acc
        sub_path = out_dir / "submission.csv"
        submission.to_csv(sub_path, index=False)

    return metrics_out, (str(sub_path) if sub_path is not None else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_type", type=str, default="random_forest",
                        choices=["logreg", "sgd_logloss", "random_forest"])
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument("--class_weight_balanced", action="store_true")
    parser.add_argument("--max_iter", type=int, default=3000)
    parser.add_argument("--l2_tol", type=float, default=1e-3)
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--min_samples_split", type=int, default=2)
    parser.add_argument("--min_samples_leaf", type=int, default=1)
    parser.add_argument("--max_features", type=str, default="sqrt")
    parser.add_argument("--class_weight_rf", type=str, default=None,
                        choices=[None, "balanced", "balanced_subsample"])

    parser.add_argument("--run_name", type=str, default="rf_run")
    parser.add_argument(
        "--no_submission",
        action="store_true",
        help="Only compute validation metrics; do not fit on full train or write submission.csv.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    train_df, test_df, sample_df = read_data(data_dir)
    train_df = feature_engineering(train_df)
    test_df = feature_engineering(test_df)
    feature_cols = get_feature_columns(train_df)

    cfg = ModelConfig(
        model_type=args.model_type,
        C=args.C if args.model_type == "logreg" else None,
        alpha=args.alpha if args.model_type == "sgd_logloss" else None,
        class_weight_balanced=args.class_weight_balanced,
        max_iter=args.max_iter,
        l2_tol=args.l2_tol,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        class_weight_rf=args.class_weight_rf
    )

    metrics_out, sub_path = fit_and_predict(
        train_df=train_df,
        test_df=test_df,
        feature_cols=feature_cols,
        sample_df=sample_df,
        cfg=cfg,
        cv_folds=args.cv_folds,
        seed=args.seed,
        run_name=args.run_name,
        out_dir=out_dir,
        make_submission=not args.no_submission,
    )

    print(json.dumps(metrics_out["metrics"], indent=2))
    if sub_path is not None:
        print(f"Saved Submission: {sub_path}")


if __name__ == "__main__":
    main()