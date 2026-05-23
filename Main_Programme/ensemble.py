"""
Ensemble multiple Kaggle submission CSVs.

Modes:
  1) Ridge on OOF predictions (best) — put oof_<run_name>.csv in INPUT_DIR
  2) Val_AUC-weighted blend — auto-reads outputs_test/gbdt_experiment_log.csv
  3) Equal weights

Usage:
  cd .../DSAI4203_Project
  mkdir ensemble_input
  copy outputs_test\\lgbm_tune2_gbdt_submission.csv ensemble_input\\
  python ensemble.py --method auc
  python ensemble.py --method ridge   # needs oof_*.csv
  python ensemble.py --method equal
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

TARGET_COL = "diagnosed_diabetes"
ID_COL = "id"


def run_name_from_submission_filename(name: str) -> str:
    """lgbm_tune2_gbdt_submission.csv -> lgbm_tune2"""
    stem = Path(name).stem
    for suf in ("_gbdt_submission", "_submission"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    return stem


def load_val_auc_map(log_path: Path) -> dict[str, float]:
    if not log_path.exists():
        return {}
    df = pd.read_csv(log_path)
    if "Run_Name" not in df.columns or "Val_AUC" not in df.columns:
        return {}
    # keep last occurrence per Run_Name (most recent run)
    out: dict[str, float] = {}
    for _, row in df.iterrows():
        out[str(row["Run_Name"])] = float(row["Val_AUC"])
    return out


def read_submissions(input_dir: Path) -> tuple[list[str], list[str], np.ndarray, pd.Series]:
    files = sorted(
        f
        for f in os.listdir(input_dir)
        if f.lower().endswith(".csv") and "submission" in f.lower()
    )
    if not files:
        raise FileNotFoundError(f"No *submission*.csv in {input_dir}")

    first = pd.read_csv(input_dir / files[0])
    if ID_COL not in first.columns:
        raise ValueError(f"Expected column '{ID_COL}' in {files[0]}")
    if TARGET_COL not in first.columns:
        raise ValueError(f"Expected column '{TARGET_COL}' in {files[0]}")
    ids = first[ID_COL].copy()

    preds = []
    names = []
    for f in files:
        df = pd.read_csv(input_dir / f)
        if not df[ID_COL].equals(ids):
            df = df.set_index(ID_COL).reindex(ids).reset_index()
        if TARGET_COL not in df.columns:
            raise ValueError(f"Missing {TARGET_COL} in {f}")
        preds.append(df[TARGET_COL].values.astype(float))
        names.append(run_name_from_submission_filename(f))

    return names, files, np.column_stack(preds), ids


def load_oof_matrix(
    input_dir: Path,
    sub_names: list[str],
    train_ids: pd.Series,
) -> np.ndarray:
    cols = []
    for name in sub_names:
        oof_path = input_dir / f"oof_{name}.csv"
        if not oof_path.exists():
            raise FileNotFoundError(
                f"Ridge mode needs {oof_path.name} for each model (missing for '{name}').\n"
                "Generate OOF files by re-training with:\n"
                f"  python main_gbdt.py ... --run_name {name} --out_dir outputs_test --save_oof\n"
                f"Then copy outputs_test/oof_{name}.csv into {input_dir}/\n"
                "Or use: python ensemble.py --method auc  (no OOF required)"
            )
        df = pd.read_csv(oof_path)
        if ID_COL not in df.columns:
            raise ValueError(f"{oof_path} must have '{ID_COL}'")
        pred_col = "oof" if "oof" in df.columns else TARGET_COL
        if pred_col not in df.columns:
            raise ValueError(f"{oof_path} needs column 'oof' or '{TARGET_COL}'")
        df = df.set_index(ID_COL).reindex(train_ids).reset_index()
        cols.append(df[pred_col].values.astype(float))
    return np.column_stack(cols)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="ensemble_input")
    parser.add_argument("--output_dir", type=str, default="ensemble_output")
    parser.add_argument(
        "--method",
        type=str,
        choices=["ridge", "auc", "equal"],
        default="auc",
        help="ridge: needs oof_<run>.csv | auc: weights from gbdt log | equal",
    )
    parser.add_argument(
        "--log_csv",
        type=str,
        default="outputs_test/gbdt_experiment_log.csv",
    )
    parser.add_argument("--train_csv", type=str, default="data/train.csv")
    parser.add_argument("--test_csv", type=str, default="data/test.csv")
    parser.add_argument("--out_name", type=str, default="ensemble_ridge.csv")
    parser.add_argument("--ridge_alpha", type=float, default=0.1)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    input_dir = root / args.input_dir
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = root / args.train_csv
    test_path = root / args.test_csv
    log_path = root / args.log_csv

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Create folder and copy submissions: {input_dir}")

    sub_names, sub_files, sub_preds, test_ids = read_submissions(input_dir)
    print(f"Found {len(sub_names)} submission files:")
    for n, f in zip(sub_names, sub_files):
        print(f"  {f} -> run_key={n}")

    train = pd.read_csv(train_path)
    y_true = train[TARGET_COL].values.astype(int)
    train_ids = train[ID_COL]

    if args.method == "ridge":
        oof_mat = load_oof_matrix(input_dir, sub_names, train_ids)
        ridge = Ridge(alpha=args.ridge_alpha, fit_intercept=False, positive=True)
        ridge.fit(oof_mat, y_true)
        w = ridge.coef_.astype(float)
        s = w.sum()
        if s > 0:
            w = w / s
        print("\nRidge weights (normalized):")
        for name, wi in zip(sub_names, w):
            print(f"  {name:<35} {wi:.6f}")
        test_pred = sub_preds @ w
        oof_blend = oof_mat @ w
        print(f"\nOOF ensemble ROC-AUC: {roc_auc_score(y_true, oof_blend):.6f}")

    elif args.method == "auc":
        auc_map = load_val_auc_map(log_path)
        raw_auc: list[float] = []
        for name in sub_names:
            if name in auc_map:
                raw_auc.append(auc_map[name])
            else:
                raw_auc.append(float("nan"))
                print(f"  [warn] No Val_AUC in log for '{name}' — using 0.5 placeholder")
        raw = np.array(raw_auc, dtype=float)
        if np.any(np.isnan(raw)):
            raw = np.where(np.isnan(raw), 0.5, raw)
        # softmax-style emphasis on stronger models (temperature via scale)
        centered = raw - raw.max()
        w = np.exp(centered * 50.0)
        w = w / w.sum()
        print(f"\nVal_AUC-based weights (from {log_path}):")
        for name, wi, auc_val in zip(sub_names, w, raw):
            print(f"  {name:<35} Val_AUC={auc_val:.6f}  weight={wi:.6f}")
        test_pred = sub_preds @ w

    else:
        w = np.ones(len(sub_names), dtype=float) / len(sub_names)
        print("\nEqual weights:")
        for name, wi in zip(sub_names, w):
            print(f"  {name:<35} {wi:.6f}")
        test_pred = sub_preds @ w

    test_df = pd.read_csv(test_path)
    out = pd.DataFrame({ID_COL: test_df[ID_COL], TARGET_COL: test_pred})
    out_path = output_dir / args.out_name
    out.to_csv(out_path, index=False)

    meta = {
        "method": args.method,
        "models": sub_names,
        "weights": w.tolist() if isinstance(w, np.ndarray) else list(w),
    }
    (output_dir / (Path(args.out_name).stem + "_meta.json")).write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
