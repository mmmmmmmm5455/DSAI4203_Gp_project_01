"""
Tail-based Ridge stacking (optional experiment).

Fits Ridge only on training rows with id >= CUTOFF (idea: tail may resemble test
distribution). Applies learned weights to full test predictions.

Notes for course / integrity:
- This can improve Public LB while hurting Private; report both metrics.
- sklearn Ridge has no random_state; results are deterministic for default solver.

Usage:
  cd .../DSAI4203_Project
  python ensemble_tail_ridge.py --cutoff_id 678260 --alpha 10.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

TARGET_COL = "diagnosed_diabetes"
ID_COL = "id"


def find_submission_for_run(input_dir: Path, run_name: str) -> Path | None:
    """Match lgbm_tune2 -> lgbm_tune2_gbdt_submission.csv or *lgbm_tune2*submission*.csv"""
    exact = input_dir / f"{run_name}_gbdt_submission.csv"
    if exact.exists():
        return exact
    for p in sorted(input_dir.glob("*.csv")):
        if "submission" not in p.name.lower():
            continue
        stem = p.stem
        if stem == f"{run_name}_gbdt_submission" or stem.startswith(f"{run_name}_"):
            if run_name in stem:
                return p
    return None


def load_oof_aligned(oof_path: Path, train_ids: np.ndarray) -> np.ndarray:
    df = pd.read_csv(oof_path)
    if ID_COL not in df.columns:
        raise ValueError(f"{oof_path} must have '{ID_COL}'")
    col = "oof" if "oof" in df.columns else TARGET_COL
    if col not in df.columns:
        raise ValueError(f"{oof_path} needs 'oof' or '{TARGET_COL}'")
    m = df.set_index(ID_COL).reindex(train_ids)
    if m[col].isna().any():
        raise ValueError(f"{oof_path}: missing OOF for some train ids")
    return m[col].values.astype(float)


def load_sub_aligned(sub_path: Path, test_ids: np.ndarray) -> np.ndarray:
    df = pd.read_csv(sub_path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"{sub_path} missing {TARGET_COL}")
    m = df.set_index(ID_COL).reindex(test_ids)
    if m[TARGET_COL].isna().any():
        raise ValueError(f"{sub_path}: missing predictions for some test ids")
    return m[TARGET_COL].values.astype(float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="ensemble_input")
    parser.add_argument("--output_dir", type=str, default="ensemble_output")
    parser.add_argument("--train_csv", type=str, default="data/train.csv")
    parser.add_argument("--test_csv", type=str, default="data/test.csv")
    parser.add_argument("--cutoff_id", type=int, default=678260)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--out_name", type=str, default="ensemble_tail_ridge.csv")
    parser.add_argument(
        "--no_intercept",
        action="store_true",
        help="Use fit_intercept=False (default is True, as in many stacking recipes)",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    input_dir = root / args.input_dir
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(root / args.train_csv)
    test = pd.read_csv(root / args.test_csv)
    train_ids = train[ID_COL].values
    test_ids = test[ID_COL].values
    y = train[TARGET_COL].values.astype(int)
    tail_mask = train_ids >= args.cutoff_id

    oof_files = sorted(input_dir.glob("oof_*.csv"))
    if not oof_files:
        raise FileNotFoundError(f"No oof_*.csv in {input_dir}")

    model_names: list[str] = []
    for p in oof_files:
        stem = p.stem
        if not stem.startswith("oof_"):
            continue
        name = stem[4:]
        sub_path = find_submission_for_run(input_dir, name)
        if sub_path is None:
            print(f"[skip] No submission for OOF {p.name} (run={name})")
            continue
        model_names.append(name)

    if not model_names:
        raise RuntimeError("No matched (oof, submission) pairs.")

    print(f"Models ({len(model_names)}): {model_names}")
    print(f"Tail rows: {tail_mask.sum()} / {len(train)} (id >= {args.cutoff_id})")

    oof_cols = []
    sub_cols = []
    for name in model_names:
        oof_path = input_dir / f"oof_{name}.csv"
        sub_path = find_submission_for_run(input_dir, name)
        assert sub_path is not None
        oof_cols.append(load_oof_aligned(oof_path, train_ids))
        sub_cols.append(load_sub_aligned(sub_path, test_ids))

    oof_mat = np.column_stack(oof_cols)
    sub_mat = np.column_stack(sub_cols)

    fit_intercept = not args.no_intercept
    ridge = Ridge(alpha=args.alpha, fit_intercept=fit_intercept)
    ridge.fit(oof_mat[tail_mask], y[tail_mask])

    w = ridge.coef_.astype(float)
    b = float(ridge.intercept_) if fit_intercept else 0.0
    print("\nRidge (tail-only train):")
    for name, wi in zip(model_names, w):
        print(f"  {name:<40} {wi:+.6f}")
    if fit_intercept:
        print(f"  {'intercept':<40} {b:+.6f}")

    oof_blend = oof_mat @ w + b
    oof_blend = np.clip(oof_blend, 1e-6, 1 - 1e-6)
    print(f"\nOOF AUC (full train):  {roc_auc_score(y, oof_blend):.6f}")
    print(f"OOF AUC (tail only):   {roc_auc_score(y[tail_mask], oof_blend[tail_mask]):.6f}")

    test_pred = sub_mat @ w + b
    test_pred = np.clip(test_pred, 0.0, 1.0)

    out = pd.DataFrame({ID_COL: test_ids, TARGET_COL: test_pred})
    out_path = output_dir / args.out_name
    out.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
