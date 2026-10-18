"""
validate_kfold_test_predictions.py
=====================================
Cross-fold validation for the per-sample MethylLlama test predictions
(extract_kfold_test_predictions.py). Runs locally, no GPU -- pure pandas/
numpy checks against the 5 CSV/JSON outputs after syncing them back from
the cluster.

Fails loudly (raises, nonzero exit) on any check failure, per spec.

Checks:
  1. Recomputed MedAE/MAE/R2 per fold match the official WandB test metrics
     (recorded below, verified directly from WandB this session) within
     tolerance.
  2. All 5 CSVs contain exactly the same 2,149 sample IDs.
  3. true_age is identical for every sample_id across all 5 folds.
  4. No duplicate IDs, no missing predictions, no NaNs, in any fold.

Usage:
  python scripts/finetune/validate_kfold_test_predictions.py \
      --dir outputs/bootstrap_predictions/methyllama
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Official test metrics per fold, verified directly from WandB
# (finetune-llama-small project, fold*-testeval runs) earlier this session.
OFFICIAL = {
    0: {"medae": 3.125000, "mae": 4.439657, "r2": 0.932128},
    1: {"medae": 3.215000, "mae": 4.457891, "r2": 0.933666},
    2: {"medae": 3.216484, "mae": 4.570466, "r2": 0.928401},
    3: {"medae": 3.156250, "mae": 4.540810, "r2": 0.929797},
    4: {"medae": 3.144424, "mae": 4.454264, "r2": 0.933833},
}
TOL = 0.01  # absolute tolerance for medae/mae (years) and r2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="outputs/bootstrap_predictions/methyllama")
    p.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    return p.parse_args()


def main():
    a = parse_args()
    d = Path(a.dir)
    failures = []

    dfs = {}
    for fold in a.folds:
        csv_path = d / f"fold_{fold}_predictions.csv"
        json_path = d / f"fold_{fold}_verification.json"
        if not csv_path.exists():
            failures.append(f"fold {fold}: missing {csv_path}")
            continue
        df = pd.read_csv(csv_path, dtype={"sample_id": str})
        dfs[fold] = df
        meta = json.loads(json_path.read_text())

        # Check 4: internal integrity -- run FIRST. A corrupted fold (dupes,
        # NaNs, wrong row count) must not crash metric computation and take
        # down the rest of the validation run with it (r2_score raises on
        # NaN input) -- report the integrity issue and skip metrics for that
        # fold only, so every other fold and the cross-fold checks still run.
        n_dupe = int(df["sample_id"].duplicated().sum())
        n_missing = int(df["predicted_age"].isna().sum())
        row_count_ok = len(df) == 2149
        if n_dupe:
            failures.append(f"fold {fold}: {n_dupe} duplicate sample_ids")
        if n_missing:
            failures.append(f"fold {fold}: {n_missing} missing predictions")
        if not row_count_ok:
            failures.append(f"fold {fold}: expected 2149 rows, got {len(df)}")

        if n_dupe or n_missing or not row_count_ok:
            print(f"fold {fold}: SKIPPING metric recomputation (integrity check failed above)")
            continue

        # Check 1: recompute metrics, compare to official WandB values
        from sklearn.metrics import r2_score
        pred = df["predicted_age"].values
        true = df["true_age"].values
        medae = float(np.median(np.abs(pred - true)))
        mae = float(np.mean(np.abs(pred - true)))
        r2 = float(r2_score(true, pred))
        off = OFFICIAL[fold]
        for name, val, off_val in [("medae", medae, off["medae"]), ("mae", mae, off["mae"]), ("r2", r2, off["r2"])]:
            if abs(val - off_val) > TOL:
                failures.append(
                    f"fold {fold}: recomputed {name}={val:.4f} vs official {off_val:.4f} "
                    f"(diff {abs(val-off_val):.4f} > tol {TOL}) -- extraction does NOT reproduce official eval!"
                )
        print(f"fold {fold}: recomputed medae={medae:.4f} (official {off['medae']:.4f}) | "
              f"mae={mae:.4f} (official {off['mae']:.4f}) | r2={r2:.4f} (official {off['r2']:.4f})")

    if len(dfs) == len(a.folds):
        # Check 2: identical sample_id sets across all folds
        id_sets = {fold: set(df["sample_id"]) for fold, df in dfs.items()}
        ref_fold = a.folds[0]
        ref_ids = id_sets[ref_fold]
        for fold, ids in id_sets.items():
            if ids != ref_ids:
                missing = ref_ids - ids
                extra = ids - ref_ids
                failures.append(
                    f"fold {fold}: sample_id set differs from fold {ref_fold} "
                    f"({len(missing)} missing, {len(extra)} extra)"
                )
        print(f"\nSample-ID set identical across all {len(dfs)} folds: "
              f"{all(ids == ref_ids for ids in id_sets.values())} ({len(ref_ids)} unique IDs)")

        # Check 3: true_age identical per sample_id across folds
        merged = None
        for fold, df in dfs.items():
            sub = df[["sample_id", "true_age"]].rename(columns={"true_age": f"true_age_fold{fold}"})
            merged = sub if merged is None else merged.merge(sub, on="sample_id", how="outer")
        age_cols = [c for c in merged.columns if c.startswith("true_age_fold")]
        age_std_per_row = merged[age_cols].std(axis=1, skipna=True)
        n_inconsistent = int((age_std_per_row > 1e-6).sum())
        if n_inconsistent:
            failures.append(f"true_age differs across folds for {n_inconsistent} sample_ids")
        print(f"true_age consistent across folds for all sample_ids: {n_inconsistent == 0}")

    print()
    if failures:
        print("=" * 70)
        print(f"VALIDATION FAILED ({len(failures)} issue(s)):")
        for f in failures:
            print(f"  - {f}")
        print("=" * 70)
        raise SystemExit(1)
    else:
        print("=" * 70)
        print("ALL VALIDATION CHECKS PASSED")
        print("=" * 70)


if __name__ == "__main__":
    main()
