"""
paired_bootstrap_comparison.py
=================================
Paired subject-level bootstrap comparison: MethylLlama vs MethylGPT on
the shared fixed 2,149-sample held-out test set.

Methodology (matches the constraint specified for this analysis):
  1. Load each model's 5 fold-level prediction CSVs. The SAME 2,149 subjects
     appear in every fold -- do NOT concatenate into 10,745 rows and treat
     them as independent samples.
  2. Average each model's 5 predictions per subject -> one prediction per
     subject per model (an ensemble-of-folds prediction).
  3. Join the two models' per-subject predictions by sample_id (inner join;
     must be exactly 2,149 matched subjects if both sides are correct).
  4. Paired bootstrap: resample SUBJECTS with replacement (the same
     resampled index set applied to both models simultaneously, preserving
     pairing), recompute MedAE/MAE/R2 for each model within each resample,
     and record the difference (MethylGPT - MethylLlama). The distribution
     of that difference across B resamples gives a 95% CI on the actual gap
     between the two models -- not just two separate per-model CIs.

Usage:
  python scripts/evaluation/paired_bootstrap_comparison.py \
      --methyllama_dir outputs/bootstrap_predictions/methyllama \
      --methylgpt_dir outputs/bootstrap_predictions/methylgpt \
      --n_boot 10000 --seed 0 \
      --outdir outputs/bootstrap_predictions
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--methyllama_dir", required=True)
    p.add_argument("--methylgpt_dir", required=True)
    p.add_argument("--n_boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default="outputs/bootstrap_predictions")
    return p.parse_args()


def load_and_average(model_dir: Path, expected_model_name: str) -> pd.DataFrame:
    """Load all 5 fold CSVs, verify consistency, average predictions per subject."""
    csvs = sorted(model_dir.glob("fold_*_predictions.csv"))
    if len(csvs) != 5:
        raise RuntimeError(f"{model_dir}: expected 5 fold_*_predictions.csv files, found {len(csvs)}")

    dfs = [pd.read_csv(f, dtype={"sample_id": str}) for f in csvs]
    all_df = pd.concat(dfs, ignore_index=True)

    models_seen = all_df["model"].unique().tolist()
    if models_seen != [expected_model_name]:
        raise RuntimeError(f"{model_dir}: expected model=={expected_model_name}, found {models_seen}")

    # Cross-fold consistency (same checks as validate_kfold_test_predictions.py)
    id_sets = [set(df["sample_id"]) for df in dfs]
    if not all(s == id_sets[0] for s in id_sets):
        raise RuntimeError(f"{model_dir}: fold_*_predictions.csv files do not share identical sample_id sets")
    if len(id_sets[0]) != 2149:
        raise RuntimeError(f"{model_dir}: expected 2149 unique sample_ids, found {len(id_sets[0])}")

    age_check = all_df.groupby("sample_id")["true_age"].nunique()
    if (age_check > 1).any():
        bad = age_check[age_check > 1].index.tolist()
        raise RuntimeError(f"{model_dir}: true_age differs across folds for {len(bad)} sample_ids, e.g. {bad[:5]}")

    avg = all_df.groupby("sample_id").agg(
        true_age=("true_age", "first"),
        predicted_age=("predicted_age", "mean"),
        n_folds=("predicted_age", "count"),
    ).reset_index()
    if (avg["n_folds"] != 5).any():
        raise RuntimeError(f"{model_dir}: not every subject has exactly 5 fold predictions")
    return avg[["sample_id", "true_age", "predicted_age"]]


def metrics(true, pred):
    return {
        "medae": float(np.median(np.abs(pred - true))),
        "mae": float(np.mean(np.abs(pred - true))),
        "r2": float(r2_score(true, pred)),
    }


def main():
    a = parse_args()
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    llama = load_and_average(Path(a.methyllama_dir), "MethylLlama")
    gpt = load_and_average(Path(a.methylgpt_dir), "MethylGPT")

    merged = llama.merge(gpt, on="sample_id", suffixes=("_llama", "_gpt"), how="inner")
    n_matched = len(merged)
    print(f"Matched subjects: {n_matched} (MethylLlama: {len(llama)}, MethylGPT: {len(gpt)})")
    if n_matched != 2149:
        raise RuntimeError(
            f"Expected 2149 matched subjects after joining by sample_id, got {n_matched}. "
            f"MethylLlama-only IDs: {len(set(llama.sample_id) - set(gpt.sample_id))}, "
            f"MethylGPT-only IDs: {len(set(gpt.sample_id) - set(llama.sample_id))}. "
            f"Do not proceed with a mismatched test set -- investigate before trusting any result."
        )

    # true_age should agree between the two models (same underlying test set / same ages)
    age_diff = (merged["true_age_llama"] - merged["true_age_gpt"]).abs()
    n_age_mismatch = int((age_diff > 1e-6).sum())
    if n_age_mismatch:
        raise RuntimeError(f"true_age disagrees between MethylLlama and MethylGPT for {n_age_mismatch} subjects")

    true = merged["true_age_llama"].values
    pred_llama = merged["predicted_age_llama"].values
    pred_gpt = merged["predicted_age_gpt"].values

    point = {
        "MethylLlama": metrics(true, pred_llama),
        "MethylGPT": metrics(true, pred_gpt),
    }
    point["gap_gpt_minus_llama"] = {
        k: point["MethylGPT"][k] - point["MethylLlama"][k] for k in ["medae", "mae", "r2"]
    }
    print(json.dumps(point, indent=2))

    # ── Paired bootstrap over subjects ────────────────────────────────────────
    n = len(true)
    boot_gap = {"medae": [], "mae": [], "r2": []}
    boot_llama = {"medae": [], "mae": [], "r2": []}
    boot_gpt = {"medae": [], "mae": [], "r2": []}
    for _ in range(a.n_boot):
        idx = rng.integers(0, n, size=n)  # same resample applied to both models -> preserves pairing
        m_l = metrics(true[idx], pred_llama[idx])
        m_g = metrics(true[idx], pred_gpt[idx])
        for k in ["medae", "mae", "r2"]:
            boot_llama[k].append(m_l[k])
            boot_gpt[k].append(m_g[k])
            boot_gap[k].append(m_g[k] - m_l[k])

    ci = {}
    for k in ["medae", "mae", "r2"]:
        arr = np.array(boot_gap[k])
        lo, hi = np.percentile(arr, [2.5, 97.5])
        excludes_zero = (lo > 0) or (hi < 0)
        ci[k] = {
            "point_estimate": point["gap_gpt_minus_llama"][k],
            "bootstrap_mean": float(arr.mean()),
            "ci_95_low": float(lo),
            "ci_95_high": float(hi),
            "significant_at_95": bool(excludes_zero),
        }

    summary = {
        "n_subjects": n_matched,
        "n_bootstrap": a.n_boot,
        "seed": a.seed,
        "point_estimates": point,
        "paired_bootstrap_gap_gpt_minus_llama": ci,
    }
    out_json = outdir / "paired_bootstrap_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    merged[["sample_id", "true_age_llama", "predicted_age_llama", "predicted_age_gpt"]].rename(
        columns={"true_age_llama": "true_age"}
    ).to_csv(outdir / "paired_per_subject_predictions.csv", index=False)

    print("\n" + "=" * 70)
    print("PAIRED BOOTSTRAP RESULT (MethylGPT - MethylLlama, positive = GPT worse)")
    print("=" * 70)
    for k in ["medae", "mae", "r2"]:
        c = ci[k]
        sig = "SIGNIFICANT (95% CI excludes 0)" if c["significant_at_95"] else "not significant"
        print(f"  {k:5s}: gap={c['point_estimate']:+.4f}  95% CI=[{c['ci_95_low']:+.4f}, {c['ci_95_high']:+.4f}]  {sig}")
    print(f"\nSaved -> {out_json}")
    print(f"Saved -> {outdir / 'paired_per_subject_predictions.csv'}")


if __name__ == "__main__":
    main()
