"""
paired_bootstrap_vs_elasticnet.py
=================================
Paired bootstrap of the MethylLlama fold ensemble against the ElasticNetCV baseline on the common
2,149-profile test set. Same procedure as paired_bootstrap_comparison.py (same resampled profiles for
both models, 10,000 resamples, seed 0, percentile 95% intervals). Isolated script; inputs are the
stored predictions, nothing is refitted.

The run fails unless the point estimates reproduce the reported values exactly:
MethylLlama MedAE 3.112 / MAE 4.333 / R2 0.9365 and ElasticNetCV MedAE 3.437 / MAE 4.731 / R2 0.9322.

Inputs (as released in results/age_benchmark/):
  paired_per_subject_predictions.csv  (predicted_age_llama = MethylLlama fold ensemble)
  elasticnet_test_predictions.csv     (written by scripts/baselines/elasticnet_age.py)
Output: results/age_benchmark/paired_bootstrap_vs_elasticnet_summary.json

Usage: python scripts/evaluation/paired_bootstrap_vs_elasticnet.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

N_BOOT, SEED = 10000, 0


def metrics(true, pred):
    return {"medae": float(np.median(np.abs(pred - true))), "mae": float(np.mean(np.abs(pred - true))),
            "r2": float(r2_score(true, pred))}


def main():
    root = Path(__file__).resolve().parents[2]
    res = root / "results/age_benchmark"
    p = pd.read_csv(res / "paired_per_subject_predictions.csv", dtype={"sample_id": str})
    e = pd.read_csv(res / "elasticnet_test_predictions.csv", dtype={"sample_id": str})
    m = p.merge(e.rename(columns={"true_age": "true_age_en", "predicted_age": "predicted_age_en"}), on="sample_id")
    assert len(m) == len(p) == len(e) == 2149, (len(m), len(p), len(e))
    assert np.max(np.abs(m.true_age - m.true_age_en)) < 1e-4
    y, a, b = m.true_age.to_numpy(float), m.predicted_age_llama.to_numpy(float), m.predicted_age_en.to_numpy(float)

    pa, pb = metrics(y, a), metrics(y, b)
    rep = (round(pa["medae"], 3), round(pa["mae"], 3), round(pa["r2"], 4), round(pb["medae"], 3), round(pb["mae"], 3), round(pb["r2"], 4))
    assert rep == (3.112, 4.333, 0.9365, 3.437, 4.731, 0.9322), f"point estimates do not reproduce: {rep}"

    rng = np.random.default_rng(SEED)
    n = len(y)
    gaps = {k: [] for k in ("medae", "mae", "r2")}
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        ma, mb = metrics(y[idx], a[idx]), metrics(y[idx], b[idx])
        for k in gaps:
            gaps[k].append(mb[k] - ma[k])       # ElasticNet minus MethylLlama (positive = MethylLlama better for errors)
    out = {"n_subjects": n, "n_bootstrap": N_BOOT, "seed": SEED,
           "point_estimates": {"MethylLlama": pa, "ElasticNetCV": pb,
                               "gap_elasticnet_minus_llama": {k: pb[k] - pa[k] for k in pa}},
           "paired_bootstrap_gap_elasticnet_minus_llama": {}}
    for k, arr in gaps.items():
        arr = np.asarray(arr); lo, hi = np.percentile(arr, [2.5, 97.5])
        out["paired_bootstrap_gap_elasticnet_minus_llama"][k] = {
            "point_estimate": pb[k] - pa[k], "bootstrap_mean": float(arr.mean()), "ci_95_low": float(lo),
            "ci_95_high": float(hi), "ci_excludes_zero": bool(lo > 0 or hi < 0)}
    dst = res / "paired_bootstrap_vs_elasticnet_summary.json"
    dst.write_text(json.dumps(out, indent=2))
    print(json.dumps(out["paired_bootstrap_gap_elasticnet_minus_llama"], indent=2))


if __name__ == "__main__":
    main()
