"""
age_restricted_paired_bootstrap.py
==================================
Nested post-hoc age restrictions of the paired MethylLlama-versus-MethylGPT comparison
(Results, Fig. 4c): the Table-1 paired bootstrap repeated on the test profiles aged at least
18 years and at least 40 years. Same statistics as paired_bootstrap_comparison.py: percentile
bootstrap over test profiles, 10,000 resamples, seed 0 (a new generator for each subset),
identical resample indices for both models. Positive values favour MethylLlama.

Pure post-processing of results/age_benchmark/paired_per_subject_predictions.csv; no GPU.
Usage:  python scripts/evaluation/age_restricted_paired_bootstrap.py
Writes  results/age_benchmark/age_restricted_paired_bootstrap_summary.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

SRC = Path("results/age_benchmark/paired_per_subject_predictions.csv")
OUT = Path("results/age_benchmark/age_restricted_paired_bootstrap_summary.json")
N_BOOT, SEED = 10000, 0
SUBSETS = (("adults_age_ge_18", 18), ("older_age_ge_40", 40))


def metrics(y, p):
    e = np.abs(p - y)
    return np.median(e), e.mean(), r2_score(y, p)


def diff(y, pl, pg):
    ml, mg = metrics(y, pl), metrics(y, pg)
    return np.array([mg[0] - ml[0], mg[1] - ml[1], ml[2] - mg[2]])


def main():
    d = pd.read_csv(SRC, dtype={"sample_id": str})
    assert len(d) == 2149
    y, pl, pg = (d[k].to_numpy(float) for k in ("true_age", "predicted_age_llama", "predicted_age_gpt"))
    out = {"source": str(SRC), "n_bootstrap": N_BOOT, "seed": SEED,
           "sign_convention": {"medae": "MethylGPT minus MethylLlama", "mae": "MethylGPT minus MethylLlama",
                               "r2": "MethylLlama minus MethylGPT"},
           "subsets": {}}
    for name, min_age in SUBSETS:
        s = y >= min_age
        ys, ls, gs = y[s], pl[s], pg[s]
        n = len(ys)
        rng = np.random.default_rng(SEED)
        boot = np.array([diff(ys[i], ls[i], gs[i]) for i in (rng.integers(0, n, n) for _ in range(N_BOOT))])
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        obs = diff(ys, ls, gs)
        r = {"n": int(n)}
        for j, k in enumerate(("medae", "mae", "r2")):
            r[f"delta_{k}"] = round(float(obs[j]), 6)
            r[f"delta_{k}_ci_95"] = [round(float(lo[j]), 6), round(float(hi[j]), 6)]
        out["subsets"][name] = r
        print(f"age >= {min_age}: n={n}  dMedAE {r['delta_medae']:.3f} {r['delta_medae_ci_95']}  "
              f"dMAE {r['delta_mae']:.3f} {r['delta_mae_ci_95']}  dR2 {r['delta_r2']:.4f} {r['delta_r2_ci_95']}")
    out["note"] = ("Nested post-hoc age restrictions; confidence intervals are paired percentile-bootstrap intervals "
                   "over held-out subjects. These are sensitivity analyses, not independent strata.")
    OUT.write_text(json.dumps(out, indent=2))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
