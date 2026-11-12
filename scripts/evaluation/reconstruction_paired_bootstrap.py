"""
reconstruction_paired_bootstrap.py
==================================
Paired bootstrap of the per-profile reconstruction error (Fig. 2c, Results): for each control
(per-CpG mean, shuffled CLS, random CLS), the difference control MSE - model MSE of every
held-out profile, its mean, and a 95% percentile interval over 10,000 resamples of the profiles
(one numpy generator, seed 0, used for the three controls in this order). Also the fraction of
profiles for which the control is worse than the model.

Pure post-processing of per_sample_metrics.csv (scripts/evaluation/reconstruction_withheld_eval.py);
no GPU, about a minute on a laptop CPU.

Usage:  python scripts/evaluation/reconstruction_paired_bootstrap.py [--dir results/pretrained_cls/reconstruction_withheld]
Writes  <dir>/paired_bootstrap_ci.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CONTROLS = (("Per-CpG mean", "b_mean_mse"), ("Shuffled CLS", "b_shuffle_mse"), ("Random CLS", "b_random_mse"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="results/pretrained_cls/reconstruction_withheld")
    p.add_argument("--n_boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    d = Path(a.dir)
    m = pd.read_csv(d / "per_sample_metrics.csv")
    model, n = m["model_mse"].values, len(m)
    rng = np.random.default_rng(a.seed)
    out = {}
    for name, col in CONTROLS:
        diff = m[col].values - model
        boot = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(a.n_boot)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out[name] = {"mean_diff": round(float(diff.mean()), 5), "ci95": [round(float(lo), 5), round(float(hi), 5)],
                     "frac_profiles_control_worse": round(float((diff > 0).mean()), 4)}
        print(f"{name:13s} mean diff {out[name]['mean_diff']:.5f}  95% CI {out[name]['ci95']}  "
              f"control worse in {100 * out[name]['frac_profiles_control_worse']:.2f}% of {n} profiles")
    (d / "paired_bootstrap_ci.json").write_text(json.dumps(out, indent=1))
    print(f"saved {d / 'paired_bootstrap_ci.json'}")


if __name__ == "__main__":
    main()
