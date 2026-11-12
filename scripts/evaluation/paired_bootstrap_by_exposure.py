#!/usr/bin/env python3
"""
paired_bootstrap_by_exposure.py
===============================
Table-1 comparison stratified by the pretraining-exposure classes of scripts/data/data_structure_audit.py
(Results: the 626 test profiles for which no profile of the same individual was identified in the
pretraining optimisation partition, classes 4-6). TCGA profiles are matched by participant and sample
type, because downstream TCGA profiles are named by participant and corpus TCGA profiles by aliquot
barcode, so matching by accession alone would count every TCGA profile as "not in corpus".

Same statistics as Table 1: percentile bootstrap over test profiles, 10,000 resamples, seed 0,
identical resample indices for both models. Positive always favours MethylLlama.

Inputs   results/age_benchmark/paired_per_subject_predictions.csv
         results/data_audit/downstream_exposure_per_profile.csv
Output   results/data_audit/paired_bootstrap_by_exposure.json

Pure post-processing; no GPU.
Usage:  python scripts/evaluation/paired_bootstrap_by_exposure.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

REPO = Path(__file__).resolve().parents[2]
N_BOOT, SEED = 10000, 0


def metrics(y, p):
    e = np.abs(p - y)
    return float(np.median(e)), float(e.mean()), float(r2_score(y, p))


def main():
    pred = pd.read_csv(REPO / "results/age_benchmark/paired_per_subject_predictions.csv", dtype={"sample_id": str})
    ex = pd.read_csv(REPO / "results/data_audit/downstream_exposure_per_profile.csv", dtype={"sample_id": str})
    df = pred.merge(ex[["sample_id", "study", "exposure"]], on="sample_id", how="left", validate="one_to_one")
    assert len(df) == 2149 and df.exposure.notna().all()
    c = df.exposure.str[0].astype(int)
    subsets = {
        "all test profiles": c >= 1,
        "1 same accession in pretraining train": c == 1,
        "1-3 profile or individual in pretraining train": c <= 3,
        "4-6 no profile of the individual found in pretraining train": c >= 4,
        "5-6 not found in the corpus by any identifier": c >= 5,
        "6 not found in the corpus, study not in pretraining train": c == 6,
    }
    out = {"n_boot": N_BOOT, "seed": SEED, "class_counts": df.exposure.value_counts().sort_index().to_dict(),
           "sign_convention": "error: MethylGPT - MethylLlama; R2: MethylLlama - MethylGPT", "subsets": {}}
    for name, mask in subsets.items():
        s = df[mask.values]
        y, pl, pg = (s[k].to_numpy(float) for k in ("true_age", "predicted_age_llama", "predicted_age_gpt"))
        n = len(s); ml, mg = metrics(y, pl), metrics(y, pg)
        rng = np.random.default_rng(SEED)
        d = np.empty((N_BOOT, 3))
        for b in range(N_BOOT):
            i = rng.integers(0, n, n)
            x, z = metrics(y[i], pl[i]), metrics(y[i], pg[i])
            d[b] = (z[0] - x[0], z[1] - x[1], x[2] - z[2])
        lo, hi = np.percentile(d, [2.5, 97.5], axis=0)
        pt = (mg[0] - ml[0], mg[1] - ml[1], ml[2] - mg[2])
        out["subsets"][name] = {"n": n, "n_studies": int(s.study.nunique()), "age_mean": float(y.mean()),
                                "methyllama": dict(zip(("medae", "mae", "r2"), ml)),
                                "methylgpt": dict(zip(("medae", "mae", "r2"), mg)),
                                "paired_difference": {k: {"estimate": float(pt[j]), "ci_95": [float(lo[j]), float(hi[j])],
                                                          "excludes_zero": bool(lo[j] > 0 or hi[j] < 0)}
                                                      for j, k in enumerate(("medae", "mae", "r2"))}}
        print(f"{name:<62}{n:>5} st {s.study.nunique():>3} | L {ml[0]:.3f}/{ml[1]:.3f}/{ml[2]:.4f}  G {mg[0]:.3f}/{mg[1]:.3f}/{mg[2]:.4f} | "
              f"dMedAE {pt[0]:.3f} [{lo[0]:.3f}, {hi[0]:.3f}]  dMAE {pt[1]:.3f} [{lo[1]:.3f}, {hi[1]:.3f}]  dR2 {pt[2]:.4f} [{lo[2]:.4f}, {hi[2]:.4f}]")
    (REPO / "results/data_audit").mkdir(parents=True, exist_ok=True)
    (REPO / "results/data_audit/paired_bootstrap_by_exposure.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
