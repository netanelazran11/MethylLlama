#!/usr/bin/env python3
"""
paired_bootstrap_grouped_sensitivity.py
=======================================
Two sensitivity analyses of the Table-1 comparison that respect the grouping of profiles.

  A. donor-disjoint test set: test profiles whose (known) individual also has a profile in the
     development pool are removed; profile-level bootstrap as in Table 1.
  B. study-level cluster bootstrap: whole studies are resampled with replacement instead of
     profiles, so that profiles of one study are not treated as independent observations.

10,000 resamples, seed 0, same resample for both models. Positive favours MethylLlama.
Inputs   results/age_benchmark/paired_per_subject_predictions.csv
         results/data_audit/downstream_exposure_per_profile.csv
Output   results/data_audit/paired_bootstrap_grouped_sensitivity.json, per_study_mae.csv

The exposure table comes from scripts/data/data_structure_audit.py. Pure post-processing; no GPU.
Usage:  python scripts/evaluation/paired_bootstrap_grouped_sensitivity.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

REPO = Path(__file__).resolve().parents[2]
N_BOOT, SEED, MAX_GROUP = 10000, 0, 20


def metrics(y, p):
    e = np.abs(p - y)
    return np.array([np.median(e), e.mean(), r2_score(y, p)])


def diff(y, pl, pg):
    l, g = metrics(y, pl), metrics(y, pg)
    return np.array([g[0] - l[0], g[1] - l[1], l[2] - g[2]])


def report(name, point, boot, n, extra):
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    print(f"{name:<46} n={n:<5} dMedAE {point[0]:.3f} [{lo[0]:.3f}, {hi[0]:.3f}]  dMAE {point[1]:.3f} "
          f"[{lo[1]:.3f}, {hi[1]:.3f}]  dR2 {point[2]:.4f} [{lo[2]:.4f}, {hi[2]:.4f}]")
    return {"n": int(n), **extra, **{k: {"estimate": float(point[j]), "ci_95": [float(lo[j]), float(hi[j])],
                                         "excludes_zero": bool(lo[j] > 0 or hi[j] < 0)}
                                     for j, k in enumerate(("medae", "mae", "r2"))}}


def main():
    pred = pd.read_csv(REPO / "results/age_benchmark/paired_per_subject_predictions.csv", dtype={"sample_id": str})
    ex = pd.read_csv(REPO / "results/data_audit/downstream_exposure_per_profile.csv", dtype={"sample_id": str})
    b = ex[ex.bench != "excluded"]
    size = b.groupby("individual").size()
    person = b.individual.where(~b.individual.isin(size[size > MAX_GROUP].index))
    dev_people = set(person[b.bench == "development"].dropna())
    df = pred.merge(ex[["sample_id", "study", "individual"]], on="sample_id", validate="one_to_one")
    df["individual"] = df.individual.where(df.individual.isin(set(person.dropna())))
    assert len(df) == 2149
    shared = df.individual.isin(dev_people)
    t = df[df.individual.notna()].groupby("individual").size()
    out = {"n_boot": N_BOOT, "seed": SEED,
           "test_profiles_with_individual_identifier": int(df.individual.notna().sum()),
           "test_profiles_sharing_an_individual_with_development": int(shared.sum()),
           "individuals_with_more_than_one_test_profile": int((t > 1).sum()),
           "test_profiles_of_such_individuals": int(t[t > 1].sum())}

    def cols(d):
        return (d[k].to_numpy(float) for k in ("true_age", "predicted_age_llama", "predicted_age_gpt"))

    def profile_boot(d):
        y, pl, pg = cols(d); n = len(d); rng = np.random.default_rng(SEED)
        bt = np.empty((N_BOOT, 3))
        for k in range(N_BOOT):
            i = rng.integers(0, n, n); bt[k] = diff(y[i], pl[i], pg[i])
        return diff(y, pl, pg), bt

    pt, bt = profile_boot(df)
    out["table_1_profile_bootstrap"] = report("Table 1 (profile bootstrap)", pt, bt, len(df), {})
    d = df[~shared]
    pt, bt = profile_boot(d)
    out["A_donor_disjoint_test_set"] = report("A. donor-disjoint test set (profile bootstrap)", pt, bt, len(d), {})

    y, pl, pg = (np.asarray(list(v)) for v in cols(df))
    groups = [np.where(df.study.to_numpy() == s)[0] for s in sorted(df.study.unique())]
    rng = np.random.default_rng(SEED); bt = np.empty((N_BOOT, 3))
    for k in range(N_BOOT):
        i = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        bt[k] = diff(y[i], pl[i], pg[i])
    out["B_study_cluster_bootstrap"] = report("B. study-level cluster bootstrap", diff(y, pl, pg), bt, len(df),
                                              {"n_studies": len(groups)})
    # per-study direction of the difference
    per = df.groupby("study").apply(lambda g: pd.Series({
        "n": len(g), "mae_llama": np.abs(g.predicted_age_llama - g.true_age).mean(),
        "mae_gpt": np.abs(g.predicted_age_gpt - g.true_age).mean()}), include_groups=False)
    big = per[per.n >= 10]
    out["per_study"] = {"studies": int(len(per)), "studies_with_lower_mae_for_methyllama": int((per.mae_llama < per.mae_gpt).sum()),
                        "studies_n_ge_10": int(len(big)),
                        "studies_n_ge_10_with_lower_mae_for_methyllama": int((big.mae_llama < big.mae_gpt).sum())}
    print(out["per_study"], {k: out[k] for k in list(out)[2:6]})
    (REPO / "results/data_audit").mkdir(parents=True, exist_ok=True)
    per.to_csv(REPO / "results/data_audit/per_study_mae.csv")
    (REPO / "results/data_audit/paired_bootstrap_grouped_sensitivity.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
