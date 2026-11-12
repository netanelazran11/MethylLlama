"""
Leave-one-dataset-out (LODO) age probe — cross-study generalization test.

Pure post-processing on embeddings_cls.npy + metadata.csv (no GPU). Quantifies
how much of the in-distribution age R² is dataset/batch-specific vs. a universal
aging signal: for each large study, hold it out ENTIRELY, train a Ridge age probe
on all other studies, and test on the held-out one.

R² is only meaningful where the held-out study has age spread, so we report the
summary over studies with age_std >= --min_age_std; MedAE is reported for all.

Studies given with --exclude_study are removed from every step (probe fitting, the in-distribution
reference and the held-out studies). The manuscript (Fig. 3e) excludes E-GEOD-56105 after the
study-level quality check of scripts/data/downstream_integrity_check.py (its downstream values
do not correlate with the same accessions in the pretraining corpus; Methods).

Writes lodo_age_probe.csv and lodo_reference.json to --outdir (default: --dir).

Usage (the manuscript run, writes results/pretrained_cls/lodo_*):
  python scripts/evaluation/leave_one_dataset_out_probe.py \
     --dir results/pretrained_cls --min_n 100 --min_age_std 10 --exclude_study E-GEOD-56105
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="results/pretrained_cls")
    p.add_argument("--emb", default="embeddings_cls.npy")
    p.add_argument("--dataset_col", default="dataset")
    p.add_argument("--age_col", default="age")
    p.add_argument("--min_n", type=int, default=100, help="min held-out study size")
    p.add_argument("--min_age_std", type=float, default=10.0, help="min age std for R² summary")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--exclude_study", action="append", default=[],
                   help="study removed from every step (repeatable); the manuscript uses E-GEOD-56105")
    p.add_argument("--outdir", default=None, help="output directory (default: --dir)")
    return p.parse_args()


def main():
    a = parse_args()
    d = Path(a.dir)
    X = np.load(d / a.emb).astype(np.float64)
    meta = pd.read_csv(d / "metadata.csv")
    age = pd.to_numeric(meta[a.age_col], errors="coerce").values
    ds = meta[a.dataset_col].astype(str).values
    ok = ~np.isnan(age) & ~np.isin(ds, a.exclude_study)
    if a.exclude_study:
        print(f"excluded studies: {a.exclude_study} ({int((np.isin(ds, a.exclude_study)).sum())} profiles)")
    X, age, ds = X[ok], age[ok], ds[ok]
    out = Path(a.outdir) if a.outdir else d
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(X)} samples, {pd.Series(ds).nunique()} datasets")

    # in-distribution reference (random 80/20)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X)); cut = int(0.8 * len(X))
    tr, te = idx[:cut], idx[cut:]
    sc = StandardScaler().fit(X[tr])
    p = Ridge(alpha=a.alpha).fit(sc.transform(X[tr]), age[tr]).predict(sc.transform(X[te]))
    id_r2, id_med = r2_score(age[te], p), float(np.median(np.abs(p - age[te])))
    print(f"[in-distribution 80/20] R2={id_r2:.3f}  MedAE={id_med:.2f}")
    note = "reference for the LODO analysis (random 80/20 split of the profiles used by the LODO probe)"
    if a.exclude_study:
        note += "; excluded: " + ", ".join(a.exclude_study) + " (study-level quality check, Methods)"
    ref = {"in_distribution_random_80_20": {"r2": round(float(id_r2), 4), "medae": round(id_med, 3),
                                            "n_train": int(len(tr)), "n_test": int(len(te)), "seed": 0,
                                            "alpha": a.alpha, "note": note}}
    (out / "lodo_reference.json").write_text(json.dumps(ref, indent=2))

    vc = pd.Series(ds).value_counts()
    rows = []
    for name in vc[vc >= a.min_n].index:
        te = ds == name; trm = ~te
        if te.sum() < 30:
            continue
        sc = StandardScaler().fit(X[trm])
        pr = Ridge(alpha=a.alpha).fit(sc.transform(X[trm]), age[trm]).predict(sc.transform(X[te]))
        av = float(age[te].var())
        r2 = r2_score(age[te], pr) if av > 1 else np.nan
        rows.append({"dataset": name, "n": int(te.sum()),
                     "age_mean": round(float(age[te].mean()), 1),
                     "age_std": round(float(np.sqrt(av)), 1),
                     "R2_LODO": round(r2, 3) if not np.isnan(r2) else None,
                     "MedAE": round(float(np.median(np.abs(pr - age[te]))), 2)})
    res = pd.DataFrame(rows).sort_values("n", ascending=False)
    res.to_csv(out / "lodo_age_probe.csv", index=False)
    print(res.to_string(index=False))

    wide = res[(res["age_std"] >= a.min_age_std) & res["R2_LODO"].notna()]
    print(f"\n=== LODO summary (age_std >= {a.min_age_std}, n={len(wide)} studies) ===")
    print(f"  in-distribution R2 : {id_r2:.3f}   MedAE {id_med:.2f}")
    print(f"  cross-study median R2 : {wide['R2_LODO'].median():.3f}")
    print(f"  cross-study median MedAE: {wide['MedAE'].median():.2f} yr")
    print(f"  R2>=0.5: {(wide['R2_LODO']>=0.5).sum()}/{len(wide)}  "
          f"0-0.5: {((wide['R2_LODO']>0)&(wide['R2_LODO']<0.5)).sum()}/{len(wide)}  "
          f"R2<0: {(wide['R2_LODO']<0).sum()}/{len(wide)}")
    print(f"  saved {out/'lodo_age_probe.csv'} and {out/'lodo_reference.json'}")


if __name__ == "__main__":
    main()
