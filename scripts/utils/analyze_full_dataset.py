#!/usr/bin/env python3
"""
analyze_full_dataset.py
========================
Full analysis of AltumAge 21k dataset before deciding exclusion strategy.

Q1. Age distribution of ALL 10,988 samples:
    - How many have negative age?
    - How many have age > 120?
    - How many are in the 0-120 valid range?
    - Is an age<0 / age>120 filter sufficient to catch all problems?

Q2. Duplicate detection across ALL 10,988 samples:
    - Find samples with near-identical methylation profiles (cosine >= 0.9999)
    - How many duplicates exist in the KEPT set (not removed)?
    - Do the 302 normal-age removed samples have their duplicate counterpart in the kept set?
    - Are there duplicates in the kept set that were NOT removed?

Q3. Comparison: outliers.csv vs age-filter-only:
    - What does age filter miss vs outliers.csv?
    - What does outliers.csv remove that age filter wouldn't?

Outputs:
  full_dataset_report.html
  full_dataset_summary.txt
  duplicate_pairs.csv  — all detected duplicate pairs
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

_BASE = "/sci/labs/benjamin.yakir/netanel.azran"
_DATA = f"{_BASE}/data"

ALT_H5AD     = f"{_DATA}/data_methyl_21k_h5ad/altumage_21k_3way.h5ad"
OUTLIERS_CSV = (f"{_BASE}/repos/BMFM-RNA/methyl/"
                "dataset_fingerprint_outputs/outliers.csv")


def load_h5ad(path, label):
    import scipy.sparse
    print(f"\n[{label}] loading {path}")
    try:
        import scanpy as sc
        adata = sc.read_h5ad(path)
    except Exception as e:
        print(f"  scanpy failed ({e}), using h5py...")
        import h5py, anndata as ad
        with h5py.File(path, "r") as f:
            X_grp = f["X"]
            if isinstance(X_grp, h5py.Dataset):
                X = X_grp[()].astype(np.float32)
            else:
                data    = X_grp["data"][()]
                indices = X_grp["indices"][()]
                indptr  = X_grp["indptr"][()]
                n_obs   = len(f["obs"]["_index"])
                n_var   = len(f["var"]["_index"])
                X = scipy.sparse.csr_matrix(
                    (data, indices, indptr), shape=(n_obs, n_var)
                ).toarray().astype(np.float32)
            def read_grp(grp, n):
                idx_key = "_index" if "_index" in grp else list(grp.keys())[0]
                idx = [x.decode() if isinstance(x, bytes) else str(x)
                       for x in grp[idx_key][:]]
                cols = {}
                for k in grp.keys():
                    if k == idx_key: continue
                    try:
                        v = grp[k]
                        if isinstance(v, h5py.Dataset) and v.ndim==1 and len(v)==n:
                            raw = v[()]
                            cols[k] = [x.decode() if isinstance(x,bytes) else x for x in raw]
                        elif isinstance(v, h5py.Group) and "categories" in v:
                            cats  = [x.decode() if isinstance(x,bytes) else str(x)
                                     for x in v["categories"][()]]
                            codes = v["codes"][()]
                            cols[k] = [cats[c] if c >= 0 else None for c in codes]
                    except Exception: pass
                return idx, pd.DataFrame(cols, index=idx)
            obs_idx, obs = read_grp(f["obs"], X.shape[0])
            var_idx, var = read_grp(f["var"], X.shape[1])
        adata = __import__("anndata").AnnData(X=X, obs=obs, var=var)
    if scipy.sparse.issparse(adata.X):
        adata.X = adata.X.toarray().astype(np.float32)
    adata.obs.index = adata.obs.index.astype(str)
    print(f"  shape: {adata.n_obs:,} × {adata.n_vars:,}")
    return adata


def find_duplicates(X, ids, batch=256, threshold=0.9999):
    """
    Find all pairs (i, j) where i < j and cosine_sim(X[i], X[j]) >= threshold.
    Returns list of (id_i, id_j, sim).
    """
    print(f"\n  Finding duplicates (cosine >= {threshold}) among {len(ids):,} samples...")
    X = np.nan_to_num(X, nan=0.0).astype(np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    X_norm = X / norms

    pairs = []
    n = len(X_norm)
    for start in range(0, n, batch):
        end = min(start + batch, n)
        S = X_norm[start:end] @ X_norm.T   # (batch, n)
        # Only look at j > i to avoid duplicates and self-matches
        for local_i, global_i in enumerate(range(start, end)):
            row = S[local_i]
            for j in range(global_i + 1, n):
                if row[j] >= threshold:
                    pairs.append((ids[global_i], ids[j], float(row[j])))
        if start % 1000 == 0:
            print(f"    progress: {start:,}/{n:,} ({100*start/n:.1f}%)", flush=True)

    print(f"  Found {len(pairs):,} duplicate pairs")
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alt_h5ad",     default=ALT_H5AD)
    ap.add_argument("--outliers_csv", default=OUTLIERS_CSV)
    ap.add_argument("--outdir",       default="dataset_fingerprint_outputs")
    ap.add_argument("--threshold",    type=float, default=0.9999)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────────
    outliers_df = pd.read_csv(args.outliers_csv)
    outlier_ids = set(outliers_df["sample_id"].astype(str))

    alt = load_h5ad(args.alt_h5ad, "AltumAge 21k")
    obs = alt.obs.copy()
    obs.index = obs.index.astype(str)
    obs["age_num"] = pd.to_numeric(obs["age"], errors="coerce")
    obs["is_outlier"] = obs.index.isin(outlier_ids)

    all_ids = obs.index.tolist()
    X = alt.X

    # ════════════════════════════════════════════════════════════════════════
    # Q1: Age distribution of ALL samples
    # ════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("Q1: AGE DISTRIBUTION — ALL 10,988 SAMPLES")
    print(f"{'='*60}")

    ages = obs["age_num"]
    neg      = obs[ages < 0]
    over120  = obs[ages > 120]
    valid_age= obs[(ages >= 0) & (ages <= 120)]
    nan_age  = obs[ages.isna()]

    print(f"\n  Total samples       : {len(obs):,}")
    print(f"  Age < 0 (negative)  : {len(neg):,}")
    print(f"  Age > 120           : {len(over120):,}")
    print(f"  Age 0–120 (valid)   : {len(valid_age):,}")
    print(f"  Age NaN/missing     : {len(nan_age):,}")

    # Age bins for ALL samples
    bins   = [(-999,-1),(-1,-0.1),(-0.1,0),(0,10),(10,20),(20,30),
              (30,40),(40,50),(50,60),(60,70),(70,80),(80,90),(90,100),(100,120),(120,999)]
    labels = ["< -1","−1 to −0.1","−0.1 to 0","0–10","10–20","20–30",
              "30–40","40–50","50–60","60–70","70–80","80–90","90–100","100–120","> 120"]

    print(f"\n  Age distribution (all 10,988 samples):")
    age_dist_all = {}
    for lbl,(lo,hi) in zip(labels,bins):
        sub = obs[(ages > lo) & (ages <= hi)]
        n_out = sub["is_outlier"].sum()
        n_kept = len(sub) - n_out
        age_dist_all[lbl] = {"total": len(sub), "kept": int(n_kept), "removed": int(n_out)}
        if len(sub) > 0:
            print(f"    {lbl:15s}: total={len(sub):4d}  kept={n_kept:4d}  removed={n_out:4d}")

    # What age-filter would do vs outliers.csv
    age_filter_remove = set(obs[~ages.between(0, 120, inclusive="both")].index)
    csv_only   = outlier_ids - age_filter_remove   # removed by CSV but not age filter
    filter_only= age_filter_remove - outlier_ids   # would be removed by filter but not in CSV
    both       = outlier_ids & age_filter_remove   # removed by both

    print(f"\n  COMPARISON: outliers.csv vs age filter (age<0 or age>120):")
    print(f"  Removed by CSV only (age-filter misses)  : {len(csv_only):,}")
    print(f"  Removed by age-filter only (not in CSV)  : {len(filter_only):,}")
    print(f"  Removed by BOTH methods                  : {len(both):,}")
    print(f"  Total in CSV                             : {len(outlier_ids):,}")
    print(f"  Total age-filter would remove            : {len(age_filter_remove):,}")

    if len(csv_only) > 0:
        csv_only_df = obs.loc[sorted(csv_only)]
        print(f"\n  Samples removed by CSV but age-filter would KEEP (sample):")
        print(f"  These have valid age (0-120) but were removed for other reasons:")
        print(csv_only_df[["age_num","tissue_type","dataset","split"]].head(10).to_string())
        print(f"  ... ({len(csv_only):,} total)")
        print(f"\n  Tissue breakdown of these {len(csv_only):,} samples:")
        print(csv_only_df["tissue_type"].value_counts().to_string())
        print(f"\n  Dataset breakdown:")
        print(csv_only_df["dataset"].value_counts().to_string())

    if len(filter_only) > 0:
        filter_only_df = obs.loc[sorted(filter_only)]
        print(f"\n  !! Samples age-filter removes but NOT in outliers.csv:")
        print(filter_only_df[["age_num","tissue_type","dataset"]].to_string())

    # ════════════════════════════════════════════════════════════════════════
    # Q2: Duplicate detection across ALL samples
    # ════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("Q2: DUPLICATE DETECTION — ALL 10,988 SAMPLES")
    print(f"{'='*60}")

    pairs = find_duplicates(X, all_ids, batch=256, threshold=args.threshold)

    # Classify each pair
    pair_rows = []
    both_kept = both_removed = one_each = 0
    for id_a, id_b, sim in pairs:
        a_out = id_a in outlier_ids
        b_out = id_b in outlier_ids
        if a_out and b_out:
            cat = "both_removed"
            both_removed += 1
        elif not a_out and not b_out:
            cat = "both_kept"
            both_kept += 1
        else:
            cat = "one_removed_one_kept"
            one_each += 1
        a_age = float(obs.loc[id_a, "age_num"]) if id_a in obs.index else None
        b_age = float(obs.loc[id_b, "age_num"]) if id_b in obs.index else None
        a_tis = obs.loc[id_a, "tissue_type"] if id_a in obs.index else None
        b_tis = obs.loc[id_b, "tissue_type"] if id_b in obs.index else None
        pair_rows.append({
            "id_a": id_a, "id_b": id_b, "cosine_sim": round(sim, 6),
            "category": cat,
            "a_removed": a_out, "b_removed": b_out,
            "a_age": a_age, "b_age": b_age,
            "a_tissue": a_tis, "b_tissue": b_tis,
        })

    pairs_df = pd.DataFrame(pair_rows)
    pairs_path = outdir / "duplicate_pairs.csv"
    pairs_df.to_csv(pairs_path, index=False)

    print(f"\n  Total duplicate pairs found    : {len(pairs):,}")
    print(f"  Both samples KEPT (problem!)   : {both_kept:,}")
    print(f"  Both samples removed           : {both_removed:,}")
    print(f"  One kept, one removed (correct): {one_each:,}")

    if both_kept > 0:
        kept_dups = pairs_df[pairs_df["category"] == "both_kept"]
        print(f"\n  !! DUPLICATES STILL IN KEPT SET ({both_kept:,} pairs):")
        print(kept_dups[["id_a","id_b","cosine_sim","a_age","b_age",
                          "a_tissue","b_tissue"]].head(20).to_string())
        # Unique sample IDs that are duplicates in the kept set
        dup_ids_kept = set(kept_dups["id_a"]) | set(kept_dups["id_b"])
        print(f"\n  Unique samples involved: {len(dup_ids_kept):,}")

    # Check: do the 302 normal-age removed samples have their dup in kept set?
    csv_only_pairs = pairs_df[
        ((pairs_df["id_a"].isin(csv_only)) & (~pairs_df["id_b"].isin(outlier_ids))) |
        ((pairs_df["id_b"].isin(csv_only)) & (~pairs_df["id_a"].isin(outlier_ids)))
    ] if len(pairs_df) > 0 and len(csv_only) > 0 else pd.DataFrame()

    print(f"\n  Of the {len(csv_only):,} normal-age removed samples:")
    print(f"  Those confirmed as duplicates of a kept sample: {len(csv_only_pairs):,} pairs")
    if len(csv_only_pairs) > 0:
        print(csv_only_pairs.head(10).to_string())

    # ════════════════════════════════════════════════════════════════════════
    # Q3: Final verdict
    # ════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("Q3: FINAL VERDICT")
    print(f"{'='*60}")
    print(f"\n  Negative-age samples in full 21k  : {len(neg):,}")
    print(f"  Would age filter catch them all?   : "
          f"{'YES' if len(filter_only)==0 else 'NO — see filter_only samples above'}")
    print(f"  Extra removals in CSV (valid age)  : {len(csv_only):,}")
    print(f"  Of those, confirmed duplicates     : {len(csv_only_pairs):,}")
    print(f"  Duplicate pairs still in kept set  : {both_kept:,}")
    print(f"\n  RECOMMENDATION:")
    if both_kept == 0 and len(csv_only_pairs) >= len(csv_only) * 0.8:
        print("  → outliers.csv is correct and complete.")
        print("    Age filter alone would miss valid-age duplicates.")
        print("    Use outliers.csv IDs for exclusion.")
    elif both_kept > 0:
        print(f"  → {both_kept:,} duplicate pairs still exist in kept set!")
        print("    Additional deduplication may be needed.")
    else:
        print("  → Investigate why normal-age samples were removed.")
        print("    Duplicate confirmation is incomplete.")

    # Write summary
    lines = [
        "="*70, "FULL DATASET ANALYSIS REPORT", "AltumAge 21k — 10,988 samples × 21,368 CpGs",
        "="*70, "",
        "Q1: AGE DISTRIBUTION OF ALL SAMPLES",
        "-"*40,
        f"  Total          : {len(obs):,}",
        f"  Negative age   : {len(neg):,}  ({100*len(neg)/len(obs):.1f}%)",
        f"  Age > 120      : {len(over120):,}",
        f"  Valid (0-120)  : {len(valid_age):,}  ({100*len(valid_age)/len(obs):.1f}%)",
        "",
        "  Age filter vs outliers.csv comparison:",
        f"  CSV removes but age-filter keeps : {len(csv_only):,}  (valid-age, likely duplicates)",
        f"  Age-filter removes but not in CSV: {len(filter_only):,}",
        f"  Removed by both                  : {len(both):,}",
        "",
        "Q2: DUPLICATE DETECTION",
        "-"*40,
        f"  Total duplicate pairs (cosine>={args.threshold}): {len(pairs):,}",
        f"  Both kept (problem)    : {both_kept:,}",
        f"  Both removed           : {both_removed:,}",
        f"  One kept, one removed  : {one_each:,}",
        f"  Normal-age removals confirmed as dups: {len(csv_only_pairs):,}/{len(csv_only):,}",
        "",
        "="*70,
    ]
    (outdir / "full_dataset_summary.txt").write_text("\n".join(lines))
    print(f"\nOutputs saved to {outdir}/")
    print(f"  full_dataset_summary.txt")
    print(f"  duplicate_pairs.csv ({len(pairs):,} pairs)")


if __name__ == "__main__":
    main()
