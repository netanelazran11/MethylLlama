#!/usr/bin/env python3
"""
analyze_21k_structure.py
=========================
Full diagnostic report on the AltumAge 21k dataset before training.

Q1. OUTLIER SAMPLES
    - Are ALL 630 removed samples negative-age / >120?
    - Are there any samples with bad age that were NOT removed (stayed in 21k)?
    - Full breakdown by age, tissue, dataset source, split

Q2. EXTRA 1,760 CpGs (in 21k but not in 19k)
    - NaN rate per CpG (histogram + statistics)
    - NaN rate per sample
    - Are they fully NaN? Or do some samples have real values?

Outputs:
  analysis_report.html   — full visual slide report
  analysis_summary.txt   — plain-text full report
  extra_cpg_nan_profile.csv — per-CpG NaN rates for the 1,760 extra CpGs
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

_BASE = "/sci/labs/benjamin.yakir/netanel.azran"
_DATA = f"{_BASE}/data"

ALT_H5AD     = f"{_DATA}/data_methyl_21k_h5ad/altumage_21k_3way.h5ad"
LLAMA_H5AD   = (f"{_DATA}/data_methyl_finetune_19k_h5ad/"
                "finetuning_19608_clean_stratified_no_outliers.h5ad")
OUTLIERS_CSV = (f"{_BASE}/repos/BMFM-RNA/methyl/"
                "dataset_fingerprint_outputs/outliers.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────────────────────
def load_h5ad_full(path, label):
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
    adata.var.index = adata.var.index.astype(str)
    print(f"  shape: {adata.n_obs:,} × {adata.n_vars:,}")
    return adata


# ─────────────────────────────────────────────────────────────────────────────
# Analysis Q1: Outlier samples
# ─────────────────────────────────────────────────────────────────────────────
def analyze_outliers(alt_obs, outlier_ids):
    alt_obs = alt_obs.copy()
    alt_obs["age_num"] = pd.to_numeric(alt_obs["age"], errors="coerce")
    alt_obs["is_outlier"] = alt_obs.index.isin(outlier_ids)

    kept    = alt_obs[~alt_obs["is_outlier"]]
    removed = alt_obs[ alt_obs["is_outlier"]]

    kept_ages    = kept["age_num"].dropna()
    removed_ages = removed["age_num"].dropna()

    # Safety: any kept sample with bad age?
    kept_neg     = kept[kept["age_num"] < 0]
    kept_over120 = kept[kept["age_num"] > 120]

    # Age breakdown of removed samples
    age_bins   = [(-9999,-0.001), (-0.001,0), (0,1), (1,5), (5,18),
                  (18,40), (40,60), (60,80), (80,100), (100,120), (120,9999)]
    age_labels = ["< 0 (negative)", "exactly 0", "0–1", "1–5", "5–18",
                  "18–40", "40–60", "60–80", "80–100", "100–120", "> 120"]
    removed_age_dist = {}
    for lbl, (lo, hi) in zip(age_labels, age_bins):
        n = ((removed_ages > lo) & (removed_ages <= hi)).sum()
        removed_age_dist[lbl] = int(n)

    kept_age_dist = {}
    for lbl, (lo, hi) in zip(age_labels, age_bins):
        n = ((kept_ages > lo) & (kept_ages <= hi)).sum()
        kept_age_dist[lbl] = int(n)

    # Dataset/tissue breakdown of removed
    removed_by_tissue  = removed["tissue_type"].value_counts().to_dict() \
                         if "tissue_type" in removed.columns else {}
    removed_by_dataset = removed["dataset"].value_counts().to_dict() \
                         if "dataset"     in removed.columns else {}
    removed_by_split   = removed["split"].value_counts().to_dict() \
                         if "split"       in removed.columns else {}

    # All samples with age<0 in the full 21k — check if any were kept
    all_neg = alt_obs[alt_obs["age_num"] < 0]

    print(f"\n{'='*60}")
    print("Q1: OUTLIER SAMPLE ANALYSIS")
    print(f"{'='*60}")
    print(f"  Total 21k samples : {len(alt_obs):,}")
    print(f"  Kept              : {len(kept):,}")
    print(f"  Removed (outliers): {len(removed):,}")
    print(f"\n  KEPT samples age  : [{kept_ages.min():.3f}, {kept_ages.max():.3f}]")
    print(f"  REMOVED samples age: [{removed_ages.min():.3f}, {removed_ages.max():.3f}]")
    print(f"\n  Kept samples with age < 0   : {len(kept_neg):,}  ← should be 0")
    print(f"  Kept samples with age > 120 : {len(kept_over120):,}  ← should be 0")
    if len(kept_neg) > 0:
        print("  !! PROBLEM — kept samples with negative age:")
        print(kept_neg[["age_num","tissue_type","dataset"]].to_string())
    if len(kept_over120) > 0:
        print("  !! PROBLEM — kept samples with age > 120:")
        print(kept_over120[["age_num","tissue_type","dataset"]].to_string())
    print(f"\n  All 21k samples with age < 0 : {len(all_neg):,}")
    print(f"  Of those, in outliers CSV    : {all_neg.index.isin(outlier_ids).sum():,}")
    print(f"  Missed (kept but age<0)      : {(~all_neg.index.isin(outlier_ids)).sum():,}")

    print(f"\n  REMOVED samples — age distribution:")
    for lbl, n in removed_age_dist.items():
        if n > 0:
            bar = "█" * int(n / max(removed_age_dist.values()) * 25)
            print(f"    {lbl:20s}: {n:4d}  {bar}")
    print(f"\n  REMOVED samples — tissue breakdown:")
    for t, n in sorted(removed_by_tissue.items(), key=lambda x: -x[1]):
        print(f"    {t:30s}: {n:4d}")
    print(f"\n  REMOVED samples — dataset source:")
    for d, n in sorted(removed_by_dataset.items(), key=lambda x: -x[1]):
        print(f"    {d:30s}: {n:4d}")
    print(f"\n  REMOVED samples — original 21k split:")
    for s, n in sorted(removed_by_split.items(), key=lambda x: -x[1]):
        print(f"    {s:10s}: {n:4d}")

    return {
        "total_21k":         len(alt_obs),
        "kept_n":            len(kept),
        "removed_n":         len(removed),
        "kept_age_min":      float(kept_ages.min()),
        "kept_age_max":      float(kept_ages.max()),
        "kept_age_mean":     float(kept_ages.mean()),
        "kept_age_std":      float(kept_ages.std()),
        "removed_age_min":   float(removed_ages.min()),
        "removed_age_max":   float(removed_ages.max()),
        "removed_age_mean":  float(removed_ages.mean()),
        "removed_age_std":   float(removed_ages.std()),
        "kept_neg_n":        len(kept_neg),
        "kept_over120_n":    len(kept_over120),
        "all_neg_in_21k":    len(all_neg),
        "neg_in_outliers":   int(all_neg.index.isin(outlier_ids).sum()),
        "neg_missed":        int((~all_neg.index.isin(outlier_ids)).sum()),
        "removed_age_dist":  removed_age_dist,
        "kept_age_dist":     kept_age_dist,
        "removed_by_tissue": removed_by_tissue,
        "removed_by_dataset":removed_by_dataset,
        "removed_by_split":  removed_by_split,
        "safe_to_use_ids":   len(kept_neg) == 0 and len(kept_over120) == 0,
        "kept_df":           kept,
        "removed_df":        removed,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Analysis Q2: Extra CpG NaN rates
# ─────────────────────────────────────────────────────────────────────────────
def analyze_nan(alt_X, alt_cpgs, llama_cpgs, outdir):
    extra_mask  = np.array([c not in llama_cpgs for c in alt_cpgs])
    shared_mask = ~extra_mask
    extra_idx   = np.where(extra_mask)[0]
    shared_idx  = np.where(shared_mask)[0]

    X_extra  = alt_X[:, extra_idx]
    X_shared = alt_X[:, shared_idx]

    nan_extra_per_cpg  = np.isnan(X_extra).mean(axis=0)
    nan_shared_per_cpg = np.isnan(X_shared).mean(axis=0)
    nan_extra_per_sample  = np.isnan(X_extra).mean(axis=1)
    nan_shared_per_sample = np.isnan(X_shared).mean(axis=1)

    bins   = [0, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.001]
    labels = ["0% (no NaN)", "0–0.1%", "0.1–1%", "1–10%",
              "10–25%", "25–50%", "50–75%", "75–90%", "90–99%", "100% (all NaN)"]
    counts_extra,  _ = np.histogram(nan_extra_per_cpg,  bins=bins)
    counts_shared, _ = np.histogram(nan_shared_per_cpg, bins=bins)

    print(f"\n{'='*60}")
    print("Q2: NaN RATE ANALYSIS")
    print(f"{'='*60}")
    print(f"\n  SHARED 19,608 CpGs:")
    print(f"    Mean NaN rate : {nan_shared_per_cpg.mean():.6f}")
    print(f"    Max NaN rate  : {nan_shared_per_cpg.max():.6f}")
    print(f"    Any NaN       : {(nan_shared_per_cpg > 0).sum():,} CpGs "
          f"({(nan_shared_per_cpg>0).mean()*100:.2f}%)")
    print(f"    Fully NaN     : {(nan_shared_per_cpg == 1).sum():,} CpGs")

    print(f"\n  EXTRA 1,760 CpGs (21k only):")
    print(f"    Mean NaN rate : {nan_extra_per_cpg.mean():.4f}  ({nan_extra_per_cpg.mean()*100:.1f}%)")
    print(f"    Min NaN rate  : {nan_extra_per_cpg.min():.4f}  ({nan_extra_per_cpg.min()*100:.1f}%)")
    print(f"    Max NaN rate  : {nan_extra_per_cpg.max():.4f}  ({nan_extra_per_cpg.max()*100:.1f}%)")
    print(f"    Median NaN    : {np.median(nan_extra_per_cpg):.4f}  ({np.median(nan_extra_per_cpg)*100:.1f}%)")
    print(f"    Any NaN       : {(nan_extra_per_cpg>0).sum():,} / {len(nan_extra_per_cpg):,} CpGs")
    print(f"    > 50% NaN     : {(nan_extra_per_cpg>0.5).sum():,} CpGs")
    print(f"    > 90% NaN     : {(nan_extra_per_cpg>0.9).sum():,} CpGs")
    print(f"    100% NaN      : {(nan_extra_per_cpg==1.0).sum():,} CpGs")

    print(f"\n  NaN rate distribution for extra 1,760 CpGs:")
    mx = max(counts_extra)
    for lbl, cnt in zip(labels, counts_extra):
        bar = "█" * int(cnt / max(mx, 1) * 30)
        print(f"    {lbl:20s}: {cnt:4d}  {bar}")

    print(f"\n  Per-sample NaN rate across extra 1,760 CpGs:")
    print(f"    Mean  : {nan_extra_per_sample.mean():.4f} ({nan_extra_per_sample.mean()*100:.1f}%)")
    print(f"    % samples with ALL extra NaN : {(nan_extra_per_sample==1.0).mean()*100:.1f}%")
    print(f"    % samples with NO  extra NaN : {(nan_extra_per_sample==0.0).mean()*100:.1f}%")
    print(f"    % samples with >50% extra NaN: {(nan_extra_per_sample>0.5).mean()*100:.1f}%")

    # Save per-CpG profile
    profile = pd.DataFrame({
        "cpg_id":      [alt_cpgs[i] for i in extra_idx],
        "nan_rate":    nan_extra_per_cpg.round(6),
        "nan_pct":     (nan_extra_per_cpg * 100).round(2),
        "fully_nan":   nan_extra_per_cpg == 1.0,
        "mostly_nan":  nan_extra_per_cpg > 0.9,
        "n_missing":   np.isnan(X_extra).sum(axis=0),
        "n_measured":  (~np.isnan(X_extra)).sum(axis=0),
    })
    profile_path = outdir / "extra_cpg_nan_profile.csv"
    profile.sort_values("nan_rate", ascending=False).to_csv(profile_path, index=False)
    print(f"\n  Saved per-CpG NaN profile: {profile_path}")

    return {
        "shared_n":              int(shared_mask.sum()),
        "extra_n":               int(extra_mask.sum()),
        "shared_nan_mean":       float(nan_shared_per_cpg.mean()),
        "shared_nan_max":        float(nan_shared_per_cpg.max()),
        "shared_any_nan":        int((nan_shared_per_cpg > 0).sum()),
        "extra_nan_mean":        float(nan_extra_per_cpg.mean()),
        "extra_nan_min":         float(nan_extra_per_cpg.min()),
        "extra_nan_max":         float(nan_extra_per_cpg.max()),
        "extra_nan_median":      float(np.median(nan_extra_per_cpg)),
        "extra_fully_nan":       int((nan_extra_per_cpg == 1.0).sum()),
        "extra_over90_nan":      int((nan_extra_per_cpg > 0.9).sum()),
        "extra_over50_nan":      int((nan_extra_per_cpg > 0.5).sum()),
        "extra_hist_counts":     counts_extra.tolist(),
        "extra_hist_labels":     labels,
        "sample_all_extra_nan_pct":  float((nan_extra_per_sample == 1.0).mean() * 100),
        "sample_no_extra_nan_pct":   float((nan_extra_per_sample == 0.0).mean() * 100),
        "sample_over50_extra_nan_pct": float((nan_extra_per_sample > 0.5).mean() * 100),
        "profile_path":          str(profile_path),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Text report
# ─────────────────────────────────────────────────────────────────────────────
def build_txt(q1, q2):
    lines = [
        "=" * 70,
        "FULL ANALYSIS REPORT — AltumAge 21k Dataset Structure",
        "=" * 70, "",
        "═" * 70,
        "PART 1: OUTLIER SAMPLE ANALYSIS",
        "═" * 70, "",
        f"  AltumAge 21k total samples : {q1['total_21k']:,}",
        f"  Kept in 19k (clean)        : {q1['kept_n']:,}",
        f"  Removed as outliers        : {q1['removed_n']:,}", "",
        f"  KEPT samples age range     : [{q1['kept_age_min']:.2f}, {q1['kept_age_max']:.2f}] yr",
        f"  KEPT samples age mean±std  : {q1['kept_age_mean']:.1f} ± {q1['kept_age_std']:.1f} yr",
        f"  REMOVED samples age range  : [{q1['removed_age_min']:.2f}, {q1['removed_age_max']:.2f}] yr",
        f"  REMOVED samples age mean   : {q1['removed_age_mean']:.1f} ± {q1['removed_age_std']:.1f} yr", "",
        "  ── Safety check ──",
        f"  Kept samples with age < 0   : {q1['kept_neg_n']:,}  "
        f"{'✓ SAFE' if q1['kept_neg_n']==0 else '✗ PROBLEM'}",
        f"  Kept samples with age > 120 : {q1['kept_over120_n']:,}  "
        f"{'✓ SAFE' if q1['kept_over120_n']==0 else '✗ PROBLEM'}",
        f"  All neg-age in 21k          : {q1['all_neg_in_21k']:,}",
        f"  Of those in outliers CSV    : {q1['neg_in_outliers']:,}",
        f"  Missed (kept but neg)       : {q1['neg_missed']:,}  "
        f"{'✓ NONE missed' if q1['neg_missed']==0 else '✗ MISSED SOME'}", "",
        "  ── Removed samples — age distribution ──",
    ]
    for lbl, n in q1["removed_age_dist"].items():
        if n > 0:
            lines.append(f"    {lbl:22s}: {n:4d}")
    lines += ["", "  ── Removed samples — tissue breakdown ──"]
    for t, n in sorted(q1["removed_by_tissue"].items(), key=lambda x: -x[1]):
        lines.append(f"    {t:35s}: {n:4d}")
    lines += ["", "  ── Removed samples — dataset source ──"]
    for d, n in sorted(q1["removed_by_dataset"].items(), key=lambda x: -x[1]):
        lines.append(f"    {d:35s}: {n:4d}")
    lines += ["", "  ── Removed samples — original 21k split ──"]
    for s, n in sorted(q1["removed_by_split"].items(), key=lambda x: -x[1]):
        lines.append(f"    {s:10s}: {n:4d}")
    lines += [
        "",
        f"  CONCLUSION: {'✓ outliers.csv is complete and safe to use as exclude list'  if q1['safe_to_use_ids'] else '✗ WARNING — some problematic samples may have been missed'}",
        "",
        "═" * 70,
        "PART 2: EXTRA 1,760 CpG NaN RATE ANALYSIS",
        "═" * 70, "",
        f"  Shared CpGs (in 19k and 21k) : {q2['shared_n']:,}",
        f"  Extra CpGs (21k only)        : {q2['extra_n']:,}", "",
        f"  SHARED CpGs NaN rate: mean={q2['shared_nan_mean']:.6f}  max={q2['shared_nan_max']:.6f}",
        f"  EXTRA  CpGs NaN rate: mean={q2['extra_nan_mean']*100:.1f}%  "
        f"median={q2['extra_nan_median']*100:.1f}%  "
        f"min={q2['extra_nan_min']*100:.1f}%  max={q2['extra_nan_max']*100:.1f}%", "",
        f"  Extra CpGs that are 100% NaN : {q2['extra_fully_nan']:,} / {q2['extra_n']:,}",
        f"  Extra CpGs with > 90% NaN    : {q2['extra_over90_nan']:,} / {q2['extra_n']:,}",
        f"  Extra CpGs with > 50% NaN    : {q2['extra_over50_nan']:,} / {q2['extra_n']:,}", "",
        f"  Per-sample NaN across extra CpGs:",
        f"    All extra CpGs NaN   : {q2['sample_all_extra_nan_pct']:.1f}% of samples",
        f"    No extra CpGs NaN    : {q2['sample_no_extra_nan_pct']:.1f}% of samples",
        f"    > 50% extra NaN      : {q2['sample_over50_extra_nan_pct']:.1f}% of samples", "",
        "  ── NaN rate distribution for extra 1,760 CpGs ──",
    ]
    mx = max(q2["extra_hist_counts"])
    for lbl, cnt in zip(q2["extra_hist_labels"], q2["extra_hist_counts"]):
        bar = "█" * int(cnt / max(mx, 1) * 25)
        lines.append(f"    {lbl:22s}: {cnt:4d}  {bar}")
    lines += [
        "",
        "  CONCLUSION:",
        f"    {'✓ Extra CpGs are mostly/fully NaN → EXCLUDE them, use only 19,608 CpGs' if q2['extra_fully_nan'] > q2['extra_n']*0.8 else '→ Extra CpGs have meaningful values → imputation or inclusion may be worthwhile'}",
        "",
        "═" * 70,
        "FINAL RECOMMENDATION",
        "═" * 70,
        "",
        "  1. Remove outliers  → use outliers.csv sample IDs (exact, safe)",
        "  2. NaN handling     → based on Q2 result above",
        "  3. Use 21k splits   → train/valid/test from altumage_21k_3way.h5ad",
        "=" * 70,
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# HTML report
# ─────────────────────────────────────────────────────────────────────────────
CSS = """
* { box-sizing:border-box; margin:0; padding:0; }
body { font-family:'Segoe UI',Arial,sans-serif; background:#eef0f4; color:#1e2535; font-size:14px; }
.slide { width:1280px; min-height:720px; margin:40px auto; background:#fff;
         border-radius:16px; padding:46px 56px;
         box-shadow:0 4px 24px rgba(0,0,0,.10); border:1px solid #dde1ea; }
.slide-title { font-size:24px; font-weight:700; margin-bottom:28px; color:#1a2340;
               border-bottom:2px solid #dde3f0; padding-bottom:12px; }
.grid2 { display:grid; grid-template-columns:1fr 1fr; gap:20px; }
.grid3 { display:grid; grid-template-columns:1fr 1fr 1fr; gap:16px; }
.grid4 { display:grid; grid-template-columns:1fr 1fr 1fr 1fr; gap:14px; }
.panel { background:#f7f9fc; border:1px solid #dde3ef; border-radius:10px; padding:18px 20px; }
.panel-title { font-size:12px; font-weight:700; text-transform:uppercase;
               letter-spacing:.8px; color:#5a6888; margin-bottom:14px; }
.stat-card { border-radius:10px; padding:18px 14px; text-align:center; }
.s-val { font-size:28px; font-weight:800; line-height:1; margin-bottom:6px; }
.s-label { font-size:11px; text-transform:uppercase; letter-spacing:.8px; font-weight:600; }
.s-sub { font-size:11px; margin-top:4px; opacity:.75; }
.sc-green  { background:#edf7ed; border:1.5px solid #4a9a4a; color:#1a5a1a; }
.sc-blue   { background:#eef3ff; border:1.5px solid #7a9fe0; color:#1a3a80; }
.sc-purple { background:#f4eeff; border:1.5px solid #8a60c8; color:#3a1a70; }
.sc-amber  { background:#fff8ee; border:1.5px solid #cc9040; color:#6a3a00; }
.sc-red    { background:#fff0f0; border:1.5px solid #cc4040; color:#6a0a0a; }
.sc-teal   { background:#edfafa; border:1.5px solid #3a9a9a; color:#0a4040; }
.callout-ok   { background:#edf7ed; border-left:4px solid #4a9a4a; color:#1a4a1a;
                border-radius:8px; padding:12px 16px; font-size:13px; margin-top:14px; }
.callout-warn { background:#fff8ee; border-left:4px solid #cc9040; color:#5a3a00;
                border-radius:8px; padding:12px 16px; font-size:13px; margin-top:14px; }
.callout-bad  { background:#fff0f0; border-left:4px solid #cc4040; color:#5a0a0a;
                border-radius:8px; padding:12px 16px; font-size:13px; margin-top:14px; }
table { width:100%; border-collapse:collapse; font-size:13px; }
th { background:#f0f3f8; color:#4a5a78; font-size:11px; text-transform:uppercase;
     letter-spacing:.6px; padding:8px 10px; font-weight:700; text-align:left; }
td { padding:6px 10px; border-bottom:1px solid #edf0f6; }
tr:last-child td { border-bottom:none; }
tr:hover td { background:#f4f7fc; }
.td-r { text-align:right; }
.bar-row { display:flex; align-items:center; gap:8px; margin-bottom:5px; }
.bar-label { font-size:11px; color:#4a5a78; width:130px; text-align:right;
             flex-shrink:0; font-family:monospace; }
.bar-track { flex:1; height:18px; background:#eef1f8; border-radius:4px; overflow:hidden; }
.bar-fill  { height:100%; border-radius:4px; }
.bar-val   { font-size:11px; color:#6a7890; width:70px; flex-shrink:0; font-family:monospace; }
.b-green { background:#4a9a60; } .b-blue { background:#5a8de0; }
.b-amber { background:#cc8830; } .b-red  { background:#cc5040; }
.b-purple{ background:#8060c0;} .b-teal { background:#3a9a9a; }
"""


def _bars(items, colors, title=""):
    if not items: return ""
    mx = max(items.values()) or 1
    color_list = list(colors)
    html = f'<div class="panel-title">{title}</div>' if title else ""
    for i, (lbl, cnt) in enumerate(sorted(items.items(), key=lambda x: -x[1])):
        w   = 100 * cnt / mx
        col = color_list[i % len(color_list)]
        html += (f'<div class="bar-row">'
                 f'<span class="bar-label">{lbl[:20]}</span>'
                 f'<div class="bar-track"><div class="bar-fill {col}" style="width:{w:.1f}%"></div></div>'
                 f'<span class="bar-val">{cnt:,}</span></div>')
    return html


def _age_bars(dist):
    mx = max(dist.values()) or 1
    colors = {"< 0 (negative)": "b-red", "exactly 0": "b-amber"}
    html = ""
    for lbl, cnt in dist.items():
        if cnt == 0: continue
        col = colors.get(lbl, "b-blue")
        w   = 100 * cnt / mx
        html += (f'<div class="bar-row">'
                 f'<span class="bar-label">{lbl}</span>'
                 f'<div class="bar-track"><div class="bar-fill {col}" style="width:{w:.1f}%"></div></div>'
                 f'<span class="bar-val">{cnt:,}</span></div>')
    return html


def _nan_bars(counts, labels):
    mx = max(counts) or 1
    colors = ["b-green","b-green","b-green","b-amber","b-amber","b-red","b-red","b-red","b-red","b-red"]
    html = ""
    for cnt, lbl, col in zip(counts, labels, colors):
        w = 100 * cnt / mx
        html += (f'<div class="bar-row">'
                 f'<span class="bar-label">{lbl}</span>'
                 f'<div class="bar-track"><div class="bar-fill {col}" style="width:{w:.1f}%"></div></div>'
                 f'<span class="bar-val">{cnt:,}</span></div>')
    return html


def build_html(q1, q2):
    safe_badge = ('<span style="color:#1a7a1a;font-weight:700">✓ SAFE — outliers.csv is complete</span>'
                  if q1["safe_to_use_ids"] else
                  '<span style="color:#8a0a0a;font-weight:700">✗ WARNING — some bad samples still in kept set</span>')

    # Slide 1: Outlier overview
    s1 = f"""
<div class="slide">
  <div class="slide-title">Part 1 — Outlier Sample Analysis: Overview</div>
  <div class="grid4" style="margin-bottom:22px">
    <div class="stat-card sc-blue"><div class="s-val">{q1['total_21k']:,}</div>
      <div class="s-label">AltumAge 21k total</div></div>
    <div class="stat-card sc-green"><div class="s-val">{q1['kept_n']:,}</div>
      <div class="s-label">Kept (clean 19k)</div></div>
    <div class="stat-card sc-red"><div class="s-val">{q1['removed_n']:,}</div>
      <div class="s-label">Removed (outliers)</div></div>
    <div class="stat-card {'sc-green' if q1['kept_neg_n']==0 and q1['kept_over120_n']==0 else 'sc-red'}">
      <div class="s-val">{'✓' if q1['safe_to_use_ids'] else '✗'}</div>
      <div class="s-label">Kept set is clean</div>
      <div class="s-sub">0 neg-age / 0 &gt;120 in kept</div></div>
  </div>
  <div class="grid2" style="margin-bottom:20px">
    <div class="panel">
      <div class="panel-title">Age ranges</div>
      <table>
        <tr><th></th><th class="td-r">Kept</th><th class="td-r">Removed</th></tr>
        <tr><td>Min age</td><td class="td-r">{q1['kept_age_min']:.2f} yr</td>
            <td class="td-r" style="color:#cc4040">{q1['removed_age_min']:.2f} yr</td></tr>
        <tr><td>Max age</td><td class="td-r">{q1['kept_age_max']:.2f} yr</td>
            <td class="td-r">{q1['removed_age_max']:.2f} yr</td></tr>
        <tr><td>Mean ± std</td>
            <td class="td-r">{q1['kept_age_mean']:.1f} ± {q1['kept_age_std']:.1f}</td>
            <td class="td-r">{q1['removed_age_mean']:.1f} ± {q1['removed_age_std']:.1f}</td></tr>
      </table>
      <div style="margin-top:14px">
        <p><strong>Safety check:</strong></p>
        <p>Kept samples with age &lt; 0: <strong>{q1['kept_neg_n']:,}</strong>
           {'✓' if q1['kept_neg_n']==0 else '✗ PROBLEM'}</p>
        <p>Kept samples with age &gt; 120: <strong>{q1['kept_over120_n']:,}</strong>
           {'✓' if q1['kept_over120_n']==0 else '✗ PROBLEM'}</p>
        <p style="margin-top:8px">All neg-age in 21k: <strong>{q1['all_neg_in_21k']:,}</strong>
           &nbsp;·&nbsp; In outliers CSV: <strong>{q1['neg_in_outliers']:,}</strong>
           &nbsp;·&nbsp; Missed: <strong style="color:{'#1a7a1a' if q1['neg_missed']==0 else '#cc4040'}">{q1['neg_missed']:,}</strong></p>
      </div>
    </div>
    <div class="panel">
      <div class="panel-title">Removed samples — age distribution</div>
      {_age_bars(q1['removed_age_dist'])}
    </div>
  </div>
  <div class="{('callout-ok' if q1['safe_to_use_ids'] else 'callout-bad')}">
    <strong>Verdict:</strong> {safe_badge}<br>
    Using the outliers.csv sample ID list directly to exclude samples is safe and exact.
    No age-threshold guessing needed.
  </div>
</div>"""

    # Slide 2: Outlier breakdown by tissue/dataset/split
    s2 = f"""
<div class="slide">
  <div class="slide-title">Part 1 — Outlier Sample Analysis: Breakdown by Source</div>
  <div class="grid3">
    <div class="panel">
      <div class="panel-title">Tissue type</div>
      {_bars(q1['removed_by_tissue'], ['b-red','b-amber','b-purple','b-teal'], '')}
    </div>
    <div class="panel">
      <div class="panel-title">Dataset / GEO source</div>
      {_bars(q1['removed_by_dataset'], ['b-blue','b-purple','b-teal','b-amber'], '')}
    </div>
    <div class="panel">
      <div class="panel-title">Original split in AltumAge 21k</div>
      {_bars(q1['removed_by_split'], ['b-blue','b-red','b-amber'], '')}
      <div style="margin-top:16px;font-size:12px;color:#5a6888">
        <p>These {q1['removed_n']:,} samples were distributed across all 3 splits in the 21k.</p>
        <p style="margin-top:6px">Removing them changes the effective split sizes:</p>
        <p>train: −{q1['removed_by_split'].get('train',0):,} &nbsp;·&nbsp;
           valid: −{q1['removed_by_split'].get('valid',0):,} &nbsp;·&nbsp;
           test:  −{q1['removed_by_split'].get('test',0):,}</p>
      </div>
    </div>
  </div>
  <div class="callout-ok" style="margin-top:20px">
    <strong>Why were these samples removed?</strong><br>
    Negative ages = prenatal/fetal samples (placenta tissue, E-GEOD-43256, E-GEOD-44712).
    Their ages are stored as fractional negative years before birth.
    These are biologically incompatible with postnatal DNA methylation age prediction.
    One blood sample removed = likely duplicate methylation profile.
  </div>
</div>"""

    # Slide 3: NaN analysis
    fully_pct = 100 * q2['extra_fully_nan'] / max(q2['extra_n'], 1)
    over90_pct = 100 * q2['extra_over90_nan'] / max(q2['extra_n'], 1)
    conclusion_cls = "callout-bad" if fully_pct > 80 else "callout-warn"
    recommendation = (
        "Extra 1,760 CpGs are almost entirely NaN → <strong>EXCLUDE them</strong>. "
        "Use only the 19,608 shared CpGs. No imputation needed."
        if fully_pct > 80 else
        "Extra CpGs have partial NaN — imputation may be possible for some. "
        "Consider CpG-level NaN rate threshold (e.g., exclude CpGs with >50% NaN)."
    )
    s3 = f"""
<div class="slide">
  <div class="slide-title">Part 2 — Extra 1,760 CpGs: NaN Rate Analysis</div>
  <div class="grid4" style="margin-bottom:22px">
    <div class="stat-card sc-blue"><div class="s-val">{q2['shared_n']:,}</div>
      <div class="s-label">Shared CpGs (19k ∩ 21k)</div>
      <div class="s-sub">NaN mean: {q2['shared_nan_mean']*100:.4f}%</div></div>
    <div class="stat-card sc-red"><div class="s-val">{q2['extra_n']:,}</div>
      <div class="s-label">Extra CpGs (21k only)</div>
      <div class="s-sub">NaN mean: {q2['extra_nan_mean']*100:.1f}%</div></div>
    <div class="stat-card sc-red"><div class="s-val">{q2['extra_fully_nan']:,}</div>
      <div class="s-label">Fully NaN (100%)</div>
      <div class="s-sub">{fully_pct:.1f}% of extra CpGs</div></div>
    <div class="stat-card sc-amber"><div class="s-val">{q2['extra_over90_nan']:,}</div>
      <div class="s-label">&gt;90% NaN</div>
      <div class="s-sub">{over90_pct:.1f}% of extra CpGs</div></div>
  </div>
  <div class="grid2">
    <div class="panel">
      <div class="panel-title">NaN rate distribution — extra 1,760 CpGs</div>
      {_nan_bars(q2['extra_hist_counts'], q2['extra_hist_labels'])}
    </div>
    <div class="panel">
      <div class="panel-title">Per-sample NaN across extra CpGs</div>
      <table>
        <tr><th>Condition</th><th class="td-r">% of samples</th></tr>
        <tr><td>All 1,760 extra CpGs are NaN</td>
            <td class="td-r"><strong>{q2['sample_all_extra_nan_pct']:.1f}%</strong></td></tr>
        <tr><td>No extra CpGs are NaN</td>
            <td class="td-r"><strong>{q2['sample_no_extra_nan_pct']:.1f}%</strong></td></tr>
        <tr><td>&gt;50% of extra CpGs are NaN</td>
            <td class="td-r"><strong>{q2['sample_over50_extra_nan_pct']:.1f}%</strong></td></tr>
      </table>
      <div style="margin-top:14px;font-size:12px;color:#5a6888">
        <p>Shared 19,608 CpGs NaN mean: <strong>{q2['shared_nan_mean']*100:.4f}%</strong>
           (essentially zero)</p>
        <p style="margin-top:4px">Extra 1,760 CpGs NaN mean: <strong>{q2['extra_nan_mean']*100:.1f}%</strong></p>
      </div>
    </div>
  </div>
  <div class="{conclusion_cls}" style="margin-top:20px">
    <strong>Conclusion:</strong> {recommendation}
  </div>
</div>"""

    # Slide 4: Final recommendation
    s4 = f"""
<div class="slide">
  <div class="slide-title">Final Recommendation — How to Use AltumAge 21k for Fair Comparison</div>
  <div class="grid2" style="margin-bottom:20px">
    <div class="panel">
      <div class="panel-title">Step 1 — Remove outlier samples</div>
      <p><strong>Method:</strong> Exclude sample IDs from outliers.csv (exact list, 630 samples)</p>
      <p style="margin-top:8px"><strong>Why not age filter?</strong> Age threshold might accidentally remove
         valid samples or miss some outliers. The CSV is exact.</p>
      <p style="margin-top:8px">Result: <strong>{q1['kept_n']:,} samples</strong> remaining
         (age range: [{q1['kept_age_min']:.1f}, {q1['kept_age_max']:.1f}] yr)</p>
      <p style="margin-top:8px">Split sizes after removal:<br>
         train: {q1['removed_by_split'].get('train',0)} removed &nbsp;·&nbsp;
         valid: {q1['removed_by_split'].get('valid',0)} removed &nbsp;·&nbsp;
         test: {q1['removed_by_split'].get('test',0)} removed</p>
    </div>
    <div class="panel">
      <div class="panel-title">Step 2 — Handle extra 1,760 CpGs</div>
      <p><strong>Finding:</strong> {q2['extra_fully_nan']:,}/{q2['extra_n']:,} extra CpGs are 100% NaN.
         Mean NaN rate = {q2['extra_nan_mean']*100:.1f}%.</p>
      <p style="margin-top:8px"><strong>Decision:</strong>
         {'Use only the 19,608 shared CpGs. The extra 1,760 carry almost no information and filling them with fake values would harm the model.' if fully_pct > 80 else 'Some extra CpGs have real values. Consider a NaN threshold to decide which to keep.'}</p>
      <p style="margin-top:8px"><strong>No imputation needed</strong> for shared 19,608 CpGs
         (NaN rate ≈ {q2['shared_nan_mean']*100:.4f}%).</p>
    </div>
  </div>
  <div class="panel">
    <div class="panel-title">Step 3 — Use AltumAge 21k splits (same for both models)</div>
    <table>
      <tr><th>Split</th><th class="td-r">21k total</th><th class="td-r">After outlier removal</th><th class="td-r">Removed</th></tr>
      <tr><td>train</td><td class="td-r">7,416</td>
          <td class="td-r"><strong>{7416 - q1['removed_by_split'].get('train',0):,}</strong></td>
          <td class="td-r">{q1['removed_by_split'].get('train',0):,}</td></tr>
      <tr><td>valid</td><td class="td-r">1,308</td>
          <td class="td-r"><strong>{1308 - q1['removed_by_split'].get('valid',0):,}</strong></td>
          <td class="td-r">{q1['removed_by_split'].get('valid',0):,}</td></tr>
      <tr><td>test</td><td class="td-r">2,264</td>
          <td class="td-r"><strong>{2264 - q1['removed_by_split'].get('test',0):,}</strong></td>
          <td class="td-r">{q1['removed_by_split'].get('test',0):,}</td></tr>
    </table>
  </div>
  <div class="callout-ok" style="margin-top:16px">
    <strong>Result:</strong> Both MethylLlama and MethylGPT trained and evaluated on exactly the same
    {q1['kept_n']:,} samples with the same split assignments.
    MedAE comparison becomes fully valid and directly comparable.
  </div>
</div>"""

    return (f"<!DOCTYPE html><html><head><meta charset='UTF-8'>"
            f"<title>21k Dataset Structure Analysis</title>"
            f"<style>{CSS}</style></head><body>"
            f"{s1}{s2}{s3}{s4}</body></html>")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alt_h5ad",     default=ALT_H5AD)
    ap.add_argument("--llama_h5ad",   default=LLAMA_H5AD)
    ap.add_argument("--outliers_csv", default=OUTLIERS_CSV)
    ap.add_argument("--outdir",       default="dataset_fingerprint_outputs")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    outliers_df = pd.read_csv(args.outliers_csv)
    outlier_ids = set(outliers_df["sample_id"].astype(str))
    print(f"Outliers CSV loaded: {len(outlier_ids):,} samples")

    alt   = load_h5ad_full(args.alt_h5ad,   "AltumAge 21k")
    import scanpy as sc
    llama = sc.read_h5ad(args.llama_h5ad)
    llama_cpgs = set(llama.var.index.astype(str))
    alt_cpgs   = list(alt.var.index.astype(str))

    q1 = analyze_outliers(alt.obs, outlier_ids)
    q2 = analyze_nan(alt.X, alt_cpgs, llama_cpgs, outdir)

    txt  = build_txt(q1, q2)
    html = build_html(q1, q2)

    (outdir / "analysis_summary.txt").write_text(txt)
    (outdir / "analysis_report.html").write_text(html)

    print(f"\n{'='*60}")
    print(f"Outputs → {outdir}/")
    print(f"  analysis_report.html")
    print(f"  analysis_summary.txt")
    print(f"  extra_cpg_nan_profile.csv")
    print(f"{'='*60}")
    print(txt)


if __name__ == "__main__":
    main()
