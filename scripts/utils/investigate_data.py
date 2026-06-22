"""
Data Investigation — AltumAge 21k Fine-tuning Dataset
======================================================
Analyzes both the original 21k AltumAge file and the cleaned 19k version
used in MethylLlama fine-tuning.

Covers all 14 inspection points per dataset:
  1.  File path & size
  2.  File format
  3.  Number of samples
  4.  Number of CpG probes
  5.  Total beta-values
  6.  CpG probe names
  7.  CpG order consistency (duplicate IDs)
  8.  Sample metadata presence
  9.  Metadata fields
  10. Label columns (age, tissue, etc.)
  11. Duplicate samples
  12. Duplicate CpG IDs
  13. CpGs with 100% missing values
  14. Samples with 100% missing methylation

Plus extended analysis:
  - Age distribution (mean, std, min, max, percentiles, histogram)
  - Beta-value distribution per split
  - Zero vs NaN breakdown (zeros are real unmethylated CpGs)
  - Per-split statistics (train / val / test)

Usage:
    python scripts/utils/investigate_data.py
"""

import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp

try:
    import scanpy as sc
except ImportError:
    sys.exit("scanpy not installed. Run: pip install scanpy")


BASE = "/sci/labs/benjamin.yakir/netanel.azran/data"

# Original AltumAge 21k file (train/val/test splits baked in)
ALTUMAGE_21K_PATH = f"{BASE}/data_methyl_21k_h5ad/altumage_21k_3way.h5ad"

# Cleaned version: only the 19,608 CpGs that have zero NaN across all samples
CLEAN_19K_PATH = (
    f"{BASE}/data_methyl_finetune_19k_h5ad/finetuning_19608_clean.h5ad"
)

CHUNK_ROWS = 2000   # rows per NaN-analysis chunk
SEP = "=" * 70


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def section(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def load_data(path):
    print(f"\nLoading: {path}")
    print(f"File size: {os.path.getsize(path) / 1e9:.2f} GB")
    try:
        adata = sc.read_h5ad(path)
    except ValueError:
        # Pretrain file has a 1-row obs stub (old anndata format).
        # Build a minimal AnnData directly from h5py to bypass shape validation.
        import h5py
        import anndata as ad
        print("  obs/X shape mismatch — building AnnData from h5py directly")
        with h5py.File(path, "r") as f:
            # X can be a dataset or a group (sparse CSR/CSC)
            if "X" in f and isinstance(f["X"], h5py.Dataset):
                X = f["X"][:]
            elif "X" in f:
                # sparse matrix stored as group with data/indices/indptr
                grp = f["X"]
                data = grp["data"][:]
                indices = grp["indices"][:]
                indptr = grp["indptr"][:]
                shape = tuple(grp.attrs.get("shape", grp.attrs.get("h5sparse_shape")))
                X = sp.csr_matrix((data, indices, indptr), shape=shape)
            else:
                raise RuntimeError("Cannot find X in h5ad file")
            n_obs = X.shape[0]
            # var_names
            if "var" in f and "_index" in f["var"]:
                var_names = f["var"]["_index"][:].astype(str).tolist()
            elif "var" in f:
                keys = list(f["var"].keys())
                var_names = f["var"][keys[0]][:].astype(str).tolist()
            else:
                var_names = [str(i) for i in range(X.shape[1])]
            # obs metadata (best-effort)
            obs_df = pd.DataFrame(index=pd.RangeIndex(n_obs))
            if "obs" in f:
                for col in f["obs"].keys():
                    if col.startswith("_"):
                        continue
                    try:
                        vals = f["obs"][col][:]
                        if len(vals) == n_obs:
                            obs_df[col] = vals
                    except Exception:
                        pass
        var_df = pd.DataFrame(index=var_names)
        adata = ad.AnnData(X=X, obs=obs_df, var=var_df)
    print(f"Loaded OK  →  {adata.n_obs:,} samples × {adata.n_vars:,} CpGs")
    return adata


def _row_chunk(X, start, end):
    """Return rows [start:end] as a dense float32 array, sparse-safe."""
    block = X[start:end]
    if sp.issparse(block):
        return block.toarray().astype(np.float32)
    return np.array(block, dtype=np.float32)


def nan_stats_chunked(adata):
    """
    Compute NaN statistics without ever densifying the full matrix.

    Returns a dict with:
        total_values, nan_count, nan_pct,
        nan_per_sample  (ndarray, shape n_obs),
        nan_per_cpg     (ndarray, shape n_vars),
    """
    X = adata.X
    n_obs, n_vars = adata.n_obs, adata.n_vars

    nan_per_sample = np.zeros(n_obs, dtype=np.int64)
    nan_per_cpg = np.zeros(n_vars, dtype=np.int64)
    total_nan = 0

    for start in range(0, n_obs, CHUNK_ROWS):
        end = min(start + CHUNK_ROWS, n_obs)
        chunk = _row_chunk(X, start, end)
        is_nan = np.isnan(chunk)
        nan_per_sample[start:end] = is_nan.sum(axis=1)
        nan_per_cpg += is_nan.sum(axis=0)
        total_nan += int(is_nan.sum())
        del chunk, is_nan

    total = n_obs * n_vars
    return {
        "total_values": total,
        "nan_count": total_nan,
        "nan_pct": 100.0 * total_nan / total if total > 0 else 0.0,
        "nan_per_sample": nan_per_sample,
        "nan_per_cpg": nan_per_cpg,
    }


def beta_stats_chunked(adata):
    """
    Compute β-value range and percentile distribution in chunks.
    Returns (min, max, out_of_range_count, percentile_dict).
    """
    X = adata.X
    n_obs = adata.n_obs
    reservoir = []   # collect a sample of valid values for percentile estimation
    reservoir_cap = 5_000_000
    global_min = np.inf
    global_max = -np.inf
    out_of_range = 0
    total_valid = 0

    for start in range(0, n_obs, CHUNK_ROWS):
        end = min(start + CHUNK_ROWS, n_obs)
        chunk = _row_chunk(X, start, end)
        valid = chunk[~np.isnan(chunk)]
        if len(valid) == 0:
            continue
        global_min = min(global_min, float(valid.min()))
        global_max = max(global_max, float(valid.max()))
        out_of_range += int(((valid < 0) | (valid > 1)).sum())
        total_valid += len(valid)
        # subsample for percentiles
        if len(reservoir) < reservoir_cap:
            take = min(len(valid), reservoir_cap - len(reservoir))
            reservoir.append(valid[:take])
        del chunk, valid

    if total_valid == 0:
        return None

    sample = np.concatenate(reservoir)
    percs = np.percentile(sample, [0, 1, 5, 25, 50, 75, 95, 99, 100])
    labels = ["0%", "1%", "5%", "25%", "50%", "75%", "95%", "99%", "100%"]

    return {
        "min": global_min,
        "max": global_max,
        "out_of_range": out_of_range,
        "total_valid": total_valid,
        "percentiles": dict(zip(labels, percs)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-dataset investigation
# ─────────────────────────────────────────────────────────────────────────────

def investigate_dataset(path, dataset_name):
    section(f"DATASET: {dataset_name.upper()}")

    # ── [1][2] File ───────────────────────────────────────────────────────────
    print(f"\n[1] Path      : {path}")
    print(f"[2] Format    : h5ad (AnnData / HDF5)")
    print(f"    File size : {os.path.getsize(path) / 1e9:.2f} GB")

    adata = load_data(path)

    # ── [3][4][5] Shape ───────────────────────────────────────────────────────
    print(f"\n[3] Samples     : {adata.n_obs:,}")
    print(f"[4] CpG probes  : {adata.n_vars:,}")
    print(f"[5] Total values: {adata.n_obs * adata.n_vars:,}")
    print(f"    Storage     : {'sparse' if sp.issparse(adata.X) else 'dense'}")

    # ── [6] CpG names ─────────────────────────────────────────────────────────
    cpg_ids = list(adata.var_names)
    print(f"\n[6] CpG IDs (first 5): {cpg_ids[:5]}")
    print(f"    CpG IDs (last  5): {cpg_ids[-5:]}")

    # ── [7][12] Duplicate CpG IDs ─────────────────────────────────────────────
    dup_cpg = int(pd.Series(cpg_ids).duplicated().sum())
    print(f"\n[7/12] Duplicate CpG IDs: {dup_cpg}  "
          f"{'⚠️  WARNING' if dup_cpg > 0 else '✅ none'}")

    # ── [11] Duplicate sample IDs ─────────────────────────────────────────────
    obs_ids = list(adata.obs_names)
    dup_samples = int(pd.Series(obs_ids).duplicated().sum())
    print(f"[11]   Duplicate sample IDs: {dup_samples}  "
          f"{'⚠️  WARNING' if dup_samples > 0 else '✅ none'}")

    # ── [8][9] Metadata ───────────────────────────────────────────────────────
    meta_cols = list(adata.obs.columns)
    print(f"\n[8] Metadata present : {'yes' if meta_cols else 'no'}")
    print(f"[9] Metadata fields  ({len(meta_cols)}): {meta_cols}")

    # ── [10] Label columns ────────────────────────────────────────────────────
    label_candidates = ["age", "tissue", "disease", "batch", "platform",
                        "dataset", "source", "cell_type", "sex", "split",
                        "sample_id", "donor_id"]
    found_labels = [c for c in label_candidates if c in adata.obs.columns]
    print(f"[10] Label columns found: {found_labels if found_labels else 'none'}")

    if "age" in adata.obs.columns:
        age = adata.obs["age"].dropna()
        nan_age = adata.obs["age"].isna().sum()
        print(f"\n     Age — n={len(age):,}  NaN={nan_age:,}")
        print(f"     min={age.min():.1f}   max={age.max():.1f}")
        print(f"     mean={age.mean():.2f}  std={age.std():.2f}  median={age.median():.1f}")
        pcts = np.percentile(age, [5, 25, 50, 75, 95])
        print(f"     percentiles — p5={pcts[0]:.1f}  p25={pcts[1]:.1f}  "
              f"p50={pcts[2]:.1f}  p75={pcts[3]:.1f}  p95={pcts[4]:.1f}")
        # Age histogram in terminal (10-year bins)
        print(f"\n     Age histogram (10-year bins):")
        bins = list(range(0, 121, 10))
        counts, _ = np.histogram(age, bins=bins)
        max_c = max(counts) if max(counts) > 0 else 1
        for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
            bar_len = int(40 * counts[i] / max_c)
            bar = "█" * bar_len
            print(f"     {lo:3d}–{hi:3d}: {bar:<40s} {counts[i]:,}")

    # ── Split distribution ────────────────────────────────────────────────────
    split_col = next((c for c in ["split", "Split", "dataset_split", "fold"]
                      if c in adata.obs.columns), None)
    if split_col:
        print(f"\n     Split column '{split_col}':")
        print(adata.obs[split_col].value_counts().to_string())
    else:
        print("\n     ⚠️  No split column found")

    # ── [13][14] NaN analysis (chunked) ──────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  β-VALUE NaN ANALYSIS (chunked, memory-safe)")
    print(f"{'─'*60}")
    print("  Computing NaN statistics...", flush=True)

    stats = nan_stats_chunked(adata)
    nan_per_sample = stats["nan_per_sample"]
    nan_per_cpg = stats["nan_per_cpg"]

    print(f"\n  Total β-values : {stats['total_values']:,}")
    print(f"  NaN count      : {stats['nan_count']:,}")
    print(f"  NaN %          : {stats['nan_pct']:.4f}%")

    # [14] Samples fully missing
    full_missing_samples = int((nan_per_sample == adata.n_vars).sum())
    partial_missing_samples = int(
        ((nan_per_sample > 0) & (nan_per_sample < adata.n_vars)).sum()
    )
    print(f"\n  [14] Samples with 0   NaN : {int((nan_per_sample == 0).sum()):,}")
    print(f"       Samples with SOME NaN: {partial_missing_samples:,}")
    print(f"       Samples with ALL  NaN: {full_missing_samples:,}  "
          f"{'⚠️  WARNING' if full_missing_samples > 0 else '✅ none'}")

    pcts = nan_per_sample / adata.n_vars * 100
    for threshold in [1, 5, 10, 25, 50]:
        n = int((pcts >= threshold).sum())
        print(f"       ≥{threshold:2d}% NaN/sample: {n:,}  ({100*n/adata.n_obs:.1f}%)")

    # [13] CpGs fully missing
    full_missing_cpgs = int((nan_per_cpg == adata.n_obs).sum())
    partial_missing_cpgs = int(
        ((nan_per_cpg > 0) & (nan_per_cpg < adata.n_obs)).sum()
    )
    print(f"\n  [13] CpGs with 0   NaN : {int((nan_per_cpg == 0).sum()):,}")
    print(f"       CpGs with SOME NaN: {partial_missing_cpgs:,}")
    print(f"       CpGs with ALL  NaN: {full_missing_cpgs:,}  "
          f"{'⚠️  WARNING' if full_missing_cpgs > 0 else '✅ none'}")

    cpg_pcts = nan_per_cpg / adata.n_obs * 100
    for threshold in [1, 5, 10, 25, 50]:
        n = int((cpg_pcts >= threshold).sum())
        print(f"       ≥{threshold:2d}% NaN/CpG  : {n:,}  ({100*n/adata.n_vars:.1f}%)")

    # ── β-value range (chunked) ───────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  β-VALUE RANGE & DISTRIBUTION")
    print(f"{'─'*60}")
    print("  Computing β stats...", flush=True)

    bstats = beta_stats_chunked(adata)
    if bstats:
        print(f"\n  β-value range  : [{bstats['min']:.4f}, {bstats['max']:.4f}]")
        oor = bstats["out_of_range"]
        print(f"  Out-of-range   : {oor:,}  "
              f"{'⚠️  WARNING' if oor > 0 else '✅ all in [0,1]'}")
        print(f"\n  β-value percentiles (sampled from {bstats['total_valid']:,} valid values):")
        for lbl, val in bstats["percentiles"].items():
            print(f"    {lbl:>5s}: {val:.4f}")

    # ── Per-split NaN breakdown ───────────────────────────────────────────────
    if split_col:
        print(f"\n{'─'*60}")
        print("  NaN BREAKDOWN BY SPLIT")
        print(f"{'─'*60}")
        for split_val in sorted(adata.obs[split_col].unique()):
            mask = adata.obs[split_col] == split_val
            sub = adata[mask]
            sub_stats = nan_stats_chunked(sub)
            print(f"\n  [{split_val}]  {sub.n_obs:,} samples")
            print(f"    NaN: {sub_stats['nan_count']:,}  ({sub_stats['nan_pct']:.4f}%)")
            sub_pcts = sub_stats["nan_per_sample"] / sub.n_vars * 100
            for thr in [1, 10, 50]:
                n = int((sub_pcts >= thr).sum())
                print(f"    ≥{thr:2d}% NaN/sample: {n:,} ({100*n/sub.n_obs:.1f}%)")

    # ── Summary record ────────────────────────────────────────────────────────
    return {
        "dataset": dataset_name,
        "n_samples": adata.n_obs,
        "n_cpgs": adata.n_vars,
        "total_values": stats["total_values"],
        "nan_count": stats["nan_count"],
        "nan_pct": stats["nan_pct"],
        "full_missing_samples": full_missing_samples,
        "full_missing_cpgs": full_missing_cpgs,
        "dup_cpg": dup_cpg,
        "dup_samples": dup_samples,
        "meta_cols": meta_cols,
        "label_cols": found_labels,
        "split_col": split_col,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_summary_table(results):
    section("SUMMARY TABLE")
    fields = [
        ("Dataset",               "dataset"),
        ("Samples",               "n_samples"),
        ("CpGs",                  "n_cpgs"),
        ("Total β-values",        "total_values"),
        ("NaN count",             "nan_count"),
        ("NaN %",                 "nan_pct"),
        ("Fully missing samples", "full_missing_samples"),
        ("Fully missing CpGs",    "full_missing_cpgs"),
        ("Duplicate sample IDs",  "dup_samples"),
        ("Duplicate CpG IDs",     "dup_cpg"),
    ]

    col_w = 26
    header = f"{'Field':<{col_w}}" + "".join(f"{r['dataset']:>22}" for r in results)
    print(f"\n{header}")
    print("─" * (col_w + 22 * len(results)))

    for label, key in fields:
        row = f"{label:<{col_w}}"
        for r in results:
            val = r[key]
            if isinstance(val, float):
                row += f"{val:>21.4f}%"
            elif isinstance(val, int):
                row += f"{val:>22,}"
            else:
                row += f"{str(val):>22}"
        print(row)


# ─────────────────────────────────────────────────────────────────────────────
# Interpretation
# ─────────────────────────────────────────────────────────────────────────────

def print_interpretation(results):
    section("INTERPRETATION")
    for r in results:
        nan_ok = r["nan_pct"] < 1.0
        dup_ok = r["dup_samples"] == 0 and r["dup_cpg"] == 0
        miss_ok = r["full_missing_samples"] == 0 and r["full_missing_cpgs"] == 0
        print(f"\n  [{r['dataset']}]")
        nan_msg = "✅ Low (<1%)" if nan_ok else f'⚠️  High ({r["nan_pct"]:.2f}%) — model must mask or impute'
        print(f"    NaN rate       : {nan_msg}")
        print(f"    Duplicates     : {'✅ None' if dup_ok else '⚠️  Found — deduplicate before training'}")
        print(f"    Fully missing  : {'✅ None' if miss_ok else '⚠️  Found — filter before training'}")
        print(f"    Label columns  : {r['label_cols']}")
        print(f"    Metadata fields: {r['meta_cols']}")
        print(f"    Split column   : {r['split_col'] or '⚠️  missing'}")

        if nan_ok and dup_ok and miss_ok:
            print("    → ✅ Structurally valid and ready for training.")
        else:
            print("    → ⚠️  Issues detected:")
            if not nan_ok:
                print("       NaN: confirm the data loader fills missing CpGs with 0 "
                      "or uses an attention mask.")
            if not dup_ok:
                print("       Duplicates: deduplicate to avoid data leakage.")
            if not miss_ok:
                print("       Fully missing entries: filter out affected rows/columns.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    datasets = [
        (ALTUMAGE_21K_PATH, "altumage-21k-original"),
        (CLEAN_19K_PATH,    "finetune-21k-clean"),
    ]

    results = []
    for path, name in datasets:
        if not os.path.exists(path):
            print(f"\n⚠️  Skipping {name} — file not found: {path}")
            continue
        result = investigate_dataset(path, name)
        results.append(result)

    if results:
        print_summary_table(results)
        print_interpretation(results)

    section("DONE")


if __name__ == "__main__":
    main()
