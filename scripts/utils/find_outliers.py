#!/usr/bin/env python3
"""
find_outliers.py
=================
Identifies samples that exist in AltumAge 21k but were removed
when creating the MethylLlama 19k filtered dataset.

Outputs:
  outliers.csv  — one row per removed sample with all available metadata
  summary.txt   — plain-text summary
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

_BASE = "/sci/labs/benjamin.yakir/netanel.azran"
_DATA = f"{_BASE}/data"

ALT_H5AD   = f"{_DATA}/data_methyl_21k_h5ad/altumage_21k_3way.h5ad"
LLAMA_H5AD = (f"{_DATA}/data_methyl_finetune_19k_h5ad/"
              "finetuning_19608_clean_stratified_no_outliers.h5ad")


def load_obs(path: str, label: str) -> pd.DataFrame:
    """Load only the obs (sample metadata) from an h5ad file."""
    print(f"\n[{label}] loading obs from {path}")
    try:
        import scanpy as sc
        adata = sc.read_h5ad(path)
        obs = adata.obs.copy()
        obs.index = obs.index.astype(str)
        print(f"  {len(obs):,} samples  columns: {obs.columns.tolist()}")
        return obs
    except Exception as e:
        print(f"  scanpy failed ({e}), using h5py...")
        import h5py
        with h5py.File(path, "r") as f:
            def read_grp(grp, n):
                idx_key = "_index" if "_index" in grp else list(grp.keys())[0]
                idx = [x.decode() if isinstance(x, bytes) else str(x)
                       for x in grp[idx_key][:]]
                cols = {}
                for k in grp.keys():
                    if k == idx_key:
                        continue
                    try:
                        v = grp[k]
                        if isinstance(v, h5py.Dataset) and v.ndim == 1 and len(v) == n:
                            raw = v[()]
                            cols[k] = [x.decode() if isinstance(x, bytes) else x for x in raw]
                        elif isinstance(v, h5py.Group) and "categories" in v:
                            cats  = [x.decode() if isinstance(x, bytes) else str(x)
                                     for x in v["categories"][()]]
                            codes = v["codes"][()]
                            cols[k] = [cats[c] if c >= 0 else None for c in codes]
                    except Exception:
                        pass
                return idx, pd.DataFrame(cols, index=idx)

            n = len(f["obs"]["_index"])
            idx, obs = read_grp(f["obs"], n)
        print(f"  {len(obs):,} samples  columns: {obs.columns.tolist()}")
        return obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alt_h5ad",   default=ALT_H5AD)
    ap.add_argument("--llama_h5ad", default=LLAMA_H5AD)
    ap.add_argument("--outdir",     default="dataset_fingerprint_outputs")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load metadata only (no X matrix needed)
    alt_obs   = load_obs(args.alt_h5ad,   "AltumAge 21k")
    llama_obs = load_obs(args.llama_h5ad, "MethylLlama 19k")

    alt_ids   = set(alt_obs.index)
    llama_ids = set(llama_obs.index)

    kept    = alt_ids & llama_ids
    removed = alt_ids - llama_ids
    added   = llama_ids - alt_ids  # should be 0

    print(f"\n{'='*60}")
    print(f"AltumAge 21k total    : {len(alt_ids):,}")
    print(f"MethylLlama 19k total : {len(llama_ids):,}")
    print(f"Kept (in both)        : {len(kept):,}")
    print(f"Removed (outliers)    : {len(removed):,}")
    print(f"In 19k but not 21k    : {len(added):,}  (should be 0)")

    # Build outlier CSV — all metadata from the 21k obs
    outlier_df = alt_obs.loc[sorted(removed)].copy()
    outlier_df.index.name = "sample_id"
    outlier_df = outlier_df.reset_index()

    # Add a column showing which 21k split they were in
    if "split" in outlier_df.columns:
        split_counts = outlier_df["split"].value_counts()
        print(f"\nOutlier split distribution in AltumAge 21k:")
        for sp, n in split_counts.items():
            print(f"  {sp}: {n:,}")

    # Age stats of outliers vs kept
    if "age" in alt_obs.columns:
        kept_ages    = pd.to_numeric(alt_obs.loc[sorted(kept),    "age"], errors="coerce").dropna()
        removed_ages = pd.to_numeric(alt_obs.loc[sorted(removed), "age"], errors="coerce").dropna()
        print(f"\nAge stats — KEPT samples   : mean={kept_ages.mean():.1f}  "
              f"std={kept_ages.std():.1f}  range=[{kept_ages.min():.0f}, {kept_ages.max():.0f}]")
        print(f"Age stats — REMOVED samples: mean={removed_ages.mean():.1f}  "
              f"std={removed_ages.std():.1f}  range=[{removed_ages.min():.0f}, {removed_ages.max():.0f}]")

        outlier_df["age"] = pd.to_numeric(outlier_df["age"], errors="coerce")

    # Save CSV
    csv_path = outdir / "outliers.csv"
    outlier_df.to_csv(csv_path, index=False)
    print(f"\nOutlier CSV saved: {csv_path}")
    print(f"Columns: {outlier_df.columns.tolist()}")
    print(f"\nFirst 10 outlier samples:")
    print(outlier_df.head(10).to_string(index=False))

    # Summary text
    lines = [
        "="*60,
        "OUTLIER SAMPLES REPORT",
        "AltumAge 21k  →  MethylLlama 19k (filtered)",
        "="*60,
        f"AltumAge 21k total    : {len(alt_ids):,}",
        f"MethylLlama 19k total : {len(llama_ids):,}",
        f"Kept (in both)        : {len(kept):,}",
        f"Removed as outliers   : {len(removed):,}",
        f"In 19k but not 21k    : {len(added):,}",
        "",
    ]
    if "split" in outlier_df.columns:
        lines.append("Outlier split distribution in AltumAge 21k:")
        for sp, n in split_counts.items():
            lines.append(f"  {sp}: {n:,}")
        lines.append("")
    if "age" in alt_obs.columns:
        lines += [
            f"Age — kept    : mean={kept_ages.mean():.1f}  std={kept_ages.std():.1f}  "
            f"range=[{kept_ages.min():.0f}, {kept_ages.max():.0f}]",
            f"Age — removed : mean={removed_ages.mean():.1f}  std={removed_ages.std():.1f}  "
            f"range=[{removed_ages.min():.0f}, {removed_ages.max():.0f}]",
            "",
        ]
    lines += [f"CSV: {csv_path}", "="*60]

    txt_path = outdir / "outliers_summary.txt"
    txt_path.write_text("\n".join(lines))
    print(f"Summary saved: {txt_path}")


if __name__ == "__main__":
    main()
