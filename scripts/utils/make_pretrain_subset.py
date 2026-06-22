"""
Create a small representative subset of the 170k × 49k pretrain dataset for smoke testing.

Strategy:
  - Samples : ~2000 total, sampled uniformly at random
  - CpGs    : All 49k kept by default (top_k_cpg=0); or top-K by variance if specified
  - Splits  : 80 / 10 / 10 train/valid/test added as obs column

Input (choose one):
  --h5ad   Path to existing AnnData h5ad (preferred — data is already in h5ad format)
  --tar    Path to .tar.gz of parquet blocks (legacy)

Output: methylgpt_pretrain_type3_subset.h5ad
        probe_ids_type3_pretrain_subset.csv

Usage (h5ad input):
    python scripts/utils/make_pretrain_subset.py \
        --h5ad    /path/to/methylgpt_pretrain_type3.h5ad \
        --out_dir /path/to/output/ \
        --n_samples 2000

Usage (parquet tar input):
    python scripts/utils/make_pretrain_subset.py \
        --tar     /path/to/processed_type3_parquet_shuffled.tar.gz \
        --probes  /path/to/probe_ids_type3_pretrain.csv \
        --out_dir /path/to/output/ \
        --n_samples 2000
"""

import argparse
import tarfile
import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    # Input — mutually exclusive paths
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--h5ad",    help="Path to existing .h5ad AnnData file (preferred)")
    grp.add_argument("--tar",     help="Path to .tar.gz of parquet blocks (legacy)")
    p.add_argument("--probes",    help="Path to probe_ids CSV — required when using --tar")
    p.add_argument("--out_dir",   required=True, help="Output directory")
    p.add_argument("--n_samples", type=int, default=2000,  help="Total samples to keep")
    p.add_argument("--top_k_cpg", type=int, default=0, help="Top-K most variable CpGs to keep (0 = all)")
    p.add_argument("--seed",      type=int, default=42)
    return p.parse_args()


# ── h5ad loading path ──────────────────────────────────────────────────────────
def load_from_h5ad(h5ad_path, n_samples, rng):
    import h5py
    print(f"[1] Reading {h5ad_path} via h5py (avoids anndata shape-inference bug)")

    with h5py.File(h5ad_path, "r") as f:
        n_total, n_cpg = f["X"].shape
        print(f"    Full dataset: {n_total} samples × {n_cpg} CpGs")

        # Sample IDs from obs/_index (the canonical AnnData index)
        obs_index = f["obs/_index"][:].astype(str)

        # CpG IDs from var/cpg_id; fall back to var/_index
        if "cpg_id" in f["var"]:
            all_cpg_ids = f["var/cpg_id"][:].astype(str).tolist()
        else:
            all_cpg_ids = f["var/_index"][:].astype(str).tolist()

        n_take = min(n_samples, n_total)
        chosen = rng.choice(n_total, size=n_take, replace=False)
        chosen_sorted = np.sort(chosen)

        print(f"[2] Sampling {n_take} rows from disk ...")
        X = f["X"][chosen_sorted, :].astype(np.float32)   # only loads selected rows

    all_ids = obs_index[chosen_sorted].tolist()
    nan_pct = np.isnan(X).mean() * 100
    print(f"    Shape: {X.shape}  NaN: {nan_pct:.1f}%")
    return X, all_ids, all_cpg_ids, nan_pct


# ── parquet tar loading path (legacy) ─────────────────────────────────────────
def load_from_tar(tar_path, probes_path, n_samples, rng):
    probe_df = pd.read_csv(probes_path)
    if "illumina_probe_id" in probe_df.columns:
        all_cpg_ids = probe_df["illumina_probe_id"].tolist()
    else:
        all_cpg_ids = probe_df.iloc[:, 0].tolist()
    n_cpg_total = len(all_cpg_ids)
    print(f"[1] Probe IDs loaded: {n_cpg_total} CpGs")

    print(f"\n[2] Opening {tar_path}")
    with tarfile.open(tar_path) as tar:
        members = sorted(
            [m for m in tar.getmembers() if m.name.endswith(".parquet")],
            key=lambda m: m.name,
        )
    n_blocks = len(members)
    print(f"    {n_blocks} parquet blocks found")

    per_block = n_samples // n_blocks
    remainder = n_samples - per_block * n_blocks
    per_block_counts = [per_block + (1 if i < remainder else 0) for i in range(n_blocks)]

    print(f"\n[3] Sampling {n_samples} samples (~{per_block} per block)")
    all_ids  = []
    all_data = []

    with tarfile.open(tar_path) as tar:
        for idx, (member, n_take) in enumerate(zip(members, per_block_counts)):
            if n_take == 0:
                continue
            f  = tar.extractfile(member)
            df = pd.read_parquet(f)
            n_available = len(df)
            n_take = min(n_take, n_available)
            chosen = rng.choice(n_available, size=n_take, replace=False)
            for _, row in df.iloc[chosen].iterrows():
                all_ids.append(row["id"])
                all_data.append(row["data"].astype(np.float32))
            if (idx + 1) % 5 == 0 or idx == n_blocks - 1:
                print(f"    block {idx+1}/{n_blocks}  collected so far: {len(all_ids)}")

    print(f"\n[4] Building data matrix ...")
    X = np.stack(all_data, axis=0)
    nan_pct = np.isnan(X).mean() * 100
    print(f"    Shape: {X.shape}  NaN: {nan_pct:.1f}%")
    return X, all_ids, all_cpg_ids, nan_pct


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    if args.tar and not args.probes:
        raise ValueError("--probes is required when using --tar")

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.h5ad:
        X, all_ids, all_cpg_ids, nan_pct = load_from_h5ad(args.h5ad, args.n_samples, rng)
    else:
        X, all_ids, all_cpg_ids, nan_pct = load_from_tar(args.tar, args.probes, args.n_samples, rng)

    n_samples_actual = X.shape[0]
    n_cpg_total = X.shape[1]

    # ── CpG filtering (optional) ───────────────────────────────────────────────
    step = "[3]" if args.h5ad else "[5]"
    if args.top_k_cpg > 0 and args.top_k_cpg < n_cpg_total:
        print(f"\n{step} Selecting top {args.top_k_cpg} most variable CpGs ...")
        cpg_var = np.nanvar(X, axis=0)
        top_idx = np.argsort(cpg_var)[-args.top_k_cpg:][::-1]
        top_idx_sorted = np.sort(top_idx)
        X_sub = X[:, top_idx_sorted]
        cpg_ids_sub = [all_cpg_ids[i] for i in top_idx_sorted]
        print(f"    Variance range (top-K): {cpg_var[top_idx].min():.4f} – {cpg_var[top_idx].max():.4f}")
    else:
        print(f"\n{step} Keeping all {n_cpg_total} CpGs (top_k_cpg=0 or >= total)")
        X_sub = X
        cpg_ids_sub = all_cpg_ids

    print(f"    Final matrix: {X_sub.shape}")

    # ── Train/valid/test split ─────────────────────────────────────────────────
    step = "[4]" if args.h5ad else "[6]"
    print(f"\n{step} Assigning train/valid/test splits (80/10/10, seed=42) ...")
    perm      = rng.permutation(n_samples_actual)
    train_end = int(0.8 * n_samples_actual)
    val_end   = int(0.9 * n_samples_actual)
    split_col = np.empty(n_samples_actual, dtype=object)
    split_col[perm[:train_end]]        = "train"
    split_col[perm[train_end:val_end]] = "valid"
    split_col[perm[val_end:]]          = "test"
    train_n = (split_col == "train").sum()
    valid_n = (split_col == "valid").sum()
    test_n  = (split_col == "test").sum()
    print(f"    train={train_n}  valid={valid_n}  test={test_n}")

    # ── Build AnnData ──────────────────────────────────────────────────────────
    step = "[5]" if args.h5ad else "[7]"
    print(f"\n{step} Building AnnData ...")
    obs   = pd.DataFrame({"sample_id": all_ids, "split": split_col}, index=all_ids)
    var   = pd.DataFrame({"cpg_id": cpg_ids_sub}, index=cpg_ids_sub)
    adata = ad.AnnData(X=X_sub, obs=obs, var=var)
    print(f"    AnnData: {adata}")

    # ── Save h5ad ──────────────────────────────────────────────────────────────
    out_h5ad = out_dir / "methylgpt_pretrain_type3_subset.h5ad"
    print(f"\n[*] Saving h5ad → {out_h5ad}")
    adata.write_h5ad(str(out_h5ad))
    size_mb = out_h5ad.stat().st_size / 1e6
    print(f"    Saved ({size_mb:.1f} MB)")

    # ── Save probe IDs CSV ─────────────────────────────────────────────────────
    out_csv = out_dir / "probe_ids_type3_pretrain_subset.csv"
    pd.DataFrame({"illumina_probe_id": cpg_ids_sub}).to_csv(out_csv, index=False)
    print(f"    Probe IDs CSV → {out_csv}  ({len(cpg_ids_sub)} CpGs)")

    # ── Summary ────────────────────────────────────────────────────────────────
    if args.top_k_cpg > 0 and args.top_k_cpg < n_cpg_total:
        cpg_summary = f"{len(cpg_ids_sub)} (top-{args.top_k_cpg} by variance from {n_cpg_total})"
    else:
        cpg_summary = f"{len(cpg_ids_sub)} (all CpGs kept)"

    print("\n" + "=" * 60)
    print("SUBSET CREATION COMPLETE")
    print("=" * 60)
    print(f"  Samples  : {n_samples_actual} ({train_n} train / {valid_n} valid / {test_n} test)")
    print(f"  CpGs     : {cpg_summary}")
    print(f"  NaN pct  : {nan_pct:.1f}%")
    print(f"  h5ad     : {out_h5ad}")
    print(f"  probes   : {out_csv}")
    print(f"\nNext step — pretrain smoke test:")
    print(f"  sbatch scripts/llama/pretrain_llama_smoke.sh")
    print("=" * 60)


if __name__ == "__main__":
    main()
