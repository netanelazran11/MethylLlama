#!/usr/bin/env python3
"""
sort_probes_genomic.py
======================
Given the HM450 hg38 manifest and probe_ids_type3.csv, compute the genomic
rank of every data column and save it as cpg_genomic_rank.npy.

The rank array is consumed by WCEDCollator (genomic_rank_path=...) so that
selected CpGs are placed in the input sequence in genomic order — making RoPE
encode true chromosomal proximity instead of arbitrary probe-ID order.

Manifest format expected (tab-separated):
  CpG_chrm  CpG_beg  CpG_end  probe_strand  probeID  ...
  chr1      15864    15866    -             cg13869341 ...

Outputs (all in --outdir):
  cpg_genomic_rank.npy         shape [n_cpgs], dtype int32
                               genomic_rank[col_i] = genomic rank of data column i
  cpg_genomic_sorted_order.csv probe_id, CpG_chrm, CpG_beg, original_col_idx
                               full sorted table for inspection

Usage:
  python scripts/data/sort_probes_genomic.py \\
      --manifest  ${METHYLLAMA_DATA}/manifests/HM450.hg38.manifest.tsv \\
      --probe_ids ${METHYLLAMA_DATA}/data_methyl_pretrain_type3_h5ad/probe_ids_type3_pretrain.csv \\
      --outdir    resources
"""

import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def chrom_sort_key(chrom: str) -> int:
    """chr1→1, chr2→2, ..., chr22→22, chrX→23, chrY→24, chrM→25, other→99."""
    c = str(chrom).strip().upper().replace("CHR", "")
    if c == "X":  return 23
    if c == "Y":  return 24
    if c == "M":  return 25
    try:
        return int(c)
    except ValueError:
        return 99


def load_probe_ids(csv_path: Path) -> list:
    """Read the probe_ids CSV → list of cg... probe ID strings."""
    df = pd.read_csv(csv_path)
    # Try known column names first
    for col in ("illumina_probe_id", "probe_id", "cpg_id", "IlmnID", "Name"):
        if col in df.columns:
            ids = df[col].dropna().astype(str).tolist()
            if ids and ids[0].startswith("cg"):
                log.info(f"Probe IDs loaded from column '{col}': {len(ids):,}  "
                         f"(first: {ids[0]}, last: {ids[-1]})")
                return ids
    # Fallback: find any column where non-NaN values all start with "cg"
    for col in df.columns:
        vals = df[col].dropna().astype(str)
        if len(vals) > 0 and vals.str.startswith("cg").all():
            ids = vals.tolist()
            log.info(f"Probe IDs loaded from column '{col}': {len(ids):,}  "
                     f"(first: {ids[0]}, last: {ids[-1]})")
            return ids
    raise ValueError(f"No cg-prefixed probe ID column found in {csv_path}.\n"
                     f"Columns: {df.columns.tolist()}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute genomic rank array for WCEDCollator"
    )
    parser.add_argument("--manifest",  required=True,
                        help="HM450.hg38.manifest.tsv path")
    parser.add_argument("--probe_ids", required=True,
                        help="probe_ids_type3_pretrain.csv (49,156 CpG IDs)")
    parser.add_argument("--outdir",    required=True,
                        help="Output directory")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load probe IDs (original data column order) ────────────────────────
    probe_ids = load_probe_ids(Path(args.probe_ids))
    n_cpgs = len(probe_ids)

    # ── 2. Load manifest (only the 3 columns we need) ─────────────────────────
    log.info(f"Loading manifest: {args.manifest}")
    manifest = pd.read_csv(
        args.manifest,
        sep="\t",
        usecols=["probeID", "CpG_chrm", "CpG_beg"],
        dtype={"probeID": str, "CpG_chrm": str, "CpG_beg": "Int64"},
        low_memory=True,
    )
    manifest = manifest[manifest["probeID"].str.startswith("cg", na=False)].copy()
    manifest = manifest.set_index("probeID")
    log.info(f"Manifest CpG probes: {len(manifest):,}")

    # ── 3. Join: attach chr + position to each of our 49k probes ─────────────
    our_df = pd.DataFrame({
        "probe_id": probe_ids,
        "col_idx":  np.arange(n_cpgs, dtype=np.int32),
    })
    our_df = our_df.join(manifest, on="probe_id")

    n_missing = our_df["CpG_chrm"].isna().sum()
    n_found   = n_cpgs - n_missing
    log.info(f"Found in manifest: {n_found:,} / {n_cpgs:,}  "
             f"({100 * n_found / n_cpgs:.2f}%)")
    if n_missing:
        missing_ids = our_df.loc[our_df["CpG_chrm"].isna(), "probe_id"].tolist()
        log.warning(f"{n_missing} probes NOT in manifest → placed at end. "
                    f"First few: {missing_ids[:5]}")

    # ── 4. Sort: chromosome numerically, then by bp position ──────────────────
    our_df["chrom_key"] = our_df["CpG_chrm"].apply(chrom_sort_key)
    our_df["CpG_beg"]   = pd.to_numeric(our_df["CpG_beg"], errors="coerce").fillna(0).astype(int)

    our_df_sorted = our_df.sort_values(
        ["chrom_key", "CpG_beg"],
        na_position="last",        # missing CpGs go to the end
        ignore_index=True,
    )

    # ── 5. Build genomic_rank array ───────────────────────────────────────────
    # genomic_rank[col_idx] = genomic rank (0-based) of data column col_idx
    genomic_rank = np.empty(n_cpgs, dtype=np.int32)
    genomic_rank[our_df_sorted["col_idx"].values] = np.arange(n_cpgs, dtype=np.int32)

    # ── 6. Save outputs ───────────────────────────────────────────────────────
    out_npy = outdir / "cpg_genomic_rank.npy"
    np.save(out_npy, genomic_rank)
    log.info(f"Saved: {out_npy}  shape={genomic_rank.shape}  dtype={genomic_rank.dtype}")

    out_csv = outdir / "cpg_genomic_sorted_order.csv"
    our_df_sorted[["probe_id", "CpG_chrm", "CpG_beg", "col_idx"]].to_csv(
        out_csv, index=False
    )
    log.info(f"Saved: {out_csv}")

    # ── 7. Verification ───────────────────────────────────────────────────────
    # Reconstruct sorted order from genomic_rank and verify it matches our_df_sorted
    reconstructed = np.argsort(genomic_rank)   # col indices in genomic order
    assert list(reconstructed) == list(our_df_sorted["col_idx"].values), \
        "BUG: reconstructed genomic order does not match sorted table"
    log.info("Verification passed: genomic_rank is self-consistent")

    # ── 8. Summary ────────────────────────────────────────────────────────────
    first = our_df_sorted.iloc[0]
    last_found = our_df_sorted[our_df_sorted["CpG_chrm"].notna()].iloc[-1]

    chrom_counts = (
        our_df_sorted[our_df_sorted["CpG_chrm"].notna()]
        .groupby("CpG_chrm").size()
        .reset_index(name="n")
        .assign(sort_key=lambda d: d["CpG_chrm"].apply(chrom_sort_key))
        .sort_values("sort_key")
    )

    print()
    print("=" * 62)
    print("GENOMIC RANK SUMMARY")
    print("=" * 62)
    print(f"  Total CpGs            : {n_cpgs:,}")
    print(f"  Found in manifest     : {n_found:,}  ({100*n_found/n_cpgs:.2f}%)")
    print(f"  Missing (end of order): {n_missing:,}")
    print(f"  Chromosomes covered   : {our_df_sorted['CpG_chrm'].nunique()}")
    print()
    print(f"  Genomically FIRST CpG : {first['probe_id']}"
          f"  ({first['CpG_chrm']}:{first['CpG_beg']:,})")
    print(f"  Genomically LAST CpG  : {last_found['probe_id']}"
          f"  ({last_found['CpG_chrm']}:{last_found['CpG_beg']:,})")
    print()
    print("  CpGs per chromosome:")
    for _, row in chrom_counts.iterrows():
        print(f"    {row['CpG_chrm']:6s} : {row['n']:>6,}")
    print()
    print(f"  Example: data col 0 ({probe_ids[0]}) → genomic rank {genomic_rank[0]}")
    print(f"  Example: data col 1 ({probe_ids[1]}) → genomic rank {genomic_rank[1]}")
    print()
    print(f"  Outputs:")
    print(f"    {out_npy}")
    print(f"    {out_csv}")
    print("=" * 62)


if __name__ == "__main__":
    main()
