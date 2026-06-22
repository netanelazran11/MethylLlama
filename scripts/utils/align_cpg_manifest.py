#!/usr/bin/env python3
"""
align_cpg_manifest.py
=====================
Validate CpG site overlap between:
  • Tokenizer vocabulary  (49,156 pretrain CpGs)
  • h5ad var_names        (19,608 finetune CpGs after NaN removal)
  • Illumina manifest(s)  (HM450K and/or EPIC)

Produces:
  cpg_annotations.tsv    — aligned annotation table for CpG-level UMAP
  alignment_report.txt   — overlap statistics + any CpG IDs that are missing

Usage:
  python scripts/utils/align_cpg_manifest.py \\
      --tokenizer  tokenizer_llama_pretrain49k \\
      --data       /path/to/finetuning_19608_clean_stratified_no_outliers.h5ad \\
      --manifests  /path/to/HumanMethylation450_15017482_v1-2.csv \\
                   /path/to/MethylationEPIC_v-1-0_B5.csv \\
      --outdir     outputs/cpg_manifest

  # Minimal: just check tokenizer vs manifest (no h5ad needed)
  python scripts/utils/align_cpg_manifest.py \\
      --tokenizer tokenizer_llama_pretrain49k \\
      --manifests /path/to/manifest.csv \\
      --outdir    outputs/cpg_manifest
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Manifest loading
# ─────────────────────────────────────────────────────────────────────────────

# Annotation columns we want from the manifest (name varies by version)
_ISLAND_ALIASES = [
    "Relation_to_UCSC_CpG_Island",
    "Relation_to_Island",
    "CpG_Island_Relation",
    "UCSC_CpG_Islands_Name",
]
_ENHANCER_ALIASES = [
    "Phantom5_Enhancers",
    "Phantom4_Enhancers",
    "Enhancer",
    "Regulatory_Feature_Group",
    "DNase_Hypersensitivity_NAME",
]
_CHR_ALIASES = ["CHR", "chr", "Chromosome", "CpG_chrm"]
_POS_ALIASES = ["MAPINFO", "Start", "Position", "MAPINFO_hg38", "CpG_beg"]
_GENE_ALIASES = ["UCSC_RefGene_Name", "Gene_Name", "RefGene_Name", "gene"]


def _find_col(df: pd.DataFrame, aliases: list[str]) -> str | None:
    for a in aliases:
        if a in df.columns:
            return a
    return None


def _skip_illumina_header(path: str) -> int:
    """Count comment/descriptor lines before the actual CSV header row."""
    skip = 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("[") or line.startswith("Heading"):
                skip += 1
            else:
                break
    return skip


def load_manifest(path: str) -> pd.DataFrame:
    """Load a single Illumina manifest (CSV or TSV) → DataFrame indexed by CpG Name."""
    path = str(path)
    log.info(f"Loading manifest: {path}")

    # Auto-detect separator from extension
    sep = "\t" if path.endswith(".tsv") or path.endswith(".txt") else ","

    # For CSV files only: skip Illumina metadata header block
    skip = 0
    if sep == ",":
        with open(path, encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh):
                stripped = line.strip()
                if stripped.startswith("IlmnID") or stripped.startswith("Name,"):
                    skip = i
                    break
                if stripped.startswith("["):
                    skip = i + 1

    log.info(f"  Separator: {'TAB' if sep == chr(9) else 'COMMA'}  skip={skip} lines")

    # Peek at the header to determine which columns to load (avoids OOM from probe sequences)
    header_df = pd.read_csv(path, sep=sep, skiprows=skip, nrows=0, encoding="utf-8")
    all_cols = list(header_df.columns)

    id_col = next((c for c in ["Name", "IlmnID", "probeID"] if c in all_cols), None)
    if id_col is None:
        raise ValueError(
            f"Cannot find CpG ID column ('Name', 'IlmnID', or 'probeID') in {path}.\n"
            f"Columns found: {all_cols[:20]}"
        )

    # Only load columns we will actually use
    wanted = (
        [id_col]
        + [c for aliases in [_ISLAND_ALIASES, _ENHANCER_ALIASES, _CHR_ALIASES, _POS_ALIASES, _GENE_ALIASES]
           for c in aliases if c in all_cols]
    )
    wanted = list(dict.fromkeys(wanted))  # deduplicate, preserve order
    log.info(f"  Loading {len(wanted)} of {len(all_cols)} columns: {wanted}")

    df = pd.read_csv(
        path,
        sep=sep,
        skiprows=skip,
        usecols=wanted,
        low_memory=True,
        encoding="utf-8",
        encoding_errors="replace",
    )

    # Keep only rows whose ID looks like a CpG (starts with cg/ch/rs)
    mask = df[id_col].astype(str).str.match(r"^(cg|ch|rs)\d+", na=False)
    n_dropped = (~mask).sum()
    if n_dropped:
        log.info(f"  Dropped {n_dropped} non-CpG rows (controls/footer)")
    df = df[mask].copy()

    df = df.set_index(id_col)
    df.index.name = "cpg_id"
    log.info(f"  → {len(df):,} CpG sites")
    return df


def merge_manifests(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Union-merge multiple manifests (HM450K + EPIC).
    If a CpG appears in both, prefer EPIC (larger / more recent annotation).
    """
    if len(dfs) == 1:
        return dfs[0]

    # Put EPIC last so it wins on duplicate index
    merged = pd.concat(dfs, axis=0)
    dups = merged.index.duplicated(keep="last")
    log.info(
        f"Merged {len(dfs)} manifests: {len(merged):,} total rows, "
        f"{dups.sum():,} duplicates (keeping last = EPIC annotation)"
    )
    merged = merged[~dups]
    log.info(f"After dedup: {len(merged):,} unique CpG sites")
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer CpG extraction
# ─────────────────────────────────────────────────────────────────────────────

def load_tokenizer_cpgs(tokenizer_path: str) -> list[str]:
    """Extract CpG site IDs from the tokenizer vocabulary.
    Filters to IDs that look like Illumina CpG names (cg/ch/rs + digits).
    """
    import os

    log.info(f"Loading tokenizer vocab: {tokenizer_path}")
    vocab_file = os.path.join(tokenizer_path, "tokenizers", "cpg_sites", "vocab.txt")

    with open(vocab_file) as f:
        tokens = [line.strip() for line in f]

    cpg_ids = [t for t in tokens if t.startswith(("cg", "ch", "rs"))]
    log.info(f"Tokenizer vocab size: {len(tokens):,}  →  CpG tokens: {len(cpg_ids):,}")
    return cpg_ids


# ─────────────────────────────────────────────────────────────────────────────
# h5ad CpG extraction
# ─────────────────────────────────────────────────────────────────────────────

def load_h5ad_cpgs(h5ad_path: str) -> list[str]:
    """Return var_names from an AnnData h5ad (the 19k finetune CpG sites)."""
    import anndata

    log.info(f"Loading h5ad var_names: {h5ad_path}")
    adata = anndata.read_h5ad(h5ad_path, backed="r")
    cpg_ids = list(adata.var_names)
    log.info(f"h5ad CpG sites: {len(cpg_ids):,}")
    return cpg_ids


# ─────────────────────────────────────────────────────────────────────────────
# Overlap analysis
# ─────────────────────────────────────────────────────────────────────────────

def overlap_stats(name_a: str, set_a: set, name_b: str, set_b: set) -> dict:
    inter = set_a & set_b
    only_a = set_a - set_b
    only_b = set_b - set_a

    pct_a = 100 * len(inter) / len(set_a) if set_a else 0.0
    pct_b = 100 * len(inter) / len(set_b) if set_b else 0.0

    return {
        "name_a": name_a,
        "name_b": name_b,
        "n_a": len(set_a),
        "n_b": len(set_b),
        "intersection": len(inter),
        "pct_a_covered": pct_a,
        "pct_b_covered": pct_b,
        "only_in_a": len(only_a),
        "only_in_b": len(only_b),
        "only_in_a_examples": sorted(only_a)[:10],
        "only_in_b_examples": sorted(only_b)[:10],
    }


def print_overlap(stats: dict, fh=None) -> None:
    lines = [
        f"\n{'─'*60}",
        f"  {stats['name_a']}  vs.  {stats['name_b']}",
        f"{'─'*60}",
        f"  {stats['name_a']:30s} : {stats['n_a']:>8,} sites",
        f"  {stats['name_b']:30s} : {stats['n_b']:>8,} sites",
        f"  Intersection                   : {stats['intersection']:>8,} sites",
        f"  % of {stats['name_a']} covered : {stats['pct_a_covered']:>7.2f}%",
        f"  % of {stats['name_b']} covered : {stats['pct_b_covered']:>7.2f}%",
        f"  Only in {stats['name_a']}        : {stats['only_in_a']:>8,}",
        f"  Only in {stats['name_b']}        : {stats['only_in_b']:>8,}",
    ]
    if stats["only_in_a_examples"]:
        lines.append(f"  Examples only in {stats['name_a']}: {stats['only_in_a_examples'][:5]}")
    if stats["only_in_b_examples"]:
        lines.append(f"  Examples only in {stats['name_b']}: {stats['only_in_b_examples'][:5]}")

    text = "\n".join(lines)
    print(text)
    if fh:
        fh.write(text + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Annotation table builder
# ─────────────────────────────────────────────────────────────────────────────

def build_annotation_table(
    manifest: pd.DataFrame,
    cpg_ids: list[str],
    source_tag: str,
) -> pd.DataFrame:
    """Build a tidy annotation DataFrame for the given CpG list.

    Columns produced (all renamed to canonical names):
      cpg_id, source, chr, position, island_relation, enhancer, gene,
      is_sex_chr  (bool)

    CpGs not in manifest get NaN annotations.
    """
    # Canonical column mapping
    island_col   = _find_col(manifest, _ISLAND_ALIASES)
    enhancer_col = _find_col(manifest, _ENHANCER_ALIASES)
    chr_col      = _find_col(manifest, _CHR_ALIASES)
    pos_col      = _find_col(manifest, _POS_ALIASES)
    gene_col     = _find_col(manifest, _GENE_ALIASES)

    keep_cols = [c for c in [island_col, enhancer_col, chr_col, pos_col, gene_col] if c]

    df = manifest.loc[manifest.index.isin(cpg_ids), keep_cols].copy()

    # Add rows for CpGs not in manifest (NaN annotations)
    missing = set(cpg_ids) - set(manifest.index)
    if missing:
        missing_df = pd.DataFrame(index=sorted(missing), columns=keep_cols)
        df = pd.concat([df, missing_df])

    # Reindex to match cpg_ids order
    df = df.reindex(cpg_ids)
    df.index.name = "cpg_id"

    # Rename to canonical names
    rename = {}
    if island_col:   rename[island_col]   = "island_relation"
    if enhancer_col: rename[enhancer_col] = "enhancer_raw"
    if chr_col:      rename[chr_col]      = "chr"
    if pos_col:      rename[pos_col]      = "position"
    if gene_col:     rename[gene_col]     = "gene"
    df = df.rename(columns=rename)

    # ── island_relation: simplify to 6 categories ──────────────────────────
    if "island_relation" in df.columns:
        island_map = {
            "Island":   "Island",
            "N_Shore":  "Shore",
            "S_Shore":  "Shore",
            "N_Shelf":  "Shelf",
            "S_Shelf":  "Shelf",
            "OpenSea":  "OpenSea",
        }
        df["island_relation"] = (
            df["island_relation"]
            .astype(str)
            .map(lambda x: island_map.get(x.strip(), "OpenSea" if x in ("nan", "", "NA") else x))
        )

    # ── enhancer: collapse to bool ──────────────────────────────────────────
    if "enhancer_raw" in df.columns:
        df["is_enhancer"] = (
            df["enhancer_raw"]
            .astype(str)
            .str.strip()
            .apply(lambda x: x not in ("", "nan", "NA", "0", "FALSE", "False"))
        )
        df = df.drop(columns=["enhancer_raw"])

    # ── chromosome grouping ─────────────────────────────────────────────────
    if "chr" in df.columns:
        df["chr"] = df["chr"].astype(str).str.strip().str.replace("chr", "", regex=False)
        df["chr_group"] = df["chr"].apply(
            lambda x: "Sex (X/Y)" if x in ("X", "Y") else ("Autosome" if x.isdigit() else "Other")
        )

    df["source"] = source_tag
    df = df.reset_index()

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Align CpG sites between tokenizer, h5ad, and Illumina manifest(s)")
    p.add_argument("--tokenizer",  required=True,  help="Path to tokenizer_llama_pretrain49k directory")
    p.add_argument("--data",       default=None,   help="Path to h5ad (finetune dataset). Optional but recommended.")
    p.add_argument("--manifests",  nargs="+", required=True, help="Illumina manifest CSV file(s). HM450K first, then EPIC.")
    p.add_argument("--outdir",     required=True,  help="Output directory")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    report_path = outdir / "alignment_report.txt"
    report_fh = open(report_path, "w")

    header = (
        "CpG Site Alignment Report\n"
        "=========================\n"
        f"Tokenizer : {args.tokenizer}\n"
        f"Data (h5ad): {args.data or '<not provided>'}\n"
        f"Manifests : {args.manifests}\n"
    )
    print(header)
    report_fh.write(header + "\n")

    # ── 1. Load sources ───────────────────────────────────────────────────────
    tokenizer_cpgs = load_tokenizer_cpgs(args.tokenizer)
    tok_set = set(tokenizer_cpgs)

    h5ad_cpgs = None
    h5ad_set  = set()
    if args.data:
        h5ad_cpgs = load_h5ad_cpgs(args.data)
        h5ad_set  = set(h5ad_cpgs)

    manifest_dfs = [load_manifest(m) for m in args.manifests]
    manifest = merge_manifests(manifest_dfs)
    manifest_set = set(manifest.index)

    # ── 2. Overlap statistics ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print(" OVERLAP STATISTICS")
    print("="*60)
    report_fh.write("\nOVERLAP STATISTICS\n" + "="*60 + "\n")

    pairs = [
        ("Tokenizer (49k)", tok_set, "Manifest", manifest_set),
    ]
    if h5ad_set:
        pairs += [
            ("h5ad (19k finetune)", h5ad_set, "Manifest", manifest_set),
            ("h5ad (19k finetune)", h5ad_set, "Tokenizer (49k)", tok_set),
        ]

    for na, sa, nb, sb in pairs:
        s = overlap_stats(na, sa, nb, sb)
        print_overlap(s, report_fh)

    # ── 3. Build annotation tables ────────────────────────────────────────────
    print("\n" + "="*60)
    print(" BUILDING ANNOTATION TABLES")
    print("="*60)
    report_fh.write("\nBUILDING ANNOTATION TABLES\n" + "="*60 + "\n")

    # Table 1: all 49k tokenizer CpGs
    log.info("Building annotation table for 49k tokenizer CpGs ...")
    tok_annot = build_annotation_table(manifest, tokenizer_cpgs, source_tag="tokenizer_49k")
    tok_annot_path = outdir / "cpg_annotations_tokenizer49k.tsv"
    tok_annot.to_csv(tok_annot_path, sep="\t", index=False)
    log.info(f"Saved → {tok_annot_path}  ({len(tok_annot):,} rows)")

    # Coverage summary for tokenizer 49k
    if "island_relation" in tok_annot.columns:
        island_counts = tok_annot["island_relation"].value_counts()
        pct_annotated = 100 * tok_annot["island_relation"].notna().sum() / len(tok_annot)
        msg = (
            f"\nTokenizer 49k  island_relation coverage: {pct_annotated:.1f}%\n"
            + island_counts.to_string()
        )
        print(msg)
        report_fh.write(msg + "\n")

    if "is_enhancer" in tok_annot.columns:
        n_enh = tok_annot["is_enhancer"].sum()
        msg = f"\nTokenizer 49k  enhancer CpGs: {n_enh:,} / {len(tok_annot):,} ({100*n_enh/len(tok_annot):.1f}%)"
        print(msg)
        report_fh.write(msg + "\n")

    if "chr_group" in tok_annot.columns:
        chr_counts = tok_annot["chr_group"].value_counts()
        msg = "\nTokenizer 49k  chromosome groups:\n" + chr_counts.to_string()
        print(msg)
        report_fh.write(msg + "\n")

    # Table 2: 19k finetune CpGs (if h5ad provided)
    if h5ad_cpgs is not None:
        log.info("Building annotation table for 19k finetune CpGs ...")
        h5ad_annot = build_annotation_table(manifest, h5ad_cpgs, source_tag="finetune_19k")
        h5ad_annot_path = outdir / "cpg_annotations_finetune19k.tsv"
        h5ad_annot.to_csv(h5ad_annot_path, sep="\t", index=False)
        log.info(f"Saved → {h5ad_annot_path}  ({len(h5ad_annot):,} rows)")

        if "island_relation" in h5ad_annot.columns:
            island_counts = h5ad_annot["island_relation"].value_counts()
            pct_annotated = 100 * h5ad_annot["island_relation"].notna().sum() / len(h5ad_annot)
            msg = (
                f"\nFinetune 19k   island_relation coverage: {pct_annotated:.1f}%\n"
                + island_counts.to_string()
            )
            print(msg)
            report_fh.write(msg + "\n")

    # ── 4. Missing CpG report ─────────────────────────────────────────────────
    tok_missing = tok_set - manifest_set
    h5ad_missing = h5ad_set - manifest_set if h5ad_set else set()

    if tok_missing:
        missing_path = outdir / "tokenizer_cpgs_not_in_manifest.txt"
        with open(missing_path, "w") as f:
            f.write("\n".join(sorted(tok_missing)))
        msg = f"\nTokenizer CpGs not in manifest: {len(tok_missing):,} → saved to {missing_path}"
        print(msg)
        report_fh.write(msg + "\n")

    if h5ad_missing:
        missing_path = outdir / "finetune_cpgs_not_in_manifest.txt"
        with open(missing_path, "w") as f:
            f.write("\n".join(sorted(h5ad_missing)))
        msg = f"\nFinetune CpGs not in manifest: {len(h5ad_missing):,} → saved to {missing_path}"
        print(msg)
        report_fh.write(msg + "\n")

    # ── 5. Summary ────────────────────────────────────────────────────────────
    summary = f"""
{'='*60}
 SUMMARY
{'='*60}
 Tokenizer 49k  in manifest : {len(tok_set & manifest_set):>8,} / {len(tok_set):,} ({100*len(tok_set & manifest_set)/len(tok_set):.1f}%)
"""
    if h5ad_set:
        summary += f" Finetune 19k   in manifest : {len(h5ad_set & manifest_set):>8,} / {len(h5ad_set):,} ({100*len(h5ad_set & manifest_set)/len(h5ad_set):.1f}%)\n"
    summary += f"""
 Annotation files saved to:
   {outdir}/cpg_annotations_tokenizer49k.tsv
"""
    if h5ad_cpgs:
        summary += f"   {outdir}/cpg_annotations_finetune19k.tsv\n"
    summary += f"""
 Use cpg_annotations_tokenizer49k.tsv as --cpg_manifest in
 extract_sample_embeddings.py for the CpG-embedding UMAP (Fig 2 style).
{'='*60}
"""
    print(summary)
    report_fh.write(summary)
    report_fh.close()

    log.info(f"Full report saved → {report_path}")


if __name__ == "__main__":
    main()
