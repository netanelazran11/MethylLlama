#!/usr/bin/env python3
"""
Merge multiple smoking h5ads (GSE50660, GSE42861, ...) into one aligned h5ad.

Each input h5ad has raw sequential cpg_ids that don't match the pretraining vocab.
This script:
  1. Loads the tokenizer to get the true CpG->token_id mapping
  2. Filters each dataset to the tokenizer vocab CpGs
  3. Concatenates all datasets
  4. Saves a single aligned h5ad ready for downstream fine-tuning

Usage:
  python data_prep/merge_smoking_datasets.py \
      --inputs /data/smoking_geo/smoking_data.h5ad /data/smoking_geo/gse42861_smoking.h5ad \
      --tokenizer_path /path/to/tokenizer \
      --output /data/smoking_geo/smoking_combined_aligned.h5ad
"""

import argparse
import logging
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def build_vocab(tokenizer_path: str) -> dict:
    from bmfm_targets.tokenization import MultiFieldTokenizer
    tok = MultiFieldTokenizer.from_pretrained(tokenizer_path)
    vocab = tok.tokenizers["cpg_sites"].get_vocab()
    vocab = {k: v for k, v in vocab.items() if k.startswith("cg")}
    logger.info(f"Tokenizer vocab: {len(vocab)} CpGs")
    return vocab


def align_one(adata: ad.AnnData, vocab: dict, gse: str) -> ad.AnnData:
    common = [c for c in adata.var_names if c in vocab]
    logger.info(f"{gse}: {len(adata)} samples, {len(common)}/{len(adata.var)} CpGs in vocab")
    aligned = adata[:, common].copy()
    aligned.var["cpg_id"] = [vocab[c] for c in common]
    aligned.obs["gse"] = gse
    return aligned


def restratify_splits(combined: ad.AnnData, seed: int = 42) -> ad.AnnData:
    """Re-assign train/val/test splits on the combined dataset using stratified sampling.

    Per-dataset splits merged naively can produce val and test sets with different
    class distributions (e.g. val=60% never, test=35% never). Re-stratifying on the
    full pool ensures all splits share the same class proportions.

    Split: 80% train / 10% val / 10% test, stratified by smoking_status.
    """
    from sklearn.model_selection import train_test_split as tts

    labels = combined.obs["smoking_status"].values
    idx = np.arange(len(combined))

    idx_tr, idx_tmp, lab_tr, lab_tmp = tts(
        idx, labels, test_size=0.20, random_state=seed, stratify=labels
    )
    idx_val, idx_te = tts(
        idx_tmp, test_size=0.50, random_state=seed, stratify=lab_tmp
    )

    split_col = np.array(["train"] * len(combined))
    split_col[idx_val] = "valid"
    split_col[idx_te] = "test"
    combined.obs["split"] = split_col

    logger.info("Re-stratified splits on combined dataset:")
    for s in ["train", "valid", "test"]:
        mask = combined.obs["split"] == s
        counts = combined.obs.loc[mask, "smoking_status"].value_counts()
        logger.info(f"  {s}: {mask.sum()} samples — {counts.to_dict()}")

    return combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="List of smoking h5ad files to merge")
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--restratify", action="store_true", default=True,
                        help="Re-stratify splits on the combined dataset (recommended)")
    args = parser.parse_args()

    vocab = build_vocab(args.tokenizer_path)

    aligned_list = []
    for path in args.inputs:
        path = Path(path)
        gse = path.stem.split("_")[0].upper()
        if "gse" not in gse.lower():
            gse = path.stem
        logger.info(f"Loading {path}...")
        adata = ad.read_h5ad(path)
        aligned = align_one(adata, vocab, gse)
        aligned_list.append(aligned)

    # Find common CpGs across all datasets
    cpg_sets = [set(a.var_names) for a in aligned_list]
    common_cpgs = sorted(cpg_sets[0].intersection(*cpg_sets[1:]))
    logger.info(f"Common CpGs across all datasets: {len(common_cpgs)}")

    # Filter all to common CpGs
    aligned_list = [a[:, common_cpgs].copy() for a in aligned_list]

    # Concatenate
    combined = ad.concat(aligned_list, join="inner", merge="first")
    combined.obs_names_make_unique()

    # Ensure cpg_id is set from vocab
    combined.var["cpg_id"] = [vocab[c] for c in combined.var_names]

    if args.restratify:
        combined = restratify_splits(combined)

    logger.info(f"Combined dataset: {combined.shape}")
    logger.info(f"Label distribution: {combined.obs['smoking_status'].value_counts().to_dict()}")
    logger.info(f"GSE distribution: {combined.obs['gse'].value_counts().to_dict()}")
    logger.info(f"Split distribution: {combined.obs['split'].value_counts().to_dict()}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.write_h5ad(out)
    logger.info(f"Saved: {out}")


if __name__ == "__main__":
    main()
