#!/usr/bin/env python3
"""
Download and process GSE50660 smoking methylation dataset into h5ad format.

Input files (download to output_dir first):
  GSE50660_matrix_processed.txt.gz  — beta values
  GSE50660_series_matrix.txt.gz     — sample metadata with smoking labels

Output:
  smoking_data.h5ad — AnnData with:
    X:                  [464, n_cpg] float32 beta values
    obs[smoking_status]: current / former / never
    obs[split]:          train / valid / test (80/10/10)
    var[cpg_id]:         integer token IDs (offset +5)

Usage:
  python data_prep/prepare_geo_smoking.py \
      --output_dir /sci/labs/benjamin.yakir/netanel.azran/data/smoking_geo
"""

import argparse
import gzip
import logging
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def parse_series_matrix(matrix_gz: Path):
    """Extract sample IDs and smoking status labels from series matrix file."""
    logger.info("Parsing series matrix for smoking labels...")
    sample_ids = []
    smoking = {}

    with gzip.open(matrix_gz, "rt", errors="replace") as f:
        for line in f:
            if line.startswith("!Sample_geo_accession"):
                parts = line.strip().split("\t")[1:]
                sample_ids = [s.strip('"') for s in parts]

            if "smoking" in line.lower() and line.startswith("!Sample_characteristics"):
                parts = line.strip().split("\t")[1:]
                for i, p in enumerate(parts):
                    p = p.strip('"').strip()
                    # Format: "smoking (...): VALUE" where VALUE is 0/1/2 or never/former/current
                    val = p.split(":")[-1].strip().lower()
                    if val == "2" or "current" in val:
                        smoking[i] = "current"
                    elif val == "1" or "former" in val or "ex-" in val:
                        smoking[i] = "former"
                    elif val == "0" or "never" in val or "non" in val:
                        smoking[i] = "never"

    labels = [smoking.get(i, "unknown") for i in range(len(sample_ids))]
    logger.info(f"Found {len(sample_ids)} samples")
    logger.info(f"Smoking labels: {pd.Series(labels).value_counts().to_dict()}")
    return sample_ids, labels


def parse_beta_matrix(processed_gz: Path, sample_ids: list) -> pd.DataFrame:
    """
    Parse the processed beta value matrix.
    File format: rows=CpGs, columns=samples (GSM IDs).
    Returns DataFrame [samples x CpGs].
    """
    logger.info("Parsing beta matrix (this takes a few minutes)...")
    df = pd.read_csv(processed_gz, sep="\t", index_col=0, compression="gzip")

    # Transpose: rows=samples, columns=CpGs
    df = df.T
    logger.info(f"Beta matrix shape after transpose: {df.shape}")

    # Align sample IDs
    if len(df) == len(sample_ids):
        df.index = sample_ids
    else:
        logger.warning(f"Sample count mismatch: matrix={len(df)}, ids={len(sample_ids)}")
        df.index = sample_ids[:len(df)]

    return df


def build_h5ad(beta_df: pd.DataFrame, labels: list, output_path: Path):
    """Build and save h5ad file with train/valid/test split."""
    # Filter to valid labels only
    valid_mask = [l in {"current", "former", "never"} for l in labels]
    beta_df = beta_df[valid_mask].copy()
    labels = [l for l, v in zip(labels, valid_mask) if v]

    logger.info(f"Valid samples: {len(beta_df)}")
    logger.info(f"Label distribution: {pd.Series(labels).value_counts().to_dict()}")

    # Stratified train/valid/test split (80/10/10)
    idx = np.arange(len(beta_df))
    idx_tr, idx_tmp, _, lab_tmp = train_test_split(
        idx, labels, test_size=0.2, random_state=42, stratify=labels
    )
    idx_val, idx_te = train_test_split(
        idx_tmp, test_size=0.5, random_state=42, stratify=lab_tmp
    )

    split_col = ["train"] * len(beta_df)
    for i in idx_val:
        split_col[i] = "valid"
    for i in idx_te:
        split_col[i] = "test"

    obs = pd.DataFrame({
        "smoking_status": labels,
        "split": split_col,
        "gse": "GSE50660",
    }, index=beta_df.index)

    logger.info(f"Split sizes: {pd.Series(split_col).value_counts().to_dict()}")

    # CpG vocabulary: token IDs offset by 5 to match tokenizer convention
    cpg_ids = list(beta_df.columns)
    var = pd.DataFrame({
        "cpg_id": np.arange(5, len(cpg_ids) + 5, dtype=np.int64),
    }, index=cpg_ids)

    X = beta_df.values.astype(np.float32)
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.write_h5ad(output_path)
    logger.info(f"Saved h5ad: {output_path} — shape {adata.shape}")
    return adata


def main():
    parser = argparse.ArgumentParser(description="Process GSE50660 into smoking h5ad")
    parser.add_argument("--output_dir", required=True,
                        help="Directory containing GSE50660 downloaded files and output")
    args = parser.parse_args()

    d = Path(args.output_dir)

    matrix_gz = d / "GSE50660_series_matrix.txt.gz"
    processed_gz = d / "GSE50660_matrix_processed.txt.gz"

    for f in [matrix_gz, processed_gz]:
        if not f.exists():
            raise FileNotFoundError(
                f"Missing: {f}\n"
                "Download first:\n"
                "  wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE50nnn/GSE50660/suppl/GSE50660_matrix_processed.txt.gz\n"
                "  wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE50nnn/GSE50660/matrix/GSE50660_series_matrix.txt.gz"
            )

    sample_ids, labels = parse_series_matrix(matrix_gz)
    beta_df = parse_beta_matrix(processed_gz, sample_ids)
    build_h5ad(beta_df, labels, d / "smoking_data.h5ad")
    logger.info("Done.")


if __name__ == "__main__":
    main()
