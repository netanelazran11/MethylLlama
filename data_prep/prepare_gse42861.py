#!/usr/bin/env python3
"""
Download and process GSE42861 smoking methylation dataset into h5ad format.
(Liu et al. 2013 RA cohort — 689 blood samples, HM450, current/former/never)

Input files (download to output_dir first):
  GSE42861_processed_methylation_matrix.txt.gz  — beta values (~1.5 GB)
  GSE42861_series_matrix.txt.gz                 — sample metadata

Output:
  gse42861_smoking.h5ad — AnnData with smoking_status, split, gse obs columns

Usage:
  python data_prep/prepare_gse42861.py \
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

DOWNLOAD_CMDS = """
wget -P {dir} "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE42nnn/GSE42861/matrix/GSE42861_series_matrix.txt.gz"
wget -P {dir} "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE42nnn/GSE42861/suppl/GSE42861_processed_methylation_matrix.txt.gz"
"""


def parse_series_matrix(matrix_gz: Path):
    """Extract sample IDs and smoking labels from GSE42861 series matrix."""
    logger.info("Parsing series matrix for smoking labels...")
    sample_ids = []
    smoking = {}

    with gzip.open(matrix_gz, "rt", errors="replace") as f:
        for line in f:
            if line.startswith("!Sample_geo_accession"):
                parts = line.strip().split("\t")[1:]
                sample_ids = [s.strip('"') for s in parts]

            if "smok" in line.lower() and line.startswith("!Sample_characteristics"):
                parts = line.strip().split("\t")[1:]
                for i, p in enumerate(parts):
                    p = p.strip('"').lower()
                    val = p.split(":")[-1].strip()
                    if "current" in val or val == "2":
                        smoking[i] = "current"
                    elif "former" in val or "ex" in val or "past" in val or val == "1":
                        smoking[i] = "former"
                    elif "never" in val or "non" in val or val == "0":
                        smoking[i] = "never"

    labels = [smoking.get(i, "unknown") for i in range(len(sample_ids))]
    logger.info(f"Found {len(sample_ids)} samples")
    logger.info(f"Smoking labels: {pd.Series(labels).value_counts().to_dict()}")
    return sample_ids, labels


def parse_beta_matrix(processed_gz: Path, sample_ids: list) -> pd.DataFrame:
    logger.info("Parsing beta matrix (this may take several minutes for ~1.5GB file)...")
    df = pd.read_csv(processed_gz, sep="\t", index_col=0, compression="gzip")
    df = df.T
    logger.info(f"Beta matrix shape after transpose: {df.shape}")

    if len(df) == len(sample_ids):
        df.index = sample_ids
    else:
        logger.warning(f"Sample count mismatch: matrix={len(df)}, ids={len(sample_ids)}")
        common = min(len(df), len(sample_ids))
        df = df.iloc[:common]
        df.index = sample_ids[:common]
    return df


def build_h5ad(beta_df: pd.DataFrame, labels: list, output_path: Path):
    valid_mask = [l in {"current", "former", "never"} for l in labels]
    beta_df = beta_df[valid_mask].copy()
    labels = [l for l, v in zip(labels, valid_mask) if v]

    logger.info(f"Valid samples: {len(beta_df)}")
    logger.info(f"Label distribution: {pd.Series(labels).value_counts().to_dict()}")

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
        "gse": "GSE42861",
    }, index=beta_df.index)

    cpg_ids = list(beta_df.columns)
    var = pd.DataFrame({
        "cpg_id": np.arange(5, len(cpg_ids) + 5, dtype=np.int64),
    }, index=cpg_ids)

    X = beta_df.values.astype(np.float32)
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.write_h5ad(output_path)
    logger.info(f"Saved: {output_path} — shape {adata.shape}")
    return adata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    d = Path(args.output_dir)
    matrix_gz = d / "GSE42861_series_matrix.txt.gz"
    processed_gz = d / "GSE42861_processed_methylation_matrix.txt.gz"

    for f in [matrix_gz, processed_gz]:
        if not f.exists():
            raise FileNotFoundError(
                f"Missing: {f}\nDownload with:\n{DOWNLOAD_CMDS.format(dir=d)}"
            )

    sample_ids, labels = parse_series_matrix(matrix_gz)
    beta_df = parse_beta_matrix(processed_gz, sample_ids)
    build_h5ad(beta_df, labels, d / "gse42861_smoking.h5ad")
    logger.info("Done.")


if __name__ == "__main__":
    main()
