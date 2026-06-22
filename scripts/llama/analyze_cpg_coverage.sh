#!/bin/bash -l
#SBATCH --job-name=analyze-cpg-coverage
#SBATCH --partition=glacier,glacier-k,catfish,catfish-k,salmon,salmon-k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00

#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.err

set -euo pipefail

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
DATA_DIR="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_pretrain_type3_h5ad"
PRETRAIN_H5AD="${DATA_DIR}/methylgpt_pretrain_type3.h5ad"
FINETUNE_H5AD="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_finetune_49k_h5ad/finetuning_49k.h5ad"
OUT_DIR="${REPO}/outputs/cpg_coverage_analysis"

mkdir -p "${OUT_DIR}"

cd "${REPO}"
source bmfm_methyl_env/bin/activate

echo "============================================================"
echo "CpG Coverage Analysis"
echo "============================================================"
echo "Job: ${SLURM_JOB_ID} | Host: $(hostname) | Time: $(date)"
echo "Pretrain data: ${PRETRAIN_H5AD}"
echo "Finetune data: ${FINETUNE_H5AD}"
echo "Output:        ${OUT_DIR}"
echo "============================================================"

python3 - <<PY
import h5py
import numpy as np
import pandas as pd

out_dir = "${OUT_DIR}"

def read_h5ad_raw(h5ad_path):
    """Read X matrix and var_names directly via h5py, bypassing anndata obs validation."""
    with h5py.File(h5ad_path, "r") as f:
        # CpG names from var index
        var = f["var"]
        index_key = var.attrs.get("_index", "_index")
        if index_key not in var:
            index_key = list(var.keys())[0]
        cpg_names = np.array(var[index_key]).astype(str)

        # X matrix — may be dense or sparse (CSR/CSC)
        X_grp = f["X"]
        if isinstance(X_grp, h5py.Dataset):
            X = X_grp[()].astype(np.float32)
        else:
            # sparse: read as CSR
            import scipy.sparse
            data    = X_grp["data"][()]
            indices = X_grp["indices"][()]
            indptr  = X_grp["indptr"][()]
            shape   = tuple(X_grp.attrs["shape"])
            X = scipy.sparse.csr_matrix((data, indices, indptr), shape=shape).toarray().astype(np.float32)

    return X, cpg_names

def analyze(h5ad_path, label):
    print(f"\n{'='*60}")
    print(f"Dataset: {label}")
    print(f"File:    {h5ad_path}")
    print('='*60)

    X, cpg_names = read_h5ad_raw(h5ad_path)
    n_cells, n_cpgs = X.shape
    print(f"Shape: {n_cells:,} cells × {n_cpgs:,} CpGs")

    measured = ~np.isnan(X)
    coverage = measured.sum(axis=0)   # [n_cpgs]

    always_nan  = int((coverage == 0).sum())
    always_meas = int((coverage == n_cells).sum())
    partial     = int(((coverage > 0) & (coverage < n_cells)).sum())

    print(f"\nPer-CpG coverage:")
    print(f"  Always measured ({n_cells:,} cells): {always_meas:>8,}  ({100*always_meas/n_cpgs:.1f}%)")
    print(f"  Partially measured:                  {partial:>8,}  ({100*partial/n_cpgs:.1f}%)")
    print(f"  Always NaN (0 cells):                {always_nan:>8,}  ({100*always_nan/n_cpgs:.1f}%)")

    if partial > 0:
        print(f"\n  Partial-coverage breakdown:")
        for pct in [99, 95, 90, 75, 50]:
            n = int((coverage >= n_cells * pct / 100).sum())
            print(f"    >= {pct:3d}% cells: {n:>8,} CpGs")

    per_cell = measured.sum(axis=1)
    print(f"\nPer-cell measurement count:")
    print(f"  Min:    {int(per_cell.min()):>8,}")
    print(f"  Median: {int(np.median(per_cell)):>8,}")
    print(f"  Max:    {int(per_cell.max()):>8,}")
    print(f"  All cells same count: {per_cell.min() == per_cell.max()}")

    tag = label.lower().replace(" ", "_")
    measured_cpgs = cpg_names[coverage == n_cells]
    nan_cpgs      = cpg_names[coverage == 0]

    pd.DataFrame({"cpg_site": measured_cpgs}).to_csv(
        f"{out_dir}/cpgs_always_measured_{tag}.csv", index=False)
    pd.DataFrame({"cpg_site": nan_cpgs}).to_csv(
        f"{out_dir}/cpgs_always_nan_{tag}.csv", index=False)
    pd.DataFrame({"cpg_site": cpg_names, "coverage_count": coverage,
                  "coverage_pct": coverage / n_cells * 100}).to_csv(
        f"{out_dir}/cpgs_full_coverage_{tag}.csv", index=False)

    print(f"\nSaved:")
    print(f"  cpgs_always_measured_{tag}.csv  ({len(measured_cpgs):,} CpGs)")
    print(f"  cpgs_always_nan_{tag}.csv        ({len(nan_cpgs):,} CpGs)")
    print(f"  cpgs_full_coverage_{tag}.csv     (all {n_cpgs:,} CpGs)")

    return set(measured_cpgs), always_meas, always_nan, partial

pt_measured, pt_meas, pt_nan, pt_partial = analyze("${PRETRAIN_H5AD}", "pretrain")
ft_measured, ft_meas, ft_nan, ft_partial = analyze("${FINETUNE_H5AD}", "finetune")

print(f"\n{'='*60}")
print("SUMMARY")
print('='*60)
print(f"Pretrain — measured: {pt_meas:,}  always-NaN: {pt_nan:,}  partial: {pt_partial:,}")
print(f"Finetune — measured: {ft_meas:,}  always-NaN: {ft_nan:,}  partial: {ft_partial:,}")

overlap = pt_measured & ft_measured
only_pt = pt_measured - ft_measured
only_ft = ft_measured - pt_measured

print(f"\nCpGs measured in BOTH pretrain and finetune: {len(overlap):,}")
print(f"Only in pretrain:                            {len(only_pt):,}")
print(f"Only in finetune:                            {len(only_ft):,}")

pd.DataFrame({"cpg_site": sorted(overlap)}).to_csv(
    "${OUT_DIR}/cpgs_pretrain_AND_finetune_measured.csv", index=False)
print(f"\nSaved: cpgs_pretrain_AND_finetune_measured.csv  ({len(overlap):,} CpGs)")
print(f"\n{'='*60}")
print(f"RECOMMENDATION: SUBSET_K = {len(overlap):,} (intersection — measured in both datasets)")
print('='*60)
PY

echo "============================================================"
echo "Analysis finished: $(date)"
echo "Results: ${OUT_DIR}/"
echo "============================================================"
