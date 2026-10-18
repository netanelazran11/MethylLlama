"""
create_pretrain_subset_h5ad.py
================================
Genomic-RoPE ablation (Run A vs Run B) — Step 0: build a fixed, reusable
sample subset of the pretrain corpus, saved as a NEW h5ad file. Read-only on
the source file; never modifies anything in the existing pipeline.

Both Run A (genomic RoPE) and Run B (no genomic RoPE) pretrain jobs point at
this SAME subset file, so the only thing that differs between the two runs is
whether wced_genomic_rank_path is set — the fixed subset is what makes the
ablation clean.

Usage (cluster):
  python scripts/rope_pilot/create_pretrain_subset_h5ad.py \
      --source ${METHYLLAMA_DATA}/data_methyl_pretrain_type3_h5ad/methylgpt_pretrain_type3.h5ad \
      --n_samples 5000 \
      --seed 42 \
      --outdir outputs/ablation_rope
"""

import argparse
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bmfm_methylation.shared.data_module import _read_h5ad_robust


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="full pretrain h5ad (read-only)")
    p.add_argument("--n_samples", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", default="outputs/ablation_rope")
    return p.parse_args()


def main():
    a = parse_args()
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"pretrain_subset_{a.n_samples}.h5ad"

    if out_path.exists():
        print(f"Subset already exists, not overwriting: {out_path}")
        print("(delete it manually first if you want to regenerate)")
        return

    print(f"Loading (read-only): {a.source}")
    adata = _read_h5ad_robust(a.source)
    n_total = adata.shape[0]
    print(f"  Source: {n_total:,} samples x {adata.shape[1]:,} CpGs")

    rng = np.random.default_rng(a.seed)
    n = min(a.n_samples, n_total)
    idx = np.sort(rng.choice(n_total, size=n, replace=False))
    subset = adata[idx].copy()

    print(f"  Subset: {subset.shape[0]:,} samples x {subset.shape[1]:,} CpGs (seed={a.seed})")
    subset.write_h5ad(out_path)
    print(f"Saved -> {out_path}")
    print("\nBoth Run A (genomic) and Run B (no-genomic) pretrain jobs should set:")
    print(f"  PRETRAIN_DATA={out_path.resolve()}")


if __name__ == "__main__":
    main()
