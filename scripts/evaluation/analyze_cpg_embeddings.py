"""
analyze_cpg_embeddings.py
=========================
CpG embedding matrix analysis: does genomic neighborhood structure appear in the learned
CpG token embeddings of the production model? (Supplementary Note 1: the static token-embedding
table shows no positive locality gap; adjacent_vs_random_gap in cpg_embedding_analysis.json.)

Pure post-processing on cpg_embedding_matrix.npy + cpg_alignment.csv (from
extract_pretrain_cls.py). No GPU.

Tests (using genomic RANK, which is chromosome+position sorted):
  1. Genomic-proximity decay — mean cosine similarity of CpG-embedding pairs as a
     function of genomic-rank distance. If the model learned position structure,
     genomically-adjacent CpGs are more similar than random pairs.
  2. Nearest-neighbor genomic locality — for each CpG, are its top-k embedding
     neighbors closer in genomic rank than random? (median rank-distance vs chance)

Note: this inspects the TOKEN embedding table. Genomic RoPE acts on attention
Q/K, not on this table directly; correlated adjacent embeddings reflect learned
co-methylation / locality. True RoPE validation is attention-distance decay
(separate analysis). Chromosome-label clustering needs an external CpG manifest
(chr per probe) — not required here; rank distance is a sufficient proxy.

Usage (step 3 of run_full_analysis.sh; CPU):
  python scripts/evaluation/analyze_cpg_embeddings.py --dir results/pretrained_cls
cpg_embedding_matrix.npy (the embedding table of the checkpoint) and cpg_alignment.csv are written
by extract_pretrain_cls.py and are not shipped; the output cpg_embedding_analysis.json is.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="results/pretrained_cls")
    p.add_argument("--n_probe", type=int, default=3000, help="CpGs sampled for NN test")
    p.add_argument("--topk", type=int, default=10)
    return p.parse_args()


def main():
    a = parse_args()
    d = Path(a.dir)
    W = np.load(d / "cpg_embedding_matrix.npy").astype(np.float64)      # [vocab, 256]
    align = pd.read_csv(d / "cpg_alignment.csv")                         # column_index, encoder_vocab_id, cpg_name, genomic_rank
    n = len(align)
    E = W[align["encoder_vocab_id"].values]                            # [n_cpgs, 256] real embedding rows
    rank = align["genomic_rank"].values.astype(np.int64)
    order = np.argsort(rank)                                            # genomic order
    Eo = E[order]                                                       # embeddings in genomic order
    En = Eo / (np.linalg.norm(Eo, axis=1, keepdims=True) + 1e-9)
    print(f"CpG embeddings: {E.shape}  (of vocab {W.shape[0]})")

    rng = np.random.default_rng(0)
    summary = {"n_cpgs": int(n), "emb_dim": int(E.shape[1])}

    # ── 1. Proximity decay: cosine sim vs genomic-rank distance (adjacent in order) ──
    bins = [1, 2, 5, 10, 50, 100, 500]
    decay = {}
    for step in bins:
        a1 = En[:-step]
        a2 = En[step:]
        decay[step] = float(np.mean(np.sum(a1 * a2, axis=1)))
    # random baseline
    i = rng.integers(0, n, 20000); j = rng.integers(0, n, 20000)
    rand_cos = float(np.mean(np.sum(En[i] * En[j], axis=1)))
    summary["proximity_cosine_by_rankstep"] = decay
    summary["random_pair_cosine"] = rand_cos
    print("\nGenomic-proximity cosine (adjacent in genomic order):")
    for k, v in decay.items():
        print(f"  rank step {k:>4}: cos={v:.4f}")
    print(f"  random pair : cos={rand_cos:.4f}")
    summary["adjacent_vs_random_gap"] = round(decay[1] - rand_cos, 4)

    # ── 2. NN genomic locality: are embedding-NN close in genomic rank? ──
    idx = rng.choice(n, min(a.n_probe, n), replace=False)
    Enf = En  # full normalized (genomic order)
    # map: position in genomic order for each sampled cpg
    med_nn_rankdist, med_rand_rankdist = [], []
    positions = np.arange(n)
    for p in idx:
        sims = Enf @ Enf[p]
        sims[p] = -np.inf
        nn = np.argpartition(-sims, a.topk)[: a.topk]
        med_nn_rankdist.append(np.median(np.abs(positions[nn] - p)))
        med_rand_rankdist.append(np.median(np.abs(rng.integers(0, n, a.topk) - p)))
    summary["nn_median_rankdist"] = round(float(np.mean(med_nn_rankdist)), 1)
    summary["random_median_rankdist"] = round(float(np.mean(med_rand_rankdist)), 1)
    summary["nn_locality_ratio"] = round(
        float(np.mean(med_nn_rankdist) / max(np.mean(med_rand_rankdist), 1e-9)), 4
    )
    print(f"\nNN genomic locality (top-{a.topk}):")
    print(f"  embedding-NN median rank distance : {summary['nn_median_rankdist']}")
    print(f"  random         median rank distance: {summary['random_median_rankdist']}")
    print(f"  ratio (lower=more local)          : {summary['nn_locality_ratio']}")

    with open(d / "cpg_embedding_analysis.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved {d/'cpg_embedding_analysis.json'}")
    verdict = ("LOCAL structure present" if summary["nn_locality_ratio"] < 0.8
               or summary["adjacent_vs_random_gap"] > 0.02 else "weak/no local structure")
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
