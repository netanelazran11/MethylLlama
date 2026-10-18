"""
Genomic RoPE ablation — final comparison: Run A (genomic positions) vs
Run B (arbitrary/sequential positions), same 5,000-sample subset, same
architecture, same hyperparameters, checkpoints picked by lowest
reconstruction loss for each.

Reuses the exact distance-sampling method from analyze_cpg_genomic_locality.py
(validated earlier against the production model's known result), applied to
both ablation runs' contextualized CpG embeddings so the two curves are
computed identically and are directly comparable.

Usage:
  python scripts/rope_pilot/analyze_rope_ablation_comparison.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

A_DIR = Path("outputs/ablation_rope/cpg_context_A")
B_DIR = Path("outputs/ablation_rope/cpg_context_B")
OUTDIR = Path("outputs/ablation_rope")

N_PAIRS_PER_BIN = 4000
N_RANDOM_PAIRS = 200_000
TOL_FRAC = 0.15
SEED = 0
BINS = [1, 2, 3, 5, 8, 11, 20, 35, 50, 75, 100, 200, 500, 1000, 2000, 5000]


def cosine_for_pairs(Xn, i, j):
    return np.sum(Xn[i] * Xn[j], axis=1)


def mean_sim_at_distance(rank, Xn, target_dist, n_pairs, tol_frac, rng):
    n = len(rank)
    order = np.argsort(rank)
    ranked = rank[order]
    tol = max(1, int(round(target_dist * tol_frac)))

    pos = rng.integers(0, n, size=n_pairs * 4)
    sign = rng.choice([-1, 1], size=n_pairs * 4)
    target_rank = ranked[pos] + sign * target_dist
    pos2 = np.searchsorted(ranked, target_rank)
    pos2 = np.clip(pos2, 0, n - 1)
    actual_gap = np.abs(ranked[pos2] - ranked[pos])
    valid = (np.abs(actual_gap - target_dist) <= tol) & (pos2 != pos)
    pos, pos2 = pos[valid], pos2[valid]
    if len(pos) > n_pairs:
        pos, pos2 = pos[:n_pairs], pos2[:n_pairs]
    if len(pos) == 0:
        return np.nan
    i_cols, j_cols = order[pos], order[pos2]
    return float(np.mean(cosine_for_pairs(Xn, i_cols, j_cols)))


def load_run(d: Path):
    ctx = np.load(d / "contextual_cpg_emb.npy").astype(np.float64)
    order_df = pd.read_csv(d / "cpg_order.csv")
    rank = order_df["genomic_rank"].values.astype(np.int64)
    meta = json.loads((d / "contextual_cpg_meta.json").read_text())
    ctx_n = ctx / (np.linalg.norm(ctx, axis=1, keepdims=True) + 1e-9)
    return ctx_n, rank, meta


def curve_for_run(ctx_n, rank, rng):
    rows = []
    for d in BINS:
        sim = mean_sim_at_distance(rank, ctx_n, d, N_PAIRS_PER_BIN, TOL_FRAC, rng)
        rows.append({"distance": d, "cos": sim})
    df = pd.DataFrame(rows)

    n = len(ctx_n)
    i_rand = rng.integers(0, n, size=N_RANDOM_PAIRS)
    j_rand = rng.integers(0, n, size=N_RANDOM_PAIRS)
    valid = i_rand != j_rand
    baseline = float(np.mean(cosine_for_pairs(ctx_n, i_rand[valid], j_rand[valid])))
    return df, baseline


def main():
    rng = np.random.default_rng(SEED)

    ctx_a, rank_a, meta_a = load_run(A_DIR)
    ctx_b, rank_b, meta_b = load_run(B_DIR)
    assert np.array_equal(rank_a, rank_b), "CpG ordering differs between A and B -- not comparable"

    print(f"Run A checkpoint: {meta_a['checkpoint']}  (fed_position_ids={meta_a['fed_position_ids']})")
    print(f"Run B checkpoint: {meta_b['checkpoint']}  (fed_position_ids={meta_b['fed_position_ids']})")
    print(f"n_samples: A={meta_a['n_samples']}  B={meta_b['n_samples']}")
    print()

    df_a, baseline_a = curve_for_run(ctx_a, rank_a, rng)
    df_b, baseline_b = curve_for_run(ctx_b, rank_b, rng)

    summary = {
        "run_A_genomic": {
            "adjacent_cos_dist1": float(df_a.loc[df_a["distance"] == 1, "cos"].iloc[0]),
            "random_pair_baseline": baseline_a,
            "cos_at_dist100": float(df_a.loc[df_a["distance"] == 100, "cos"].iloc[0]),
        },
        "run_B_nogenomic": {
            "adjacent_cos_dist1": float(df_b.loc[df_b["distance"] == 1, "cos"].iloc[0]),
            "random_pair_baseline": baseline_b,
            "cos_at_dist100": float(df_b.loc[df_b["distance"] == 100, "cos"].iloc[0]),
        },
    }
    with open(OUTDIR / "rope_ablation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    df_a.to_csv(OUTDIR / "rope_ablation_curve_A.csv", index=False)
    df_b.to_csv(OUTDIR / "rope_ablation_curve_B.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(df_a["distance"], df_a["cos"], "o-", color="#1a4ab0", ms=5, lw=2,
            label="Run A -- genomic RoPE (trained WITH genomic positions)")
    ax.axhline(baseline_a, color="#1a4ab0", linestyle="--", lw=1.1, alpha=0.6)
    ax.plot(df_b["distance"], df_b["cos"], "o-", color="#8090a8", ms=5, lw=2,
            label="Run B -- no genomic RoPE (trained WITHOUT genomic positions)")
    ax.axhline(baseline_b, color="#8090a8", linestyle="--", lw=1.1, alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("genomic-rank distance between CpGs")
    ax.set_ylabel("mean cosine similarity of contextualized embeddings")
    ax.set_title("Genomic RoPE ablation: does the model learn chromosomal locality?\n"
                 "(same 5k-sample subset, same architecture, only genomic positions differ)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTDIR / "rope_ablation_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved -> {OUTDIR / 'rope_ablation_comparison.png'}")


if __name__ == "__main__":
    main()
