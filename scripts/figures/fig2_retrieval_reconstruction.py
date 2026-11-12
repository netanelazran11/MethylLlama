#!/usr/bin/env python3
"""
fig2_retrieval_reconstruction.py -- Figure 2 of the manuscript.

  a  ECDF of the contrastive margin, cos(positive) - cos(hardest of 31 negatives), per negative
     pool, disjoint views (seed 0). The height of each curve at margin 0 is its error rate.
  b  Top-1 retrieval (1 positive vs 31 negatives) per pool, overlapping vs disjoint views;
     points are means of 3 view draws x 5 negative draws, bars the range over view draws.
  c  Per-profile reconstruction MSE at withheld CpGs: model vs per-CpG mean, shuffled CLS, random CLS.

Inputs:
  results/pretrained_cls/hard_negative_heldout/{summary.json, per_query.npz}
      (scripts/evaluation/hard_negative_heldout.py)
  results/pretrained_cls/reconstruction_withheld/per_sample_metrics.csv
      (scripts/evaluation/reconstruction_withheld_eval.py)
Usage:  python scripts/figures/fig2_retrieval_reconstruction.py
Output: figures/paper/fig2_retrieval_reconstruction.{pdf,png,svg} + _provenance.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_style import COL_GPT, COL_LLAMA, apply_style, panel_label, save  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
HN = ROOT / "results/pretrained_cls/hard_negative_heldout"
RECON = ROOT / "results/pretrained_cls/reconstruction_withheld/per_sample_metrics.csv"
OUT = ROOT / "figures/paper/fig2_retrieval_reconstruction"

POOLS = ["random", "same tissue", "same study", "same study+tissue", "same study+tissue+sex",
         "K nearest neighbours of target"]
LABEL = {"random": "Random", "same tissue": "Same tissue", "same study": "Same study",
         "same study+tissue": "Same study + tissue", "same study+tissue+sex": "Same study + tissue + sex",
         "K nearest neighbours of target": "31 nearest neighbours"}
# Panel a: four pools with distinct hue + line style (Okabe-Ito; no blue/orange, which mean overlap/disjoint in b).
# "Same study" and "+ sex" are omitted from a because they overlap "same study + tissue" and "31 nearest
# neighbours"; all six pools are shown in b.
POOLS_A = [("random", "#9a9a9a", (0, (4, 2))), ("same tissue", "#009E73", "-"),
           ("same study+tissue", "#CC79A7", "-"), ("K nearest neighbours of target", "#000000", "-")]


def main():
    apply_style()
    s = json.load(open(HN / "summary.json"))
    z = np.load(HN / "per_query.npz")
    rec = pd.read_csv(RECON)
    assert len(rec) == 16912, "panel c must use all held-out profiles"

    fig = plt.figure(figsize=(7.2, 3.35))
    gs = GridSpec(1, 3, width_ratios=[1.15, 1.05, 1.0], wspace=0.6, left=0.07, right=0.99, bottom=0.36, top=0.93)

    # ---- a: margin ECDF, disjoint views --------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    cond = "disjoint_seed0"
    pos = z[f"{cond}__pos"]
    err = {}
    for p, col, ls in POOLS_A:
        h = z[f"{cond}__{p}"]
        ok = ~np.isnan(h)
        m = np.sort(pos[ok] - h[ok])
        y = np.arange(1, len(m) + 1) / len(m)
        err[p] = float((m <= 0).mean())
        ax.plot(m, y, color=col, ls=ls, lw=1.6, label=f"{LABEL[p]}  ({100 * err[p]:.1f}% errors)")
    ax.axvline(0, color="#b03030", lw=0.8, ls="--")
    ax.set_xlim(-0.02, 0.15)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Margin: cos(positive) − cos(hardest negative)")
    ax.set_ylabel("Cumulative fraction of profiles")
    ax.text(0.003, 0.985, "margin 0", color="#b03030", fontsize=6.8, va="top")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(-0.28, -0.24), ncol=2, fontsize=6.8,
              handlelength=1.4, columnspacing=1.2, title="Negative pool (disjoint views; % with margin ≤ 0)",
              title_fontsize=6.8, alignment="left")
    panel_label(ax, "a", dx=-0.2)

    # ---- b: top-1 by pool, dumbbell (overlapping vs disjoint on the same negatives) ----------------
    ax = fig.add_subplot(gs[0, 1])
    ys = np.arange(len(POOLS))[::-1]
    def fmt(m):  # same rounding as the text; a value that would round to 100 keeps two decimals
        return ">99.99" if (round(m, 2) >= 99.995 and m < 100) else (f"{np.floor(m * 100) / 100:.2f}" if (round(m, 1) >= 99.95 and m < 100) else f"{m:.1f}")
    v = {c: np.array([[100 * s["conditions"][f"{c}_seed{k}"][p]["top1"] for k in range(3)] for p in POOLS])
         for c in ("overlap", "disjoint")}
    for xg in (90, 92, 94, 96, 98, 100):
        ax.axvline(xg, color="#e6e8ec", lw=0.6, zorder=0)
    for i, y in enumerate(ys):
        mo, md = v["overlap"][i].mean(), v["disjoint"][i].mean()
        ax.plot([md, mo], [y, y], color="#b8bcc4", lw=2.0, solid_capstyle="round", zorder=1)
        for c, col, m in (("overlap", COL_LLAMA, mo), ("disjoint", COL_GPT, md)):
            ax.plot(m, y, "o", ms=5.2, color=col, mec="white", mew=0.6, zorder=3)
        ax.text(md - 0.35, y, fmt(md), ha="right", va="center", fontsize=6.6, color=COL_GPT)
        ax.text(mo + 0.35, y, fmt(mo), ha="left", va="center", fontsize=6.6, color=COL_LLAMA)
    ax.set_yticks(ys)
    # short nested labels so the longest one does not run into panel a
    LABEL_B = {"random": "Random", "same tissue": "Same tissue", "same study": "Same study",
               "same study+tissue": "  + tissue", "same study+tissue+sex": "  + tissue + sex",
               "K nearest neighbours of target": "31 nearest\nneighbours"}
    ax.set_yticklabels([LABEL_B[p] for p in POOLS])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(87.5, 101.8)
    ax.set_xticks([88, 90, 92, 94, 96, 98, 100])
    ax.set_xlabel("Top-1 retrieval (%), 1 vs 31 negatives\n(chance 3.1%)")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="none", color=COL_LLAMA, ms=5, label="Overlapping views"),
                       Line2D([], [], marker="o", ls="none", color=COL_GPT, ms=5, label="Disjoint views")],
              frameon=False, loc="lower center", fontsize=6.8, handletextpad=0.2, borderaxespad=0.0,
              ncol=2, columnspacing=1.0, bbox_to_anchor=(0.5, 1.0))
    ax.set_ylim(-0.5, len(POOLS) - 0.5)
    panel_label(ax, "b", dx=-0.5)

    # ---- c: reconstruction vs controls -------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    cols = [("model_mse", "Model", COL_LLAMA), ("b_mean_mse", "CpG\nmean", "#8a8a8a"),
            ("b_shuffle_mse", "Shuffled\nCLS", "#8a8a8a"), ("b_random_mse", "Random\nvectors", "#8a8a8a")]
    data = [rec[c].values for c, _, _ in cols]
    bp = ax.boxplot(data, widths=0.55, showfliers=False, patch_artist=True, whis=(5, 95),
                    medianprops=dict(color="black", lw=1.0), whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8))
    for patch, (_, _, col) in zip(bp["boxes"], cols):
        patch.set_facecolor(col); patch.set_alpha(0.75); patch.set_edgecolor("black"); patch.set_linewidth(0.6)
    ax.set_yscale("log")
    ax.set_xticks(range(1, 5))
    ax.set_xticklabels([l for _, l, _ in cols], fontsize=6.8, rotation=0)
    ax.tick_params(axis="x", pad=2)
    ax.set_ylabel("Per-profile MSE at withheld CpGs")
    panel_label(ax, "c", dx=-0.32)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save(fig, str(OUT))

    prov = {
        "file": "fig2_retrieval_reconstruction.pdf",
        "panel_a": {"source": "results/pretrained_cls/hard_negative_heldout/per_query.npz",
                    "condition": cond, "negative_draw": 0,
                    "pools_shown": [LABEL[p] for p, _, _ in POOLS_A],
                    "error_rate_at_margin_0": {LABEL[p]: round(err[p], 4) for p, _, _ in POOLS_A}},
        "panel_b": {"source": "results/pretrained_cls/hard_negative_heldout/summary.json",
                    "top1_percent_mean_and_range_over_3_view_draws": {
                        c: {LABEL[p]: [round(float(np.mean(v)), 2), round(float(np.min(v)), 2), round(float(np.max(v)), 2)]
                            for p in POOLS for v in [[100 * s["conditions"][f"{c}_seed{k}"][p]["top1"] for k in range(3)]]}
                        for c in ("overlap", "disjoint")},
                    "n_queries": {LABEL[p]: s["conditions"]["disjoint_seed0"][p]["n_queries"] for p in POOLS}},
        "panel_c": {"source": "results/pretrained_cls/reconstruction_withheld/per_sample_metrics.csv",
                    "n_profiles": int(len(rec)), "box": "IQR, median line, whiskers 5th-95th percentile, log scale",
                    "median_mse": {l.replace("\n", " "): round(float(np.median(rec[c])), 5) for c, l, _ in cols}},
    }
    Path(f"{OUT}_provenance.json").write_text(json.dumps(prov, indent=1))
    print(json.dumps(prov["panel_a"]["error_rate_at_margin_0"], indent=1))


if __name__ == "__main__":
    main()
