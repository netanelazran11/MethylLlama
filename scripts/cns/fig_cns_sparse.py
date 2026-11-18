"""
fig_cns_sparse.py  (CNS task, stage 5 - sparse-input figure)
============================================================
Draws the sparse-input panels from the CSVs written by compare_sparse.py (nothing is recomputed here).

  e  balanced accuracy against % of CpGs measured (log scale), Scenario A (head trained on full profiles, tested on
     sparse ones), full-data setting: MethylLlama, MethylGPT and the raw baselines with 95 % bootstrap CI bands;
     below it, MethylLlama minus MethylGPT and MethylLlama minus the best raw baseline, with CI bands and a zero line
     (filled markers = CI excludes zero)
  f  self-retrieval: fraction of profiles whose sparse CLS has its own full-profile CLS as nearest neighbour among
     all 3,902 (comparable across models, unlike raw cosine); markers = mean over mask seeds, bars = seed range
  g  Scenario B (head trained and tested at the same sparse level), full-data setting, 95 % CI

Outputs  outputs/cns/figures/{pdf,png,svg}/fig_cns_sparse.*, and the plotted rows in figures/data/
  python scripts/cns/fig_cns_sparse.py [--compare outputs/cns/sparse/compare_wideC_all]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/figures"))
from common_style import COL_GPT, COL_LLAMA, apply_style, panel_label  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

COL_RAW = "#8a8a8a"
LINES = {"llama_cls": ("MethylLlama", COL_LLAMA, "o", "-"), "gpt_cls": ("MethylGPT", COL_GPT, "o", "-"),
         "raw_top5k": ("Raw, 5k CpGs", COL_RAW, "s", "-"), "raw_pca256": ("Raw, PCA-256", COL_RAW, "D", "--"),
         "raw_top32k": ("Raw, 32k CpGs", COL_RAW, "^", ":"), "raw_all": ("Raw, all CpGs", COL_RAW, "v", "-.")}
CHANCE = 1 / 91


def save(fig, out, stem):
    for ext in ("pdf", "png", "svg"):
        (out / ext).mkdir(parents=True, exist_ok=True)
        fig.savefig(out / ext / f"{stem}.{ext}", bbox_inches="tight", dpi=300)
    print("saved", stem)


def level_axis(ax, levels):
    ax.set_xscale("log")
    ax.set_xticks(levels, [f"{x:g}" for x in levels])
    ax.minorticks_off()
    ax.set_xlim(min(levels) / 1.35, max(levels) * 1.35)


def panel_e(ax, axd, acc, dif, levels):
    a = acc[(acc.scenario == "A") & (acc.setting == "full")]
    for rep, (lab, col, mk, ls) in LINES.items():
        t = a[a.rep == rep].sort_values("level")
        if not len(t):
            continue
        model = not rep.startswith("raw")
        ax.fill_between(t.level, t.ci_lo, t.ci_hi, color=col, alpha=0.18 if model else 0.10, lw=0)
        ax.plot(t.level, t.ba, ls=ls, marker=mk, color=col, ms=3.2 if model else 2.8, lw=1.3 if model else 0.9,
                mfc=col if model else "white", label=lab, zorder=3 if model else 2)
    ax.axhline(CHANCE, color="#b0b0b0", lw=0.6, ls=":")
    ax.text(min(levels), CHANCE + 0.015, "chance", fontsize=6.8, color="#8a8a8a", ha="left", va="bottom")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Balanced accuracy (91 classes)")
    level_axis(ax, levels)
    ax.tick_params(labelbottom=False)
    ax.legend(frameon=False, loc="lower right", handlelength=1.8, ncol=1)

    d = dif[(dif.scenario == "A") & (dif.setting == "full")]
    for key, lab, col in (("llama_cls - gpt_cls", "vs MethylGPT", COL_GPT), ("llama_cls - best raw", "vs best raw", COL_RAW)):
        t = d[d.comparison.str.startswith(key)].sort_values("level")
        if not len(t):
            continue
        axd.fill_between(t.level, t.ci_lo, t.ci_hi, color=col, alpha=0.2, lw=0)
        axd.plot(t.level, t["diff"], color=col, lw=1.1, label=f"MethylLlama {lab}")
        sig = t.excludes_zero.to_numpy()
        axd.scatter(t.level[sig], t["diff"][sig], color=col, s=10, zorder=3)
        axd.scatter(t.level[~sig], t["diff"][~sig], facecolor="white", edgecolor=col, s=10, zorder=3, lw=0.8)
    axd.axhline(0, color="black", lw=0.6)
    axd.set_ylabel("Difference")
    axd.set_xlabel("CpGs measured per profile (% of 49,156)")
    level_axis(axd, levels)
    axd.set_ylim(min(-0.2, axd.get_ylim()[0]), 1.2)
    axd.legend(frameon=False, loc="upper left", handlelength=1.4, ncol=2, columnspacing=1.0)


def panel_f(ax, stab, levels):
    for m, rep in (("llama", "llama_cls"), ("gpt", "gpt_cls")):
        t = stab[stab.model == m].groupby("level").top1.agg(["mean", "min", "max"]).reset_index()
        if not len(t):
            continue
        lab, col = LINES[rep][:2]
        ax.errorbar(t.level, t["mean"], yerr=[t["mean"] - t["min"], t["max"] - t["mean"]], color=col, marker="o",
                    ms=3.2, lw=1.3, capsize=1.5, elinewidth=0.7, label=lab)
    ax.set_ylim(-0.02, 1.04)
    ax.set_ylabel("Sparse profile retrieves its own\nfull profile (top-1, of 3,902)")
    ax.set_xlabel("CpGs measured (%)")
    level_axis(ax, levels)
    ax.legend(frameon=False, loc="lower right", handlelength=1.4)


def panel_g(ax, acc):
    b = acc[(acc.scenario == "B") & (acc.setting == "full") & (acc.level < 100)]
    lv = sorted(b.level.unique(), reverse=True)
    if len(lv) >= 3:                                         # enough levels for a curve (probe_sparse_wideC grid)
        for rep, (lab, col, mk, ls) in LINES.items():
            t = b[b.rep == rep].sort_values("level")
            if not len(t):
                continue
            model = not rep.startswith("raw")
            ax.fill_between(t.level, t.ci_lo, t.ci_hi, color=col, alpha=0.18 if model else 0.10, lw=0)
            ax.plot(t.level, t.ba, ls=ls, marker=mk, color=col, ms=3.0 if model else 2.6, lw=1.2 if model else 0.9,
                    mfc=col if model else "white", label=lab)
        ax.axhline(CHANCE, color="#b0b0b0", lw=0.6, ls=":")
        level_axis(ax, sorted(lv))
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Balanced accuracy")
        ax.set_xlabel("CpGs measured, train and test (%)")
        ax.legend(frameon=False, loc="lower right", fontsize=6.3, handlelength=1.6)
        return
    reps = [r for r in LINES if r in set(b.rep)]
    w = 0.8 / max(len(reps), 1)
    for j, rep in enumerate(reps):
        lab, col, _, _ = LINES[rep]
        t = b[b.rep == rep].set_index("level").reindex(lv)
        x = np.arange(len(lv)) - 0.4 + w * (j + 0.5)
        raw = rep.startswith("raw")
        ax.bar(x, t.ba, w * 0.92, color="white" if raw else col, edgecolor=col, lw=0.8,
               hatch={"raw_top5k": "", "raw_pca256": "////", "raw_top32k": "....", "raw_all": "xxx"}.get(rep, ""),
               label=lab)
        ax.errorbar(x, t.ba, yerr=[t.ba - t.ci_lo, t.ci_hi - t.ba], fmt="none", ecolor="black", elinewidth=0.6,
                    capsize=1.3)
    ax.axhline(CHANCE, color="#b0b0b0", lw=0.6, ls=":")
    ax.set_xticks(np.arange(len(lv)), [f"{x:g}%" for x in lv])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Balanced accuracy")
    ax.set_xlabel("CpGs measured, train and test")
    ax.legend(frameon=False, loc="upper right", fontsize=6.8, handlelength=1.2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--compare", default=str(REPO / "outputs/cns/sparse/compare_wideC_all"))
    p.add_argument("--root", default=str(REPO / "outputs/cns/figures"))
    p.add_argument("--stem", default="fig_cns_sparse")
    a = p.parse_args()
    c = Path(a.compare)
    acc, dif = pd.read_csv(c / "accuracy.csv"), pd.read_csv(c / "differences.csv")
    stab = pd.read_csv(c / "stability.csv")
    levels = sorted(acc[acc.scenario == "A"].level.unique())
    out = Path(a.root)
    (out / "data").mkdir(parents=True, exist_ok=True)
    acc[acc.setting == "full"].to_csv(out / "data" / f"{a.stem}_accuracy.csv", index=False)
    dif[dif.setting == "full"].to_csv(out / "data" / f"{a.stem}_difference.csv", index=False)
    stab.to_csv(out / "data" / f"{a.stem}_stability.csv", index=False)

    apply_style()
    fig = plt.figure(figsize=(7.2, 4.9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1.0], height_ratios=[1.7, 1.0], hspace=0.12, wspace=0.42)
    axe = fig.add_subplot(gs[0, 0])
    axd = fig.add_subplot(gs[1, 0], sharex=axe)
    gsr = gs[:, 1].subgridspec(2, 1, hspace=0.55)
    axf = fig.add_subplot(gsr[0])
    axg = fig.add_subplot(gsr[1])
    panel_e(axe, axd, acc, dif, levels)
    panel_f(axf, stab, levels)
    panel_g(axg, acc)
    panel_label(axe, "a", dx=-0.13)
    panel_label(axf, "b", dx=-0.27)
    panel_label(axg, "c", dx=-0.27)
    save(fig, out, a.stem)
    plt.close(fig)


if __name__ == "__main__":
    main()
