"""
fig_cns_sparse_B.py  (CNS task, stage 5 - sparse training and test data)
========================================================================
The figure of the question "when every profile, in training and in test, is measured at only part of its CpGs, which
representation classifies best, and down to / up to which level?" Capper reference profiles, 5 x 5 CV, the same linear
head on every representation; each level keeps a random share of each profile's measured CpGs (3 mask seeds).

  a  balanced accuracy against % of CpGs measured (train and test, Scenario B), 95 % CI; shaded: levels where
     MethylLlama is above the best raw baseline (Holm-adjusted p < 0.05); the crossover lies between the highest such
     level and the lowest level where the best raw baseline is significantly above MethylLlama
  b  MethylLlama minus the best raw baseline and minus MethylGPT, 95 % CI (filled = Holm-adjusted p < 0.05)
  c  the 50-100 % range on a linear axis: where the best raw baseline overtakes MethylLlama (levels 60-90 % from
     make_sparse_masks_high.py, nested with the lower levels)
  d  at 5 % of CpGs: head trained on full profiles (Scenario A) against head trained at 5 % (Scenario B)
  e  stability of the embedding itself: share of profiles whose sparse CLS has its own full-profile CLS as nearest
     neighbour among all 3,902 full profiles (mean over seeds; comparable between models, unlike raw cosine)

Statistics: per-profile correctness pooled over CV repeats and mask seeds; paired bootstrap over profiles (B = 2,000,
the same resamples for every representation at a level, seeds as compare_sparse.py); two-sided bootstrap p =
2 x min(share of differences <= 0, share >= 0), floor 1/B; Holm over all Scenario B levels within each comparison. "Best raw"
is picked per level on the point estimate, which favours the raw side.

Outputs  outputs/cns/figures/{pdf,png,svg}/fig_cns_sparse_B.*, figures/data/fig_cns_sparse_B_{accuracy,tests}.csv
  python scripts/cns/fig_cns_sparse_B.py [--probe outputs/cns/sparse/probe_wideC]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/figures"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_style import COL_GPT, COL_LLAMA, apply_style, panel_label  # noqa: E402
from compare_sparse import MODELS, RAW, SETTINGS, SPARSE, boot, load_setting, stability  # noqa: E402
from make_sparse_masks import LEVELS, tag  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

COL_RAW, COL_WIN = "#8a8a8a", "#dcebf7"
NAMES = {"llama_cls": "MethylLlama", "gpt_cls": "MethylGPT", "raw_top5k": "Raw β, 5,000 most variable CpGs",
         "raw_pca256": "Raw β, PCA-256"}
STYLE = {"llama_cls": (COL_LLAMA, "-", "o"), "gpt_cls": (COL_GPT, "-", "o"), "raw_top5k": (COL_RAW, "-", "s"),
         "raw_pca256": (COL_RAW, "--", "D")}
REPS = ("llama_cls", "gpt_cls", "raw_top5k", "raw_pca256")
HIGH = (60.0, 70.0, 80.0, 90.0)
ALL = tuple(sorted(set(LEVELS) | set(HIGH)))
LABELLED = (100, 50, 20, 10, 5, 2, 1, 0.5)


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for k, i in enumerate(o):
        run = max(run, min(1.0, (len(p) - k) * p[i]))
        adj[i] = run
    return adj


def tests(probe, seeds):
    import anndata as ad
    obs = ad.read_h5ad(REPO / "outputs/cns/data/cns_capper.h5ad", backed="r").obs
    gsm = obs.index.astype(str).to_numpy()[obs.set == "reference"]
    y = obs.methylation_class.astype(str).to_numpy()[obs.set == "reference"]
    classes = np.array(sorted(set(y)))
    ci = np.searchsorted(classes, y)
    A, T = [], []
    for sc in ("A", "B"):
        for lv in ALL:
            if sc == "A" and lv not in LEVELS:
                continue
            cbar = {}
            for r in REPS:
                rows = [load_setting(Path(probe) / ("A" if lv == 100 else sc) / r / tag(lv, s), "full", gsm)
                        for s in ([0] if lv == 100 else seeds)]
                if any(x is None for x in rows):
                    raise SystemExit(f"missing cell {sc} {r} L{lv:g}")
                cbar[r] = np.vstack(rows).mean(0)
            # same seed as compare_sparse.py, so the CIs match its output
            seed0 = LEVELS.index(lv) * 100 if lv in LEVELS else 900 + int(lv)
            rng = np.random.default_rng(seed0 + SETTINGS.index("full") * 10 + (sc == "B"))
            est, bs = boot([cbar[r] for r in REPS], ci, len(classes), rng)
            for j, r in enumerate(REPS):
                lo, hi = np.percentile(bs[:, j], [2.5, 97.5])
                A.append({"scenario": sc, "level": lv, "rep": r, "ba": est[j], "ci_lo": lo, "ci_hi": hi})
            best = max(RAW[:2], key=lambda r: est[REPS.index(r)])
            for name, z in (("best raw", best), ("MethylGPT", "gpt_cls")):
                d = bs[:, 0] - bs[:, REPS.index(z)]
                lo, hi = np.percentile(d, [2.5, 97.5])
                p = min(1.0, max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1 / len(d)))
                T.append({"scenario": sc, "level": lv, "vs": name, "against": z, "diff": est[0] - est[REPS.index(z)],
                          "ci_lo": lo, "ci_hi": hi, "p": p})
    A, T = pd.DataFrame(A), pd.DataFrame(T)
    T["p_holm"] = np.nan
    for (sc, vs), g in T.groupby(["scenario", "vs"]):
        T.loc[g.index, "p_holm"] = holm(g.p.to_numpy())
    T["significant"] = T.p_holm < 0.05
    return A, T


def log_axis(ax, levels, reverse=True):
    ax.set_xscale("log")
    ax.set_xticks([x for x in levels if x in LABELLED], [f"{x:g}" for x in levels if x in LABELLED])
    ax.set_xticks([x for x in levels if x not in LABELLED], minor=True)
    ax.tick_params(axis="x", which="minor", length=2, labelbottom=False)
    lo, hi = min(levels) / 1.4, max(levels) * 1.4
    ax.set_xlim((hi, lo) if reverse else (lo, hi))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--probe", default=str(REPO / "outputs/cns/sparse/probe_wideC"))
    p.add_argument("--stability", default=str(REPO / "outputs/cns/sparse/compare_wideC_all/stability.csv"))
    p.add_argument("--root", default=str(REPO / "outputs/cns/figures"))
    p.add_argument("--seeds", default="0,1,2")
    a = p.parse_args()
    out = Path(a.root)
    (out / "data").mkdir(parents=True, exist_ok=True)
    A, T = tests(a.probe, [int(s) for s in a.seeds.split(",")])
    A.to_csv(out / "data" / "fig_cns_sparse_B_accuracy.csv", index=False)
    T.to_csv(out / "data" / "fig_cns_sparse_B_tests.csv", index=False)
    print(T[T.scenario == "B"].round(4).to_string(index=False), flush=True)

    levels = sorted(ALL)
    B_ = A[A.scenario == "B"]
    tb = T[(T.scenario == "B") & (T.vs == "best raw")].set_index("level")
    win = [lv for lv in levels if tb.loc[lv, "significant"] and tb.loc[lv, "diff"] > 0]
    lose = [lv for lv in levels if tb.loc[lv, "significant"] and tb.loc[lv, "diff"] < 0]
    win_band = max([lv for lv in levels if lv <= min(lose, default=1e9) and lv in win], default=None)

    apply_style()
    fig = plt.figure(figsize=(7.2, 6.1))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1.15, 1], wspace=0.5, hspace=0.55)
    ax = fig.add_subplot(gs[0, :2])
    # zones: MethylLlama significantly better / no significant difference / best raw significantly better; borders
    # halfway (geometric mean on the log axis) between neighbouring tested levels
    def border(x0, x1, geo):
        return np.sqrt(x0 * x1) if geo else (x0 + x1) / 2
    hi_win = max(win) if win else None
    lo_lose = min(lose) if lose else None
    nxt = {lv: levels[i + 1] for i, lv in enumerate(levels[:-1])}
    prv = {lv: levels[i - 1] for i, lv in enumerate(levels) if i}

    def zones(axx, geo, xmin, xmax, text_y=None):
        if hi_win:
            b = border(hi_win, nxt[hi_win], geo)
            axx.axvspan(xmin, b, color=COL_WIN, lw=0, zorder=0)
        if lo_lose:
            b2 = border(prv[lo_lose], lo_lose, geo)
            axx.axvspan(b2, xmax, color="#ececec", lw=0, zorder=0)
        return (border(hi_win, nxt[hi_win], geo) if hi_win else None,
                border(prv[lo_lose], lo_lose, geo) if lo_lose else None)

    b1, b2 = zones(ax, True, min(levels) / 1.4, max(levels) * 1.4)
    if b1:
        ax.text(np.sqrt(min(levels) * b1), 1.03, "MethylLlama most accurate (Holm p < 0.05)", ha="center",
                va="bottom", fontsize=6.6, color="#2c5d8a")
    if b2:
        ax.text(np.sqrt(b2 * max(levels) * 1.4), 1.03, "raw β\nbetter", ha="center", va="bottom", fontsize=6.2,
                color="#666666", linespacing=1.0)
    for r in REPS:
        col, ls, mk = STYLE[r]
        t = B_[B_.rep == r].sort_values("level")
        model = r in MODELS
        ax.fill_between(t.level, t.ci_lo, t.ci_hi, color=col, alpha=0.2 if model else 0.12, lw=0)
        ax.plot(t.level, t.ba, color=col, ls=ls, marker=mk, ms=3.0 if model else 2.4, lw=1.4 if model else 0.9,
                mfc=col if model else "white", label=NAMES[r], zorder=3 if model else 2)
    ax.axhline(1 / 91, color="#bbbbbb", lw=0.6, ls=":")
    log_axis(ax, levels)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("CpGs measured per profile, training and test (%)")
    ax.set_ylabel("Balanced accuracy (91 classes)")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.42, 0.0), fontsize=6.4, handlelength=1.8)
    panel_label(ax, "a", dx=-0.09)

    axb = fig.add_subplot(gs[0, 2])
    for vs, col in (("MethylGPT", COL_GPT), ("best raw", COL_RAW)):
        t = T[(T.scenario == "B") & (T.vs == vs)].sort_values("level")
        axb.fill_between(t.level, t.ci_lo, t.ci_hi, color=col, alpha=0.22, lw=0)
        axb.plot(t.level, t["diff"], color=col, lw=1.2, label=f"− {'best raw β' if vs == 'best raw' else vs}")
        sg = t.significant.to_numpy()
        axb.scatter(t.level[sg], t["diff"][sg], color=col, s=9, zorder=3)
        axb.scatter(t.level[~sg], t["diff"][~sg], facecolor="white", edgecolor=col, s=9, zorder=3, lw=0.8)
    axb.axhline(0, color="black", lw=0.6)
    log_axis(axb, levels)
    axb.set_ylim(-0.1, 0.45)
    axb.set_xlabel("CpGs measured (%)")
    axb.set_ylabel("MethylLlama minus …")
    axb.legend(frameon=False, loc="upper left", fontsize=6.3, handlelength=1.4)
    panel_label(axb, "b", dx=-0.3)

    axz = fig.add_subplot(gs[1, 0])
    z1, z2 = zones(axz, False, 47, 103)
    if z1 and z2:
        axz.text((z1 + z2) / 2, 0.981, "n.s.", ha="center", va="top", fontsize=6.2, color="#666666")
        axz.text(101.5, 0.981, "raw β better", ha="left", va="top", fontsize=5.8, color="#666666")
    hi = [lv for lv in levels if lv >= 50]
    for r in ("llama_cls", "raw_top5k", "raw_pca256"):
        col, ls, mk = STYLE[r]
        t = B_[(B_.rep == r) & (B_.level >= 50)].sort_values("level")
        axz.fill_between(t.level, t.ci_lo, t.ci_hi, color=col, alpha=0.2 if r in MODELS else 0.12, lw=0)
        axz.plot(t.level, t.ba, color=col, ls=ls, marker=mk, ms=3.0 if r in MODELS else 2.4, lw=1.3,
                 mfc=col if r in MODELS else "white", label=NAMES[r].replace("5,000 most variable CpGs", "top 5,000"))
    d = tb.loc[hi, "diff"]
    cross = None
    for x0, x1 in zip(hi[:-1], hi[1:]):
        if d[x0] > 0 >= d[x1]:
            cross = x0 + (x1 - x0) * d[x0] / (d[x0] - d[x1])
            break
    if cross is not None:
        axz.axvline(cross, color="#555555", lw=0.7, ls=":")
        axz.text(cross, 0.9755, f"≈{cross:.0f} % ", fontsize=6.0, color="#555555", va="top", ha="right")
    axz.set_ylim(0.92, 0.982)
    axz.set_xlim(103, 47)
    axz.set_xticks(hi, [f"{x:g}" for x in hi], fontsize=6.3)
    axz.set_xlabel("CpGs measured (%)")
    axz.set_ylabel("Balanced accuracy")
    axz.legend(frameon=False, loc="lower left", fontsize=6.0, handlelength=1.6)
    panel_label(axz, "c", dx=-0.3)

    axc = fig.add_subplot(gs[1, 1])
    lv = 5.0
    w = 0.38
    short = {"llama_cls": "Methyl-\nLlama", "gpt_cls": "Methyl-\nGPT", "raw_top5k": "Raw β\n5,000",
             "raw_pca256": "Raw β\nPCA-256"}
    for i, r in enumerate(REPS):
        col = STYLE[r][0]
        for k, (sc, fill) in enumerate((("A", False), ("B", True))):
            v = A[(A.scenario == sc) & (A.level == lv) & (A.rep == r)].iloc[0]
            x = i + (k - 0.5) * w
            axc.bar(x, v.ba, w * 0.92, color=col if fill else "white", edgecolor=col, lw=0.9,
                    hatch=None if fill else "////")
            axc.errorbar(x, v.ba, yerr=[[v.ba - v.ci_lo], [v.ci_hi - v.ba]], color="black", capsize=1.2,
                         elinewidth=0.6)
            axc.text(x, v.ci_hi + 0.02, f"{v.ba:.2f}", ha="center", va="bottom", fontsize=5.6, color=col, rotation=90)
    axc.set_xticks(range(len(REPS)), [short[r] for r in REPS], fontsize=5.8)
    for t, r in zip(axc.get_xticklabels(), REPS):
        t.set_color(STYLE[r][0])
    axc.set_ylim(0, 1.12)
    axc.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axc.set_ylabel("Balanced accuracy\n(test profiles at 5 % of CpGs)")
    axc.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc="white", ec="#555555", hatch="////"),
                        plt.Rectangle((0, 0), 1, 1, fc="#555555", ec="#555555")],
               labels=["classifier trained on complete profiles", "classifier trained on 5 % profiles"],
               frameon=False, fontsize=5.8, loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=1,
               handlelength=1.2)
    panel_label(axc, "d", dx=-0.3)

    axd = fig.add_subplot(gs[1, 2])
    st = pd.read_csv(a.stability)
    sd = [int(x) for x in a.seeds.split(",")]
    st = pd.concat([st, pd.DataFrame([r for m in ("llama", "gpt") for r in stability(str(SPARSE), m, list(HIGH), sd)])])
    st.to_csv(out / "data" / "fig_cns_sparse_B_stability.csv", index=False)
    for m, r in (("llama", "llama_cls"), ("gpt", "gpt_cls")):
        t = st[st.model == m].groupby("level").top1.agg(["mean", "min", "max"]).reset_index()
        col = STYLE[r][0]
        axd.errorbar(t.level, t["mean"], yerr=[t["mean"] - t["min"], t["max"] - t["mean"]], color=col, marker="o",
                     ms=2.6, lw=1.2, capsize=1.2, elinewidth=0.6, label=NAMES[r])
    log_axis(axd, sorted(st.level.unique()))
    axd.set_ylim(-0.02, 1.04)
    axd.set_xlabel("CpGs measured (%)")
    axd.set_ylabel("Sparse profile nearest to its\nown full profile (of 3,902)")
    axd.legend(frameon=False, loc="lower left", fontsize=6.0)
    panel_label(axd, "e", dx=-0.3)

    for ext in ("pdf", "png", "svg"):
        (out / ext).mkdir(parents=True, exist_ok=True)
        fig.savefig(out / ext / f"fig_cns_sparse_B.{ext}", bbox_inches="tight", dpi=400)
    print("saved fig_cns_sparse_B")


if __name__ == "__main__":
    main()
