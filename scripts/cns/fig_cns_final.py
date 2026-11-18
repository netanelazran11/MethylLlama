"""
fig_cns_final.py  (CNS task - the two main-text figures)
=======================================================
Fig. 5  the frozen representation, no classifier
  a  UMAP of MethylLlama CLS for the 3,902 Capper profiles, 14 tumour categories (fig_cns_landscape.py)
  b  ependymal tumours by anatomical compartment; medulloblastoma by molecular subgroup (re-embedded per family)
  c  each model's full-profile map kept fixed; the same profiles seen at 5 % of their CpGs placed onto it
  d  share of profiles whose 10 nearest full-profile neighbours are mostly of their own category (solid) and share of
     those neighbours of their own class (dashed), against % of CpGs; mean and range over 3 mask seeds
Fig. 6  classification, classifier trained and tested at the same share of CpGs (fig_cns_sparse_B.py)
  a  balanced accuracy 0.5-100 %, zones: MethylLlama significantly most accurate / n.s. / raw beta significantly better
  b  the 50-100 % range on a linear axis, where the curves cross
  c  at 5 % of CpGs: classifier trained on complete profiles against trained on 5 % profiles

Reads the data written by fig_cns_landscape.py (coords, neighbours) and fig_cns_sparse_B.py (accuracy, tests); refits
only the two full-profile UMAPs of panel 5a/5c (same settings and seed as fig_cns_landscape.py).
Outputs  outputs/cns/figures/final/Fig5_cns_representation.*, Fig6_cns_sparse_input.*
  python scripts/cns/fig_cns_final.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig_cns_landscape as L  # noqa: E402
import fig_cns_sparse_B as S  # noqa: E402
from common_style import COL_GPT, COL_LLAMA, apply_style, panel_label  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

REPO = L.REPO
FIG = REPO / "outputs/cns/figures"
OUT = FIG / "final"
NAME = {"llama": "MethylLlama", "gpt": "MethylGPT"}
COL = {"llama": COL_LLAMA, "gpt": COL_GPT}


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight", dpi=400)
    print("saved", stem)


def frame(ax):
    L.clean(ax)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color("#d0d0d0")
        sp.set_linewidth(0.6)


def fig5():
    import anndata as ad
    obs = ad.read_h5ad(REPO / "outputs/cns/data/cns_capper.h5ad", backed="r").obs
    ids = obs.index.astype(str).to_numpy()
    cls = obs.methylation_class.astype(str).to_numpy()
    pre = obs.prefix_group.astype(str).to_numpy()
    cat = np.array([L.HGNET.get(c) or L.CATEGORY[f] for c, f in zip(cls, pre)])
    full = {k: L.aligned(REPO / f"outputs/cns/cls_methyl{n}", ids) for k, n in (("llama", "llama"), ("gpt", "gpt"))}
    sc = {k: StandardScaler().fit(v) for k, v in full.items()}
    maps = {k: L.umap_fit(sc[k].transform(v)) for k, v in full.items()}
    sparse5 = {k: maps[k].transform(sc[k].transform(L.aligned(REPO / f"outputs/cns/sparse/emb_{k}/L5_s0", ids)))
               for k in full}
    coords = pd.read_csv(FIG / "data/fig_cns_landscape_coords.csv")
    nb = pd.read_csv(FIG / "data/fig_cns_landscape_neighbours.csv")

    apply_style()
    fig = plt.figure(figsize=(7.2, 7.4))
    G = fig.add_gridspec(2, 1, height_ratios=[1.35, 1], hspace=0.16)
    top = G[0].subgridspec(2, 2, width_ratios=[1.85, 1], hspace=0.22, wspace=0.04)

    # a: landscape
    ax = fig.add_subplot(top[:, 0])
    xy = maps["llama"].embedding_
    L.scatter_cat(ax, xy, cat, s=4.0)
    span, lo = np.ptp(xy, 0).max(), xy.min(0)
    for c in L.COLORS:
        ax.annotate(L.SHORT.get(c, c), L.anchor(xy[cat == c], 0.04 * span), lo + span * np.array(L.LABEL_AT[c]),
                    fontsize=6.6, color=L.COLORS[c], fontweight="bold", ha="center", va="center",
                    path_effects=L.HALO, arrowprops=dict(arrowstyle="-", lw=0.5, color=L.COLORS[c], shrinkA=2,
                                                         shrinkB=1.5))
    ax.set_xlim(lo[0] - 0.17 * span, lo[0] + 1.13 * span)
    ax.set_ylim(lo[1] - 0.05 * span, lo[1] + 1.08 * span)
    ax.set_aspect("equal", adjustable="box")
    L.clean(ax)
    full_row = nb[(nb.model == "llama") & (nb.level == 100)].iloc[0]
    gpt_row = nb[(nb.model == "gpt") & (nb.level == 100)].iloc[0]
    ax.set_title("MethylLlama: 3,902 profiles, 91 methylation classes", fontsize=8.0, loc="left", pad=14)
    ax.text(0.0, 1.01, f"10 nearest neighbours of the same class: {full_row.class_knn:.0%} "
                       f"(MethylGPT {gpt_row.class_knn:.0%})", transform=ax.transAxes, fontsize=6.6,
            color="#555555", va="bottom")
    panel_label(ax, "a", dx=-0.01)

    # b: two families re-embedded
    axe = fig.add_subplot(top[0, 1])
    e = coords[coords.panel == "b_epn"]
    xe, ce = e[["x", "y"]].to_numpy(), e.category.astype(str).to_numpy()
    for c in L.EPN_SITE:
        m = ce == c
        axe.scatter(*xe[m].T, s=5, c=L.EPN_COL[L.EPN_SITE[c]], lw=0, alpha=0.85,
                    marker="o" if c.startswith("EPN") else "^", rasterized=True)
    masks = {c: ce == c for c in L.EPN_SITE if c.startswith("EPN")}
    masks["SE"] = np.array([c.startswith("SUBEPN") for c in ce])
    sp_, lo_ = L.callouts(axe, xe, masks, {**L.EPN_LAB, "SE": "subependymoma\n(all sites)"},
                          {**{c: L.EPN_COL[L.EPN_SITE[c]] for c in L.EPN_SITE}, "SE": "#555555"}, L.EPN_AT, fs=6.0)
    axe.set_xlim(lo_[0] - 0.12 * sp_, lo_[0] + 1.2 * sp_)
    axe.set_ylim(lo_[1] - 0.13 * sp_, lo_[1] + 0.8 * sp_)
    axe.set_title("Ependymal tumours, by site", fontsize=7.4, loc="left")
    axe.legend(handles=[plt.Line2D([], [], marker="o", ls="", ms=3.2, color=v, label=L.EPN_SITE_NAME[k])
                        for k, v in L.EPN_COL.items()] +
               [plt.Line2D([], [], marker="^", ls="", ms=3.2, color="#777777", label="subependymoma")],
               frameon=False, fontsize=6.0, loc="upper left", bbox_to_anchor=(1.0, 1.0), handletextpad=0.1,
               borderaxespad=0)
    axm = fig.add_subplot(top[1, 1])
    mb = coords[coords.panel == "b_mb"]
    xm, cm = mb[["x", "y"]].to_numpy(), mb.category.astype(str).to_numpy()
    for c, col in L.MB_COL.items():
        axm.scatter(*xm[cm == c].T, s=5, c=col, lw=0, alpha=0.85, rasterized=True)
    sp_, lo_ = L.callouts(axm, xm, {c: cm == c for c in L.MB_COL}, L.MB_LAB, L.MB_COL, L.MB_AT, fs=6.0)
    axm.set_xlim(lo_[0] - 0.2 * sp_, lo_[0] + 1.08 * sp_)
    axm.set_ylim(lo_[1] - 0.13 * sp_, lo_[1] + 1.06 * sp_)
    axm.set_title("Medulloblastoma, by subgroup", fontsize=7.4, loc="left")
    for x in (axe, axm):
        x.set_aspect("equal", adjustable="box")
        frame(x)
    panel_label(axe, "b", dx=-0.04)

    # c: complete vs 5 % on each model's fixed map
    bot = G[1].subgridspec(1, 2, width_ratios=[1.25, 1], wspace=0.28)
    cg = bot[0].subgridspec(2, 2, hspace=0.12, wspace=0.04)
    for i, k in enumerate(("llama", "gpt")):
        base = maps[k].embedding_
        lim = None
        for j, (lab, pts) in enumerate((("complete profile", base), ("5 % of CpGs", sparse5[k]))):
            x = fig.add_subplot(cg[i, j])
            if j:
                x.scatter(*base.T, s=1.2, c="#ebebeb", lw=0, rasterized=True)
            L.scatter_cat(x, pts, cat, s=1.8)
            if lim is None:
                x.set_aspect("equal", adjustable="datalim")
                x.margins(0.04)
                fig.canvas.draw()
                lim = x.get_xlim(), x.get_ylim()
            else:
                x.set_xlim(lim[0])
                x.set_ylim(lim[1])
                x.set_aspect("equal", adjustable="box")
            L.clean(x)
            r = nb[(nb.model == k) & (nb.level == (100 if j == 0 else 5)) & (nb.seed == 0)].iloc[0]
            x.text(0.5, -0.02, f"{r.category_majority:.0%} near own category", transform=x.transAxes, fontsize=6.3,
                   va="top", ha="center", color=COL[k])
            if i == 0:
                x.set_title(lab, fontsize=7.0)
            if j == 0:
                x.set_ylabel(NAME[k], fontsize=7.4, color=COL[k])
                x.yaxis.label.set_visible(True)
                if i == 0:
                    panel_label(x, "c", dx=-0.12)

    # d: neighbourhood retention
    axd = fig.add_subplot(bot[1])
    for k in ("llama", "gpt"):
        for meas, ls, mk in (("category_majority", "-", "o"), ("class_knn", "--", "s")):
            t = nb[nb.model == k].groupby("level")[meas].agg(["mean", "min", "max"]).reset_index()
            axd.fill_between(t.level, t["min"], t["max"], color=COL[k], alpha=0.18, lw=0)
            axd.plot(t.level, t["mean"], ls=ls, marker=mk, ms=2.6, lw=1.2, color=COL[k],
                     mfc=COL[k] if meas == "category_majority" else "white")
    lv = sorted(nb.level.unique())
    axd.set_xscale("log")
    axd.set_xticks(lv, [f"{v:g}" for v in lv])
    axd.minorticks_off()
    axd.set_xlim(max(lv) * 1.3, min(lv) / 1.3)
    axd.set_ylim(0, 1.02)
    axd.set_xlabel("CpGs measured per profile (%)")
    axd.set_ylabel("Share of profiles / neighbours")
    axd.legend(handles=[plt.Line2D([], [], color=COL_LLAMA, lw=1.4, label="MethylLlama"),
                        plt.Line2D([], [], color=COL_GPT, lw=1.4, label="MethylGPT"),
                        plt.Line2D([], [], color="#555555", marker="o", ms=3, lw=1, label="own category (majority of 10)"),
                        plt.Line2D([], [], color="#555555", ls="--", marker="s", ms=3, mfc="white", lw=1,
                                   label="own class (share of 10)")],
               frameon=False, fontsize=5.9, loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=2, handlelength=1.8,
               columnspacing=0.8, borderaxespad=0)
    panel_label(axd, "d", dx=-0.22, dy=1.1)
    save(fig, "Fig5_cns_representation")
    plt.close(fig)


def fig6():
    A = pd.read_csv(FIG / "data/fig_cns_sparse_B_accuracy.csv")
    T = pd.read_csv(FIG / "data/fig_cns_sparse_B_tests.csv")
    B_ = A[A.scenario == "B"]
    levels = sorted(B_.level.unique())
    tb = T[(T.scenario == "B") & (T.vs == "best raw")].set_index("level")
    win = [v for v in levels if tb.loc[v, "significant"] and tb.loc[v, "diff"] > 0]
    lose = [v for v in levels if tb.loc[v, "significant"] and tb.loc[v, "diff"] < 0]
    nxt = {v: levels[i + 1] for i, v in enumerate(levels[:-1])}
    prv = {v: levels[i - 1] for i, v in enumerate(levels) if i}
    b1 = np.sqrt(max(win) * nxt[max(win)])
    b2 = np.sqrt(prv[min(lose)] * min(lose))

    apply_style()
    fig = plt.figure(figsize=(7.2, 2.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.75, 1, 1.15], wspace=0.42)

    ax = fig.add_subplot(gs[0])
    ax.axvspan(min(levels) / 1.4, b1, color=S.COL_WIN, lw=0, zorder=0)
    ax.axvspan(b2, max(levels) * 1.4, color="#ececec", lw=0, zorder=0)
    ax.text(np.sqrt(min(levels) / 1.4 * b1), 1.03, "MethylLlama most accurate (Holm p < 0.05)", ha="center",
            va="bottom", fontsize=6.3, color="#2c5d8a")
    ax.text(np.sqrt(b2 * max(levels) * 1.4), 1.03, "raw β\nbetter", ha="center", va="bottom", fontsize=5.8,
            color="#666666", linespacing=1.0)
    for r in S.REPS:
        col, ls, mk = S.STYLE[r]
        t = B_[B_.rep == r].sort_values("level")
        model = r in S.MODELS
        ax.fill_between(t.level, t.ci_lo, t.ci_hi, color=col, alpha=0.2 if model else 0.12, lw=0)
        ax.plot(t.level, t.ba, color=col, ls=ls, marker=mk, ms=2.8 if model else 2.2, lw=1.4 if model else 0.9,
                mfc=col if model else "white", label=S.NAMES[r], zorder=3 if model else 2)
    S.log_axis(ax, levels)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("CpGs measured per profile, training and test (%)")
    ax.set_ylabel("Balanced accuracy (91 classes)")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.45, 0.0), fontsize=6.0, handlelength=1.8)
    panel_label(ax, "a", dx=-0.13)

    axz = fig.add_subplot(gs[1])
    axz.axvspan(47, (max(win) + nxt[max(win)]) / 2, color=S.COL_WIN, lw=0, zorder=0)
    axz.axvspan((prv[min(lose)] + min(lose)) / 2, 103, color="#ececec", lw=0, zorder=0)
    for r in ("llama_cls", "raw_top5k", "raw_pca256"):
        col, ls, mk = S.STYLE[r]
        t = B_[(B_.rep == r) & (B_.level >= 50)].sort_values("level")
        axz.fill_between(t.level, t.ci_lo, t.ci_hi, color=col, alpha=0.2 if r in S.MODELS else 0.12, lw=0)
        axz.plot(t.level, t.ba, color=col, ls=ls, marker=mk, ms=2.8 if r in S.MODELS else 2.2, lw=1.3,
                 mfc=col if r in S.MODELS else "white")
    hi = [v for v in levels if v >= 50]
    d = tb.loc[hi, "diff"]
    for x0, x1 in zip(hi[:-1], hi[1:]):
        if d[x0] > 0 >= d[x1]:
            cross = x0 + (x1 - x0) * d[x0] / (d[x0] - d[x1])
            axz.axvline(cross, color="#555555", lw=0.7, ls=":")
            axz.text(cross, 0.9755, f"≈{cross:.0f} % ", fontsize=5.8, color="#555555", va="top", ha="right")
            break
    axz.text(((max(win) + nxt[max(win)]) / 2 + (prv[min(lose)] + min(lose)) / 2) / 2, 0.981, "n.s.", ha="center",
             va="top", fontsize=6.0, color="#666666")
    axz.set_ylim(0.92, 0.982)
    axz.set_xlim(103, 47)
    axz.set_xticks(hi, [f"{v:g}" for v in hi], fontsize=6.0)
    axz.set_xlabel("CpGs measured (%)")
    axz.set_ylabel("Balanced accuracy")
    panel_label(axz, "b", dx=-0.32)

    axc = fig.add_subplot(gs[2])
    w = 0.38
    short = {"llama_cls": "Methyl-\nLlama", "gpt_cls": "Methyl-\nGPT", "raw_top5k": "Raw β\n5,000",
             "raw_pca256": "Raw β\nPCA-256"}
    for i, r in enumerate(S.REPS):
        col = S.STYLE[r][0]
        for k, (sc_, fill) in enumerate((("A", False), ("B", True))):
            v = A[(A.scenario == sc_) & (A.level == 5.0) & (A.rep == r)].iloc[0]
            x = i + (k - 0.5) * w
            axc.bar(x, v.ba, w * 0.92, color=col if fill else "white", edgecolor=col, lw=0.9,
                    hatch=None if fill else "////")
            axc.errorbar(x, v.ba, yerr=[[v.ba - v.ci_lo], [v.ci_hi - v.ba]], color="black", capsize=1.1,
                         elinewidth=0.6)
            axc.text(x, v.ci_hi + 0.02, f"{v.ba:.2f}", ha="center", va="bottom", fontsize=5.4, color=col, rotation=90)
    axc.set_xticks(range(len(S.REPS)), [short[r] for r in S.REPS], fontsize=5.8)
    for t, r in zip(axc.get_xticklabels(), S.REPS):
        t.set_color(S.STYLE[r][0])
    axc.set_ylim(0, 1.12)
    axc.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axc.set_ylabel("Balanced accuracy, test at 5 % of CpGs")
    axc.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc="white", ec="#555555", hatch="////"),
                        plt.Rectangle((0, 0), 1, 1, fc="#555555", ec="#555555")],
               labels=["trained on complete profiles", "trained on 5 % profiles"], frameon=False, fontsize=5.7,
               loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.2)
    panel_label(axc, "c", dx=-0.3)
    save(fig, "Fig6_cns_sparse_input")
    plt.close(fig)


if __name__ == "__main__":
    fig6()
    fig5()
