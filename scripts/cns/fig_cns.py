"""
fig_cns.py  (CNS task, stage 2 - figures from existing results)
===============================================================
Draws the CNS figure panels from the tidy CSVs written by fig_cns_data.py (nothing is recomputed here).

  fig_cns_main   c  balanced accuracy, MethylLlama vs MethylGPT (full / 5-shot / 3-shot), 95% bootstrap CI;
                    raw-data probes as faint reference ticks
                 b  reference set: profiles and classes per family (families with >= MIN_FAM profiles; rest pooled)
                 d  per-family balanced accuracy (mean per-class recall), MethylLlama vs MethylGPT, same row order as b
  fig_cns_s1     per-class recall, MethylGPT (x) vs MethylLlama (y), marker area ~ class size

Outputs  outputs/cns/figures/{pdf,png,svg}/
  python scripts/cns/fig_cns.py
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

MIN_FAM = 40
SETTINGS = ["full", "5-shot", "3-shot"]
SETTING_LABEL = {"full": "All training\nprofiles", "5-shot": "5 per class", "3-shot": "3 per class"}
RAW = {"raw_all": ("Raw, all CpGs", "o"), "raw_top5k": ("Raw, 5k CpGs", "s"), "raw_pca256": ("Raw, PCA-256", "D")}
COL_RAW = "#9a9a9a"


def save(fig, out, stem):
    for ext in ("pdf", "png", "svg"):
        (out / ext).mkdir(parents=True, exist_ok=True)
        fig.savefig(out / ext / f"{stem}.{ext}", bbox_inches="tight", dpi=300)
    print("saved", stem)


def family_table(d):
    """Families with >= MIN_FAM profiles, the rest pooled as 'Other'; per-class rows from s1 give the pooled BA."""
    s1, fam = d["s1"], d["b"]
    big = fam[fam.n_profiles >= MIN_FAM].family.tolist()
    s1 = s1.assign(fam2=np.where(s1.family.isin(big), s1.family, "Other"))
    t = s1.groupby("fam2").agg(n_classes=("class", "size"), n_profiles=("n_profiles", "sum"),
                               n_lt20=("n_profiles", lambda x: int((x < 20).sum())),
                               ba_llama=("recall_llama", "mean"), ba_gpt=("recall_gpt", "mean"))
    n_other = int((~fam.family.isin(big)).sum())
    t = t.rename(index={"Other": f"Other ({n_other} families)"})
    order = t.drop(index=t.index[t.index.str.startswith("Other")]).sort_values("n_profiles").index.tolist()
    return t.loc[[i for i in t.index if i.startswith("Other")] + order]


def panel_c(ax, d):
    acc = d["c"].set_index(["rep", "setting"])
    dif = d["dc"].set_index("setting")
    x = np.arange(len(SETTINGS))
    w = 0.34
    for off, rep, col, lab in ((-w / 2, "llama_cls", COL_LLAMA, "MethylLlama"), (w / 2, "gpt_cls", COL_GPT, "MethylGPT")):
        v = np.array([acc.loc[(rep, s), "ba"] for s in SETTINGS])
        lo = v - np.array([acc.loc[(rep, s), "ci_lo"] for s in SETTINGS])
        hi = np.array([acc.loc[(rep, s), "ci_hi"] for s in SETTINGS]) - v
        ax.bar(x + off, v, w, color=col, label=lab, zorder=2)
        ax.errorbar(x + off, v, yerr=[lo, hi], fmt="none", ecolor="black", elinewidth=0.7, capsize=1.6, zorder=3)
    for rep, (lab, mk) in RAW.items():
        v = [acc.loc[(rep, s), "ba"] for s in SETTINGS]
        ax.scatter(x + 0.30 + 0.07 * list(RAW).index(rep), v, marker=mk, s=9, facecolor="white", edgecolor=COL_RAW,
                   linewidth=0.7, zorder=4, label=lab)
    for i, s in enumerate(SETTINGS):
        r = dif.loc[s]
        top = acc.loc[("llama_cls", s), "ci_hi"]
        ax.text(i, top + 0.035, f"+{r['diff']:.2f}\n[{r.ci_lo:.2f}, {r.ci_hi:.2f}]", ha="center", va="bottom",
                fontsize=6.8, linespacing=1.0)
    ax.set_xticks(x, [SETTING_LABEL[s] for s in SETTINGS])
    ax.set_ylim(0.5, 1.12)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_ylabel("Balanced accuracy (91 classes)")
    ax.set_xlabel("Training profiles per class for the linear probe")
    h, l = ax.get_legend_handles_labels()
    order = [l.index(k) for k in ("MethylLlama", "MethylGPT")] + [l.index(v[0]) for v in RAW.values()]
    ax.legend([h[i] for i in order], [l[i] for i in order], frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0),
              handlelength=1.2, borderaxespad=0)


def panel_b(ax, t):
    y = np.arange(len(t))
    ax.barh(y, t.n_profiles, color="#b8c4d8", height=0.7)
    for i, (_, r) in enumerate(t.iterrows()):
        lab = f"{r.n_classes:.0f} cl." + (f" ({r.n_lt20:.0f} <20)" if r.n_lt20 else "")
        ax.text(r.n_profiles + 6, i, lab, va="center", fontsize=6.8)
    ax.set_yticks(y, t.index)
    ax.set_xlabel("Reference profiles")
    ax.set_xlim(0, t.n_profiles.max() * 1.45)
    ax.set_ylim(-0.6, len(t) - 0.4)


def panel_d(ax, t):
    y = np.arange(len(t))
    ax.hlines(y, t.ba_gpt, t.ba_llama, color="#c8c8c8", linewidth=1.2, zorder=1)
    ax.scatter(t.ba_gpt, y, s=14, color=COL_GPT, zorder=2, label="MethylGPT")
    ax.scatter(t.ba_llama, y, s=14, color=COL_LLAMA, zorder=3, label="MethylLlama")
    ax.set_yticks(y, [""] * len(t))
    ax.set_xlim(0.6, 1.02)
    ax.set_ylim(-0.6, len(t) - 0.4)
    ax.set_xlabel("Balanced accuracy within family\n(mean per-class recall, all training profiles)")
    ax.legend(frameon=False, loc="lower left", handletextpad=0.2, borderaxespad=0.2)


def fig_main(d, out):
    t = family_table(d)
    fig = plt.figure(figsize=(7.2, 6.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.55], width_ratios=[1.0, 1.0], hspace=0.42, wspace=0.08)
    axc = fig.add_subplot(gs[0, :])
    panel_c(axc, d)
    axb = fig.add_subplot(gs[1, 0])
    panel_b(axb, t)
    axd = fig.add_subplot(gs[1, 1])
    panel_d(axd, t)
    panel_label(axc, "a", dx=-0.07)
    panel_label(axb, "b", dx=-0.22)
    panel_label(axd, "c", dx=-0.04)
    save(fig, out, "fig_cns_main")
    plt.close(fig)


def fig_s1(d, out):
    s1 = d["s1"]
    fig, ax = plt.subplots(figsize=(3.5, 3.3))
    better = s1.recall_llama > s1.recall_gpt
    worse = s1.recall_llama < s1.recall_gpt
    for m, col, lab in ((better, COL_LLAMA, f"MethylLlama higher ({better.sum()})"),
                        (worse, COL_GPT, f"MethylGPT higher ({worse.sum()})"),
                        (~better & ~worse, "#8a8a8a", f"Equal ({(~better & ~worse).sum()})")):
        ax.scatter(s1.recall_gpt[m], s1.recall_llama[m], s=s1.n_profiles[m] * 0.35, color=col, alpha=0.7,
                   edgecolor="white", linewidth=0.4)
        ax.scatter([], [], s=14, color=col, label=lab)
    ax.plot([0.4, 1.05], [0.4, 1.05], color="#888888", linewidth=0.7, linestyle="--", zorder=0)
    lo = 0.45
    g = s1.assign(g=s1.recall_llama - s1.recall_gpt).sort_values("g")
    for (_, r), off in zip(g.iloc[[0, -1, -2]].iterrows(), ((6, -10), (-6, 8), (-6, 8))):
        ax.annotate(r["class"], (r.recall_gpt, r.recall_llama), xytext=off, textcoords="offset points", fontsize=6.8,
                    ha="left" if off[0] > 0 else "right", arrowprops=dict(arrowstyle="-", lw=0.5, color="#555555"))
    ax.set_xlim(lo, 1.02)
    ax.set_ylim(lo, 1.02)
    ax.set_aspect("equal")
    ax.set_xlabel("Per-class recall, MethylGPT")
    ax.set_ylabel("Per-class recall, MethylLlama")
    ax.legend(frameon=False, loc="lower right", handletextpad=0.2,
              title="Marker area ~ class size", title_fontsize=6.8)
    save(fig, out, "fig_cns_s1_per_class")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(REPO / "outputs/cns/figures"))
    a = p.parse_args()
    out = Path(a.root)
    dd = out / "data"
    d = {"b": pd.read_csv(dd / "panel_b_families.csv"), "c": pd.read_csv(dd / "panel_c_accuracy.csv"),
         "dc": pd.read_csv(dd / "panel_c_difference.csv"), "s1": pd.read_csv(dd / "s1_per_class.csv")}
    apply_style()
    fig_main(d, out)
    fig_s1(d, out)


if __name__ == "__main__":
    main()
