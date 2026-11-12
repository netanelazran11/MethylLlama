#!/usr/bin/env python3
"""
figS_crossview.py -- Supplementary Fig. S1 of the manuscript (file name figS2_crossview.pdf, as in the
manuscript sources).

All panels use the complete held-out pretraining partition (16,912 profiles) and every saved
view draw; nothing is subsampled here.
  a  Cosine similarity between CLS representations of two views: matched views of the same
     profile (all 16,912 per draw) and all unmatched pairs (16,912 x 16,911 per draw), for the
     overlapping and disjoint constructions; every draw is drawn (3 + 3), they coincide.
  b  Retrieval of the matched view among all 16,912 candidates, top-1 and top-10, for
     overlapping and disjoint views (3 draws each) and the pattern-only control (1 draw, the only
     one computed); dashed lines are chance (k / 16,912).
  c  Observed versus predicted beta at all withheld CpGs of all 16,912 profiles (200 x 200 histogram,
     logarithmic colour scale).
Inputs: results/pretrained_cls/view_consistency_heldout_full/{summary.json, per_run/*.npz}
        results/pretrained_cls/reconstruction_withheld/{reconstruction_withheld_summary.json, obs_pred_hist2d.npz}
Usage:  python scripts/figures/figS_crossview.py
Output: figures/paper/figS2_crossview.{pdf,png,svg} + _provenance.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_style import COL_GPT, COL_LLAMA, apply_style, panel_label, save  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, NullFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
VC = ROOT / "results/pretrained_cls/view_consistency_heldout_full"
OUT = ROOT / "figures/paper/figS2_crossview"
GREY = "#8a8a8a"
COLS = {"overlap": COL_LLAMA, "disjoint": COL_GPT, "pattern_only": GREY}
NAMES = {"overlap": "Overlapping\nviews", "disjoint": "Disjoint\nviews", "pattern_only": "Pattern-only\ncontrol"}


def main():
    apply_style()
    s = json.load(open(VC / "summary.json"))
    runs = {c: [np.load(VC / f"per_run/{c}_seed{k}.npz") for k in s["conditions"][c]["seeds"]] for c in COLS}
    n = int(s["n_profiles"])
    assert all(len(z["all_matched_cos"]) == n and int(z["all_unmatched_hist"].sum()) == n * (n - 1)
               for zs in runs.values() for z in zs), "every run must cover all profiles and all unmatched pairs"

    fig = plt.figure(figsize=(7.2, 2.7))
    gs = GridSpec(1, 3, width_ratios=[1.2, 1.0, 1.0], wspace=0.72, left=0.07, right=0.99, bottom=0.24, top=0.92)

    # ---- a: cumulative distribution of cosine distance, all profiles, all pairs, all draws --------
    ax = fig.add_subplot(gs[0, 0])
    edges = runs["overlap"][0]["hist_bins"]          # 0.01-wide bins in cosine similarity, -1 ... 1
    for c, ls in (("overlap", "-"), ("disjoint", (0, (3, 2)))):
        for z in runs[c]:
            h = z["all_unmatched_hist"].astype(float)
            # fraction of unmatched pairs with cosine >= edge, evaluated at distance 1 - edge (exact on bin edges)
            frac_ge = np.cumsum(h[::-1])[::-1] / h.sum()
            d = 1 - edges[:-1]
            keep = d >= 0.01
            ax.plot(d[keep][::-1], frac_ge[keep][::-1], color=GREY, lw=1.1, ls=ls, drawstyle="steps-post")
            m = np.sort(1 - z["all_matched_cos"].astype(float))
            ax.plot(m, np.arange(1, len(m) + 1) / len(m), color=COLS[c], lw=1.2, ls=ls)
    ax.set_xscale("log")
    ax.set_xlim(5e-4, 2)
    ax.set_ylim(0, 1.01)
    ax.xaxis.set_major_locator(FixedLocator([1e-3, 1e-2, 1e-1, 1]))
    ax.set_xticklabels(["0.001", "0.01", "0.1", "1"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Cosine distance between CLS representations\n(1 − cosine similarity)")
    ax.set_ylabel("Cumulative fraction")
    handles = [Line2D([], [], color=COL_LLAMA, lw=1.3, label="Same profile,\noverlapping views"),
               Line2D([], [], color=COL_GPT, lw=1.3, ls=(0, (3, 2)), label="Same profile,\ndisjoint views"),
               Line2D([], [], color=GREY, lw=1.3, label="Different profiles")]
    ax.legend(handles=handles, frameon=False, loc="center", bbox_to_anchor=(0.645, 0.66), fontsize=6.4,
              handlelength=1.2, labelspacing=0.45, handletextpad=0.35)
    panel_label(ax, "a", dx=-0.17)

    # ---- b: retrieval among all candidates, horizontal bars ----------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    conds = ["overlap", "disjoint", "pattern_only"]
    names = {"overlap": "Overlapping views", "disjoint": "Disjoint views", "pattern_only": "Pattern-only control"}
    vals = {}
    for i, c in enumerate(conds):
        t1 = np.array([100 * (z["all_rank"] == 0).mean() for z in runs[c]])
        t10 = np.array([100 * (z["all_rank"] < 10).mean() for z in runs[c]])
        vals[c] = (t1, t10)
        y = len(conds) - 1 - i
        for dy, v, alpha in ((0.17, t1, 1.0), (-0.17, t10, 0.4)):
            ax.barh(y + dy, v.mean(), height=0.3, color=COLS[c], alpha=alpha, edgecolor="none")
            if len(v) > 1:
                ax.hlines(y + dy, v.min(), v.max(), color="black", lw=0.8)
            mv = v.mean()  # same rounding as the text; never shown as 100 unless it is 100
            if round(mv, 1) >= 99.95 and mv < 100:
                lab = f"{np.floor(mv * 100) / 100:.2f}"
            elif mv < 1:
                lab = f"{mv:.2f}"
            else:
                lab = f"{mv:.1f}"
            ax.text(v.mean() + 2, y + dy, lab, va="center", fontsize=6.8, color="#333333")
    ax.set_yticks(range(len(conds)))
    ax.set_yticklabels([names[c] for c in conds[::-1]], fontsize=7.0)
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel(f"Matched view retrieved (%)\namong all {n:,} candidates")
    leg = [plt.Rectangle((0, 0), 1, 1, color="#555555", label="top-1"),
           plt.Rectangle((0, 0), 1, 1, color="#555555", alpha=0.4, label="top-10")]
    ax.legend(handles=leg, frameon=False, loc="lower right", fontsize=6.8, handlelength=1.0, borderaxespad=0.1)
    panel_label(ax, "b", dx=-0.62)

    # ---- c: observed vs predicted beta at withheld CpGs, every position of all profiles -----------
    ax = fig.add_subplot(gs[0, 2])
    RF = ROOT / "results/pretrained_cls/reconstruction_withheld"
    rs = json.load(open(RF / "reconstruction_withheld_summary.json"))
    hz = np.load(RF / "obs_pred_hist2d.npz")
    counts, hedges = hz["counts"].astype(float), hz["edges"]
    assert rs["n_profiles"] == n and not rs["subsampled"] and int(counts.sum()) == rs["n_withheld_positions_total"]
    from matplotlib.colors import LogNorm
    im = ax.pcolormesh(hedges, hedges, np.where(counts > 0, counts, np.nan).T, cmap="Blues",
                       norm=LogNorm(vmin=1, vmax=counts.max()), rasterized=True)
    ax.plot([0, 1], [0, 1], color="#555555", lw=0.7, ls=(0, (3, 2)))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel(r"Observed $\beta$ (withheld CpGs)")
    ax.set_ylabel(r"Predicted $\beta$")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Positions", fontsize=6.8); cb.ax.tick_params(labelsize=6.4)
    panel_label(ax, "c", dx=-0.3)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save(fig, str(OUT))
    prov = {"file": "figS2_crossview.pdf (Supplementary Fig. S1)",
            "population": f"all {n} held-out pretraining profiles; every saved view draw",
            "panel_a": {"draws": {c: s["conditions"][c]["seeds"] for c in ("overlap", "disjoint")},
                        "matched_pairs_per_draw": n, "unmatched_pairs_per_draw": n * (n - 1),
                        "matched_cos_mean": {c: s["conditions"][c]["pools"]["all"]["matched_cos"]["mean"] for c in ("overlap", "disjoint")},
                        "unmatched_cos_mean": {c: s["conditions"][c]["pools"]["all"]["unmatched_cos"]["mean"] for c in ("overlap", "disjoint")}},
            "panel_b": {c: {"draws": s["conditions"][c]["seeds"],
                            "top1_percent_per_draw": [round(100 * float((z["all_rank"] == 0).mean()), 4) for z in runs[c]],
                            "top10_percent_per_draw": [round(100 * float((z["all_rank"] < 10).mean()), 4) for z in runs[c]]}
                        for c in conds},
            "chance_percent": {"top1": 100 / n, "top10": 1000 / n},
            "panel_c": {"source": "results/pretrained_cls/reconstruction_withheld/obs_pred_hist2d.npz",
                        "n_profiles": rs["n_profiles"], "n_positions": rs["n_withheld_positions_total"],
                        "pearson_all_positions": rs["pearson_withheld_all_positions"], "bins": "200 x 200 on [0, 1]"}}
    Path(f"{OUT}_provenance.json").write_text(json.dumps(prov, indent=1))


if __name__ == "__main__":
    main()
