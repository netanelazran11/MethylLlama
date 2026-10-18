"""
Supplementary figure -- view construction and the contrastive negative set.

Panels:
  A  Cross-view consistency under overlapping vs disjoint views, over several
     random view draws (matched and unmatched cosine similarity, mean +- SD
     across seeds). Answers whether the reported alignment survives when the
     two views share no CpG.
  B  Retrieval@k for both view constructions, with per-seed spread shaded.
  C  Similarity matrices for one seed under each construction, on a shared
     colour scale -- the visual counterpart of panel A.
  D  Pretraining pilot: reconstruction loss for the four view-design conditions
     (rendered only once the pilot runs exist).

Panels A-C need only the inference-only evaluation on a fixed subset of the
profile-level held-out pretraining partition
(results/pretrained_cls/view_design_heldout/, from run_view_design_heldout.sh).
Panel D additionally needs outputs/ablation_viewdesign/*/ from the pilot.
Missing inputs are skipped with a printed note rather than faked.

Usage:  python scripts/figures/figS_view_design.py
Output: figures/paper/figS_view_design.{pdf,png}
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common_style import COL_ENET, COL_GPT, COL_LLAMA, apply_style, panel_label, save
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

REPO = Path(__file__).resolve().parents[2]
VD = REPO / "results/pretrained_cls/view_design_heldout"
PILOT = REPO / "outputs/ablation_viewdesign"
OUTDIR = REPO / "figures/paper"

KS = [1, 5, 10, 20, 50, 100]
COND_COLOR = {"overlap": COL_LLAMA, "disjoint": COL_GPT}
COND_LABEL = {"overlap": "Overlapping views\n(canonical training setup)",
              "disjoint": "Disjoint views\n(no shared CpG)"}


def main():
    apply_style()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    per_seed_csv = VD / "view_design_per_seed.csv"
    if not per_seed_csv.exists():
        print(f"MISSING: {per_seed_csv}\n"
              f"Run scripts/evaluation/run_view_design_heldout.sh first, "
              f"then rsync results/pretrained_cls/view_design_heldout/ back.")
        return
    df = pd.read_csv(per_seed_csv)
    summary = json.loads((VD / "view_design_summary.json").read_text())
    n_seeds = int(df["seed"].nunique())
    print(f"Loaded {len(df)} rows, {n_seeds} seeds per condition")

    pilot_rows = []
    if PILOT.exists():
        for d in sorted(PILOT.glob("viewdesign-*")):
            js = d / "pilot_summary.json"
            if js.exists():
                pilot_rows.append(json.loads(js.read_text()))
    have_pilot = len(pilot_rows) > 0
    if not have_pilot:
        print("NOTE: pilot runs not found -- rendering panels a-c only.")

    n_prof = int(summary["n_profiles"])
    assert summary["split"] != "train"
    nrow = 2 if have_pilot else 1
    fig = plt.figure(figsize=(7.2, 5.8 if have_pilot else 3.1))
    gs = GridSpec(nrow, 3, figure=fig, hspace=0.65, wspace=0.5,
                  width_ratios=[0.95, 1.1, 1.25])

    # ── Panel A: matched / unmatched similarity, mean +- SD over seeds ───────
    # y-axis is not zero-anchored: the scientific point is that matched
    # similarity barely moves between constructions, which a 0-1 axis hides.
    axA = fig.add_subplot(gs[0, 0])
    width = 0.30
    for j, cond in enumerate(["overlap", "disjoint"]):
        sub = df[df.condition == cond]
        for i, metric in enumerate(["pos_cos", "neg_cos"]):
            x = i + (-0.22 if j == 0 else 0.22)
            m, s = sub[metric].mean(), sub[metric].std(ddof=1)
            axA.bar(x, m, width=width, color=COND_COLOR[cond],
                    alpha=1.0 if metric == "pos_cos" else 0.45,
                    label=COND_LABEL[cond].replace("\n", " ") if i == 0 else None)
            axA.errorbar(x, m, yerr=s, color="0.15", lw=0.8, capsize=2)
            axA.text(x, m + (0.016 if j == 0 else 0.004), f"{m:.3f}",
                     ha="center", va="bottom", fontsize=5.8,
                     color=COND_COLOR[cond])
    axA.set_xticks([0, 1])
    axA.set_xticklabels(["matched", "unmatched"])
    axA.set_ylabel("Cosine similarity")
    axA.set_ylim(0.25, 1.045)
    axA.text(0.5, -0.34, "bars: mean; error bars: SD across five draws",
             transform=axA.transAxes,
             ha="center", va="top", fontsize=5.8, color="0.45", style="italic")
    axA.set_yticks([0.3, 0.5, 0.7, 0.9, 1.0])
    panel_label(axA, "a", dx=-0.34)

    # ── Panel B: retrieval@k, per-seed spread ────────────────────────────────
    axB = fig.add_subplot(gs[0, 1])
    for cond in ["overlap", "disjoint"]:
        sub = df[df.condition == cond]
        ks = [k for k in KS if f"retrieval_at{k}" in sub.columns]
        cols = [f"retrieval_at{k}" for k in ks]
        mean = [sub[c].mean() for c in cols]
        lo = [sub[c].min() for c in cols]
        hi = [sub[c].max() for c in cols]
        axB.fill_between(ks, lo, hi, color=COND_COLOR[cond], alpha=0.25, linewidth=0)
        axB.plot(ks, mean, "o-", color=COND_COLOR[cond], ms=3.5, lw=1.2,
                 label=COND_LABEL[cond].split("\n")[0], zorder=3)
        yoff = -2 if cond == "overlap" else -12
        axB.annotate(f"{mean[0]*100:.1f}%", (ks[0], mean[0]), xytext=(4, yoff),
                     textcoords="offset points", fontsize=7.0,
                     color=COND_COLOR[cond], va="top")
    axB.plot(ks, [k / n_prof for k in ks], "--", color="#b03030", lw=0.9,
             label="chance")
    axB.set_xscale("log")
    axB.set_xticks(ks); axB.set_xticklabels([str(k) for k in ks]); axB.minorticks_off()
    axB.set_xlabel(f"k (of {n_prof:,} candidate profiles)")
    axB.set_ylabel("Correct partner within top-k")
    axB.set_ylim(0, 1.04)
    axB.legend(frameon=False, fontsize=6.2, loc="center",
               bbox_to_anchor=(0.66, 0.55), handlelength=1.4)
    panel_label(axB, "b", dx=-0.3)

    # ── Panel C: similarity matrices, shared colour scale ────────────────────
    # Few enough profiles that the diagonal is legible at print size.
    n_show = 22
    gsC = gs[0, 2].subgridspec(1, 2, wspace=0.18)
    rng = np.random.default_rng(0)
    sel = None
    ims, axes_c = [], []
    for j, cond in enumerate(["overlap", "disjoint"]):
        f = VD / f"simmatrix_{cond}_seed0.npy"
        ax = fig.add_subplot(gsC[0, j])
        axes_c.append(ax)
        if not f.exists():
            ax.axis("off"); continue
        sim = np.load(f)
        if sel is None:
            sel = np.sort(rng.choice(sim.shape[0], size=min(n_show, sim.shape[0]),
                                     replace=False))
        im = ax.imshow(sim[np.ix_(sel, sel)], cmap="magma", vmin=0, vmax=1,
                       interpolation="nearest")
        ims.append(im)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(COND_LABEL[cond].split("\n")[0], fontsize=7.2, pad=3)
        if j == 0:
            ax.set_ylabel("View 1", fontsize=7.2)
            panel_label(ax, "c", dx=-0.14)
        ax.set_xlabel("View 2", fontsize=7.2)
    if ims:
        cb = fig.colorbar(ims[0], ax=axes_c, location="bottom",
                          fraction=0.06, pad=0.16, aspect=30)
        cb.set_label("cosine similarity", fontsize=7.0)
        cb.ax.tick_params(labelsize=6.8)

    # ── Panel D: pilot reconstruction loss ───────────────────────────────────
    if have_pilot:
        axD = fig.add_subplot(gs[1, :])
        order = ["baseline", "disjoint", "sameview", "both"]
        rows = sorted(pilot_rows, key=lambda r: order.index(r.get("condition", "baseline")))
        xs = np.arange(len(rows))
        axD.bar(xs, [r["best_recon_loss"] for r in rows], width=0.55,
                color=[COL_LLAMA if r["condition"] == "baseline" else COL_ENET
                       for r in rows])
        for x, r in zip(xs, rows):
            axD.text(x, r["best_recon_loss"], f"{r['best_recon_loss']:.4f}",
                     ha="center", va="bottom", fontsize=6.8)
        axD.set_xticks(xs)
        axD.set_xticklabels([r["condition"] for r in rows])
        axD.set_ylabel("Best held-out reconstruction loss")
        panel_label(axD, "d", dx=-0.06)

    save(fig, str(OUTDIR / "figS_view_design"))

    prov = {
        "panels_a_c": str(VD.relative_to(REPO)),
        "n_seeds": n_seeds,
        "disjoint_minus_overlap": summary.get("disjoint_minus_overlap"),
        "panel_d": (str(PILOT.relative_to(REPO)) if have_pilot else "PENDING pilot runs"),
    }
    (OUTDIR / "figS_view_design_provenance.json").write_text(json.dumps(prov, indent=2))


if __name__ == "__main__":
    main()
