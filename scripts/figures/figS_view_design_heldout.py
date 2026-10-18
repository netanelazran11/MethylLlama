"""
figS_view_design_heldout.py -- Supplementary Fig. S2 from the HELD-OUT pretraining partition.

Same figure as figS_view_design.py (which is left unchanged and draws the AltuMAge run), with two
differences: the input is results/pretrained_cls/view_design_heldout/ (2,000 held-out
pretraining profiles, 5 view draws; run_view_design_heldout.sh, job 46246524), and panel a starts
at 0 because unmatched similarity on these profiles is ~0.3.

Output: figures/paper/figS_view_design_heldout.{pdf,png,svg}
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
              f"then rsync results/pretrained_cls/view_design/ back.")
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

    n_prof = 2000
    nrow = 2 if have_pilot else 1
    fig = plt.figure(figsize=(7.2, 5.8 if have_pilot else 3.1))
    gs = GridSpec(nrow, 3, figure=fig, hspace=0.65, wspace=0.5,
                  width_ratios=[0.95, 1.1, 1.25])

    # ── Panel A: matched / unmatched similarity, mean +- SD over seeds ───────
    # On held-out profiles unmatched similarity is ~0.3, so the axis starts at 0.
    axA = fig.add_subplot(gs[0, 0])
    width = 0.36
    for j, cond in enumerate(["overlap", "disjoint"]):
        sub = df[df.condition == cond]
        for i, metric in enumerate(["pos_cos", "neg_cos"]):
            x = i + (j - 0.5) * width
            m, s = sub[metric].mean(), sub[metric].std(ddof=1)
            axA.bar(x, m, width=width, color=COND_COLOR[cond],
                    alpha=1.0 if metric == "pos_cos" else 0.45,
                    label=COND_LABEL[cond].replace("\n", " ") if i == 0 else None)
            axA.errorbar(x, m, yerr=s, color="0.15", lw=0.8, capsize=2)
            axA.text(x, m + 0.015, f"{m:.3f}", ha="center", va="bottom", fontsize=6.0, rotation=90)
    axA.set_xticks([0, 1])
    axA.set_xticklabels(["matched", "unmatched"])
    axA.set_ylabel("Cosine similarity")
    axA.set_ylim(0, 1.12)
    axA.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    panel_label(axA, "a", dx=-0.34)

    # ── Panel B: top-1 retrieval, one point per view draw ────────────────────
    # Top-10 is 100% for both constructions on these profiles, so only top-1 is drawn.
    axB = fig.add_subplot(gs[0, 1])
    for j, cond in enumerate(["overlap", "disjoint"]):
        v = df[df.condition == cond]["retrieval_at1"].to_numpy() * 100
        axB.bar(j, v.mean(), width=0.6, color=COND_COLOR[cond], zorder=2)
        axB.vlines(j, v.min(), v.max(), color="0.15", lw=0.9, zorder=4)
        axB.scatter(np.full(len(v), j), v, s=5, color="white", edgecolors="0.15", linewidths=0.4, zorder=5)
        axB.text(j, v.max() + 1.5, f"{v.mean():.1f}%", ha="center", va="bottom", fontsize=6.8, color=COND_COLOR[cond])
    axB.axhline(100.0 / n_prof, color="#b03030", ls="--", lw=0.9, label=f"chance ({100.0 / n_prof:.2f}%)")
    axB.set_xticks([0, 1]); axB.set_xticklabels(["Overlapping\nviews", "Disjoint\nviews"])
    axB.set_ylabel(f"Top-1 retrieval among {n_prof:,} (%)")
    axB.set_ylim(0, 112)
    axB.legend(frameon=False, fontsize=6.8, loc="upper center", bbox_to_anchor=(0.5, -0.24), handlelength=1.4)
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

    save(fig, str(OUTDIR / "figS_view_design_heldout"))

    prov = {
        "panels_a_c": str(VD.relative_to(REPO)),
        "n_seeds": n_seeds,
        "disjoint_minus_overlap": summary.get("disjoint_minus_overlap"),
        "panel_d": (str(PILOT.relative_to(REPO)) if have_pilot else "PENDING pilot runs"),
    }
    (OUTDIR / "figS_view_design_heldout_provenance.json").write_text(json.dumps(prov, indent=2))


if __name__ == "__main__":
    main()
