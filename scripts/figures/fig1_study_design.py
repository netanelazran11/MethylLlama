"""
Figure 1 -- Study design and the WCED pretraining objective.

Rebuilt to match what Results section 1 actually argues. That section is
titled "Study design and self-supervised pretraining" and spends two of its
three paragraphs on populations and partitions -- which cohort supports which
claim -- ending with "This separation is important." Earlier versions of this
figure were model-architecture diagrams and showed none of that, which is why
they read as dense and off-topic.

  a  Study-design flow. The pretraining corpus and its fixed partition, then
     the two downstream evaluation branches: the original AltuMAge split used
     ONLY for frozen-representation probes, and the filtered, deduplicated
     cohort used for every supervised comparison. Each branch is annotated
     with the figure it supports.
  b  The WCED objective, compactly: two views of one profile, one shared
     encoder, and the two losses applied to the same CLS representation.

Transformer internals and token construction are implementation detail and
live in Methods; nothing in Results section 1 depends on them.

Counts verified: 169,120 x 49,156 (data header, SLURM job 45888987);
80/10/10 -> 135,296 / 16,912 / 16,912; AltuMAge 10,988 x 21,368 splitting
7,416/1,308/2,264; after removing 328 age outliers and 71 duplicates, 10,589
splitting 7,177/1,263/2,149.

Usage:  python scripts/figures/fig1_study_design.py
Output: figures/paper/fig1_study_design_wced.{pdf,png,svg}
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common_style import COL_ACCENT, COL_GPT, COL_LLAMA, apply_style, panel_label, save
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.gridspec import GridSpec

OUTDIR = Path(__file__).resolve().parents[2] / "figures/paper"

V1, V2 = "#c2557a", "#3a6acc"
INK, MUTE = "#243044", "#7c8797"
FILL_DATA, FILL_MODEL, FILL_USE = "#fdf3e3", "#e7edf9", "#f3f5f8"


def box(ax, x, y, w, h, title, sub=None, fc=FILL_DATA, ec="#c9d1de",
        fs=7.4, subfs=6.9, tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.004,rounding_size=0.014",
                                facecolor=fc, edgecolor=ec, linewidth=0.9, zorder=3))
    if sub:
        ax.text(x + w / 2, y + h - 0.013, title, ha="center", va="top",
                fontsize=fs, color=tc, zorder=4, fontweight="bold")
        ax.text(x + w / 2, y + 0.013, sub, ha="center", va="bottom",
                fontsize=subfs, color=MUTE, zorder=4, linespacing=1.35)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=fs, color=tc, zorder=4, linespacing=1.35,
                fontweight="bold")


def arr(ax, x1, y1, x2, y2, color="#6b7686", lw=1.0, ms=8):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=ms, color=color, linewidth=lw,
                                 zorder=2, shrinkA=0, shrinkB=0))


def strip(ax, x, y, w, h, n, color, on=None):
    cw = w / n
    for i in range(n):
        lit = True if on is None else on[i]
        ax.add_patch(Rectangle((x + i * cw, y), cw * 0.8, h,
                               facecolor=color if lit else "#dfe3ea",
                               edgecolor="white", linewidth=0.25, zorder=3))


def main():
    apply_style()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7.2, 6.2))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.34, 0.66], hspace=0.05,
                  left=0.02, right=0.98, top=0.97, bottom=0.02)
    axA = fig.add_subplot(gs[0]); axA.set_xlim(0, 1); axA.set_ylim(0, 1); axA.axis("off")
    axB = fig.add_subplot(gs[1]); axB.set_xlim(0, 1); axB.set_ylim(0, 1); axB.axis("off")

    # ---------- Panel a: study-design flow ----------
    box(axA, 0.29, 0.878, 0.42, 0.107, "Pretraining corpus",
        "169,120 profiles $\\times$ 49,156 CpG sites\nno phenotype supervision")
    arr(axA, 0.50, 0.878, 0.50, 0.845)
    axA.text(0.715, 0.861, "80/10/10, fixed seed", ha="left", va="center",
             fontsize=6.9, color=MUTE)

    for x0, lab, n in [(0.150, "train", "135,296"), (0.395, "validation", "16,912"),
                       (0.640, "held out", "16,912")]:
        held = lab == "held out"
        box(axA, x0, 0.752, 0.21, 0.073, lab, n,
            fc="#fff6e8" if held else FILL_DATA,
            ec=COL_GPT if held else "#c9d1de", fs=7.2)
    axA.plot([0.255, 0.745], [0.845, 0.845], color="#6b7686", lw=1.0, zorder=2)
    for x in (0.255, 0.500, 0.745):
        arr(axA, x, 0.845, x, 0.825)
    arr(axA, 0.855, 0.789, 0.885, 0.789, color=COL_GPT)
    axA.text(0.892, 0.789, "never seen in pretraining;\nreconstruction eval (Fig. 2c,d)",
             ha="left", va="center", fontsize=6.9, color=COL_GPT, linespacing=1.3)

    arr(axA, 0.255, 0.752, 0.255, 0.718)
    axA.plot([0.255, 0.500], [0.718, 0.718], color="#6b7686", lw=1.0, zorder=2)
    arr(axA, 0.500, 0.718, 0.500, 0.692)
    box(axA, 0.275, 0.596, 0.45, 0.096, "WCED self-supervised pretraining",
        "partial-view reconstruction  $+$  cross-view contrastive",
        fc=FILL_MODEL, ec=COL_LLAMA, fs=7.6)
    arr(axA, 0.50, 0.596, 0.50, 0.562)
    box(axA, 0.300, 0.466, 0.40, 0.096, "Pretrained MethylLlama encoder",
        "6 blocks $\\cdot$ 256-d $\\cdot$ genomic-rank RoPE",
        fc=FILL_MODEL, ec=COL_LLAMA)

    arr(axA, 0.50, 0.466, 0.50, 0.438)
    axA.text(0.5, 0.428, "AltuMAge collection  $\\cdot$  10,988 profiles $\\times$ 21,368 CpGs",
             ha="center", va="center", fontsize=6.9, color=MUTE)
    axA.plot([0.235, 0.765], [0.412, 0.412], color="#6b7686", lw=1.0, zorder=2)
    arr(axA, 0.235, 0.412, 0.235, 0.393)
    arr(axA, 0.765, 0.412, 0.765, 0.393)

    box(axA, 0.020, 0.275, 0.43, 0.118, "Original split, unfiltered",
        "7,416 train / 1,308 valid / 2,264 test", fc=FILL_DATA, ec="#c9d1de")
    arr(axA, 0.235, 0.275, 0.235, 0.252)
    box(axA, 0.020, 0.090, 0.43, 0.162, "Frozen-representation probes",
        "age, tissue and study identity;\nleave-one-dataset-out transfer\n(Fig. 4)",
        fc=FILL_USE, ec=COL_ACCENT)

    box(axA, 0.550, 0.275, 0.43, 0.118, "Filtered and deduplicated",
        "$-$328 age outliers, $-$71 duplicates\n$\\rightarrow$ 10,589 profiles",
        fc=FILL_DATA, ec="#c9d1de")
    arr(axA, 0.765, 0.275, 0.765, 0.252)
    box(axA, 0.550, 0.090, 0.43, 0.162, "Supervised age benchmark",
        "5 folds: 7,177 train / 1,263 valid\nfixed test set: 2,149 profiles\n"
        "vs MethylGPT and ElasticNet (Fig. 3)",
        fc=FILL_USE, ec=COL_LLAMA)

    axA.text(0.5, 0.036,
             "frozen probes use the 2,264-profile split; every supervised comparison "
             "uses the same 2,149 profiles",
             ha="center", va="center", fontsize=6.9, color=MUTE, style="italic")
    panel_label(axA, "a", dx=0.02, dy=0.972)

    # ---------- Panel b: the WCED objective ----------
    rng = np.random.default_rng(3)
    m = rng.random(30) < 0.5
    strip(axB, 0.030, 0.480, 0.125, 0.105, 30, "#5a6472")
    axB.text(0.092, 0.440, "one profile", ha="center", va="top",
             fontsize=6.9, color=MUTE)
    arr(axB, 0.162, 0.533, 0.200, 0.533)

    strip(axB, 0.205, 0.630, 0.125, 0.085, 30, V1, on=m)
    strip(axB, 0.205, 0.420, 0.125, 0.085, 30, V2, on=~m)
    axB.text(0.267, 0.740, "two 50% CpG views", ha="center", va="bottom",
             fontsize=6.9, color=MUTE)
    arr(axB, 0.336, 0.672, 0.378, 0.610)
    arr(axB, 0.336, 0.462, 0.378, 0.525)

    box(axB, 0.383, 0.420, 0.130, 0.295, "Shared\nencoder", fc=FILL_MODEL,
        ec=COL_LLAMA, fs=7.4)
    arr(axB, 0.513, 0.568, 0.552, 0.568)
    for y, colr, lab in [(0.600, V1, "$\\mathrm{CLS}_1$"),
                          (0.480, V2, "$\\mathrm{CLS}_2$")]:
        axB.add_patch(FancyBboxPatch((0.556, y), 0.086, 0.070,
                                     boxstyle="round,pad=0.003,rounding_size=0.012",
                                     facecolor=colr, edgecolor=colr, zorder=3))
        axB.text(0.599, y + 0.035, lab, ha="center", va="center", fontsize=6.9,
                 color="white", zorder=4)

    arr(axB, 0.646, 0.638, 0.700, 0.735, color=COL_ACCENT)
    arr(axB, 0.646, 0.512, 0.700, 0.400, color=COL_GPT)
    box(axB, 0.705, 0.635, 0.275, 0.240, "Contrastive (InfoNCE)",
        "the two CLS agree for one profile,\nand differ across profiles",
        fc="#eafaf0", ec=COL_ACCENT, fs=7.2)
    box(axB, 0.705, 0.250, 0.275, 0.240, "Reconstruction",
        "predict the CpGs withheld\nfrom each view",
        fc="#fdf0e6", ec=COL_GPT, fs=7.2)
    axB.text(0.842, 0.150, "$\\mathcal{L} = \\mathcal{L}_{\\mathrm{recon}} + "
                            "0.05\\,\\mathcal{L}_{\\mathrm{InfoNCE}}$",
             ha="center", va="center", fontsize=8.2, color=INK)
    axB.text(0.33, 0.215, "both losses act on the\nsame CLS representation",
             ha="center", va="center", fontsize=6.9, color=MUTE, style="italic",
             linespacing=1.35)
    panel_label(axB, "b", dx=0.02, dy=0.95)

    save(fig, str(OUTDIR / "fig1_study_design_wced"))

    prov = {
        "figure_type": "schematic; no plotted data",
        "rationale": "matches Results section 1, which is principally about which "
                      "population supports which claim",
        "counts": {
            "pretraining": "169,120 x 49,156 (data header, SLURM job 45888987)",
            "pretrain_partition": "80/10/10 fixed seed -> 135,296 / 16,912 / 16,912",
            "altumage": "10,988 x 21,368; original split 7,416 / 1,308 / 2,264",
            "filtered": "-328 age outliers, -71 duplicates -> 10,589",
            "folds": "7,177 train / 1,263 valid; fixed test 2,149",
            "architecture": "6 blocks, 256-d, genomic-rank RoPE; lambda=0.05",
        },
    }
    (OUTDIR / "fig1_provenance.json").write_text(json.dumps(prov, indent=2))


if __name__ == "__main__":
    main()
