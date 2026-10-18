"""Supplementary Figure S1 -- exact MethylLlama/WCED architecture schematic.

The diagram is intentionally vector-native so that every label and connection
can be checked against the implementation. In particular, genomic RoPE acts on
the query and key projections inside attention and is never added to the input
token embedding.

Output: figures/paper/figS1_architecture.{pdf,png,svg}
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

sys.path.insert(0, str(Path(__file__).parent))
from common_style import COL_ACCENT, COL_GPT, COL_LLAMA, apply_style, save


REPO = Path(__file__).resolve().parents[2]
OUTDIR = REPO / "figures/paper"

INK = "#1f2937"
MUTED = "#667085"
BG_A = "#f6f2f8"
BG_B = "#eef4fb"
BG_C = "#f8f1f7"
BG_D = "#eef7f0"
TOKEN = "#d8d7f2"
BETA = "#f5c8a8"
ROPE = "#cce8d2"
CLS = "#b8dfb9"
WITHHELD = "#f5d6bd"


def rounded(ax, xy, w, h, face, edge="#8a94a6", radius=0.012, lw=0.9,
            z=1, alpha=1.0):
    patch = FancyBboxPatch(
        xy, w, h,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        facecolor=face, edgecolor=edge, linewidth=lw, zorder=z, alpha=alpha,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end, color=INK, lw=1.0, style="-|>", rad=0.0, z=4):
    patch = FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=8, linewidth=lw,
        color=color, connectionstyle=f"arc3,rad={rad}", zorder=z,
    )
    ax.add_patch(patch)
    return patch


def token_strip(ax, x, y, w, h, values, masked=None, hatch=False):
    masked = set(masked or [])
    n = len(values)
    cmap = plt.get_cmap("coolwarm")
    for i, val in enumerate(values):
        xi = x + i * w / n
        if i in masked:
            face = "#f7f7f7"
            rect = Rectangle((xi, y), w / n, h, facecolor=face,
                             edgecolor="#9aa1ad", linewidth=0.45,
                             hatch="////" if hatch else None, zorder=3)
        else:
            face = cmap(0.08 + 0.84 * val)
            rect = Rectangle((xi, y), w / n, h, facecolor=face,
                             edgecolor="white", linewidth=0.35, zorder=3)
        ax.add_patch(rect)
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="#556070",
                           linewidth=0.75, zorder=4))


def heading(ax, x, y, letter, title):
    ax.text(x, y, letter, ha="left", va="top", fontsize=10.0,
            fontweight="bold", color="black")
    ax.text(x + 0.022, y, title, ha="left", va="top", fontsize=9.0,
            fontweight="bold", color="black")


def main():
    apply_style()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 4.25))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Four high-level regions; the architecture panel is deliberately widest.
    panels = [
        (0.012, 0.025, 0.225, 0.95, BG_A),
        (0.247, 0.025, 0.305, 0.95, BG_B),
        (0.562, 0.025, 0.205, 0.95, BG_C),
        (0.777, 0.025, 0.211, 0.95, BG_D),
    ]
    for x, y, w, h, face in panels:
        rounded(ax, (x, y), w, h, face, edge="none", radius=0.018, lw=0)

    # a -- partial view and token construction
    heading(ax, 0.025, 0.95, "a", "Partial views")
    ax.text(0.125, 0.875, "one methylation profile", ha="center", va="center",
            fontsize=7.2, color=MUTED)
    vals = [0.10, 0.78, 0.30, 0.62, 0.18, 0.90, 0.48, 0.72, 0.25, 0.83, 0.37, 0.66]
    token_strip(ax, 0.038, 0.815, 0.174, 0.045, vals)
    # Arrows stop above the view labels so the two never overlap.
    arrow(ax, (0.125, 0.802), (0.075, 0.772), color=MUTED, rad=0.12)
    arrow(ax, (0.125, 0.802), (0.175, 0.772), color=MUTED, rad=-0.12)

    ax.text(0.072, 0.738, "view 1", ha="center", va="bottom", fontsize=7.0,
            fontweight="bold")
    ax.text(0.178, 0.738, "view 2", ha="center", va="bottom", fontsize=7.0,
            fontweight="bold")
    token_strip(ax, 0.032, 0.690, 0.086, 0.040, vals[:8], masked=[1, 4, 6], hatch=True)
    token_strip(ax, 0.132, 0.690, 0.086, 0.040, vals[4:], masked=[0, 3, 5], hatch=True)
    ax.text(0.125, 0.645, "independent 50% CpG views", ha="center",
            va="center", fontsize=7.0)

    ax.text(0.125, 0.555, "Input token", ha="center", va="center",
            fontsize=7.8, fontweight="bold")
    rounded(ax, (0.025, 0.465), 0.077, 0.060, TOKEN, radius=0.009)
    ax.text(0.0635, 0.495, "CpG\nidentity", ha="center", va="center",
            fontsize=6.8, linespacing=1.05)
    ax.text(0.111, 0.495, "+", ha="center", va="center", fontsize=10)
    rounded(ax, (0.120, 0.465), 0.088, 0.060, BETA, radius=0.009)
    ax.text(0.164, 0.495, "$\\beta$-value\nencoding", ha="center", va="center",
            fontsize=6.8, linespacing=1.05)
    arrow(ax, (0.125, 0.445), (0.125, 0.385), color=COL_LLAMA)
    rounded(ax, (0.063, 0.315), 0.124, 0.062, "#e8e6f5", radius=0.010)
    ax.text(0.125, 0.346, "CpG input token", ha="center", va="center",
            fontsize=7.2, fontweight="bold")
    ax.text(0.125, 0.245, "Genomic position is not\nadded to this token",
            ha="center", va="center", fontsize=6.8, color=MUTED, linespacing=1.15)

    # b -- shared transformer encoder, with RoPE attached only to Q and K.
    heading(ax, 0.260, 0.95, "b", "Shared encoder")
    ax.text(0.399, 0.884, "6 layers  |  hidden size 256  |  4 heads",
            ha="center", va="center", fontsize=7.0, color=MUTED)

    rounded(ax, (0.272, 0.735), 0.055, 0.050, CLS, radius=0.009)
    rounded(ax, (0.272, 0.650), 0.055, 0.050, CLS, radius=0.009)
    ax.text(0.2995, 0.760, r"[CLS]$_1$", ha="center", va="center", fontsize=6.8)
    ax.text(0.2995, 0.675, r"[CLS]$_2$", ha="center", va="center", fontsize=6.8)
    token_strip(ax, 0.337, 0.742, 0.105, 0.036, vals[:8])
    token_strip(ax, 0.337, 0.657, 0.105, 0.036, vals[4:])
    arrow(ax, (0.444, 0.760), (0.470, 0.730))
    arrow(ax, (0.444, 0.675), (0.470, 0.705))

    rounded(ax, (0.464, 0.615), 0.070, 0.150, "#cdd9ee", radius=0.012)
    ax.text(0.499, 0.690, "shared\nweights", ha="center", va="center",
            fontsize=7.2, fontweight="bold")
    arrow(ax, (0.499, 0.610), (0.499, 0.565))

    rounded(ax, (0.277, 0.255), 0.245, 0.300, "#ffffff", edge="#697386",
            radius=0.014, lw=0.9)
    ax.text(0.399, 0.530, "Transformer layer  x6", ha="center", va="center",
            fontsize=7.5, fontweight="bold")

    # Q/K/V projections and explicit RoPE routes.
    for x, lab, face in [(0.300, "Q", "#f2c7cd"), (0.370, "K", "#d6d0ef"),
                         (0.440, "V", "#cae3e6")]:
        rounded(ax, (x, 0.305), 0.045, 0.048, face, radius=0.008)
        ax.text(x + 0.0225, 0.329, lab, ha="center", va="center",
                fontsize=7.5, fontweight="bold")
    rounded(ax, (0.319, 0.380), 0.137, 0.060, ROPE, radius=0.009)
    ax.text(0.3875, 0.410, "Genomic RoPE\n(Q and K only)", ha="center", va="center",
            fontsize=6.8, fontweight="bold", linespacing=1.0)
    arrow(ax, (0.3225, 0.355), (0.342, 0.380), color=COL_ACCENT)
    arrow(ax, (0.3925, 0.355), (0.411, 0.380), color=COL_ACCENT)
    arrow(ax, (0.3875, 0.440), (0.3875, 0.462), color=INK, lw=0.85)
    arrow(ax, (0.4625, 0.355), (0.438, 0.462), color=INK, lw=0.85)
    rounded(ax, (0.344, 0.462), 0.110, 0.050, "#f5d2a5", radius=0.009)
    ax.text(0.399, 0.487, "multi-head attention", ha="center", va="center",
            fontsize=6.9)
    ax.text(0.3475, 0.285, "rotates Q, K", ha="center", va="top",
            fontsize=6.8, color=MUTED)
    ax.text(0.4625, 0.285, "V unchanged", ha="center", va="top",
            fontsize=6.8, color=MUTED)

    # Encoder outputs used by both objectives.
    arrow(ax, (0.534, 0.705), (0.570, 0.705), color=COL_LLAMA)
    arrow(ax, (0.534, 0.665), (0.570, 0.665), color=COL_LLAMA)

    # c -- objectives
    heading(ax, 0.575, 0.95, "c", "Objectives")
    ax.text(0.665, 0.855, "Reconstruction", ha="center", va="center",
            fontsize=7.6, fontweight="bold")
    rounded(ax, (0.578, 0.755), 0.050, 0.048, CLS, radius=0.008)
    rounded(ax, (0.578, 0.690), 0.050, 0.048, CLS, radius=0.008)
    ax.text(0.603, 0.779, r"CLS$_1$", ha="center", va="center", fontsize=6.8)
    ax.text(0.603, 0.714, r"CLS$_2$", ha="center", va="center", fontsize=6.8)
    rounded(ax, (0.646, 0.707), 0.067, 0.082, "#d9cee8", radius=0.010)
    ax.text(0.6795, 0.748, "shared\ndecoder", ha="center", va="center",
            fontsize=6.8, linespacing=1.05)
    arrow(ax, (0.630, 0.779), (0.646, 0.765))
    arrow(ax, (0.630, 0.714), (0.646, 0.730))
    arrow(ax, (0.714, 0.748), (0.745, 0.748), color=COL_GPT)
    token_strip(ax, 0.720, 0.674, 0.034, 0.050, vals[:5], masked=[0, 2, 4], hatch=True)
    ax.text(0.665, 0.635, "predict CpGs withheld\nfrom each input view",
            ha="center", va="center", fontsize=6.8, linespacing=1.1)

    ax.plot([0.580, 0.750], [0.585, 0.585], color="#d4c8d3", lw=0.8)
    ax.text(0.665, 0.545, "Cross-view contrast", ha="center", va="center",
            fontsize=7.6, fontweight="bold")
    rounded(ax, (0.586, 0.455), 0.060, 0.050, "#e2d4eb", radius=0.008)
    rounded(ax, (0.586, 0.375), 0.060, 0.050, "#e2d4eb", radius=0.008)
    ax.text(0.616, 0.480, "head 1", ha="center", va="center", fontsize=6.8)
    ax.text(0.616, 0.400, "head 2", ha="center", va="center", fontsize=6.8)
    rounded(ax, (0.680, 0.458), 0.040, 0.045, "#9b6ec4", radius=0.008)
    rounded(ax, (0.680, 0.378), 0.040, 0.045, "#9b6ec4", radius=0.008)
    ax.text(0.700, 0.480, r"$z_1$", ha="center", va="center", fontsize=7.0,
            color="white", fontweight="bold")
    ax.text(0.700, 0.400, r"$z_2$", ha="center", va="center", fontsize=7.0,
            color="white", fontweight="bold")
    arrow(ax, (0.648, 0.480), (0.678, 0.480), color="#7d4ba0")
    arrow(ax, (0.648, 0.400), (0.678, 0.400), color="#7d4ba0")
    arrow(ax, (0.700, 0.455), (0.700, 0.425), color=COL_ACCENT, style="<->")
    ax.text(0.665, 0.322, "align views of the same profile;\nseparate different profiles",
            ha="center", va="center", fontsize=6.8, linespacing=1.1)
    ax.text(0.665, 0.210, r"$\mathcal{L}_{recon}+0.05\,\mathcal{L}_{InfoNCE}$",
            ha="center", va="center", fontsize=6.4)

    # d -- reusable encoder outputs
    heading(ax, 0.790, 0.95, "d", "Downstream reuse")
    rounded(ax, (0.812, 0.710), 0.140, 0.105, "#cbd8ed", radius=0.012)
    ax.text(0.882, 0.763, "pretrained\nMethylLlama encoder", ha="center",
            va="center", fontsize=7.3, fontweight="bold", linespacing=1.05)
    arrow(ax, (0.882, 0.700), (0.882, 0.620), color=COL_LLAMA)
    rounded(ax, (0.806, 0.515), 0.152, 0.090, "#e5f2e6", radius=0.012)
    ax.text(0.882, 0.560, "global CLS representation", ha="center", va="center",
            fontsize=7.1, fontweight="bold")
    arrow(ax, (0.882, 0.505), (0.835, 0.405), color=COL_ACCENT, rad=0.12)
    arrow(ax, (0.882, 0.505), (0.930, 0.405), color=COL_ACCENT, rad=-0.12)
    rounded(ax, (0.791, 0.285), 0.087, 0.105, "#ffffff", radius=0.010)
    rounded(ax, (0.887, 0.285), 0.087, 0.105, "#ffffff", radius=0.010)
    ax.text(0.8345, 0.338, "age\nfine-tuning", ha="center", va="center",
            fontsize=7.0, linespacing=1.05)
    ax.text(0.9305, 0.338, "tissue/study\nprobes", ha="center", va="center",
            fontsize=6.8, linespacing=1.05)

    save(fig, str(OUTDIR / "figS1_architecture"))
    plt.close(fig)

    provenance = {
        "figure": "Supplementary Figure S1",
        "type": "vector schematic; no measured data",
        "verified_architecture": {
            "input_token": "cpg_scale * cpg_embed(cpg_ids) + ScaleAdapt(beta)",
            "genomic_rope": "applied to query and key projections inside attention only",
            "encoder": "6 Transformer layers; hidden size 256; 4 attention heads",
            "objectives": "withheld-CpG reconstruction plus symmetric cross-view InfoNCE",
            "loss": "L_recon + 0.05 * L_InfoNCE",
        },
        "implementation_reference": "bmfm_methylation/llama/model.py:300,503",
        "status": "scientifically corrected; no positional encoding is added to the input token",
    }
    (OUTDIR / "figS1_architecture_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
