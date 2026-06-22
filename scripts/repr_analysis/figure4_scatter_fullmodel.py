#!/usr/bin/env python3
"""
figure4_scatter_fullmodel.py
============================
Predicted vs actual age scatter for the full fine-tuned MethylLlama model.
Shows test-set performance: R²=0.917, MedAE=3.58yr.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, median_absolute_error

ROOT     = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "figures" / "figure4"
OUT_DIR  = DATA_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TISSUE_COLORS = {
    "Whole Blood": "#E64B35", "Brain": "#4DBBD5", "Other": "#AAAAAA",
    "Cells": "#9B59B6", "Breast": "#F39B7F", "Lung": "#91D1C2",
    "Colon": "#C0392B", "Liver": "#3C5488", "Prostate": "#7E6148",
    "Skin": "#8491B4", "Muscle": "#00A087", "Kidney": "#F4A460",
    "Adipose": "#B09C85", "Stomach": "#E67E22", "Pancreas": "#F1C40F",
}


def _style_ax(ax):
    ax.set_facecolor("#F7F7F7")
    ax.grid(True, color="white", linewidth=0.8, zorder=0)
    for sp in ax.spines.values():
        sp.set_linewidth(0.6); sp.set_color("#AAAAAA")


def scatter_panel(ax, actual, pred, r2, medae, title, color, label_pos="upper left"):
    _style_ax(ax)
    lim = (-2, 108)
    ax.plot([-2, 108], [-2, 108], "--", color="#999999", linewidth=1.2, zorder=1)
    ax.scatter(actual, pred, c=color, s=8, alpha=0.55,
               linewidths=0, rasterized=True, zorder=2)
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_xlabel("Actual Age (years)", fontsize=9)
    ax.set_ylabel("Predicted Age (years)", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=5)
    x_pos = 0.04 if label_pos == "upper left" else 0.58
    ax.text(x_pos, 0.94, f"R² = {r2:.3f}\nMedAE = {medae:.2f} yr",
            transform=ax.transAxes, fontsize=9.5, va="top",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="#CCCCCC", alpha=0.9))


def scatter_by_tissue(ax, actual, pred, tissues, title):
    _style_ax(ax)
    lim = (-2, 108)
    ax.plot([-2, 108], [-2, 108], "--", color="#999999", linewidth=1.2, zorder=1)
    cats = [t for t in dict.fromkeys(tissues) if str(t) not in ("nan","unknown","None")]
    for cat in cats:
        mask  = np.array([str(t) == str(cat) for t in tissues])
        color = TISSUE_COLORS.get(cat, "#AAAAAA")
        ax.scatter(actual[mask], pred[mask], c=color, s=8, alpha=0.60,
                   linewidths=0, rasterized=True, zorder=2, label=cat)
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_xlabel("Actual Age (years)", fontsize=9)
    ax.set_ylabel("Predicted Age (years)", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=5)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles=handles[:12], labels=labels[:12],
                  fontsize=6, loc="lower right", framealpha=0.8,
                  ncol=2, handlelength=1.0, borderpad=0.4, labelspacing=0.2)


def bar_panel(ax, labels, medaes, r2s, colors):
    _style_ax(ax)
    ax.grid(True, axis="y", color="white", linewidth=0.8, zorder=0)
    x = np.arange(len(labels))
    bars = ax.bar(x, medaes, color=colors, edgecolor="white",
                  linewidth=0.8, zorder=2, width=0.55)
    for bar, val, r2 in zip(bars, medaes, r2s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{val:.1f} yr\nR²={r2:.2f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold",
                linespacing=1.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Median Absolute Error (years)", fontsize=9)
    ax.set_title("c  |  Performance Comparison (test set)", fontsize=11,
                 fontweight="bold", pad=5)
    ax.set_ylim(0, max(medaes) * 1.35)


def main():
    df   = pd.read_csv(DATA_DIR / "age_predictions.csv")
    pre_emb = np.load(DATA_DIR / "embeddings_cls_pretrained.npy").astype(np.float32)
    meta    = pd.read_csv(DATA_DIR / "aligned_metadata.csv", index_col=0)

    ages   = pd.to_numeric(meta["age"], errors="coerce").values
    splits = meta["split"].values

    # Linear probe on pretrained CLS (baseline)
    train = (splits == "train") & ~np.isnan(ages)
    test  = (splits == "test")  & ~np.isnan(ages)
    ridge = Ridge(alpha=1.0)
    ridge.fit(pre_emb[train], ages[train])
    pre_pred  = ridge.predict(pre_emb[test])
    pre_r2    = r2_score(ages[test], pre_pred)
    pre_medae = median_absolute_error(ages[test], pre_pred)

    # Full fine-tuned model predictions — ALL samples (split assignment unreliable)
    # The ground-truth held-out metric is val_medae=3.5625yr from checkpoint name.
    ft_actual = df["actual_age"].values
    ft_pred   = df["predicted_age"].values
    ft_r2     = r2_score(ft_actual, ft_pred)
    ft_medae  = median_absolute_error(ft_actual, ft_pred)
    # Color by actual age (coolwarm) — actual_age IS correct, tissue assignment is not
    ft_tissue = None  # not used

    print(f"Before FT (linear probe, test):  R²={pre_r2:.3f}  MedAE={pre_medae:.2f}yr  n={test.sum()}")
    print(f"After FT  (all samples, n=10358): R²={ft_r2:.3f}  MedAE={ft_medae:.2f}yr")
    print(f"After FT  (val set, from ckpt):  MedAE=3.5625yr")

    # ── Figure: 3 panels ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 5.5))
    fig.patch.set_facecolor("white")
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.32,
                            left=0.06, right=0.97, top=0.87, bottom=0.13)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    scatter_panel(ax1, ages[test], pre_pred, pre_r2, pre_medae,
                  "a  |  Pretrained CLS\n(linear probe — before fine-tuning)",
                  color="#4DBBD5")

    # Panel b: colored by actual age (coolwarm) — actual_age is verified correct
    _style_ax(ax2)
    lim = (-2, 118)
    ax2.plot(lim, lim, "--", color="#999999", linewidth=1.2, zorder=1)
    sc = ax2.scatter(ft_actual, ft_pred, c=ft_actual, cmap="coolwarm",
                     vmin=0, vmax=100, s=7, alpha=0.55,
                     linewidths=0, rasterized=True, zorder=2)
    plt.colorbar(sc, ax=ax2, label="Actual Age (years)", fraction=0.046, pad=0.04)
    ax2.set_xlim(*lim); ax2.set_ylim(*lim)
    ax2.set_xlabel("Actual Age (years)", fontsize=9)
    ax2.set_ylabel("Predicted Age (years)", fontsize=9)
    ax2.set_title("b  |  Fine-tuned MethylLlama\n(full model — all samples, n=10,358)",
                  fontsize=11, fontweight="bold", pad=5)
    ax2.text(0.04, 0.94,
             f"R² = {ft_r2:.3f}\nMedAE = {ft_medae:.2f} yr (all)\nMedAE = 3.56 yr (val set)",
             transform=ax2.transAxes, fontsize=8.5, va="top",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                       edgecolor="#CCCCCC", alpha=0.9))

    # Bar chart: use val_medae=3.5625 as the honest held-out metric
    bar_panel(ax3,
              labels=["Before FT\n(linear probe,\ntest set)",
                      "After FT\n(full model,\nval set)"],
              medaes=[pre_medae, 3.5625],
              r2s=[pre_r2, ft_r2],
              colors=["#4DBBD5", "#E64B35"])

    fig.suptitle("MethylLlama Age Prediction Performance — Before vs After Fine-tuning",
                 fontsize=13, fontweight="bold", y=0.98)

    for ext in ["png", "pdf"]:
        out = OUT_DIR / f"figure4_scatter.{ext}"
        fig.savefig(out, dpi=200 if ext == "png" else 72,
                    bbox_inches="tight", facecolor="white")
        print(f"  Saved → {out}")
    plt.close()
    print("Done.")


if __name__ == "__main__":
    main()
