"""
Scatter plot + error-by-age analysis for ElasticNet age baseline.

Generates 3 figures:
  1. Predicted vs actual age scatter (train / valid / test)
  2. MAE per age decade (0-10, 10-20, ..., 80-120)
  3. Absolute error distribution histogram (test set)
"""

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad",     required=True)
    parser.add_argument("--outdir",   required=True)
    parser.add_argument("--alpha",    type=float, default=0.01)
    parser.add_argument("--l1_ratio", type=float, default=0.5)
    parser.add_argument("--age_col",   default="age")
    parser.add_argument("--split_col", default="split")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load & split ───────────────────────────────────────────────────────
    logger.info(f"Loading {args.h5ad}")
    adata = sc.read_h5ad(args.h5ad)

    splits = adata.obs[args.split_col]

    def to_dense(X):
        if hasattr(X, "toarray"):
            X = X.toarray()
        return X.astype(np.float32)

    idx = {s: splits == s for s in ("train", "valid", "test")}
    X   = {s: to_dense(adata[idx[s]].X) for s in idx}
    y   = {s: adata[idx[s]].obs[args.age_col].values.astype(np.float32) for s in idx}

    # ── 2. Scale + fit ────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X["train"] = scaler.fit_transform(X["train"])
    X["valid"]  = scaler.transform(X["valid"])
    X["test"]   = scaler.transform(X["test"])

    logger.info(f"Fitting ElasticNet(alpha={args.alpha}, l1_ratio={args.l1_ratio})")
    model = ElasticNet(alpha=args.alpha, l1_ratio=args.l1_ratio,
                       max_iter=10000, tol=1e-4, random_state=42, selection="random")
    model.fit(X["train"], y["train"])

    pred = {s: model.predict(X[s]) for s in idx}
    err  = {s: np.abs(pred[s] - y[s]) for s in idx}
    logger.info(f"  test  MedAE={np.median(err['test']):.2f}yr  MAE={err['test'].mean():.2f}yr")

    COLORS = {"train": "#4C72B0", "valid": "#DD8452", "test": "#55A868"}
    ALPHA_PT = {"train": 0.15, "valid": 0.4, "test": 0.4}
    SIZE_PT  = {"train": 6,    "valid": 10,  "test": 10}

    # ── Figure 1: Scatter predicted vs actual ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 7))

    age_min = min(y[s].min() for s in idx)
    age_max = max(y[s].max() for s in idx)

    for s in ("train", "valid", "test"):
        ax.scatter(y[s], pred[s],
                   c=COLORS[s], s=SIZE_PT[s], alpha=ALPHA_PT[s],
                   label=f"{s}  (MedAE={np.median(err[s]):.2f}yr)")

    lims = [min(age_min, pred["test"].min()) - 2,
            max(age_max, pred["test"].max()) + 2]
    ax.plot(lims, lims, "k--", lw=1.2, label="Perfect prediction")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Actual age (years)", fontsize=13)
    ax.set_ylabel("Predicted age (years)", fontsize=13)
    ax.set_title(f"ElasticNet  α={args.alpha}  l1_ratio={args.l1_ratio}\n"
                 f"Predicted vs Actual Age", fontsize=13)
    ax.legend(fontsize=11)
    ax.set_aspect("equal")
    plt.tight_layout()
    fig.savefig(outdir / "scatter_predicted_vs_actual.png", dpi=150)
    plt.close(fig)
    logger.info("Saved scatter_predicted_vs_actual.png")

    # ── Figure 2: MAE by age decade ───────────────────────────────────────────
    bins   = list(range(0, 110, 10)) + [130]
    labels = [f"{b}-{b+10}" for b in range(0, 100, 10)] + ["100+"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x_pos = np.arange(len(labels))
    width = 0.28

    for i, s in enumerate(("train", "valid", "test")):
        mae_bins = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (y[s] >= lo) & (y[s] < hi)
            if mask.sum() > 0:
                mae_bins.append(err[s][mask].mean())
            else:
                mae_bins.append(np.nan)
        bars = ax.bar(x_pos + (i - 1) * width, mae_bins,
                      width=width, color=COLORS[s], label=s, alpha=0.85)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_xlabel("Age bin (years)", fontsize=12)
    ax.set_ylabel("MAE (years)", fontsize=12)
    ax.set_title(f"ElasticNet  α={args.alpha}  —  MAE by Age Decade", fontsize=13)
    ax.legend(fontsize=11)
    ax.axhline(np.median(err["test"]), color="red", linestyle="--",
               lw=1.2, label=f"test MedAE={np.median(err['test']):.2f}yr")
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(outdir / "mae_by_age_bin.png", dpi=150)
    plt.close(fig)
    logger.info("Saved mae_by_age_bin.png")

    # ── Figure 3: Absolute error histogram (test set) ─────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(err["test"], bins=50, color=COLORS["test"], edgecolor="white",
            alpha=0.85, label="test")
    ax.axvline(np.median(err["test"]), color="red", lw=2,
               label=f"MedAE = {np.median(err['test']):.2f}yr")
    ax.axvline(err["test"].mean(), color="orange", lw=2, linestyle="--",
               label=f"MAE = {err['test'].mean():.2f}yr")
    ax.set_xlabel("Absolute error (years)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"ElasticNet  α={args.alpha}  —  Test Error Distribution", fontsize=13)
    ax.legend(fontsize=11)
    plt.tight_layout()
    fig.savefig(outdir / "error_distribution_test.png", dpi=150)
    plt.close(fig)
    logger.info("Saved error_distribution_test.png")

    logger.info(f"All figures saved to {outdir}/")


if __name__ == "__main__":
    main()
