#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""W&B Analysis for BMFM Methylation Fine-tuning (Age Regression).

Downloads history from WandB and creates visualizations to analyze:
- Loss curves (train vs validation)
- MAE and MedAE curves (in years)
- MAE vs MedAE gap (outlier impact)
- R² curves
- Learning rate schedule
- Train-val gap (overfitting analysis)
- Convergence analysis

Usage:
    python scripts/utils/analyze_finetune.py

    # Or specify run directly:
    WANDB_RUN_ID=bvt444p3 python scripts/utils/analyze_finetune.py

Environment variables:
    WANDB_ENTITY: Your WandB entity (default: netanelazran11-hebrew-university-of-jerusalem)
    WANDB_PROJECT: Project name (default: finetune-llama-small)
    WANDB_RUN_ID: Specific run ID to analyze (optional, lists runs if not set)
    OUTDIR: Output directory for plots and CSVs
"""

import os
import json
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

# ==============================
# CONFIG (env overrides)
# ==============================
ENTITY  = os.getenv("WANDB_ENTITY",  "netanelazran11-hebrew-university-of-jerusalem")
PROJECT = os.getenv("WANDB_PROJECT", "finetune-llama-small")
RUN_ID  = os.getenv("WANDB_RUN_ID",  "8mjxsoez")   # V4b scratch             bvt444p3
                                                     # V4b warmstart:  1w1rk694
                                                     # V4b scratch:    8mjxsoez

# Output saved to wandb_analysis/finetune/<run_id>/ — separate folder per run
_BASE_OUTDIR = os.path.join(os.path.dirname(__file__), "..", "wandb_analysis", "finetune")
OUTDIR = os.path.abspath(os.path.expanduser(
    os.getenv("OUTDIR", os.path.join(_BASE_OUTDIR, RUN_ID))
))

WANDB_TIMEOUT = int(os.getenv("WANDB_TIMEOUT", "600"))

# Reference baselines (set to None to hide from plots)
RIDGE_BASELINE_MAE  = None
RIDGE_BASELINE_R2   = None
V1_BEST_MAE         = 6.81   # previous best (V1, mean pooling, epoch ~98)

# Metric names logged by MethylationAgeRegressorLlama
TRAIN_METRICS = [
    "train/loss_epoch",
    "train/loss",
    "train/mae",
    "train/medae",
    "train/r2",
]

VAL_METRICS = [
    "val/loss",
    "val/mae",
    "val/medae",
    "val/r2",
]

TEST_METRICS = [
    "test/mae",
    "test/medae",
    "test/r2",
]

# 4-group AdamW: head_decay, head_no_decay, enc_decay, enc_no_decay
LR_METRICS = [
    "lr-AdamW/pg1",
    "lr-AdamW/pg2",
    "lr-AdamW/pg3",
    "lr-AdamW/pg4",
    "lr-AdamW",
]
LR_LABELS = {
    "lr-AdamW/pg1": "head (decay)",
    "lr-AdamW/pg2": "head (no-decay)",
    "lr-AdamW/pg3": "encoder (decay)",
    "lr-AdamW/pg4": "encoder (no-decay)",
    "lr-AdamW":     "AdamW",
}


# ==============================
# Helpers
# ==============================

def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def fetch_runs(api: wandb.Api, entity: str, project: str) -> List[wandb.apis.public.Run]:
    path = f"{entity}/{project}"
    return list(api.runs(path=path))


def download_run_history(run: wandb.apis.public.Run) -> pd.DataFrame:
    print(f"Downloading history for run: {run.name} ({run.id})")
    try:
        hist = run.history(pandas=True, samples=50000)
    except Exception as e:
        print(f"Error downloading history: {e}")
        hist = pd.DataFrame()
    return hist


def get_epoch_data(df: pd.DataFrame) -> pd.DataFrame:
    if "epoch" not in df.columns:
        return df
    val_cols = [c for c in df.columns if c.startswith("val/")]
    if not val_cols:
        return df
    epoch_df = df.groupby("epoch").agg(
        lambda x: x.dropna().iloc[-1] if len(x.dropna()) > 0 else np.nan
    ).reset_index()
    return epoch_df


def find_best_checkpoint(df: pd.DataFrame) -> Dict[str, Any]:
    """Find best checkpoint by val/mae and also record best val/medae."""
    metric = "val/mae"
    if metric not in df.columns:
        for alt in ["val/mae_epoch"]:
            if alt in df.columns:
                metric = alt
                break
        else:
            return {}

    valid_df = df[df[metric].notna()].copy()
    if len(valid_df) == 0:
        return {}

    best_idx = valid_df[metric].idxmin()
    best_row = valid_df.loc[best_idx]

    result = {
        "best_mae":   best_row.get(metric),
        "best_epoch": best_row.get("epoch"),
        "best_step":  best_row.get("trainer/global_step", best_row.get("_step")),
    }

    for col in ["val/loss", "val/r2", "val/medae"]:
        if col in best_row and pd.notna(best_row[col]):
            key = col.replace("val/", "best_val_")
            result[key] = best_row[col]

    # Also find global best medae (might be at a different epoch)
    if "val/medae" in df.columns:
        med_df = df[df["val/medae"].notna()]
        if len(med_df) > 0:
            best_med_idx = med_df["val/medae"].idxmin()
            result["best_medae"]       = med_df.loc[best_med_idx, "val/medae"]
            result["best_medae_epoch"] = med_df.loc[best_med_idx, "epoch"]

    return result


def detect_early_stopping(df: pd.DataFrame, patience: int = 50) -> Dict[str, Any]:
    metric = "val/mae"
    if metric not in df.columns:
        return {"triggered": False}
    val_data = df[df[metric].notna()][metric]
    if len(val_data) == 0:
        return {"triggered": False}
    best_idx     = val_data.idxmin()
    best_pos     = list(val_data.index).index(best_idx)
    epochs_after = len(val_data) - best_pos - 1
    return {
        "triggered":         epochs_after >= patience,
        "best_at_position":  best_pos,
        "epochs_after_best": epochs_after,
        "total_epochs":      len(val_data),
    }


def compute_train_val_gap(df: pd.DataFrame) -> pd.DataFrame:
    epoch_df = get_epoch_data(df)
    train_col = next((c for c in ["train/mae", "train/mae_epoch"] if c in epoch_df.columns), None)
    val_col   = next((c for c in ["val/mae",   "val/mae_epoch"]   if c in epoch_df.columns), None)
    if train_col is None or val_col is None:
        return pd.DataFrame()
    gap_data = []
    for _, row in epoch_df.iterrows():
        if pd.notna(row.get(train_col)) and pd.notna(row.get(val_col)):
            gap_data.append({
                "epoch":     row.get("epoch", row.name),
                "train_mae": row[train_col],
                "val_mae":   row[val_col],
                "gap":       row[val_col] - row[train_col],
                "ratio":     row[val_col] / row[train_col] if row[train_col] > 0 else np.nan,
            })
    return pd.DataFrame(gap_data)


def _epoch_col(df):
    return "epoch" if "epoch" in df.columns else "_step"


# ==============================
# Plotting
# ==============================

def plot_loss_curves(df: pd.DataFrame, outdir: str, run_name: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ec = _epoch_col(df)

    ax = axes[0]
    for col, label, color in [
        ("train/loss", "Train Loss",  "blue"),
        ("val/loss",   "Val Loss",    "orange"),
    ]:
        if col in df.columns:
            data = df[df[col].notna()]
            if len(data):
                ax.plot(data[ec], data[col], label=label, alpha=0.8, color=color, linewidth=2)
    # Annotate best val loss
    if "val/loss" in df.columns:
        val_df = df[df["val/loss"].notna()]
        if len(val_df):
            bi = val_df["val/loss"].idxmin()
            ax.axvline(x=val_df.loc[bi, ec], color="green", linestyle="--", alpha=0.4)
            ax.annotate(f'Best: {val_df.loc[bi,"val/loss"]:.4f}\n@ epoch {val_df.loc[bi,ec]:.0f}',
                        xy=(val_df.loc[bi, ec], val_df.loc[bi, "val/loss"]),
                        xytext=(val_df.loc[bi, ec] + 3, val_df.loc[bi, "val/loss"] * 1.3),
                        fontsize=9, color="green", arrowprops=dict(arrowstyle="->", color="green", alpha=0.6))
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Loss Curves")
    ax.legend(); ax.grid(True, alpha=0.3)

    # Step-level training loss
    ax = axes[1]
    sc = "trainer/global_step" if "trainer/global_step" in df.columns else "_step"
    if "train/loss" in df.columns:
        data = df[df["train/loss"].notna()]
        if len(data):
            ax.plot(data[sc], data["train/loss"], alpha=0.2, color="blue", linewidth=0.5, label="Train Loss")
            if len(data) > 20:
                rolling = data["train/loss"].rolling(window=20, min_periods=1).mean()
                ax.plot(data[sc], rolling, color="darkblue", linewidth=1.5, alpha=0.9, label="Train Loss (MA-20)")
    ax.set_xlabel("Step"); ax.set_ylabel("Loss"); ax.set_title("Training Loss (step-level)")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.suptitle(f"Loss Analysis — {run_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(outdir, "loss_curves.png")
    plt.savefig(path, dpi=150); plt.close()
    return path


def plot_mae_curves(df: pd.DataFrame, outdir: str, run_name: str) -> str:
    """MAE + MedAE together — shows both average and median performance."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    ec = _epoch_col(df)

    # Left: MAE
    ax = axes[0]
    for col, label, color, lw in [
        ("train/mae", "Train MAE",  "blue",       1.5),
        ("val/mae",   "Val MAE",    "orange",      2.0),
    ]:
        if col in df.columns:
            data = df[df[col].notna()]
            if len(data):
                ax.plot(data[ec], data[col], label=label, alpha=0.85, color=color, linewidth=lw)

    ax.axhline(y=V1_BEST_MAE, color="purple", linestyle="--", alpha=0.7, linewidth=1.5,
               label=f"V1 best ({V1_BEST_MAE:.2f} yr)")
    if RIDGE_BASELINE_MAE is not None:
        ax.axhline(y=RIDGE_BASELINE_MAE, color="red", linestyle="--", alpha=0.5, linewidth=1.5,
                   label=f"Ridge ({RIDGE_BASELINE_MAE:.2f} yr)")

    if "val/mae" in df.columns:
        val_df = df[df["val/mae"].notna()]
        if len(val_df):
            bi = val_df["val/mae"].idxmin()
            bm, be = val_df.loc[bi, "val/mae"], val_df.loc[bi, ec]
            ax.annotate(f"Best Val MAE: {bm:.2f} yr\n@ epoch {be:.0f}",
                        xy=(be, bm), xytext=(be + 5, bm + 1.0), fontsize=10,
                        color="darkorange", fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color="darkorange", lw=1.5))

    ax.set_xlabel("Epoch", fontsize=12); ax.set_ylabel("MAE (years)", fontsize=12)
    ax.set_title("Mean Absolute Error (MAE)", fontsize=12)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Right: MedAE
    ax = axes[1]
    for col, label, color, lw in [
        ("train/medae", "Train MedAE", "steelblue", 1.5),
        ("val/medae",   "Val MedAE",   "darkorange", 2.0),
    ]:
        if col in df.columns:
            data = df[df[col].notna()]
            if len(data):
                ax.plot(data[ec], data[col], label=label, alpha=0.85, color=color, linewidth=lw)

    if RIDGE_BASELINE_MAE is not None:
        ax.axhline(y=RIDGE_BASELINE_MAE, color="red", linestyle="--", alpha=0.5, linewidth=1.5,
                   label=f"Ridge MAE ({RIDGE_BASELINE_MAE:.2f} yr)")

    if "val/medae" in df.columns:
        val_df = df[df["val/medae"].notna()]
        if len(val_df):
            bi = val_df["val/medae"].idxmin()
            bm, be = val_df.loc[bi, "val/medae"], val_df.loc[bi, ec]
            ax.annotate(f"Best Val MedAE: {bm:.2f} yr\n@ epoch {be:.0f}",
                        xy=(be, bm), xytext=(be + 5, bm + 0.5), fontsize=10,
                        color="darkorange", fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color="darkorange", lw=1.5))

    ax.set_xlabel("Epoch", fontsize=12); ax.set_ylabel("MedAE (years)", fontsize=12)
    ax.set_title("Median Absolute Error (MedAE — robust to outliers)", fontsize=12)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.suptitle(f"MAE & MedAE — {run_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(outdir, "mae_curves.png")
    plt.savefig(path, dpi=150); plt.close()
    return path


def plot_mae_medae_gap(df: pd.DataFrame, outdir: str, run_name: str) -> str:
    """MAE – MedAE gap over training: shows outlier impact on the mean error."""
    if "val/mae" not in df.columns or "val/medae" not in df.columns:
        return ""

    ec   = _epoch_col(df)
    epdf = get_epoch_data(df)
    m1   = "val/mae"
    m2   = "val/medae"

    rows = []
    for _, r in epdf.iterrows():
        if pd.notna(r.get(m1)) and pd.notna(r.get(m2)):
            rows.append({"epoch": r[ec], "mae": r[m1], "medae": r[m2], "gap": r[m1] - r[m2]})
    if not rows:
        return ""
    gdf = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: MAE and MedAE on same axis
    ax = axes[0]
    ax.plot(gdf["epoch"], gdf["mae"],   label="Val MAE",   color="orange",    linewidth=2)
    ax.plot(gdf["epoch"], gdf["medae"], label="Val MedAE", color="steelblue", linewidth=2)
    ax.fill_between(gdf["epoch"], gdf["medae"], gdf["mae"], alpha=0.15, color="red",
                    label="MAE–MedAE gap (outlier impact)")
    ax.axhline(y=V1_BEST_MAE, color="purple", linestyle="--", alpha=0.6, linewidth=1.5,
               label=f"V1 best MAE ({V1_BEST_MAE:.2f} yr)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Error (years)")
    ax.set_title("Val MAE vs Val MedAE")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Right: gap magnitude only
    ax = axes[1]
    ax.plot(gdf["epoch"], gdf["gap"], color="red", linewidth=2, alpha=0.85)
    ax.fill_between(gdf["epoch"], 0, gdf["gap"], alpha=0.2, color="red")
    ax.axhline(y=0,   color="gray",   linestyle="-",  alpha=0.3)
    ax.axhline(y=0.5, color="green",  linestyle="--", alpha=0.5, label="0.5 yr gap (good)")
    ax.axhline(y=1.0, color="orange", linestyle="--", alpha=0.5, label="1.0 yr gap (mild)")
    ax.axhline(y=2.0, color="red",    linestyle="--", alpha=0.5, label="2.0 yr gap (outlier-driven)")

    if len(gdf):
        final = gdf.iloc[-1]
        ax.annotate(f'Final gap: {final["gap"]:.2f} yr',
                    xy=(final["epoch"], final["gap"]),
                    xytext=(final["epoch"] - 30, final["gap"] + 0.3),
                    fontsize=10, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="red"))

    ax.set_xlabel("Epoch"); ax.set_ylabel("MAE − MedAE (years)")
    ax.set_title("Outlier Impact: MAE − MedAE gap\n(smaller = fewer outliers dominating the mean)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.suptitle(f"Outlier Impact Analysis — {run_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(outdir, "mae_medae_gap.png")
    plt.savefig(path, dpi=150); plt.close()
    return path


def plot_r2_curves(df: pd.DataFrame, outdir: str, run_name: str) -> str:
    fig, ax = plt.subplots(figsize=(14, 7))
    ec = _epoch_col(df)

    for col, label, color, lw in [
        ("train/r2", "Train R²", "blue",   1.5),
        ("val/r2",   "Val R²",   "orange", 2.0),
    ]:
        if col in df.columns:
            data = df[df[col].notna()]
            if len(data):
                ax.plot(data[ec], data[col], label=label, alpha=0.8, color=color,
                        linewidth=lw, marker="o", markersize=2)

    if RIDGE_BASELINE_R2 is not None:
        ax.axhline(y=RIDGE_BASELINE_R2, color="red", linestyle="--", alpha=0.7,
                   linewidth=1.5, label=f"Ridge Baseline (R²={RIDGE_BASELINE_R2:.2f})")
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.3)
    ax.axhline(y=0.0, color="gray", linestyle=":", alpha=0.3)

    if "val/r2" in df.columns:
        val_df = df[df["val/r2"].notna()]
        if len(val_df):
            bi = val_df["val/r2"].idxmax()
            br, be = val_df.loc[bi, "val/r2"], val_df.loc[bi, ec]
            ax.annotate(f"Best Val R²: {br:.4f}\n@ epoch {be:.0f}",
                        xy=(be, br), xytext=(be - 20, br - 0.07), fontsize=10,
                        color="darkorange", fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color="darkorange", lw=1.5))

    ax.set_xlabel("Epoch", fontsize=12); ax.set_ylabel("R²", fontsize=12)
    ax.set_title(f"R-squared — {run_name}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3); ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    path = os.path.join(outdir, "r2_curves.png")
    plt.savefig(path, dpi=150); plt.close()
    return path


def plot_lr_schedule(df: pd.DataFrame, outdir: str, run_name: str) -> str:
    fig, ax = plt.subplots(figsize=(14, 6))
    sc = "trainer/global_step" if "trainer/global_step" in df.columns else "_step"

    plotted = False
    colors = ["blue", "steelblue", "orange", "darkorange", "green"]
    for i, lr_col in enumerate(LR_METRICS):
        if lr_col in df.columns:
            lr_df = df[df[lr_col].notna()]
            if len(lr_df):
                label = LR_LABELS.get(lr_col, lr_col)
                ax.plot(lr_df[sc], lr_df[lr_col], label=label, alpha=0.85,
                        color=colors[i % len(colors)], linewidth=1.5)
                plotted = True

    if not plotted:
        for lr_col in [c for c in df.columns if "lr" in c.lower()]:
            lr_df = df[df[lr_col].notna()]
            if len(lr_df):
                ax.plot(lr_df[sc], lr_df[lr_col], label=lr_col, alpha=0.8)

    ax.set_xlabel("Step", fontsize=12); ax.set_ylabel("Learning Rate", fontsize=12)
    ax.set_title(f"Learning Rate Schedule (4-group AdamW) — {run_name}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_yscale("log")
    plt.tight_layout()
    path = os.path.join(outdir, "lr_schedule.png")
    plt.savefig(path, dpi=150); plt.close()
    return path


def plot_train_val_gap(df: pd.DataFrame, outdir: str, run_name: str) -> str:
    gap_df = compute_train_val_gap(df)
    if gap_df.empty:
        print("  No train-val gap data, skipping.")
        return ""

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    ax.plot(gap_df["epoch"], gap_df["train_mae"], label="Train MAE", color="blue",   linewidth=1.5)
    ax.plot(gap_df["epoch"], gap_df["val_mae"],   label="Val MAE",   color="orange", linewidth=2)
    ax.fill_between(gap_df["epoch"], gap_df["train_mae"], gap_df["val_mae"],
                    alpha=0.15, color="red", label="Gap")
    ax.axhline(y=V1_BEST_MAE, color="purple", linestyle="--", alpha=0.5, linewidth=1,
               label=f"V1 best ({V1_BEST_MAE:.2f} yr)")
    if RIDGE_BASELINE_MAE is not None:
        ax.axhline(y=RIDGE_BASELINE_MAE, color="red", linestyle="--", alpha=0.4, linewidth=1,
                   label=f"Ridge ({RIDGE_BASELINE_MAE:.2f} yr)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MAE (years)")
    ax.set_title("Train vs Validation MAE")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(gap_df["epoch"], gap_df["gap"], color="red", linewidth=1.5)
    ax.fill_between(gap_df["epoch"], 0, gap_df["gap"], alpha=0.2, color="red")
    ax.axhline(y=0,   color="gray",   linestyle="-",  alpha=0.3)
    ax.axhline(y=1.0, color="orange", linestyle="--", alpha=0.5, label="1-yr gap (mild)")
    ax.axhline(y=3.0, color="red",    linestyle="--", alpha=0.5, label="3-yr gap (concerning)")
    if len(gap_df):
        final = gap_df.iloc[-1]
        ax.annotate(f'Final gap: {final["gap"]:.2f} yr',
                    xy=(final["epoch"], final["gap"]),
                    xytext=(final["epoch"] - 20, final["gap"] + 0.5),
                    fontsize=10, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="red"))
    ax.set_xlabel("Epoch"); ax.set_ylabel("Val MAE − Train MAE (years)")
    ax.set_title("Generalization Gap")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.suptitle(f"Overfitting Analysis — {run_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(outdir, "train_val_gap.png")
    plt.savefig(path, dpi=150); plt.close()
    return path


def plot_convergence(df: pd.DataFrame, outdir: str, run_name: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ec = _epoch_col(df)

    mae_col = next((c for c in ["val/mae", "val/mae_epoch"] if c in df.columns), None)

    ax = axes[0]
    if mae_col:
        val_df = df[df[mae_col].notna()].copy()
        if len(val_df) > 1:
            epochs = val_df[ec].values
            maes   = val_df[mae_col].values
            ax.plot(epochs, maes, color="orange", linewidth=2, label="Val MAE")
            thresholds = [
                (V1_BEST_MAE, f"V1 best ({V1_BEST_MAE:.2f} yr)", "purple"),
                (5.0, "5-year MAE", "blue"),
                (4.0, "4-year MAE", "green"),
            ]
            if RIDGE_BASELINE_MAE is not None:
                thresholds.insert(1, (RIDGE_BASELINE_MAE, f"Ridge ({RIDGE_BASELINE_MAE:.2f} yr)", "red"))
            for threshold, label, color in thresholds:
                ax.axhline(y=threshold, color=color, linestyle="--", alpha=0.4, linewidth=1)
                below = np.where(maes <= threshold)[0]
                if len(below):
                    fe = epochs[below[0]]
                    ax.axvline(x=fe, color=color, linestyle=":", alpha=0.3)
                    ax.annotate(f"{label}\n@ ep {fe:.0f}", xy=(fe, threshold),
                                xytext=(fe + 3, threshold + 0.4), fontsize=7, color=color)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Val MAE (years)")
    ax.set_title("Convergence Speed"); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    ax = axes[1]
    if mae_col:
        val_df = df[df[mae_col].notna()].copy()
        if len(val_df) > 1:
            epochs = val_df[ec].values
            maes   = val_df[mae_col].values
            start_mae, final_mae = maes[0], maes[-1]
            total_improvement = start_mae - final_mae
            if total_improvement > 0:
                pct_done = np.clip(((start_mae - maes) / total_improvement) * 100, 0, 100)
                ax.plot(epochs, pct_done, color="green", linewidth=2)
                ax.fill_between(epochs, 0, pct_done, alpha=0.1, color="green")
                above_90 = np.where(pct_done >= 90)[0]
                if len(above_90):
                    e90 = epochs[above_90[0]]
                    ax.axvline(x=e90, color="green", linestyle="--", alpha=0.5)
                    ax.annotate(f"90% converged\n@ epoch {e90:.0f}",
                                xy=(e90, 90), xytext=(e90 + 5, 75), fontsize=9,
                                color="green", fontweight="bold",
                                arrowprops=dict(arrowstyle="->", color="green"))
                ax.axhline(y=90,  color="green", linestyle=":", alpha=0.3)
                ax.axhline(y=100, color="gray",  linestyle=":", alpha=0.3)
    ax.set_xlabel("Epoch"); ax.set_ylabel("% of Total Improvement")
    ax.set_title("Convergence Progress"); ax.grid(True, alpha=0.3); ax.set_ylim(-5, 105)

    plt.suptitle(f"Convergence Analysis — {run_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(outdir, "convergence.png")
    plt.savefig(path, dpi=150); plt.close()
    return path


def plot_all_metrics_combined(df: pd.DataFrame, outdir: str, run_name: str) -> str:
    """2×4 overview: Loss, MAE, MedAE, MAE-MedAE gap, R², LR, overfitting, summary bar."""
    fig, axes = plt.subplots(2, 4, figsize=(28, 12))
    ec = _epoch_col(df)
    sc = "trainer/global_step" if "trainer/global_step" in df.columns else "_step"

    # 1. Loss
    ax = axes[0, 0]
    for col, label, color in [("train/loss", "Train", "blue"), ("val/loss", "Valid", "orange")]:
        if col in df.columns:
            data = df[df[col].notna()]
            if len(data):
                ax.plot(data[ec], data[col], label=label, color=color, linewidth=1.5, alpha=0.8)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Loss")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 2. MAE
    ax = axes[0, 1]
    for col, label, color in [("train/mae", "Train MAE", "blue"), ("val/mae", "Val MAE", "orange")]:
        if col in df.columns:
            data = df[df[col].notna()]
            if len(data):
                ax.plot(data[ec], data[col], label=label, color=color, linewidth=1.5, alpha=0.8)
    ax.axhline(y=V1_BEST_MAE, color="purple", linestyle="--", alpha=0.6, linewidth=1,
               label=f"V1 ({V1_BEST_MAE:.2f})")
    if RIDGE_BASELINE_MAE is not None:
        ax.axhline(y=RIDGE_BASELINE_MAE, color="red", linestyle="--", alpha=0.4, linewidth=1,
                   label=f"Ridge ({RIDGE_BASELINE_MAE:.2f})")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MAE (years)"); ax.set_title("MAE")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # 3. MedAE
    ax = axes[0, 2]
    for col, label, color in [("train/medae", "Train MedAE", "steelblue"),
                               ("val/medae",   "Val MedAE",   "darkorange")]:
        if col in df.columns:
            data = df[df[col].notna()]
            if len(data):
                ax.plot(data[ec], data[col], label=label, color=color, linewidth=1.5, alpha=0.8)
    ax.set_xlabel("Epoch"); ax.set_ylabel("MedAE (years)"); ax.set_title("Median AE (robust)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 4. MAE – MedAE gap
    ax = axes[0, 3]
    epdf = get_epoch_data(df)
    if "val/mae" in epdf.columns and "val/medae" in epdf.columns:
        rows = [(r[ec], r["val/mae"] - r["val/medae"])
                for _, r in epdf.iterrows()
                if pd.notna(r.get("val/mae")) and pd.notna(r.get("val/medae"))]
        if rows:
            gs, gaps = zip(*rows)
            ax.plot(gs, gaps, color="red", linewidth=2)
            ax.fill_between(gs, 0, gaps, alpha=0.15, color="red")
            ax.axhline(y=1.0, color="orange", linestyle="--", alpha=0.5, label="1 yr")
            ax.axhline(y=2.0, color="red",    linestyle="--", alpha=0.4, label="2 yr")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MAE − MedAE (years)")
    ax.set_title("Outlier Impact (MAE−MedAE gap)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 5. R²
    ax = axes[1, 0]
    for col, label, color in [("train/r2", "Train R²", "blue"), ("val/r2", "Val R²", "orange")]:
        if col in df.columns:
            data = df[df[col].notna()]
            if len(data):
                ax.plot(data[ec], data[col], label=label, color=color, linewidth=1.5, alpha=0.8,
                        marker="o", markersize=2)
    if RIDGE_BASELINE_R2 is not None:
        ax.axhline(y=RIDGE_BASELINE_R2, color="red", linestyle="--", alpha=0.4, linewidth=1,
                   label=f"Ridge ({RIDGE_BASELINE_R2:.2f})")
    ax.set_xlabel("Epoch"); ax.set_ylabel("R²"); ax.set_title("R²")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_ylim(-0.05, 1.05)

    # 6. LR schedule
    ax = axes[1, 1]
    colors = ["blue", "steelblue", "orange", "darkorange", "green"]
    for i, lr_col in enumerate(LR_METRICS):
        if lr_col in df.columns:
            lr_df = df[df[lr_col].notna()]
            if len(lr_df):
                label = LR_LABELS.get(lr_col, lr_col)
                ax.plot(lr_df[sc], lr_df[lr_col], label=label, alpha=0.85,
                        color=colors[i % len(colors)], linewidth=1.2)
    if not any(c in df.columns for c in LR_METRICS):
        for lr_col in [c for c in df.columns if "lr" in c.lower()][:4]:
            lr_df = df[df[lr_col].notna()]
            if len(lr_df):
                ax.plot(lr_df[sc], lr_df[lr_col], label=lr_col, alpha=0.8)
    ax.set_xlabel("Step"); ax.set_ylabel("LR"); ax.set_title("LR Schedule (4-group AdamW)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3); ax.set_yscale("log")

    # 7. Train-val gap
    ax = axes[1, 2]
    gap_df = compute_train_val_gap(df)
    if not gap_df.empty:
        ax.plot(gap_df["epoch"], gap_df["gap"], color="red", linewidth=1.5)
        ax.fill_between(gap_df["epoch"], 0, gap_df["gap"], alpha=0.15, color="red")
        ax.axhline(y=0,   color="gray",   linestyle="-",  alpha=0.3)
        ax.axhline(y=1.0, color="orange", linestyle="--", alpha=0.4, label="1-yr gap")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Val−Train MAE (years)")
    ax.set_title("Generalization Gap")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 8. Summary bar chart: MAE and MedAE comparison
    ax = axes[1, 3]
    best = find_best_checkpoint(df)
    best_mae   = best.get("best_mae",   np.nan)
    best_medae = best.get("best_medae", np.nan)

    test_mae = test_medae = test_r2 = np.nan
    for col in ["test/mae"]:
        if col in df.columns:
            td = df[df[col].notna()]
            if len(td): test_mae = td[col].iloc[-1]
    for col in ["test/medae"]:
        if col in df.columns:
            td = df[df[col].notna()]
            if len(td): test_medae = td[col].iloc[-1]
    if "test/r2" in df.columns:
        td = df[df["test/r2"].notna()]
        if len(td): test_r2 = td["test/r2"].iloc[-1]

    if RIDGE_BASELINE_MAE is not None:
        x      = np.arange(4)
        values = [V1_BEST_MAE, RIDGE_BASELINE_MAE, test_mae, test_medae]
        labels = ["V1\nMAE", "Ridge\nMAE", "V4\ntest MAE", "V4\ntest MedAE"]
        colors_bar = ["purple", "#e74c3c", "#f39c12", "#27ae60"]
    else:
        x      = np.arange(3)
        values = [V1_BEST_MAE, test_mae, test_medae]
        labels = ["V1\nMAE", "V4\ntest MAE", "V4\ntest MedAE"]
        colors_bar = ["purple", "#f39c12", "#27ae60"]
    bars = ax.bar(x, values, color=colors_bar, alpha=0.85, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, values):
        if val is not None and not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.05,
                    f"{val:.2f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Error (years)"); ax.set_title("Results Summary")
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(f"Fine-tuning Analysis — {run_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(outdir, "all_metrics_combined.png")
    plt.savefig(path, dpi=150); plt.close()
    return path


# ==============================
# Analysis Report
# ==============================

def generate_report(df: pd.DataFrame, run: wandb.apis.public.Run, outdir: str) -> str:
    best      = find_best_checkpoint(df)
    early_s   = detect_early_stopping(df)
    gap_df    = compute_train_val_gap(df)

    def _last(col):
        if col not in df.columns: return None
        td = df[df[col].notna()]
        return td[col].iloc[-1] if len(td) else None

    test_mae   = _last("test/mae")
    test_medae = _last("test/medae")
    test_r2    = _last("test/r2")
    train_mae  = _last("train/mae")

    final_gap = gap_df.iloc[-1]["gap"] if not gap_df.empty else None

    def fmt(val, d=4):
        if val is None or (isinstance(val, float) and np.isnan(val)): return "N/A"
        return f"{val:.{d}f}"

    ridge_imp = ((RIDGE_BASELINE_MAE - test_mae) / RIDGE_BASELINE_MAE * 100) if (RIDGE_BASELINE_MAE and test_mae) else None
    v1_imp    = ((V1_BEST_MAE - test_mae) / V1_BEST_MAE * 100) if test_mae else None
    mae_med_gap = (test_mae - test_medae) if (test_mae and test_medae) else None

    gap_assessment = "N/A"
    if final_gap is not None:
        if   final_gap < 1.0: gap_assessment = "Healthy (gap < 1 yr)"
        elif final_gap < 2.0: gap_assessment = "Mild (gap 1–2 yr)"
        else:                  gap_assessment = "Concerning (gap > 2 yr)"

    report = f"""# Fine-tuning Analysis Report — V4

**Run:** {run.name}
**Run ID:** {run.id}
**URL:** {run.url}
**State:** {run.state}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Final Results

| Metric | Train (final) | Best Val | Test | V1 best |
|--------|--------------|----------|------|---------|
| MAE (yr)   | {fmt(train_mae, 2)} | {fmt(best.get('best_mae'), 2)} | {fmt(test_mae, 2)} | {V1_BEST_MAE:.2f} |
| MedAE (yr) | — | {fmt(best.get('best_medae'), 2)} | {fmt(test_medae, 2)} | — |
| R²         | — | {fmt(best.get('best_val_r2'), 4)} | {fmt(test_r2, 4)} | 0.862 |

## Performance Summary

- **Best Val MAE:**   {fmt(best.get('best_mae'), 2)} yr  @ epoch {fmt(best.get('best_epoch'), 0)}
- **Best Val MedAE:** {fmt(best.get('best_medae'), 2)} yr  @ epoch {fmt(best.get('best_medae_epoch'), 0)}
- **Test MAE:**       {fmt(test_mae, 2)} yr
- **Test MedAE:**     {fmt(test_medae, 2)} yr
- **Test MAE−MedAE gap:** {fmt(mae_med_gap, 2)} yr  ← outlier impact on the mean
- **Test R²:**        {fmt(test_r2, 4)}
- **vs V1 (6.81 yr):**    {fmt(v1_imp, 1)}% improvement in MAE

## Training Details

- **Total Epochs:** {early_s.get('total_epochs', 'N/A')}
- **Early Stopping Triggered:** {'Yes' if early_s.get('triggered') else 'No'}
- **Epochs After Best:** {early_s.get('epochs_after_best', 'N/A')}

## Overfitting Analysis

- **Train-Val MAE Gap (final):** {fmt(final_gap, 2)} yr
- **Assessment:** {gap_assessment}

## Plots

- `all_metrics_combined.png`  — 2×4 overview of all key metrics
- `loss_curves.png`           — Train vs Validation loss
- `mae_curves.png`            — MAE and MedAE curves side by side
- `mae_medae_gap.png`         — MAE–MedAE gap (outlier impact over training)
- `r2_curves.png`             — R² over training
- `lr_schedule.png`           — 4-group AdamW LR schedule
- `train_val_gap.png`         — Overfitting analysis
- `convergence.png`           — Convergence speed analysis
"""

    path = os.path.join(outdir, "analysis_report.md")
    with open(path, "w") as f:
        f.write(report)
    return path


# ==============================
# Main
# ==============================

def main() -> int:
    ensure_outdir(OUTDIR)

    print("=" * 80)
    print("BMFM MethylLlama Fine-tuning — WandB Analysis")
    print(f"ENTITY:  {ENTITY}")
    print(f"PROJECT: {PROJECT}")
    print(f"RUN_ID:  {RUN_ID}")
    print(f"OUTDIR:  {OUTDIR}")
    print("=" * 80)

    api  = wandb.Api(timeout=WANDB_TIMEOUT)
    runs = fetch_runs(api, ENTITY, PROJECT)
    print(f"\nFound {len(runs)} runs in project.")

    print("\nAvailable runs (recent 20):")
    print("-" * 80)
    for i, r in enumerate(runs[:20]):
        print(f"{i+1:3d}. [{r.state:10s}] {r.name[:55]:55s} | {r.id}")
    print("-" * 80)

    if RUN_ID:
        run = api.run(f"{ENTITY}/{PROJECT}/{RUN_ID}")
    else:
        try:
            choice = input("\nEnter run number (or Enter for most recent): ").strip()
            run = runs[int(choice) - 1] if choice else runs[0]
        except (ValueError, IndexError):
            run = runs[0]

    print(f"\nAnalyzing: {run.name}  ({run.id})  state={run.state}")

    df = download_run_history(run)
    if df.empty:
        print("No history data found!")
        return 1

    print(f"Downloaded {len(df)} history rows.")
    print(f"\nAvailable metrics (non-empty columns):")
    for col in sorted(df.columns):
        nn = df[col].notna().sum()
        if nn > 0 and not col.startswith("_"):
            print(f"  {col:50s}  ({nn} values)")

    # Save raw history
    csv_path = os.path.join(OUTDIR, "finetune_history_raw.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved raw history: {csv_path}")

    # Generate plots
    print("\nGenerating plots...")
    plots = [
        ("Combined overview",       plot_all_metrics_combined(df, OUTDIR, run.name)),
        ("Loss curves",             plot_loss_curves(df, OUTDIR, run.name)),
        ("MAE & MedAE curves",      plot_mae_curves(df, OUTDIR, run.name)),
        ("MAE–MedAE gap",           plot_mae_medae_gap(df, OUTDIR, run.name)),
        ("R² curves",               plot_r2_curves(df, OUTDIR, run.name)),
        ("LR schedule",             plot_lr_schedule(df, OUTDIR, run.name)),
        ("Train-val gap",           plot_train_val_gap(df, OUTDIR, run.name)),
        ("Convergence",             plot_convergence(df, OUTDIR, run.name)),
    ]
    for name, p in plots:
        print(f"  {'Saved' if p else 'Skipped'}: {name}  {p or ''}")

    # Generate report
    report_path = generate_report(df, run, OUTDIR)
    print(f"\nSaved report: {report_path}")

    # Print full summary
    best    = find_best_checkpoint(df)
    early_s = detect_early_stopping(df)

    def _last(col):
        if col not in df.columns: return None
        td = df[df[col].notna()]
        return td[col].iloc[-1] if len(td) else None

    test_mae   = _last("test/mae")
    test_medae = _last("test/medae")
    test_r2    = _last("test/r2")

    print("\n" + "=" * 80)
    print(f"ANALYSIS SUMMARY — {run.name}")
    print("=" * 80)
    print(f"Best Val MAE:         {best.get('best_mae', 'N/A'):.4f} yr  @ epoch {best.get('best_epoch', '?')}")
    print(f"Best Val MedAE:       {best.get('best_medae', 'N/A'):.4f} yr  @ epoch {best.get('best_medae_epoch', '?')}")
    print(f"Best Val R²:          {best.get('best_val_r2', 'N/A')}")
    print()
    if test_mae   is not None: print(f"Test MAE:             {test_mae:.4f} yr")
    if test_medae is not None: print(f"Test MedAE:           {test_medae:.4f} yr")
    if test_mae and test_medae:
        gap = test_mae - test_medae
        print(f"Test MAE−MedAE gap:   {gap:.4f} yr  ← outlier impact")
    if test_r2    is not None: print(f"Test R²:              {test_r2:.4f}")
    print()
    print(f"V1 best MAE:          {V1_BEST_MAE:.2f} yr")
    if RIDGE_BASELINE_MAE is not None:
        print(f"Ridge baseline MAE:   {RIDGE_BASELINE_MAE:.2f} yr")
    if test_mae:
        print(f"Improvement vs V1:    {((V1_BEST_MAE - test_mae)/V1_BEST_MAE*100):+.1f}%")
        if RIDGE_BASELINE_MAE is not None:
            print(f"Improvement vs Ridge: {((RIDGE_BASELINE_MAE - test_mae)/RIDGE_BASELINE_MAE*100):+.1f}%")
    print()
    print(f"Early Stopping:       {'Triggered' if early_s.get('triggered') else 'Not triggered'}")
    print(f"Total Epochs:         {early_s.get('total_epochs', 'N/A')}")
    print(f"Epochs after best:    {early_s.get('epochs_after_best', 'N/A')}")
    print(f"\nAll outputs saved to: {OUTDIR}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
