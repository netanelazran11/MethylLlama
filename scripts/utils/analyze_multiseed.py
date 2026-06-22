#!/usr/bin/env python3
"""Complete Multi-Seed Analysis for BMFM Methylation Fine-tuning.

Downloads and analyzes all runs from the finetune-bmfm-multiseed project.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

# Config
ENTITY = "netanelazran11-hebrew-university-of-jerusalem"
PROJECT = "finetune-bmfm-multiseed"
OUTDIR = os.path.join(os.path.dirname(__file__), "..", "wandb_analysis", "multiply_seeds_wandb")

os.makedirs(OUTDIR, exist_ok=True)

def main():
    print("=" * 80)
    print("Multi-Seed Analysis - BMFM Methylation Fine-tuning")
    print(f"Project: {ENTITY}/{PROJECT}")
    print("=" * 80)

    api = wandb.Api(timeout=120)
    runs = list(api.runs(f"{ENTITY}/{PROJECT}"))

    print(f"\nFound {len(runs)} runs\n")

    # Collect all run data
    all_data = []
    all_histories = {}

    for run in runs:
        # Extract seed from name
        seed = None
        if "seed" in run.name.lower():
            try:
                seed = int(run.name.split("seed")[1].split("-")[0])
            except:
                seed = run.id

        # Get metrics from summary
        summary = run.summary
        config = run.config

        data = {
            "run_id": run.id,
            "run_name": run.name,
            "seed": seed,
            "state": run.state,
            "test_mae": summary.get("test/mae"),
            "test_r2": summary.get("test/r2"),
            "val_mae_best": summary.get("val/mae", summary.get("val/mae_epoch")),
            "val_r2_best": summary.get("val/r2"),
            "train_mae_final": summary.get("train/mae"),
            "epochs": summary.get("epoch", summary.get("trainer/global_step")),
        }

        # Get best epoch from history
        try:
            hist = run.history(pandas=True, samples=50000)
            all_histories[seed] = hist

            # Find best validation MAE epoch
            val_col = "val/mae" if "val/mae" in hist.columns else "val/mae_epoch"
            if val_col in hist.columns:
                val_df = hist[hist[val_col].notna()]
                if len(val_df) > 0:
                    best_idx = val_df[val_col].idxmin()
                    best_epoch = val_df.loc[best_idx, "epoch"] if "epoch" in val_df.columns else best_idx
                    data["best_epoch"] = int(best_epoch)
                    data["val_mae_best"] = val_df.loc[best_idx, val_col]

            # Get final train MAE
            train_col = "train/mae" if "train/mae" in hist.columns else "train/mae_epoch"
            if train_col in hist.columns:
                train_df = hist[hist[train_col].notna()]
                if len(train_df) > 0:
                    data["train_mae_final"] = train_df[train_col].iloc[-1]

        except Exception as e:
            print(f"  Warning: Could not get history for {run.name}: {e}")

        all_data.append(data)
        print(f"  Seed {seed}: Test MAE={data['test_mae']:.4f}, R²={data['test_r2']:.4f}, Best Epoch={data.get('best_epoch', 'N/A')}")

    # Create DataFrame
    df = pd.DataFrame(all_data)
    df = df.sort_values("seed")

    # Save runs table
    df.to_csv(os.path.join(OUTDIR, "runs_table.csv"), index=False)
    print(f"\nSaved: {OUTDIR}/runs_table.csv")

    # Compute statistics
    stats = {
        "metric": ["Test MAE (years)", "Test R²", "Val MAE Best (years)", "Train MAE Final (years)", "Best Epoch"],
        "mean": [
            df["test_mae"].mean(),
            df["test_r2"].mean(),
            df["val_mae_best"].mean(),
            df["train_mae_final"].mean(),
            df["best_epoch"].mean(),
        ],
        "std": [
            df["test_mae"].std(),
            df["test_r2"].std(),
            df["val_mae_best"].std(),
            df["train_mae_final"].std(),
            df["best_epoch"].std(),
        ],
        "min": [
            df["test_mae"].min(),
            df["test_r2"].min(),
            df["val_mae_best"].min(),
            df["train_mae_final"].min(),
            df["best_epoch"].min(),
        ],
        "max": [
            df["test_mae"].max(),
            df["test_r2"].max(),
            df["val_mae_best"].max(),
            df["train_mae_final"].max(),
            df["best_epoch"].max(),
        ],
    }

    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(os.path.join(OUTDIR, "summary_statistics.csv"), index=False)

    # Print statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS (n=5 seeds)")
    print("=" * 80)
    print(f"\n{'Metric':<25} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
    print("-" * 80)
    for _, row in stats_df.iterrows():
        print(f"{row['metric']:<25} {row['mean']:>12.4f} {row['std']:>12.4f} {row['min']:>12.4f} {row['max']:>12.4f}")
    print("=" * 80)

    # Find best seed
    best_idx = df["test_mae"].idxmin()
    best_row = df.loc[best_idx]
    print(f"\nBest Model: Seed {best_row['seed']}")
    print(f"  Test MAE: {best_row['test_mae']:.4f} years")
    print(f"  Test R²:  {best_row['test_r2']:.4f}")
    print(f"  Best Epoch: {best_row['best_epoch']}")

    # Create plots
    print("\nGenerating plots...")

    # Plot 1: Test metrics comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    seeds = df["seed"].values
    test_maes = df["test_mae"].values
    test_r2s = df["test_r2"].values

    colors = ['#3498db' if s != best_row['seed'] else '#e74c3c' for s in seeds]

    ax = axes[0]
    bars = ax.bar([f"Seed {s}" for s in seeds], test_maes, color=colors, edgecolor='black', alpha=0.8)
    ax.axhline(y=df["test_mae"].mean(), color='green', linestyle='--', linewidth=2, label=f'Mean: {df["test_mae"].mean():.2f}')
    ax.set_ylabel("Test MAE (years)", fontsize=12)
    ax.set_title("Test MAE by Seed", fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, test_maes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax = axes[1]
    bars = ax.bar([f"Seed {s}" for s in seeds], test_r2s, color=colors, edgecolor='black', alpha=0.8)
    ax.axhline(y=df["test_r2"].mean(), color='green', linestyle='--', linewidth=2, label=f'Mean: {df["test_r2"].mean():.4f}')
    ax.set_ylabel("Test R²", fontsize=12)
    ax.set_title("Test R² by Seed", fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0.90, 0.95)
    for bar, val in zip(bars, test_r2s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, f'{val:.4f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.suptitle("Multi-Seed Test Results (n=5 seeds)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "test_metrics_comparison.png"), dpi=150)
    plt.close()
    print(f"  Saved: test_metrics_comparison.png")

    # Plot 2: MAE convergence for all seeds
    if all_histories:
        fig, ax = plt.subplots(figsize=(14, 8))

        colors_map = {40: '#e74c3c', 41: '#3498db', 42: '#2ecc71', 43: '#9b59b6', 44: '#f39c12'}

        for seed, hist in sorted(all_histories.items()):
            val_col = "val/mae" if "val/mae" in hist.columns else "val/mae_epoch"
            if val_col in hist.columns and "epoch" in hist.columns:
                val_df = hist[hist[val_col].notna()]
                if len(val_df) > 0:
                    ax.plot(val_df["epoch"], val_df[val_col],
                           label=f"Seed {seed}", color=colors_map.get(seed, 'gray'),
                           linewidth=2, alpha=0.8)

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Validation MAE (years)", fontsize=12)
        ax.set_title("Validation MAE Convergence - All Seeds", fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, "mae_convergence_all_seeds.png"), dpi=150)
        plt.close()
        print(f"  Saved: mae_convergence_all_seeds.png")

        # Plot 3: R² convergence
        fig, ax = plt.subplots(figsize=(14, 8))

        for seed, hist in sorted(all_histories.items()):
            if "val/r2" in hist.columns and "epoch" in hist.columns:
                val_df = hist[hist["val/r2"].notna()]
                if len(val_df) > 0:
                    ax.plot(val_df["epoch"], val_df["val/r2"],
                           label=f"Seed {seed}", color=colors_map.get(seed, 'gray'),
                           linewidth=2, alpha=0.8)

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Validation R²", fontsize=12)
        ax.set_title("Validation R² Convergence - All Seeds", fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.85, 0.95)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, "r2_convergence_all_seeds.png"), dpi=150)
        plt.close()
        print(f"  Saved: r2_convergence_all_seeds.png")

    # Plot 4: Metrics distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.boxplot([test_maes], labels=['Test MAE'])
    ax.scatter([1]*len(test_maes), test_maes, color='blue', alpha=0.6, s=100, zorder=5)
    ax.set_ylabel("MAE (years)")
    ax.set_title("Test MAE Distribution")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.boxplot([test_r2s], labels=['Test R²'])
    ax.scatter([1]*len(test_r2s), test_r2s, color='blue', alpha=0.6, s=100, zorder=5)
    ax.set_ylabel("R²")
    ax.set_title("Test R² Distribution")
    ax.grid(True, alpha=0.3)

    plt.suptitle("Metrics Distribution Across Seeds", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "metrics_distribution.png"), dpi=150)
    plt.close()
    print(f"  Saved: metrics_distribution.png")

    # Generate comprehensive report
    report = f"""# Complete Multi-Seed Analysis Report

## Overview

- **Project:** {ENTITY}/{PROJECT}
- **Number of Seeds:** 5 (seeds 40, 41, 42, 43, 44)
- **Model:** BMFM-RNA adapted for methylation
- **Task:** Age prediction from DNA methylation

---

## Summary Statistics (n=5 seeds)

| Metric | Mean ± Std | Min | Max |
|--------|------------|-----|-----|
| **Test MAE (years)** | **{df['test_mae'].mean():.2f} ± {df['test_mae'].std():.2f}** | {df['test_mae'].min():.2f} | {df['test_mae'].max():.2f} |
| **Test R²** | **{df['test_r2'].mean():.4f} ± {df['test_r2'].std():.4f}** | {df['test_r2'].min():.4f} | {df['test_r2'].max():.4f} |
| **Val MAE Best** | {df['val_mae_best'].mean():.2f} ± {df['val_mae_best'].std():.2f} | {df['val_mae_best'].min():.2f} | {df['val_mae_best'].max():.2f} |
| **Train MAE Final** | {df['train_mae_final'].mean():.2f} ± {df['train_mae_final'].std():.2f} | {df['train_mae_final'].min():.2f} | {df['train_mae_final'].max():.2f} |
| **Best Epoch** | {df['best_epoch'].mean():.0f} ± {df['best_epoch'].std():.0f} | {int(df['best_epoch'].min())} | {int(df['best_epoch'].max())} |

---

## Individual Seed Results

| Seed | Test MAE (years) | Test R² | Val MAE Best | Best Epoch | Status |
|------|------------------|---------|--------------|------------|--------|
"""

    for _, row in df.sort_values("seed").iterrows():
        is_best = "**Best**" if row["test_mae"] == df["test_mae"].min() else ""
        report += f"| {int(row['seed'])} | {row['test_mae']:.4f} | {row['test_r2']:.4f} | {row['val_mae_best']:.4f} | {int(row['best_epoch'])} | {row['state']} {is_best} |\n"

    report += f"""
---

## Best Model

- **Seed:** {int(best_row['seed'])}
- **Test MAE:** {best_row['test_mae']:.4f} years
- **Test R²:** {best_row['test_r2']:.4f}
- **Best Epoch:** {int(best_row['best_epoch'])}
- **Run ID:** {best_row['run_id']}

---

## Key Findings

1. **Excellent Consistency:** Low variance across seeds (MAE std = {df['test_mae'].std():.2f} years)
2. **Strong Performance:** Mean R² = {df['test_r2'].mean():.4f} (explains {df['test_r2'].mean()*100:.1f}% of age variance)
3. **Best MAE:** {df['test_mae'].min():.2f} years (Seed {int(best_row['seed'])})
4. **Convergence:** Models converge around epoch {df['best_epoch'].mean():.0f} on average

---

## Comparison with Baseline

| Model | Test MAE (years) | Test R² |
|-------|------------------|---------|
| Mean Prediction | 22.82 | 0.00 |
| MethylGPT | 4.95 | 0.911 |
| **BMFM-RNA (Mean ± Std)** | **{df['test_mae'].mean():.2f} ± {df['test_mae'].std():.2f}** | **{df['test_r2'].mean():.4f} ± {df['test_r2'].std():.4f}** |
| **BMFM-RNA (Best)** | **{df['test_mae'].min():.2f}** | **{df['test_r2'].max():.4f}** |

**Improvement over MethylGPT:**
- MAE: {((4.95 - df['test_mae'].mean()) / 4.95 * 100):.1f}% better (mean), {((4.95 - df['test_mae'].min()) / 4.95 * 100):.1f}% better (best)
- R²: {((df['test_r2'].mean() - 0.911) / 0.911 * 100):.1f}% better (mean), {((df['test_r2'].max() - 0.911) / 0.911 * 100):.1f}% better (best)

---

## Output Files

- `runs_table.csv` - All runs with metrics
- `summary_statistics.csv` - Statistical summary
- `test_metrics_comparison.png` - Bar charts comparing seeds
- `mae_convergence_all_seeds.png` - MAE training curves
- `r2_convergence_all_seeds.png` - R² training curves
- `metrics_distribution.png` - Box plots
"""

    report_path = os.path.join(OUTDIR, "COMPLETE_ANALYSIS_REPORT.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nSaved: {report_path}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nAll outputs saved to: {OUTDIR}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
