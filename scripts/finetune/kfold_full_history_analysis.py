#!/usr/bin/env python3
"""
kfold_full_history_analysis.py
===============================
Full-history analysis of the MethylLlama pretrain -> 5-fold finetune pipeline,
compared against the MethylGPT 21k baseline (see docs/presentations/methylgpt_21k_baseline.md).

Produces:
  - Pretrain curve (encoder recon loss / PCC) for the winning MethylLlama WCED run
  - Per-fold finetune curves (val MedAE/MAE/R2) for all 5 folds, overlaid
  - Final comparison figure: MethylLlama 5-fold test MedAE/MAE/R2 (mean +/- 95% CI) vs MethylGPT 21k
  - CSV tables for all of the above

Usage:
  python \
      scripts/finetune/kfold_full_history_analysis.py
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb
from scipy import stats

warnings.filterwarnings("ignore")

ENTITY = os.environ.get("WANDB_ENTITY", "")  # W&B account that holds the runs (configs/paths.example.sh)
OUT_DIR = Path("kfold_full_history_analysis")
OUT_DIR.mkdir(exist_ok=True)

# ── Run registry ─────────────────────────────────────────────────────────────
PRETRAIN_PROJECT = "pretrain-llama-wced"
PRETRAIN_RUN = "iujgrcvs"          # llama-6L-all49k-r0.5-w0.05-genomic-45468861 (winning MethylLlama encoder)

FINETUNE_PROJECT = "finetune-llama-small"
FOLD_TRAIN_RUNS = {
    0: "a154lzjy",   # methyllama-kfold-fold0-ep300-45586010 (note: first attempt 6s4ogdfo was a restart)
    1: "aas23t9m",   # fold1-ep300-45586011
    2: "vw5f9nh4",   # fold2-ep300-45586012
    3: "vqogel6f",   # fold3-ep300-45586013
    4: "6qt4ma56",   # fold4-ep300-45586014
}
FOLD_TESTEVAL_RUNS = {
    0: "tx88qk8j",   # fold0-testeval
    1: "ukklfbu4",   # fold1-testeval
    2: "uehgwttf",   # fold2-testeval
    3: "acwfu618",   # fold3-testeval (standalone testeval_kfold.sh, job 45633084)
    4: "7e1cb9bw",   # fold4-testeval (standalone testeval_kfold.sh, job 45633084)
}

METHYLGPT_TRAJECTORY_CSV = Path("docs/presentations/methylgpt_21k_trajectory.csv")
if not METHYLGPT_TRAJECTORY_CSV.exists():
    METHYLGPT_TRAJECTORY_CSV = Path("../docs/presentations/methylgpt_21k_trajectory.csv")

MethylGPT_BASELINE = {"medae": 3.839, "mae": 5.521, "r2": 0.9044}  # best-val_medae ckpt (epoch 253), see methylgpt_21k_baseline.md

FOLD_COLORS = {0: "#4DBBD5", 1: "#00A087", 2: "#3C5488", 3: "#F39B7F", 4: "#8491B4"}


def api():
    return wandb.Api(timeout=120)


def cached_history(run_id, project, keys, cache_name):
    """Fetch per-column histories independently (wandb's history(keys=) intersects
    non-null rows across all keys, which is empty when metrics log at different
    steps), then merge by taking the last value per epoch for each metric."""
    cache_path = OUT_DIR / f"{cache_name}.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path)
    run = api().run(f"{ENTITY}/{project}/{run_id}")
    other_keys = [k for k in keys if k != "epoch"]
    per_col = {}
    for k in other_keys:
        s = run.history(keys=["epoch", k], pandas=True).dropna(subset=[k])
        if not s.empty:
            per_col[k] = s.groupby("epoch")[k].last()
    df = pd.DataFrame(per_col).reset_index()
    df.to_csv(cache_path, index=False)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 1. Pretrain curve
# ─────────────────────────────────────────────────────────────────────────────

def analyze_pretrain():
    print("== Pretrain history (MethylLlama winning encoder) ==")
    df = cached_history(
        PRETRAIN_RUN, PRETRAIN_PROJECT,
        ["epoch", "train/recon_loss", "train/pcc", "validation/loss", "validation/pcc"],
        "pretrain_history",
    )
    df = df.dropna(subset=["epoch"]).sort_values("epoch")
    df.to_csv(OUT_DIR / "pretrain_history.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.patch.set_facecolor("white")
    fig.suptitle("Pretrain — MethylLlama encoder (llama-6L-all49k-r0.5-w0.05-genomic)", fontsize=11, fontweight="bold")

    ax = axes[0]
    for col, label, color in [("train/recon_loss", "train recon", "#4DBBD5"),
                                ("validation/loss", "val loss", "#E64B35")]:
        if col in df.columns:
            s = df[[  "epoch", col]].dropna()
            ax.plot(s["epoch"], s[col], color=color, label=label, linewidth=1.5)
    ax.set_title("Reconstruction loss"); ax.set_xlabel("Epoch"); ax.legend(fontsize=8)
    ax.set_facecolor("#F8F8F8"); ax.grid(True, color="white")

    ax = axes[1]
    for col, label, color in [("train/pcc", "train PCC", "#4DBBD5"),
                                ("validation/pcc", "val PCC", "#E64B35")]:
        if col in df.columns:
            s = df[["epoch", col]].dropna()
            ax.plot(s["epoch"], s[col], color=color, label=label, linewidth=1.5)
    ax.set_title("Reconstruction PCC"); ax.set_xlabel("Epoch"); ax.legend(fontsize=8)
    ax.set_facecolor("#F8F8F8"); ax.grid(True, color="white")

    plt.tight_layout()
    fig.savefig(OUT_DIR / "01_pretrain_curve.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  saved -> {OUT_DIR / '01_pretrain_curve.png'}  ({len(df)} epochs logged)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Per-fold finetune curves
# ─────────────────────────────────────────────────────────────────────────────

def analyze_finetune_curves():
    print("\n== Finetune k-fold curves (5 folds) ==")
    fold_dfs = {}
    for fold, run_id in FOLD_TRAIN_RUNS.items():
        df = cached_history(
            run_id, FINETUNE_PROJECT,
            ["epoch", "val/medae", "val/mae", "val/r2"],
            f"fold{fold}_train_history",
        )
        df = df.dropna(subset=["val/medae"]).sort_values("epoch")
        df["fold"] = fold
        fold_dfs[fold] = df
        best = df.loc[df["val/medae"].idxmin()]
        print(f"  fold {fold}: {len(df)} val epochs logged, best val_medae={best['val/medae']:.4f} @ epoch {int(best['epoch'])}")

    combined = pd.concat(fold_dfs.values(), ignore_index=True)
    combined.to_csv(OUT_DIR / "finetune_all_folds_history.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.patch.set_facecolor("white")
    fig.suptitle("MethylLlama 5-fold finetune — validation curves", fontsize=11, fontweight="bold")

    for ax, metric, ylabel, hline in [
        (axes[0], "val/medae", "Val MedAE (years)", MethylGPT_BASELINE["medae"]),
        (axes[1], "val/mae",   "Val MAE (years)",   MethylGPT_BASELINE["mae"]),
        (axes[2], "val/r2",    "Val R2",            MethylGPT_BASELINE["r2"]),
    ]:
        ax.set_facecolor("#F8F8F8"); ax.grid(True, color="white")
        for fold, df in fold_dfs.items():
            ax.plot(df["epoch"], df[metric], color=FOLD_COLORS[fold], linewidth=1.2, alpha=0.85, label=f"fold {fold}")
        ax.axhline(hline, color="black", linestyle="--", linewidth=1.0, alpha=0.6, label="MethylGPT 21k (test)")
        ax.set_title(ylabel); ax.set_xlabel("Epoch")
        ax.legend(fontsize=6.5, ncol=2)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "02_finetune_kfold_curves.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  saved -> {OUT_DIR / '02_finetune_kfold_curves.png'}")
    return fold_dfs


# ─────────────────────────────────────────────────────────────────────────────
# 3. Test-eval summary + final comparison
# ─────────────────────────────────────────────────────────────────────────────

def analyze_test_results(fold_dfs):
    print("\n== Test-eval results (fixed 2149-sample test set) ==")
    rows = []
    for fold, run_id in FOLD_TESTEVAL_RUNS.items():
        run = api().run(f"{ENTITY}/{FINETUNE_PROJECT}/{run_id}")
        s = run.summary
        rows.append({
            "fold": fold, "status": "done",
            "test_medae": s.get("test/medae"), "test_mae": s.get("test/mae"), "test_r2": s.get("test/r2"),
            "best_val_medae": float(fold_dfs[fold]["val/medae"].min()) if fold in fold_dfs else None,
        })
    for fold in FOLD_TRAIN_RUNS:
        if fold not in FOLD_TESTEVAL_RUNS:
            best_val = float(fold_dfs[fold]["val/medae"].min()) if fold in fold_dfs else None
            rows.append({
                "fold": fold, "status": "PENDING test-eval (training converged, val_medae logged)",
                "test_medae": np.nan, "test_mae": np.nan, "test_r2": np.nan,
                "best_val_medae": best_val,
            })

    df = pd.DataFrame(rows).sort_values("fold")
    df.to_csv(OUT_DIR / "fold_test_results.csv", index=False)
    print(df.to_string(index=False))

    done = df.dropna(subset=["test_medae"])
    n = len(done)
    summary = {}
    for metric in ["test_medae", "test_mae", "test_r2"]:
        vals = done[metric].values
        mean = vals.mean()
        if n > 1:
            sem = stats.sem(vals)
            ci = sem * stats.t.ppf(0.975, n - 1)
        else:
            ci = np.nan
        summary[metric] = (mean, ci)

    print(f"\n  MethylLlama test results, n={n} fold(s) evaluated:")
    for metric, (mean, ci) in summary.items():
        ci_str = f" +/- {ci:.3f}" if not np.isnan(ci) else " (n=1, no CI)"
        print(f"    {metric}: {mean:.3f}{ci_str}")

    pending = df[df["status"] != "done"]
    if len(pending):
        print(f"\n  PENDING ({len(pending)} fold(s)): {pending['fold'].tolist()} "
              f"— training converged (best val_medae logged) but test-eval not yet run.")
        print("  To finish: on the cluster, run")
        print(f"    FOLDS=\"{' '.join(str(f) for f in pending['fold'])}\" "
              f"CHECKPOINT='outputs/pretrain-llama-wced/llama-6L-all49k-r0.5-w0.05-genomic-45468861/"
              f"checkpoints/epoch=85-recon=0.0552-pcc=0.9713.ckpt' sbatch scripts/finetune/testeval_kfold.sh")

    return df, summary, n


def make_final_comparison_figure(summary, n):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    fig.patch.set_facecolor("white")
    fig.suptitle(f"MethylLlama k-fold (n={n}) vs MethylGPT 21k — fixed 2149-sample test set", fontsize=11, fontweight="bold")

    panels = [
        (axes[0], "test_medae", "MedAE (years)", MethylGPT_BASELINE["medae"], True),
        (axes[1], "test_mae",   "MAE (years)",   MethylGPT_BASELINE["mae"],   True),
        (axes[2], "test_r2",    "R2",            MethylGPT_BASELINE["r2"],   False),
    ]
    for ax, metric, ylabel, gpt_val, lower_better in panels:
        mean, ci = summary[metric]
        ci = 0 if np.isnan(ci) else ci
        ax.bar(["MethylGPT\n21k"], [gpt_val], color="#F39B7F", width=0.5)
        ax.bar([f"MethylLlama\n(n={n})"], [mean], yerr=[ci], color="#E64B35", width=0.5, capsize=6)
        pct = 100 * (gpt_val - mean) / gpt_val if lower_better else 100 * (mean - gpt_val) / gpt_val
        ax.set_title(f"{ylabel}\n({pct:+.0f}% {'better' if pct > 0 else 'worse'})", fontsize=9.5)
        ax.set_facecolor("#F8F8F8"); ax.grid(True, axis="y", color="white")

    plt.tight_layout()
    fig.savefig(OUT_DIR / "03_final_comparison.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\n  saved -> {OUT_DIR / '03_final_comparison.png'}")


def main():
    analyze_pretrain()
    fold_dfs = analyze_finetune_curves()
    df, summary, n = analyze_test_results(fold_dfs)
    make_final_comparison_figure(summary, n)
    print(f"\nAll outputs in: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
