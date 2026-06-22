#!/usr/bin/env python3
"""
robustness.py  —  Figure 4f equivalent
=======================================
Evaluate age prediction robustness under increasing levels of missing CpG data.
Systematically masks 0-90% of CpG inputs at inference time, measures MedAE,
and compares MethylLlama against ElasticNet and Ridge baselines.

Replicates MethylGPT Figure 4f:
  x-axis: % of CpG sites masked (10, 20, ..., 90%)
  y-axis: Median Absolute Error in years
  lines:  MethylLlama (CLS) | ElasticNet | Ridge | Null (predict-mean)

The key insight: a good pretrained model uses redundant signals across many
CpG sites, so masking 50% of inputs should degrade performance only slightly.
A simple linear model (ElasticNet) relies on specific sites and degrades sharply.

Usage:
  python scripts/repr_analysis/robustness.py \\
      --checkpoint outputs/finetune-llama-small/.../epoch=117-val_mae=6.3071.ckpt \\
      --data       /path/to/finetuning_19608_clean_stratified_no_outliers.h5ad \\
      --tokenizer  tokenizer_llama_pretrain49k \\
      --outdir     outputs/repr_analysis/robustness \\
      --mask_levels 0 10 20 30 40 50 60 70 80 90
"""

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Missing data robustness (Fig 4f)")
    p.add_argument("--checkpoint", required=True, help="Fine-tuned .ckpt")
    p.add_argument("--data",       required=True, help="Finetune h5ad (19k)")
    p.add_argument("--tokenizer",  required=True)
    p.add_argument("--outdir",     default="outputs/repr_analysis/robustness")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--mask_levels", nargs="+", type=int,
                   default=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
                   help="% CpG sites to mask (0=no masking)")
    p.add_argument("--age_col",    default="age")
    p.add_argument("--split_col",  default="split")
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_finetune_model(ckpt_path: str):
    from bmfm_methylation.llama.finetune_llama import load_finetune_llama_checkpoint
    log.info(f"Loading fine-tuned model: {ckpt_path}")
    return load_finetune_llama_checkpoint(ckpt_path)


# ─────────────────────────────────────────────────────────────────────────────
# Masked inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_with_masking(model, data_path, tokenizer_path, batch_size, device,
                          mask_fraction: float, seed: int):
    """
    Run model forward pass with `mask_fraction` of CpG tokens randomly zeroed.
    Returns (y_true, y_pred) arrays for the TEST split.
    """
    from bmfm_targets.tokenization import MultiFieldTokenizer
    from bmfm_methylation.shared.data_module import MethylationDataset, WCEDCollator

    tok = MultiFieldTokenizer.from_pretrained(tokenizer_path)
    dataset = MethylationDataset(h5ad_path=data_path, split="test", normalize_age=False)
    cpg_sites = dataset.cpg_sites

    collator = WCEDCollator(
        tokenizer=tok, cpg_sites=cpg_sites,
        vocab_size=len(cpg_sites), input_ratio=1.0, contrastive=False,
    )
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator,
                        shuffle=False, num_workers=2, pin_memory=(device == "cuda"))

    model = model.to(device)
    rng = np.random.default_rng(seed)

    y_true_list, y_pred_list = [], []
    for batch in loader:
        cpg_ids     = batch["cpg_ids"].to(device)
        beta_values = batch["beta_values"].to(device)
        attn_mask   = batch["attention_mask"].to(device)
        ages        = batch["age"].cpu().numpy()

        # Random masking: zero out beta_values for mask_fraction of CpG tokens
        if mask_fraction > 0:
            B, L = beta_values.shape
            # Skip position 0 (CLS), mask among positions 1..L-1
            n_mask = int(mask_fraction * (L - 1))
            for b in range(B):
                mask_idx = rng.choice(L - 1, size=n_mask, replace=False) + 1
                beta_values[b, mask_idx] = 0.0  # zero = "unknown"

        # MethylationAgeRegressorLlama has no forward() — use internal methods
        cls   = model._encode_cls(cpg_ids, beta_values, attn_mask)  # [B, D]
        preds = model.age_head(cls).squeeze(-1)                      # [B] z-scored
        # Denormalize from z-score to years
        preds = (preds * model.age_std + model.age_mean).cpu().numpy()

        y_true_list.append(ages)
        y_pred_list.append(preds)

    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)
    return y_true, y_pred


# ─────────────────────────────────────────────────────────────────────────────
# Baseline models (ElasticNet / Ridge)
# ─────────────────────────────────────────────────────────────────────────────

def train_baseline(data_path, age_col, split_col):
    """
    Fit ElasticNet and Ridge on the raw methylation matrix (train split).
    Returns fitted (elasticnet, ridge, scaler, X_test, y_test).
    """
    import anndata
    from sklearn.linear_model import ElasticNetCV, RidgeCV
    from sklearn.preprocessing import StandardScaler

    log.info("Loading h5ad for baseline training ...")
    try:
        adata = anndata.read_h5ad(data_path)
    except Exception:
        import h5py
        import anndata as ad
        with h5py.File(data_path, "r") as f:
            X = f["X"][:]
            obs_names = [s.decode() if isinstance(s, bytes) else s
                         for s in f["obs"]["_index"][:]]
        adata = ad.AnnData(X=X)
        adata.obs_names = obs_names

    split = adata.obs[split_col].values if split_col in adata.obs.columns else None
    ages  = adata.obs[age_col].values.astype(float)

    X_all = adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray()
    X_all = np.nan_to_num(X_all.astype(np.float32), nan=0.0)

    if split is not None:
        tr = np.isin(split, ["train", "valid"])
        te = split == "test"
    else:
        rng = np.random.default_rng(42)
        idx = rng.permutation(len(ages))
        tr = np.zeros(len(ages), dtype=bool)
        te = np.zeros(len(ages), dtype=bool)
        tr[idx[:int(0.8 * len(idx))]] = True
        te[idx[int(0.8 * len(idx)):]] = True

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_all[tr])
    X_te = scaler.transform(X_all[te])
    y_tr = ages[tr]
    y_te = ages[te]

    log.info(f"Fitting ElasticNetCV (train={tr.sum()}) ...")
    en = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 1.0], cv=3, max_iter=2000, n_jobs=4)
    en.fit(X_tr, y_tr)

    log.info(f"Fitting RidgeCV ...")
    ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0], cv=5)
    ridge.fit(X_tr, y_tr)

    return en, ridge, scaler, X_te, y_te


def predict_baseline_masked(model, scaler, X_te, mask_fraction: float, seed: int):
    """Predict with baseline after masking `mask_fraction` of features."""
    rng = np.random.default_rng(seed)
    X = X_te.copy()
    if mask_fraction > 0:
        n_mask = int(mask_fraction * X.shape[1])
        for i in range(X.shape[0]):
            idx = rng.choice(X.shape[1], size=n_mask, replace=False)
            X[i, idx] = 0.0
    X_scaled = scaler.transform(X) if hasattr(scaler, "transform") else X
    return model.predict(X_scaled)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def medae(y_true, y_pred):
    return float(np.median(np.abs(y_true - y_pred)))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_robustness(results: dict, outdir: Path):
    """
    Line plot: MedAE vs % missingness for all methods.
    Style matches MethylGPT Fig 4f.
    """
    fig_dir = outdir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    colors = {
        "MethylLlama": "#4DBBD5",
        "Null":        "#ADB6B6",
    }
    linestyles = {
        "MethylLlama": "-",
        "Null":        ":",
    }

    fig, ax = plt.subplots(figsize=(7, 5))
    mask_levels = sorted(results[list(results.keys())[0]].keys())

    for method, data in results.items():
        x = [m for m in mask_levels]
        y = [data[m] for m in mask_levels]
        ax.plot(x, y,
                label=method,
                color=colors.get(method, "#888888"),
                linestyle=linestyles.get(method, "-"),
                marker="o", markersize=5, linewidth=2)
        # Annotate last point
        ax.annotate(f"{y[-1]:.1f}", xy=(x[-1], y[-1]),
                    xytext=(3, 0), textcoords="offset points",
                    fontsize=7, color=colors.get(method, "#888888"))

    ax.set_xlabel("Input Data Missingness (%)", fontsize=11)
    ax.set_ylabel("Median Absolute Error (years)", fontsize=11)
    ax.set_title("Age Prediction Robustness to Missing Data", fontsize=12, fontweight="bold")
    ax.set_xticks(mask_levels)
    ax.legend(fontsize=10, framealpha=0.5)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = fig_dir / "robustness_missing_data.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved {out.name}")

    # Also save MAE version
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    results = {}  # method → {mask_level → MedAE}

    # ── MethylLlama robustness ────────────────────────────────────────────────
    log.info("\n[1] Loading fine-tuned MethylLlama ...")
    model = load_finetune_model(args.checkpoint)

    results["MethylLlama"] = {}
    for pct in args.mask_levels:
        log.info(f"  Masking {pct}% ...")
        y_true, y_pred = predict_with_masking(
            model, args.data, args.tokenizer,
            args.batch_size, args.device,
            mask_fraction=pct / 100.0,
            seed=args.seed,
        )
        med = medae(y_true, y_pred)
        results["MethylLlama"][pct] = med
        log.info(f"    MedAE = {med:.2f} yr")

    # ── Null baseline ─────────────────────────────────────────────────────────
    import anndata
    adata = anndata.read_h5ad(args.data, backed="r")
    test_mask = adata.obs["split"] == "test" if "split" in adata.obs.columns else slice(None)
    y_te = adata.obs.loc[test_mask, args.age_col].values.astype(float)
    null_pred = np.full_like(y_te, float(y_te.mean()))
    results["Null"] = {pct: medae(y_te, null_pred) for pct in args.mask_levels}

    # ── Save results table ────────────────────────────────────────────────────
    rows = []
    for method, data in results.items():
        for pct, med in data.items():
            rows.append({"method": method, "mask_pct": pct, "MedAE_yr": med})
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "robustness_results.csv", index=False)
    log.info(f"\nResults table → {outdir}/robustness_results.csv")
    print(df.pivot(index="mask_pct", columns="method", values="MedAE_yr").round(2).to_string())

    # ── Plot ──────────────────────────────────────────────────────────────────
    log.info("\nGenerating plots ...")
    plot_robustness(results, outdir)

    log.info(f"\nDone. Outputs → {outdir}/")
    log.info(f"  robustness_results.csv")
    log.info(f"  figures/robustness_missing_data.png")


if __name__ == "__main__":
    main()
