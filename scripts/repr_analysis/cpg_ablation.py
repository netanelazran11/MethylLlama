#!/usr/bin/env python3
"""
cpg_ablation.py  —  CpG Ablation Test
=======================================
Tests whether MethylLlama depends on the most-attended CpG sites.

At inference time, CpG positions are zeroed (masked) according to different
strategies, then age prediction performance is measured.  If removing the
top-attended CpGs causes a similar drop to removing the same number of random
CpGs, the model does NOT depend on a sparse set of sites.

Conditions tested
-----------------
  baseline          — no masking
  top-10            — mask 10 highest-attention CpGs
  top-100           — mask 100 highest-attention CpGs
  top-1000          — mask 1 000 highest-attention CpGs
  random-10         — mask 10 randomly chosen CpGs   (3 seeds → mean ± std)
  random-100        — mask 100 randomly chosen CpGs
  random-1000       — mask 1 000 randomly chosen CpGs

Metrics
-------
  R²   MAE (yr)   MedAE (yr)   (age prediction on test split)

Usage
-----
  python scripts/repr_analysis/cpg_ablation.py \\
      --finetune_checkpoint  outputs/finetune-llama-small/.../epoch=127-val_medae=3.5625.ckpt \\
      --attention_npy        outputs/repr_analysis/attention_XXXXX/cpg_attention.npy \\
      --data                 /path/to/finetuning_19608_clean_stratified_no_outliers.h5ad \\
      --tokenizer            tokenizer_llama_pretrain49k \\
      --outdir               outputs/repr_analysis/ablation
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

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="CpG ablation test — distributed attention proof")
    p.add_argument("--finetune_checkpoint", required=True,
                   help="Fine-tuned .ckpt (MethylationAgeRegressorLlama)")
    p.add_argument("--attention_npy",       required=True,
                   help="cpg_attention.npy from attention_analysis.py [N x n_cpg]")
    p.add_argument("--data",                required=True,
                   help="Finetune h5ad (19k, with 'split' column)")
    p.add_argument("--tokenizer",           required=True)
    p.add_argument("--outdir",              default="outputs/repr_analysis/ablation")
    p.add_argument("--batch_size",          type=int, default=32)
    p.add_argument("--device",              default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--age_col",             default="age")
    p.add_argument("--split_col",           default="split")
    p.add_argument("--top_k_sizes",         nargs="+", type=int,
                   default=[10, 100, 1000],
                   help="Sizes of top-k ablation sets")
    p.add_argument("--n_random_seeds",      type=int, default=3,
                   help="Seeds for random ablation (mean ± std reported)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Model + data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_finetune_model(ckpt_path: str):
    from bmfm_methylation.llama.finetune_llama import load_finetune_llama_checkpoint
    log.info(f"Loading fine-tuned model: {ckpt_path}")
    model = load_finetune_llama_checkpoint(ckpt_path)
    model.eval()
    return model


def build_loader(data_path, tokenizer_path, batch_size):
    from bmfm_targets.tokenization import MultiFieldTokenizer
    from bmfm_methylation.shared.data_module import MethylationDataset, WCEDCollator

    tok     = MultiFieldTokenizer.from_pretrained(tokenizer_path)
    dataset = MethylationDataset(h5ad_path=data_path, split="test", normalize_age=False)
    cpg_sites = dataset.cpg_sites
    collator = WCEDCollator(
        tokenizer=tok, cpg_sites=cpg_sites,
        vocab_size=len(cpg_sites), input_ratio=1.0, contrastive=False,
    )
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator,
                        shuffle=False, num_workers=2, pin_memory=False)
    return loader, cpg_sites


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-12))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def medae(y_true, y_pred):
    return float(np.median(np.abs(y_true - y_pred)))


# ─────────────────────────────────────────────────────────────────────────────
# Masked inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_masked(model, loader, device, mask_indices: np.ndarray):
    """
    Run model on test set with specific CpG positions zeroed out.
    mask_indices: int array of CpG positions (0-indexed among CpG tokens, not counting CLS).
    Returns (y_true, y_pred).
    """
    model = model.to(device)
    y_true_list, y_pred_list = [], []

    for batch in loader:
        cpg_ids     = batch["cpg_ids"].to(device)
        beta_values = batch["beta_values"].to(device)
        attn_mask   = batch["attention_mask"].to(device)
        ages        = batch["age"].cpu().numpy()

        # Zero out the specified CpG positions (offset +1 for CLS token)
        if len(mask_indices) > 0:
            tok_indices = mask_indices + 1  # +1 because position 0 is CLS
            # Clip to valid range
            tok_indices = tok_indices[tok_indices < beta_values.shape[1]]
            beta_values[:, tok_indices] = 0.0

        cls   = model._encode_cls(cpg_ids, beta_values, attn_mask)
        preds = model.age_head(cls).squeeze(-1)
        preds = (preds * model.age_std + model.age_mean).cpu().numpy()

        y_true_list.append(ages)
        y_pred_list.append(preds)

    return np.concatenate(y_true_list), np.concatenate(y_pred_list)


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation(model, loader, device,
                 mean_attn: np.ndarray, top_k_sizes, n_random_seeds):
    """
    Run all ablation conditions and return a results dict.
    mean_attn: [n_cpg] average attention per CpG site.
    """
    n_cpg    = len(mean_attn)
    ranked   = np.argsort(mean_attn)[::-1]   # CpGs sorted high→low attention
    results  = {}

    # ── Baseline (no masking) ────────────────────────────────────────────────
    log.info("  Baseline (no masking) ...")
    y_true, y_pred = predict_masked(model, loader, device, np.array([], dtype=int))
    results["baseline"] = {
        "label":  "Baseline\n(no masking)",
        "R2":     r2(y_true, y_pred),
        "MAE":    mae(y_true, y_pred),
        "MedAE":  medae(y_true, y_pred),
        "R2_std":    0.0, "MAE_std":   0.0, "MedAE_std": 0.0,
    }
    log.info(f"    R²={results['baseline']['R2']:.3f}  "
             f"MedAE={results['baseline']['MedAE']:.2f}yr")

    for k in top_k_sizes:
        # ── Top-k ablation ───────────────────────────────────────────────────
        key = f"top-{k}"
        log.info(f"  Masking top-{k} attention CpGs ...")
        mask_idx = ranked[:k]
        y_true, y_pred = predict_masked(model, loader, device, mask_idx)
        results[key] = {
            "label":  f"Remove top-{k}\nattention CpGs",
            "R2":     r2(y_true, y_pred),
            "MAE":    mae(y_true, y_pred),
            "MedAE":  medae(y_true, y_pred),
            "R2_std":    0.0, "MAE_std": 0.0, "MedAE_std": 0.0,
        }
        log.info(f"    R²={results[key]['R2']:.3f}  MedAE={results[key]['MedAE']:.2f}yr")

        # ── Random-k ablation (multiple seeds) ───────────────────────────────
        rkey = f"random-{k}"
        log.info(f"  Masking random-{k} CpGs ({n_random_seeds} seeds) ...")
        r2s, maes, medaes = [], [], []
        for seed in range(n_random_seeds):
            rng      = np.random.default_rng(seed)
            rand_idx = rng.choice(n_cpg, size=k, replace=False)
            yt, yp   = predict_masked(model, loader, device, rand_idx)
            r2s.append(r2(yt, yp))
            maes.append(mae(yt, yp))
            medaes.append(medae(yt, yp))

        results[rkey] = {
            "label":     f"Remove random-{k}\nCpGs (mean±std)",
            "R2":        float(np.mean(r2s)),
            "MAE":       float(np.mean(maes)),
            "MedAE":     float(np.mean(medaes)),
            "R2_std":    float(np.std(r2s)),
            "MAE_std":   float(np.std(maes)),
            "MedAE_std": float(np.std(medaes)),
        }
        log.info(f"    R²={results[rkey]['R2']:.3f}±{results[rkey]['R2_std']:.3f}  "
                 f"MedAE={results[rkey]['MedAE']:.2f}±{results[rkey]['MedAE_std']:.2f}yr")

    return results, y_true  # y_true same across all (test set is fixed)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_ablation(results: dict, outdir: Path, top_k_sizes):
    fig_dir = outdir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Order: baseline, then for each k: top-k, random-k
    keys_ordered = ["baseline"]
    for k in top_k_sizes:
        keys_ordered += [f"top-{k}", f"random-{k}"]

    labels  = [results[k]["label"]  for k in keys_ordered]
    r2_vals = [results[k]["R2"]     for k in keys_ordered]
    r2_errs = [results[k]["R2_std"] for k in keys_ordered]
    me_vals = [results[k]["MedAE"]  for k in keys_ordered]
    me_errs = [results[k]["MedAE_std"] for k in keys_ordered]

    colors = []
    for k in keys_ordered:
        if k == "baseline":       colors.append("#117A65")
        elif k.startswith("top"): colors.append("#E64B35")
        else:                     colors.append("#4DBBD5")

    x = np.arange(len(keys_ordered))
    w = 0.4

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # R² panel
    bars1 = ax1.bar(x, r2_vals, width=w, color=colors, alpha=0.85,
                    yerr=r2_errs, capsize=4, error_kw={"linewidth": 1.5})
    ax1.axhline(r2_vals[0], color="#117A65", linewidth=1.5, linestyle="--",
                alpha=0.6, label="Baseline R²")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=9, linespacing=1.3)
    ax1.set_ylabel("R²", fontsize=12); ax1.set_title("Age Prediction R²", fontsize=13, fontweight="bold")
    ax1.set_ylim(0, 1.05); ax1.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars1, r2_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    # MedAE panel
    bars2 = ax2.bar(x, me_vals, width=w, color=colors, alpha=0.85,
                    yerr=me_errs, capsize=4, error_kw={"linewidth": 1.5})
    ax2.axhline(me_vals[0], color="#117A65", linewidth=1.5, linestyle="--",
                alpha=0.6, label="Baseline MedAE")
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=9, linespacing=1.3)
    ax2.set_ylabel("Median Absolute Error (years)", fontsize=12)
    ax2.set_title("Age Prediction MedAE", fontsize=13, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars2, me_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#117A65", alpha=0.85, label="Baseline"),
        Patch(facecolor="#E64B35", alpha=0.85, label="Top-k attention CpGs removed"),
        Patch(facecolor="#4DBBD5", alpha=0.85, label="Random CpGs removed"),
    ]
    ax1.legend(handles=legend_elements, fontsize=9, loc="lower right")

    plt.suptitle("CpG Ablation Test: Does MethylLlama Depend on Specific CpG Sites?\n"
                 "If top-k ≈ random-k → distributed representation confirmed",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = fig_dir / "cpg_ablation.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved {out.name}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1. Load pre-computed average attention
    log.info(f"[1/4] Loading attention scores: {args.attention_npy}")
    cpg_attention = np.load(args.attention_npy)          # [N, n_cpg]
    mean_attn     = cpg_attention.mean(axis=0)           # [n_cpg]
    n_cpg         = mean_attn.shape[0]
    log.info(f"  CpG attention matrix: {cpg_attention.shape}  "
             f"mean={mean_attn.mean():.2e}  max={mean_attn.max():.2e}  "
             f"uniform_ref={1/n_cpg:.2e}")
    np.save(outdir / "mean_cpg_attention.npy", mean_attn)

    # 2. Load model + dataloader
    log.info("[2/4] Loading fine-tuned model ...")
    model = load_finetune_model(args.finetune_checkpoint)

    log.info("[3/4] Building test dataloader ...")
    loader, cpg_sites = build_loader(args.data, args.tokenizer, args.batch_size)
    log.info(f"  Test samples: {len(loader.dataset):,}  CpG sites: {n_cpg:,}")

    # 3. Run ablation
    log.info("[4/4] Running ablation conditions ...")
    results, y_true = run_ablation(
        model, loader, args.device,
        mean_attn, args.top_k_sizes, args.n_random_seeds,
    )

    # 4. Save results table
    rows = []
    for key, res in results.items():
        rows.append({
            "condition": key,
            "label":     res["label"].replace("\n", " "),
            "R2":        res["R2"],    "R2_std":    res["R2_std"],
            "MAE":       res["MAE"],   "MAE_std":   res["MAE_std"],
            "MedAE":     res["MedAE"], "MedAE_std": res["MedAE_std"],
        })
    df = pd.DataFrame(rows)
    csv_path = outdir / "ablation_results.csv"
    df.to_csv(csv_path, index=False)
    log.info(f"\nResults table saved → {csv_path}")
    print("\n" + df.to_string(index=False))

    # 5. Plot
    plot_ablation(results, outdir, args.top_k_sizes)

    # 6. Print interpretation
    baseline_r2 = results["baseline"]["R2"]
    for k in args.top_k_sizes:
        top_r2  = results[f"top-{k}"]["R2"]
        rand_r2 = results[f"random-{k}"]["R2"]
        drop_top  = baseline_r2 - top_r2
        drop_rand = baseline_r2 - rand_r2
        ratio     = drop_top / (drop_rand + 1e-6)
        log.info(f"\n  k={k}: top-k drop={drop_top:.3f}  random drop={drop_rand:.3f}  "
                 f"ratio={ratio:.2f}  "
                 f"({'similar — DISTRIBUTED' if ratio < 2.0 else 'top-k worse — SPARSE'})")

    log.info(f"\nDone. Outputs → {outdir}/")
    log.info(f"  ablation_results.csv")
    log.info(f"  figures/cpg_ablation.png")


if __name__ == "__main__":
    main()
