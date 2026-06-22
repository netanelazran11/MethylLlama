"""
Evaluate a MethylationAgeRegressorLlama checkpoint on val and test splits.

Prints and saves MedAE, MAE, R², PCC, SCC for both splits.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader

from bmfm_methylation.llama.finetune_llama import load_finetune_llama_checkpoint
from bmfm_methylation.shared.data_module import MethylationDataset, WCEDCollator
from bmfm_targets.tokenization import MultiFieldTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def median_absolute_error(y_true, y_pred):
    return float(np.median(np.abs(np.array(y_true) - np.array(y_pred))))


@torch.no_grad()
def run_split(model, dataset, collator, batch_size, device, split_name):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    all_pred, all_true = [], []
    for batch in loader:
        cpg_ids        = batch["cpg_ids"].to(device)
        beta_values    = batch["beta_values"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        ages           = batch["age"].to(device)

        out = model._shared_step(
            {"cpg_ids": cpg_ids, "beta_values": beta_values,
             "attention_mask": attention_mask, "age": ages},
            stage="test",
        )
        pred_years = out["age_pred_years"]
        true_years = out["age_label_years"]

        all_pred.extend(pred_years.detach().cpu().numpy())
        all_true.extend(true_years.detach().cpu().numpy())

    y_pred = np.array(all_pred)
    y_true = np.array(all_true)

    medae = median_absolute_error(y_true, y_pred)
    mae   = float(mean_absolute_error(y_true, y_pred))
    r2    = float(r2_score(y_true, y_pred))
    pcc,_ = pearsonr(y_true, y_pred)
    scc,_ = spearmanr(y_true, y_pred)
    rmse  = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    logger.info(
        f"  {split_name:5s}  MedAE={medae:.4f}yr  MAE={mae:.4f}yr  "
        f"RMSE={rmse:.4f}yr  R²={r2:.4f}  PCC={pcc:.4f}  SCC={scc:.4f}  n={len(y_true)}"
    )
    return {
        "split": split_name,
        "n": int(len(y_true)),
        "medae": medae,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "pcc": float(pcc),
        "scc": float(scc),
    }, y_pred, y_true


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--h5ad",       required=True)
    parser.add_argument("--tokenizer",  required=True)
    parser.add_argument("--outdir",     required=True)
    parser.add_argument("--subset_k",   type=int, default=49156)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--age_col",   default="age")
    parser.add_argument("--split_col", default="split")
    parser.add_argument("--filter_age_outliers", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Load checkpoint ───────────────────────────────────────────────────────
    model = load_finetune_llama_checkpoint(args.checkpoint)
    model.eval().to(device)
    logger.info(f"age_mean={model.age_mean:.4f}  age_std={model.age_std:.4f}")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = MultiFieldTokenizer.from_pretrained(args.tokenizer)

    # ── Load data (need cpg_sites for collator) ───────────────────────────────
    logger.info(f"Loading {args.h5ad}")

    # Build a dataset just to get cpg_sites list
    ds_ref = MethylationDataset(
        h5ad_path=args.h5ad,
        split="train",
        age_column=args.age_col,
        split_column=args.split_col,
        filter_age_outliers=args.filter_age_outliers,
    )
    cpg_sites = ds_ref.cpg_sites
    logger.info(f"CpG sites: {len(cpg_sites)}")

    # ── Collator (mirrors finetune_llama.py — input_ratio=1.0 for eval) ───────
    collator = WCEDCollator(
        tokenizer=tokenizer,
        cpg_sites=cpg_sites,
        vocab_size=args.subset_k,
        input_ratio=1.0,
        fixed_subset_seed=args.seed,
        contrastive=False,
    )

    # ── Evaluate each split ───────────────────────────────────────────────────
    # CRITICAL: override each split's age_mean/age_std with the training statistics
    # stored in the checkpoint. MethylationDataModule does this during training
    # (lines 551-568 of data_module.py) — we must replicate it here.
    results = []

    for split in ("valid", "test"):
        logger.info(f"Evaluating {split} split ...")
        ds = MethylationDataset(
            h5ad_path=args.h5ad,
            split=split,
            age_column=args.age_col,
            split_column=args.split_col,
            filter_age_outliers=args.filter_age_outliers,
        )
        ds.age_mean = model.age_mean
        ds.age_std  = model.age_std
        logger.info(f"  {split}: {len(ds)} samples  age_mean={ds.age_mean:.2f}  age_std={ds.age_std:.2f}")

        metrics, _, _ = run_split(model, ds, collator, args.batch_size, device, split)
        results.append(metrics)

    # ── Save ──────────────────────────────────────────────────────────────────
    with open(outdir / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    summary_lines = [
        "=" * 65,
        "MethylLlama Checkpoint Evaluation",
        f"Checkpoint: {args.checkpoint}",
        "=" * 65,
        f"{'Split':<8} {'N':>6} {'MedAE':>8} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'PCC':>8} {'SCC':>8}",
        "-" * 65,
    ]
    for r in results:
        summary_lines.append(
            f"{r['split']:<8} {r['n']:>6} {r['medae']:>8.4f} {r['mae']:>8.4f} "
            f"{r['rmse']:>8.4f} {r['r2']:>8.4f} {r['pcc']:>8.4f} {r['scc']:>8.4f}"
        )
    summary_lines.append("=" * 65)
    summary = "\n".join(summary_lines)
    print("\n" + summary)

    with open(outdir / "eval_summary.txt", "w") as f:
        f.write(summary + "\n")

    logger.info(f"Results saved to {outdir}/")


if __name__ == "__main__":
    main()
