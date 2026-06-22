#!/usr/bin/env python3
"""
extract_age_predictions.py
==========================
Run the full fine-tuned MethylationAgeRegressorLlama model on the 19k
finetune dataset and save per-sample predicted age to CSV.

Output: <outdir>/age_predictions.csv
  columns: sample_id, actual_age, predicted_age, split, tissue

Usage (cluster):
  python scripts/repr_analysis/extract_age_predictions.py \
      --checkpoint  outputs/finetune-llama-small/.../epoch=127-val_medae=3.5625.ckpt \
      --data        /path/to/finetuning_19608.h5ad \
      --tokenizer   tokenizer_llama_pretrain49k \
      --outdir      outputs/repr_analysis/age_predictions_JOBID \
      --batch_size  64 --device cuda
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",      required=True)
    p.add_argument("--data",            required=True)
    p.add_argument("--tokenizer",       required=True)
    p.add_argument("--outdir",          default="outputs/repr_analysis/age_predictions")
    p.add_argument("--batch_size",      type=int, default=64)
    p.add_argument("--device",          default="cuda")
    p.add_argument("--metadata",        default=None)
    p.add_argument("--metadata_id_col", default="GSM_ID")
    return p.parse_args()


def main():
    args   = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    from bmfm_methylation.llama.finetune_llama import load_finetune_llama_checkpoint
    model = load_finetune_llama_checkpoint(args.checkpoint)
    model.eval()
    model.to(device)
    log.info(f"age_mean={model.age_mean:.2f}  age_std={model.age_std:.2f}")

    # ── Build dataloader (same pattern as cls_probing_analysis.py) ────────────
    from bmfm_targets.tokenization import MultiFieldTokenizer
    from bmfm_methylation.shared.data_module import MethylationDataset, WCEDCollator

    tokenizer = MultiFieldTokenizer.from_pretrained(args.tokenizer)
    dataset   = MethylationDataset(h5ad_path=args.data, split=None, normalize_age=False)
    cpg_sites = dataset.cpg_sites
    log.info(f"Dataset: {len(dataset)} samples × {len(cpg_sites)} CpGs")

    collator = WCEDCollator(
        tokenizer=tokenizer, cpg_sites=cpg_sites,
        vocab_size=len(cpg_sites), input_ratio=1.0, contrastive=False,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collator,
                        shuffle=False, num_workers=4,
                        pin_memory=(device == "cuda"))

    # ── Inference ─────────────────────────────────────────────────────────────
    all_sample_ids, all_actual, all_predicted = [], [], []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            cpg_ids     = batch["cpg_ids"].to(device)
            beta_values = batch["beta_values"].to(device)
            attn_mask   = batch["attention_mask"].to(device)
            input_ids   = torch.stack([cpg_ids.float(), beta_values], dim=1)

            # Forward through encoder
            out = model.encoder(input_ids=input_ids, attention_mask=attn_mask)
            cls = out.pooler_output  # [B, hidden_size]

            # Age head: z-score → years
            age_pred_norm = model.age_head(cls).squeeze(-1)
            age_pred_yr   = (age_pred_norm * model.age_std + model.age_mean).cpu().numpy()

            # Actual age (raw, not normalized)
            actual_age = batch.get("age", None)
            if actual_age is not None:
                actual_yr = actual_age.float().numpy()
            else:
                actual_yr = np.full(len(age_pred_yr), float("nan"))

            sample_ids = batch.get("sample_id", [f"sample_{i*args.batch_size+j}"
                                                   for j in range(len(age_pred_yr))])
            if not isinstance(sample_ids, list):
                sample_ids = sample_ids.tolist()

            all_sample_ids.extend(sample_ids)
            all_predicted.extend(age_pred_yr.tolist())
            all_actual.extend(actual_yr.tolist())

            if (i + 1) % 20 == 0:
                log.info(f"  batch {i+1}/{len(loader)}")

    log.info(f"Inference done — {len(all_predicted)} samples")

    # ── Build output dataframe ────────────────────────────────────────────────
    df = pd.DataFrame({
        "sample_id":     all_sample_ids,
        "actual_age":    all_actual,
        "predicted_age": all_predicted,
    })

    # Join split + tissue from h5ad obs
    import anndata
    adata = anndata.read_h5ad(args.data, backed="r")
    obs   = adata.obs[["split"] + [c for c in ["tissue", "age"] if c in adata.obs.columns]].copy()
    obs.index.name = "sample_id"
    df = df.set_index("sample_id").join(obs, how="left").reset_index()

    # Join tissue from external metadata if not in h5ad
    if "tissue" not in df.columns and args.metadata and Path(args.metadata).exists():
        ext = pd.read_csv(args.metadata)
        ext = ext.drop_duplicates(subset=args.metadata_id_col).set_index(args.metadata_id_col)
        if "tissue" in ext.columns:
            df = df.set_index("sample_id").join(ext[["tissue"]], how="left").reset_index()

    out_path = outdir / "age_predictions.csv"
    df.to_csv(out_path, index=False)
    log.info(f"Saved → {out_path}  ({len(df)} rows)")

    # ── Quick metrics ─────────────────────────────────────────────────────────
    from sklearn.metrics import r2_score, median_absolute_error
    for split_name in ["train", "val", "test", None]:
        if split_name is None:
            mask = df["actual_age"].notna()
            label = "all"
        else:
            mask = (df["split"] == split_name) & df["actual_age"].notna()
            label = split_name
        if mask.sum() < 10:
            continue
        r2    = r2_score(df.loc[mask, "actual_age"], df.loc[mask, "predicted_age"])
        medae = median_absolute_error(df.loc[mask, "actual_age"], df.loc[mask, "predicted_age"])
        log.info(f"  [{label:5s}]  R²={r2:.3f}  MedAE={medae:.2f} yr  n={mask.sum()}")

    log.info("Done.")


if __name__ == "__main__":
    main()
