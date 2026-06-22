#!/usr/bin/env python3
"""
reconstruction_baselines.py
============================
Diagnostic experiment B: Does the WCED decoder actually use the CLS embedding,
or does it reconstruct methylation by memorising per-CpG population means?

Three baselines compared against the real model reconstruction:

  B1 — Per-CpG training-mean baseline
       Replace decoder output with E[beta_i] computed on the training set.
       This is the trivial floor: a decoder that ignores CLS entirely.
       If model MSE ≈ B1 MSE → decoder has learned nothing sample-specific.

  B3 — Shuffled-CLS baseline
       Run real encoder on each sample, then shuffle the CLS vectors across
       the batch before passing to decoder.  CLS still contains real embeddings
       but they are disconnected from the correct sample.  If model MSE ≈ B3 MSE
       → decoder reconstructs from population statistics stored in its weights,
       not from individual CLS information.

  B4 — Random-Gaussian CLS baseline
       Replace CLS with N(0,1) noise of matching dimension.  A stronger test:
       if model MSE ≈ B4 MSE → decoder has entirely learned to ignore CLS.

Shape alignment:
  The WCED decoder was pretrained on all 49,156 CpGs → outputs [B, 49,156].
  The fine-tuning dataset has 19,608 CpGs → batch beta_values shape [B, 19,609].
  We align using cpg_ids: each CpG's tokenizer vocab ID gives its decoder index
  via  decoder_idx = vocab_id - n_special_tokens  (n_special = 49161 - 49156 = 5).

Output (saved to --outdir):
  reconstruction_baselines.json   — scalar metrics for all conditions
  reconstruction_baselines.csv    — per-sample MSE for model / B1 / B3 / B4

Usage (cluster, see run_reconstruction_baselines.sh):
  python scripts/repr_analysis/reconstruction_baselines.py \\
      --checkpoint outputs/pretrain-llama-wced/.../epoch=98-val_loss=0.0059.ckpt \\
      --data /path/to/finetuning_19608_clean_stratified_no_outliers.h5ad \\
      --tokenizer tokenizer_llama_pretrain49k \\
      --outdir outputs/repr_analysis/reconstruction_baselines_JOBID \\
      --batch_size 64 --device cuda
"""

import argparse
import json
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
    p.add_argument("--checkpoint",  required=True,
                   help="WCEDLlamaModule checkpoint (.ckpt)")
    p.add_argument("--data",        required=True,
                   help="h5ad path (finetuning_19608_clean_stratified_no_outliers.h5ad)")
    p.add_argument("--tokenizer",   required=True,
                   help="tokenizer_llama_pretrain49k directory")
    p.add_argument("--outdir",      default="outputs/repr_analysis/reconstruction_baselines")
    p.add_argument("--batch_size",  type=int, default=64)
    p.add_argument("--device",      default="cuda")
    p.add_argument("--n_batches",   type=int, default=0,
                   help="Limit to N batches for quick test (0 = all)")
    return p.parse_args()


def masked_mse(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    """Per-sample MSE over valid (masked) positions only."""
    diff2 = (pred - tgt).pow(2) * mask.float()
    return (diff2.sum(dim=1) / mask.float().sum(dim=1).clamp(min=1)).cpu().numpy()


@torch.no_grad()
def run_baselines(model, loader, device, n_batches=0):
    """
    Returns per-sample arrays:
      model_mse  — MSE(decoder(real_cls),     beta_values)
      b3_mse     — MSE(decoder(shuffled_cls), beta_values)
      b4_mse     — MSE(decoder(random_cls),   beta_values)
      labels     — age labels (NaN if missing)
      splits     — split strings

    B1 (per-CpG training mean) is computed after the loop.

    Shape alignment:
      decoder output:  [B, decoder_vocab_size=49156]
      beta target:     [B, n_cpg=19608]  from beta_values[:, 1:]
      cpg_ids[:, 1:]:  [B, n_cpg]  — tokenizer vocab IDs for the 19608 CpGs

      decoder_index = vocab_id - n_special_tokens
      where n_special_tokens = emb_vocab_size - decoder_vocab_size = 49161 - 49156 = 5
    """
    # Compute vocab offset: embedding table covers special tokens + CpGs,
    # decoder only covers CpGs.  n_special = difference in sizes.
    emb_vocab_size    = model.encoder.embeddings.cpg_sites_embeddings.weight.shape[0]
    # Get decoder vocab_size from the final Linear layer in WCEDDecoder
    decoder_final_linear = [m for m in model.decoder.decoder.modules()
                             if isinstance(m, torch.nn.Linear)][-1]
    decoder_vocab_size = decoder_final_linear.out_features
    n_special = emb_vocab_size - decoder_vocab_size
    log.info(f"Embedding vocab={emb_vocab_size}, decoder vocab={decoder_vocab_size}, "
             f"n_special_tokens={n_special}")

    model_mses, b3_mses, b4_mses = [], [], []
    all_target_betas = []
    label_list, split_list = [], []

    for i, batch in enumerate(loader):
        if n_batches > 0 and i >= n_batches:
            break

        cpg_ids     = batch["cpg_ids"].to(device)       # [B, L]
        beta_values = batch["beta_values"].to(device)   # [B, L]
        attn_mask   = batch["attention_mask"].to(device)

        # Encoder → real CLS
        input_ids = torch.stack([cpg_ids.float(), beta_values], dim=1)
        enc_out   = model.encoder(input_ids=input_ids, attention_mask=attn_mask)
        cls_real  = enc_out.pooler_output   # [B, D]

        # Targets: actual beta values for all input CpGs (skip CLS at position 0)
        # With input_ratio=1.0 all n_cpg positions hold real measured betas in [0, 1].
        target = beta_values[:, 1:].float()          # [B, n_cpg]
        valid  = target >= 0                         # True for real CpG values
        B, n_cpg = target.shape

        # Align decoder output [B, decoder_vocab_size] to our n_cpg CpGs.
        # cpg_ids[:, 1:] are tokenizer vocab IDs (1-indexed CpGs, 0..n_special-1 = special).
        cpg_vocab_ids = cpg_ids[:, 1:].long()                               # [B, n_cpg]
        dec_idx       = (cpg_vocab_ids - n_special).clamp(0,
                         decoder_vocab_size - 1)                             # [B, n_cpg]

        # Model: real CLS → decoder → gather CpG positions
        recon_real_all = model.decoder(cls_real)                             # [B, 49156]
        recon_real     = recon_real_all.gather(1, dec_idx)                   # [B, 19608]
        model_mses.append(masked_mse(recon_real, target, valid))

        # B3: shuffled CLS (real embeddings, wrong samples)
        perm       = torch.randperm(B, device=device)
        cls_shuf   = cls_real[perm]
        recon_b3   = model.decoder(cls_shuf).gather(1, dec_idx)
        b3_mses.append(masked_mse(recon_b3, target, valid))

        # B4: random Gaussian CLS
        cls_rand   = torch.randn_like(cls_real)
        recon_b4   = model.decoder(cls_rand).gather(1, dec_idx)
        b4_mses.append(masked_mse(recon_b4, target, valid))

        all_target_betas.append(target.cpu().numpy())

        age = batch.get("age", None)
        label_list.extend(age.float().numpy().tolist() if age is not None
                          else [float("nan")] * B)
        split = batch.get("split", ["unknown"] * B)
        split_list.extend(split.tolist() if isinstance(split, torch.Tensor) else split)

        if (i + 1) % 20 == 0:
            log.info(f"  batch {i+1}/{len(loader)}")

    # B1: per-CpG training-set mean (trivial floor)
    all_np   = np.concatenate(all_target_betas, axis=0)   # [N, n_cpg]
    cpg_mean = all_np.mean(axis=0, keepdims=True)
    b1_mse   = np.mean((cpg_mean - all_np) ** 2, axis=1)

    return {
        "model_mse": np.concatenate(model_mses),
        "b1_mse":    b1_mse,
        "b3_mse":    np.concatenate(b3_mses),
        "b4_mse":    np.concatenate(b4_mses),
        "labels":    np.array(label_list),
        "splits":    split_list,
    }


def summarise(results: dict) -> dict:
    summary = {}
    for key in ["model_mse", "b1_mse", "b3_mse", "b4_mse"]:
        arr = results[key]
        summary[key] = {
            "mean":   float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std":    float(np.std(arr)),
            "p10":    float(np.percentile(arr, 10)),
            "p90":    float(np.percentile(arr, 90)),
        }

    for base in ["b1", "b3", "b4"]:
        ratio = summary["model_mse"]["mean"] / summary[f"{base}_mse"]["mean"]
        summary[f"ratio_model_vs_{base}"] = float(ratio)

    log.info("=== Reconstruction Baseline Summary ===")
    log.info(f"  Model MSE (real CLS):    {summary['model_mse']['mean']:.6f}")
    log.info(f"  B1  MSE  (cpg mean):     {summary['b1_mse']['mean']:.6f}")
    log.info(f"  B3  MSE  (shuffled CLS): {summary['b3_mse']['mean']:.6f}")
    log.info(f"  B4  MSE  (random CLS):   {summary['b4_mse']['mean']:.6f}")
    log.info(f"  model / B1  = {summary['ratio_model_vs_b1']:.4f}")
    log.info(f"  model / B3  = {summary['ratio_model_vs_b3']:.4f}")
    log.info(f"  model / B4  = {summary['ratio_model_vs_b4']:.4f}")
    if summary["ratio_model_vs_b3"] > 0.95:
        log.warning("  *** model ≈ shuffled CLS → decoder may not use CLS ***")
    return summary


def main():
    args   = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    from bmfm_methylation.llama.finetune_llama import load_wced_llama_checkpoint
    module = load_wced_llama_checkpoint(args.checkpoint)
    module.eval()
    module.to(device)

    from bmfm_targets.tokenization import MultiFieldTokenizer
    from bmfm_methylation.shared.data_module import MethylationDataset, WCEDCollator

    tokenizer  = MultiFieldTokenizer.from_pretrained(args.tokenizer)
    dataset    = MethylationDataset(h5ad_path=args.data, split=None, normalize_age=False)
    cpg_sites  = dataset.cpg_sites
    vocab_size = len(cpg_sites)
    log.info(f"Dataset: {len(dataset)} samples × {vocab_size} CpGs")

    collator = WCEDCollator(
        tokenizer=tokenizer, cpg_sites=cpg_sites,
        vocab_size=vocab_size, input_ratio=1.0, contrastive=False,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collator,
                        shuffle=False, num_workers=4,
                        pin_memory=(device == "cuda"))

    log.info("Running reconstruction baselines ...")
    results = run_baselines(module, loader, device, n_batches=args.n_batches)
    summary = summarise(results)

    json_path = outdir / "reconstruction_baselines.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Saved → {json_path}")

    n  = len(results["model_mse"])
    df = pd.DataFrame({
        "split":     results["splits"][:n],
        "label_age": results["labels"][:n],
        "model_mse": results["model_mse"],
        "b1_mse":    results["b1_mse"],
        "b3_mse":    results["b3_mse"],
        "b4_mse":    results["b4_mse"],
    })
    csv_path = outdir / "reconstruction_baselines.csv"
    df.to_csv(csv_path, index=False)
    log.info(f"Saved → {csv_path}")
    log.info("Done.")


if __name__ == "__main__":
    main()
