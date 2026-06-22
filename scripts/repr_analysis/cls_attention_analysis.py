#!/usr/bin/env python3
"""
cls_attention_analysis.py
==========================
Correct CLS-to-CpG attention extraction for MethylLlama.

MOTIVATION
----------
The existing attention_analysis.py computes a "column sum" over ALL query
positions and averages across the full sequence length (L=19,609).  At that
scale, even a 10× attention spike on 10 specific CpGs contributes only
10 × 10 / 19,609² ≈ 0 to the column average.  The result always looks
near-uniform — not because the CLS is truly non-selective, but because the
signal is geometrically diluted by 19k other query rows.

This script extracts only ROW 0 of the attention matrix — the CLS token
querying all CpG keys.  This is the correct way to ask:
  "Which CpGs does the CLS token attend to?"

WHAT IS COMPUTED (per checkpoint, per layer, per head)
------------------------------------------------------
  • mean_cls_attn[cpg]    — average CLS attention weight over the dataset
  • entropy               — Shannon entropy of the mean attention distribution
  • normalized_entropy    — entropy / log(n_cpg), in [0,1] (1 = uniform)
  • top_k_mass[k]         — fraction of total attention in top-k CpGs
                            for k ∈ {10, 50, 100, 353, 500, 1000}
  • gini                  — Gini coefficient (0 = uniform, 1 = single spike)

COMPARISON
----------
Runs on both the WCED pretrained and the fine-tuned checkpoints so you can see
whether fine-tuning makes attention more or less selective.

OUTPUT
------
  cls_attention_summary.json     — per-(checkpoint, layer, head) scalar metrics
  cls_attn_pretrained.npy        — (n_samples, n_layers, n_heads, n_cpg) float16
  cls_attn_finetuned.npy         — same for fine-tuned checkpoint
  top_cpg_indices.npz            — top-k CpG indices per (ckpt, layer, head)

Usage (cluster, see run_cls_attention.sh):
  python scripts/repr_analysis/cls_attention_analysis.py \\
      --pretrained  outputs/pretrain-llama-wced/.../epoch=98-val_loss=0.0059.ckpt \\
      --finetuned   outputs/finetune-llama-small/.../epoch=127-val_medae=3.5625.ckpt \\
      --data        /path/to/finetuning_19608_clean_stratified_no_outliers.h5ad \\
      --tokenizer   tokenizer_llama_pretrain49k \\
      --outdir      outputs/repr_analysis/cls_attention_JOBID \\
      --batch_size  16 --device cuda
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

TOP_K_VALUES = [10, 50, 100, 353, 500, 1000]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained",  required=True,
                   help="WCED pretrained checkpoint (.ckpt)")
    p.add_argument("--finetuned",   required=True,
                   help="Fine-tuned (age regression) checkpoint (.ckpt)")
    p.add_argument("--data",        required=True)
    p.add_argument("--tokenizer",   required=True)
    p.add_argument("--outdir",      default="outputs/repr_analysis/cls_attention")
    p.add_argument("--batch_size",  type=int, default=16)
    p.add_argument("--device",      default="cuda")
    p.add_argument("--n_batches",   type=int, default=0,
                   help="Limit to N batches for quick test (0 = all)")
    p.add_argument("--max_samples", type=int, default=2000,
                   help="Max samples for attention averaging (memory constraint)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Attention hook
# ---------------------------------------------------------------------------

class CLSAttentionHook:
    """
    Patches a MethylLlamaAttention layer to capture attention weights
    for the CLS query row only.

    MethylLlamaAttention uses F.scaled_dot_product_attention which does not
    return weights.  We patch the forward method to recompute CLS-query
    attention explicitly at fp32 without going through flash attention.

    Captured: self.last_attn  shape [B, n_heads, n_cpg]
              (CLS token = position 0; CpG tokens = positions 1..L-1)
    """

    def __init__(self, attn_module):
        self.module    = attn_module
        self.last_attn = None
        self._orig_forward = attn_module.forward
        attn_module.forward = self._hooked_forward

    def _hooked_forward(self, hidden_states, attention_mask=None, *args, **kwargs):
        # Run the original forward to get the normal output
        output = self._orig_forward(hidden_states, attention_mask, *args, **kwargs)

        # Recompute CLS-query attention weights manually
        # hidden_states: [B, L, D]
        with torch.no_grad():
            B, L, D = hidden_states.shape
            m = self.module

            # Project queries/keys — same as the real forward path
            q = m.q_proj(hidden_states)   # [B, L, D]
            k = m.k_proj(hidden_states)   # [B, L, D]

            H  = m.num_heads
            Dh = D // H

            # Reshape to [B, H, L, Dh]
            q = q.view(B, L, H, Dh).transpose(1, 2)
            k = k.view(B, L, H, Dh).transpose(1, 2)

            # Apply RoPE — rotary_emb takes seq_len int, returns [L, Dh] cos/sin tables
            if hasattr(m, "rotary_emb"):
                from bmfm_methylation.llama.model import apply_rotary_pos_emb as _rope
                cos, sin = m.rotary_emb(L)
                q, k = _rope(q, k, cos, sin)

            # CLS query row only: [B, H, 1, Dh]
            q_cls = q[:, :, 0:1, :]

            # Attention scores for CLS query against all keys: [B, H, 1, L]
            scale  = Dh ** -0.5
            scores = torch.matmul(q_cls, k.transpose(-2, -1)) * scale  # [B, H, 1, L]

            # Mask padding tokens (attention_mask: [B, L], 1=attend, 0=pad)
            if attention_mask is not None:
                am = attention_mask.unsqueeze(1).unsqueeze(2).float()  # [B, 1, 1, L]
                scores = scores + (1.0 - am) * -1e9

            # Softmax → weights
            weights = F.softmax(scores.float(), dim=-1)  # [B, H, 1, L]

            # Drop position 0 (CLS attending to itself): [B, H, n_cpg]
            self.last_attn = weights[:, :, 0, 1:].cpu().float()

        return output

    def remove(self):
        self.module.forward = self._orig_forward


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def load_model_for_attention(checkpoint_path: str, is_pretrained: bool):
    """Load encoder from either a WCED or fine-tuned checkpoint."""
    if is_pretrained:
        from bmfm_methylation.llama.finetune_llama import load_wced_llama_checkpoint
        module = load_wced_llama_checkpoint(checkpoint_path)
        encoder = module.encoder
    else:
        from bmfm_methylation.llama.finetune_llama import load_finetune_llama_checkpoint
        module = load_finetune_llama_checkpoint(checkpoint_path)
        encoder = module.encoder
    return encoder


@torch.no_grad()
def extract_cls_attention(encoder, loader, device, n_batches=0, max_samples=2000):
    """
    Returns:
      mean_attn  — (n_layers, n_heads, n_cpg) mean CLS attention over dataset
      all_attn   — (n_samples, n_layers, n_heads, n_cpg) float16, up to max_samples
    """
    encoder.eval()
    encoder.to(device)

    layers = encoder.encoder.layers
    n_layers = len(layers)

    # Install hooks on every attention layer (attribute is 'attn', not 'self_attn')
    hooks = [CLSAttentionHook(layer.attn) for layer in layers]

    attn_accum = None   # (n_layers, n_heads, n_cpg) running sum
    attn_count = 0
    all_attn   = []     # list of (B, n_layers, n_heads, n_cpg) float16

    for i, batch in enumerate(loader):
        if n_batches > 0 and i >= n_batches:
            break
        if attn_count >= max_samples:
            break

        cpg_ids     = batch["cpg_ids"].to(device)
        beta_values = batch["beta_values"].to(device)
        attn_mask   = batch["attention_mask"].to(device)
        input_ids   = torch.stack([cpg_ids.float(), beta_values], dim=1)

        encoder(input_ids=input_ids, attention_mask=attn_mask)

        # Collect: hooks[l].last_attn = [B, H, n_cpg]
        batch_attn = torch.stack([h.last_attn for h in hooks], dim=1)  # [B, L, H, n_cpg]
        B = batch_attn.shape[0]

        if attn_accum is None:
            _, n_L, n_H, n_C = batch_attn.shape
            attn_accum = torch.zeros(n_L, n_H, n_C)

        attn_accum += batch_attn.sum(dim=0)
        attn_count += B

        # Store raw (as float16 to save memory) up to max_samples
        remaining = max_samples - len(all_attn) * batch_attn.shape[0] if all_attn else max_samples
        if remaining > 0:
            keep = min(B, remaining)
            all_attn.append(batch_attn[:keep].half().numpy())

        if (i + 1) % 10 == 0:
            log.info(f"  batch {i+1}/{len(loader)}  samples={attn_count}")

    for h in hooks:
        h.remove()

    mean_attn = (attn_accum / attn_count).numpy()    # (n_layers, n_heads, n_cpg)
    all_attn_arr = np.concatenate(all_attn, axis=0)  # (n_samples, n_layers, n_heads, n_cpg)

    log.info(f"Extracted attention from {attn_count} samples: "
             f"mean shape={mean_attn.shape}, all shape={all_attn_arr.shape}")
    return mean_attn, all_attn_arr


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_attention_metrics(mean_attn: np.ndarray) -> Dict:
    """
    mean_attn: (n_layers, n_heads, n_cpg)
    Returns dict keyed by (layer_idx, head_idx).
    """
    n_layers, n_heads, n_cpg = mean_attn.shape
    metrics = {}
    for l in range(n_layers):
        for h in range(n_heads):
            w = mean_attn[l, h]   # (n_cpg,) — should already sum to ~1
            w = w / (w.sum() + 1e-12)

            # Shannon entropy
            w_pos   = w[w > 0]
            entropy = float(-np.sum(w_pos * np.log(w_pos)))
            norm_entropy = float(entropy / np.log(n_cpg))

            # Gini coefficient
            sw  = np.sort(w)
            n   = len(sw)
            cum = np.cumsum(sw)
            gini = float(1 - 2 * cum.sum() / (n * sw.sum() + 1e-12))

            # Top-k attention mass
            top_k_mass = {}
            sorted_idx = np.argsort(w)[::-1]
            for k in TOP_K_VALUES:
                top_k_mass[k] = float(w[sorted_idx[:k]].sum())

            metrics[(l, h)] = {
                "entropy":           entropy,
                "normalized_entropy": norm_entropy,
                "gini":              gini,
                "top_k_mass":        top_k_mass,
                "top_10_cpg_idx":    sorted_idx[:10].tolist(),
            }

    return metrics


def metrics_to_serialisable(metrics: Dict) -> List:
    """Convert (int, int) keys to strings for JSON serialisation."""
    out = []
    for (l, h), v in metrics.items():
        entry = {"layer": l, "head": h}
        entry.update(v)
        out.append(entry)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    from bmfm_targets.tokenization import MultiFieldTokenizer
    from bmfm_methylation.shared.data_module import MethylationDataset, WCEDCollator

    tokenizer = MultiFieldTokenizer.from_pretrained(args.tokenizer)
    dataset   = MethylationDataset(h5ad_path=args.data, split=None, normalize_age=False)
    cpg_sites = dataset.cpg_sites
    vocab_size = len(cpg_sites)
    log.info(f"Dataset: {len(dataset)} samples × {vocab_size} CpGs")

    collator = WCEDCollator(
        tokenizer=tokenizer, cpg_sites=cpg_sites,
        vocab_size=vocab_size, input_ratio=1.0, contrastive=False,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collator,
                        shuffle=False, num_workers=4,
                        pin_memory=(device == "cuda"))

    summary = {}

    for ckpt_name, ckpt_path, is_pretrained in [
        ("pretrained", args.pretrained, True),
        ("finetuned",  args.finetuned,  False),
    ]:
        log.info(f"\n{'='*60}")
        log.info(f"Processing: {ckpt_name}  ({ckpt_path})")
        encoder = load_model_for_attention(ckpt_path, is_pretrained)

        mean_attn, all_attn = extract_cls_attention(
            encoder, loader, device,
            n_batches=args.n_batches,
            max_samples=args.max_samples,
        )

        np.save(outdir / f"cls_attn_{ckpt_name}.npy", all_attn)
        log.info(f"Saved → {outdir}/cls_attn_{ckpt_name}.npy")

        metrics = compute_attention_metrics(mean_attn)
        summary[ckpt_name] = metrics_to_serialisable(metrics)

        # Print per-layer/head summary
        log.info(f"\n  {ckpt_name} — attention selectivity:")
        log.info(f"  {'L':>2}  {'H':>2}  {'norm_ent':>9}  {'gini':>7}  {'top10_mass':>11}")
        for entry in summary[ckpt_name]:
            log.info(
                f"  {entry['layer']:>2}  {entry['head']:>2}  "
                f"{entry['normalized_entropy']:9.4f}  "
                f"{entry['gini']:7.4f}  "
                f"{entry['top_k_mass'][10]:11.4f}"
            )

    json_path = outdir / "cls_attention_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"\nSaved → {json_path}")

    # Save top-CpG indices for mapping back to site names
    top_cpg = {}
    for ckpt_name in ["pretrained", "finetuned"]:
        for entry in summary[ckpt_name]:
            key = f"{ckpt_name}_L{entry['layer']}_H{entry['head']}"
            top_cpg[key] = entry["top_10_cpg_idx"]
    np.savez(outdir / "top_cpg_indices.npz", **{k: np.array(v) for k, v in top_cpg.items()})
    log.info(f"Saved → {outdir}/top_cpg_indices.npz")

    # Print interpretation hint
    log.info("\n=== INTERPRETATION GUIDE ===")
    log.info("normalized_entropy=1.0 → uniform attention (no CpG selectivity)")
    log.info("normalized_entropy<0.7 → moderate selectivity")
    log.info("gini>0.5              → attention concentrated on few CpGs")
    log.info("top10_mass>0.05       → 10 CpGs account for >5% of total attention")
    log.info("Compare pretrained vs finetuned: fine-tuning should increase selectivity")
    log.info("if age-informative CpGs exist and the model learns to use them.")
    log.info("Done.")


if __name__ == "__main__":
    main()
