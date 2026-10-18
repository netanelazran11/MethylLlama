"""
reconstruction_withheld_eval.py
=================================
Canonical reconstruction evaluation for Figure 2C/2D (+ Supplementary S3):
does the WCED decoder reconstruct *withheld* CpGs from the CLS bottleneck,
and does it beat trivial controls under identical masks?

Replaces the legacy reconstruction_baselines.py results, which are unusable
for the paper for three reasons (all fixed here):
  1. Wrong checkpoint  -- legacy run used the old no-contrastive model
     (llama-small-all49k-r0.5-w0.0-44450919), not the canonical ep85 model.
  2. Wrong objective   -- legacy run used input_ratio=1.0, i.e. it measured
     reconstruction of CpGs the encoder could SEE. Here input_ratio=0.5 and
     the loss mask is (valid & ~input): only CpGs that were measured but
     withheld from the encoder input are evaluated -- the actual WCED loss.
  3. Missing positions -- legacy run never passed position_ids, but the
     canonical checkpoint was trained with genomic-rank RoPE. Here the
     collator gets genomic_rank_path and position_ids reach the encoder.

Data: the PRETRAINING corpus' held-out split (not AltumAge) -- this measures
the pretraining objective on its own held-out profiles, comparable to the
checkpoint's recorded recon=0.0552 / pcc=0.9713 validation diagnostics.

Conditions (identical samples, identical withheld masks):
  model     -- decoder(real CLS)
  b_mean    -- per-CpG mean predictor (computed on the evaluated cohort
               itself, which ADVANTAGES this baseline -- conservative)
  b_shuffle -- decoder(CLS shuffled across samples within batch)
  b_random  -- decoder(random N(0,1) CLS)

Outputs (--outdir):
  reconstruction_withheld_summary.json  -- all metrics + provenance
  per_sample_metrics.csv                -- per-sample MSE per condition
  scatter_sample.npz                    -- subsampled (observed, predicted)
                                            pairs for the Fig 2C hexbin

Usage (cluster, GPU -- see run_reconstruction_withheld.sh):
  python scripts/evaluation/reconstruction_withheld_eval.py \
      --checkpoint checkpoints/pretrained/methyllama_pretrained_epoch85.ckpt --data <pretrain h5ad> \
      --tokenizer tokenizer \
      --genomic_rank resources/cpg_genomic_rank.npy \
      --split test --max_samples 5000 \
      --outdir results/pretrained_cls/reconstruction_withheld
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True, help="pretraining h5ad (49,156 CpGs)")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--genomic_rank", required=True,
                    help="resources/cpg_genomic_rank.npy (pretrain, 49,156)")
    p.add_argument("--split", default="test",
                    help="which held-out split of the pretraining h5ad to evaluate")
    p.add_argument("--max_samples", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--input_ratio", type=float, default=0.5,
                    help="fraction of valid CpGs shown to the encoder (pretraining value)")
    p.add_argument("--n_scatter_pairs", type=int, default=500_000,
                    help="max (observed, predicted) pairs saved for the hexbin panel")
    p.add_argument("--outdir", default="results/pretrained_cls/reconstruction_withheld")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def per_sample_masked_mse(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    diff2 = (pred - tgt).pow(2) * mask.float()
    return (diff2.sum(dim=1) / mask.float().sum(dim=1).clamp(min=1)).cpu().numpy()


def per_sample_normalized_mse(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    """z-score pred and target per sample over withheld positions before MSE --
    matches the wced_normalize_loss=true convention of the training loss, so this
    number is on the same scale as the checkpoint's recon=0.0552 diagnostic."""
    out = []
    for i in range(pred.shape[0]):
        m = mask[i]
        if m.sum() < 2:
            out.append(float("nan"))
            continue
        p = pred[i][m]
        t = tgt[i][m]
        p = (p - p.mean()) / (p.std() + 1e-8)
        t = (t - t.mean()) / (t.std() + 1e-8)
        out.append(float(((p - t) ** 2).mean().item()))
    return np.array(out)


def main():
    a = parse_args()
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    from bmfm_targets.tokenization import MultiFieldTokenizer
    from bmfm_methylation.shared.data_module import MethylationDataset, WCEDCollator
    from bmfm_methylation.llama.finetune_llama import load_wced_llama_checkpoint

    print(f"[1/6] Loading checkpoint: {a.checkpoint}", flush=True)
    module = load_wced_llama_checkpoint(a.checkpoint)
    module = module.to(a.device).eval()
    encoder, decoder = module.encoder, module.decoder

    print(f"[2/6] Loading dataset: {a.data} (split={a.split})", flush=True)
    ds = MethylationDataset(h5ad_path=a.data, split=a.split, normalize_age=False)
    if len(ds) == 0:
        raise RuntimeError(
            f"split='{a.split}' selected 0 rows -- check the h5ad's split column values "
            f"before burning GPU time."
        )
    n_total = len(ds)
    n_eval = min(a.max_samples, n_total)
    idx = rng.choice(n_total, size=n_eval, replace=False) if n_eval < n_total else np.arange(n_total)
    sub = Subset(ds, sorted(idx.tolist()))
    print(f"      split rows={n_total}, evaluating n={n_eval}", flush=True)

    n_cpgs = len(ds.cpg_sites)
    tok = MultiFieldTokenizer.from_pretrained(a.tokenizer)
    collator = WCEDCollator(
        tokenizer=tok, cpg_sites=ds.cpg_sites, vocab_size=n_cpgs,
        input_ratio=a.input_ratio, contrastive=False,
        genomic_rank_path=a.genomic_rank,
    )
    loader = DataLoader(sub, batch_size=a.batch_size, collate_fn=collator,
                        shuffle=False, num_workers=0)

    # Decoder output is indexed by (tokenizer vocab id - n_special); collator
    # tensors (all_betas, masks) are indexed by data column. Build the
    # column -> decoder-index map once and verify it is a proper permutation.
    emb_vocab = encoder.embeddings.cpg_sites_embeddings.weight.shape[0]
    dec_final = [m for m in decoder.modules() if isinstance(m, torch.nn.Linear)][-1]
    dec_vocab = dec_final.out_features
    n_special = emb_vocab - dec_vocab
    col_to_dec = np.array(collator.vocab_cpg_ids, dtype=np.int64) - n_special
    assert dec_vocab == n_cpgs, f"decoder vocab {dec_vocab} != n_cpgs {n_cpgs}"
    assert col_to_dec.min() >= 0 and col_to_dec.max() < dec_vocab, "decoder index out of range"
    assert len(np.unique(col_to_dec)) == n_cpgs, "column->decoder map is not a permutation"
    col_to_dec_t = torch.from_numpy(col_to_dec).to(a.device)
    print(f"[3/6] Vocab map verified: emb={emb_vocab}, decoder={dec_vocab}, "
          f"n_special={n_special}, permutation OK", flush=True)

    ps = {k: [] for k in ["model", "b_mean", "b_shuffle", "b_random"]}
    ps_norm_model = []
    sum_obs = torch.zeros(n_cpgs, dtype=torch.float64)
    cnt_obs = torch.zeros(n_cpgs, dtype=torch.float64)
    stash_tgt, stash_pred, stash_mask = [], [], []
    n_withheld_total = 0
    n_seen = 0

    print(f"[4/6] Pass 1: forward passes ({n_eval} samples)", flush=True)
    with torch.no_grad():
        for batch in loader:
            cpg_ids = batch["cpg_ids"].to(a.device)
            betas = batch["beta_values"].to(a.device)
            attn = batch["attention_mask"].to(a.device)
            pos = batch.get("position_ids")
            pos = pos.to(a.device) if pos is not None else None
            all_betas = batch["all_betas"].to(a.device)          # [B, n_cpgs] column order
            valid = batch["valid_mask"].to(a.device)             # measured
            inp = batch["input_mask"].to(a.device)               # shown to encoder
            withheld = valid & ~inp                              # the WCED loss mask

            input_ids = torch.stack([cpg_ids.float(), betas], dim=1)
            cls_real = encoder(input_ids=input_ids, attention_mask=attn,
                               position_ids=pos).pooler_output

            B = cls_real.shape[0]
            # decoder outputs in vocab order -> gather into column order
            dec_gather = col_to_dec_t.unsqueeze(0).expand(B, -1)
            pred_model = decoder(cls_real).gather(1, dec_gather)
            pred_shuf = decoder(cls_real[torch.randperm(B, device=a.device)]).gather(1, dec_gather)
            pred_rand = decoder(torch.randn_like(cls_real)).gather(1, dec_gather)

            ps["model"].append(per_sample_masked_mse(pred_model, all_betas, withheld))
            ps["b_shuffle"].append(per_sample_masked_mse(pred_shuf, all_betas, withheld))
            ps["b_random"].append(per_sample_masked_mse(pred_rand, all_betas, withheld))
            ps_norm_model.append(per_sample_normalized_mse(pred_model, all_betas, withheld))

            # accumulate per-CpG observed means over withheld positions (for b_mean)
            wf = withheld.float()
            sum_obs += (all_betas * wf).sum(dim=0).double().cpu()
            cnt_obs += wf.sum(dim=0).double().cpu()
            n_withheld_total += int(withheld.sum().item())

            # stash CPU copies for pass 2 (b_mean) + scatter subsample
            stash_tgt.append(all_betas.cpu())
            stash_pred.append(pred_model.cpu())
            stash_mask.append(withheld.cpu())

            n_seen += B
            if n_seen % 320 < a.batch_size:
                print(f"      {n_seen}/{n_eval} samples", flush=True)

    print(f"[5/6] Pass 2: per-CpG mean baseline + scatter subsample", flush=True)
    cpg_mean = (sum_obs / cnt_obs.clamp(min=1)).float()          # [n_cpgs]
    obs_pairs, pred_pairs = [], []
    pairs_per_batch = max(1, a.n_scatter_pairs // max(1, len(stash_tgt)))
    for tgt, prd, msk in zip(stash_tgt, stash_pred, stash_mask):
        mean_pred = cpg_mean.unsqueeze(0).expand_as(tgt)
        ps["b_mean"].append(per_sample_masked_mse(mean_pred, tgt, msk))
        flat_idx = torch.nonzero(msk.reshape(-1), as_tuple=False).squeeze(-1)
        take = flat_idx[torch.randperm(len(flat_idx))[:pairs_per_batch]]
        obs_pairs.append(tgt.reshape(-1)[take])
        pred_pairs.append(prd.reshape(-1)[take])

    obs_pairs = torch.cat(obs_pairs).numpy()
    pred_pairs = torch.cat(pred_pairs).numpy()
    pearson_scatter = float(np.corrcoef(obs_pairs, pred_pairs)[0, 1])

    results = {k: np.concatenate(v) for k, v in ps.items()}
    norm_model = np.concatenate(ps_norm_model)

    summary = {
        "checkpoint": a.checkpoint,
        "data": a.data,
        "split": a.split,
        "n_samples": int(n_eval),
        "input_ratio": a.input_ratio,
        "seed": a.seed,
        "n_withheld_positions_total": n_withheld_total,
        "loss_mask": "valid & ~input (measured CpGs withheld from encoder input only)",
        "b_mean_source": "per-CpG mean over the evaluated cohort's withheld positions "
                          "(advantages the baseline -- conservative for the model)",
        "pearson_withheld_obs_vs_pred": pearson_scatter,
        "normalized_mse_model": {"mean": float(np.nanmean(norm_model)),
                                   "median": float(np.nanmedian(norm_model))},
    }
    for k, arr in results.items():
        summary[f"raw_mse_{k}"] = {
            "mean": float(np.mean(arr)), "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "p10": float(np.percentile(arr, 10)), "p90": float(np.percentile(arr, 90)),
        }
    for b in ["b_mean", "b_shuffle", "b_random"]:
        summary[f"ratio_model_vs_{b}"] = float(np.mean(results["model"]) / np.mean(results[b]))

    print(f"[6/6] Writing outputs", flush=True)
    with open(outdir / "reconstruction_withheld_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame({
        "model_mse": results["model"],
        "b_mean_mse": results["b_mean"],
        "b_shuffle_mse": results["b_shuffle"],
        "b_random_mse": results["b_random"],
        "model_normalized_mse": norm_model,
    }).to_csv(outdir / "per_sample_metrics.csv", index=False)
    np.savez_compressed(outdir / "scatter_sample.npz",
                        observed=obs_pairs.astype(np.float16),
                        predicted=pred_pairs.astype(np.float16))

    print(json.dumps({k: v for k, v in summary.items()
                       if not isinstance(v, dict)}, indent=2))
    for k in ["raw_mse_model", "raw_mse_b_mean", "raw_mse_b_shuffle", "raw_mse_b_random"]:
        print(f"  {k:22s} mean={summary[k]['mean']:.6f}")
    print(f"  normalized model MSE   mean={summary['normalized_mse_model']['mean']:.4f} "
          f"(training-loss scale, cf. checkpoint recon=0.0552)")
    print(f"  Pearson (withheld)     {pearson_scatter:.4f} (cf. checkpoint pcc=0.9713)")
    print(f"\nSaved -> {outdir}/", flush=True)


if __name__ == "__main__":
    main()
