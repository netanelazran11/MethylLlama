"""
reconstruction_withheld_eval.py
===============================
Withheld-CpG reconstruction on the COMPLETE held-out partition of the pretraining corpus (all
16,912 profiles; Fig. 2c and Supplementary Fig. S1c). Every statistic is computed over every
withheld position; nothing is subsampled.

Does the WCED decoder reconstruct CpGs withheld from the encoder input from the CLS bottleneck,
and does it beat trivial controls under identical masks?
  * input_ratio 0.5 (the pretraining construction); genomic-rank position_ids reach the encoder
  * loss mask (valid & ~input): only CpGs that were measured but withheld from the encoder input
  * every profile of the split is evaluated (asserted with --expected_n)
  * Pearson r(observed, predicted) accumulated exactly over ALL withheld positions (float64 sums)
  * a second, CpG-centred Pearson r: (observed - CpG mean) vs (predicted - CpG mean) over all
    positions, which removes the between-CpG variance that dominates the raw r
  * a 200 x 200 histogram of (observed, predicted) over ALL withheld positions (Supplementary
    Fig. S1c). Only bin counts are stored; no individual beta value is written.

Conditions (identical profiles, identical withheld masks):
  model     -- decoder(real CLS)
  b_mean    -- per-CpG mean predictor (computed on the evaluated cohort itself, which
               ADVANTAGES this baseline -- conservative)
  b_shuffle -- decoder(CLS shuffled across profiles within the batch)
  b_random  -- decoder(random N(0,1) CLS)

Outputs (--outdir, default results/pretrained_cls/reconstruction_withheld):
  reconstruction_withheld_summary.json  all metrics + provenance
  per_sample_metrics.csv                per-profile MSE per condition (row = dataset index of the split)
  obs_pred_hist2d.npz                   200 x 200 counts of (observed, predicted) and the bin edges
The paired bootstrap of the control-minus-model differences (paired_bootstrap_ci.json) is computed
from per_sample_metrics.csv by scripts/evaluation/reconstruction_paired_bootstrap.py (CPU).

Usage (cluster, GPU): sbatch scripts/evaluation/run_reconstruction_withheld.sh
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

N_BINS = 200


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True, help="pretraining h5ad (49,156 CpGs)")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--genomic_rank", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--expected_n", type=int, default=16912,
                   help="abort if the split does not have exactly this many profiles (0 = no check)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--input_ratio", type=float, default=0.5)
    p.add_argument("--outdir", default="results/pretrained_cls/reconstruction_withheld")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def per_sample_masked_mse(pred, tgt, mask):
    diff2 = (pred - tgt).pow(2) * mask.float()
    return (diff2.sum(dim=1) / mask.float().sum(dim=1).clamp(min=1)).cpu().numpy()


def per_sample_normalized_mse(pred, tgt, mask):
    out = []
    for i in range(pred.shape[0]):
        m = mask[i]
        if m.sum() < 2:
            out.append(float("nan"))
            continue
        p, t = pred[i][m], tgt[i][m]
        p = (p - p.mean()) / (p.std() + 1e-8)
        t = (t - t.mean()) / (t.std() + 1e-8)
        out.append(float(((p - t) ** 2).mean().item()))
    return np.array(out)


class PearsonAcc:
    """Exact streaming Pearson correlation in float64."""

    def __init__(self, device):
        self.s = torch.zeros(6, dtype=torch.float64, device=device)  # n, sx, sy, sxx, syy, sxy

    def add(self, x, y):
        x, y = x.double(), y.double()
        self.s += torch.stack([torch.tensor(float(x.numel()), dtype=torch.float64, device=x.device),
                               x.sum(), y.sum(), (x * x).sum(), (y * y).sum(), (x * y).sum()])

    def r(self):
        n, sx, sy, sxx, syy, sxy = self.s.tolist()
        cov = sxy - sx * sy / n
        return cov / np.sqrt((sxx - sx * sx / n) * (syy - sy * sy / n)), int(n)


def hist2d_add(hist, x, y):
    # float64 so values on a bin edge fall in the same bin as numpy.histogram2d
    ix = (x.double().clamp(0, 1) * N_BINS).long().clamp(max=N_BINS - 1)
    iy = (y.double().clamp(0, 1) * N_BINS).long().clamp(max=N_BINS - 1)
    hist += torch.bincount(ix * N_BINS + iy, minlength=N_BINS * N_BINS).view(N_BINS, N_BINS)


def main():
    a = parse_args()
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)

    from bmfm_targets.tokenization import MultiFieldTokenizer
    from bmfm_methylation.shared.data_module import MethylationDataset, WCEDCollator
    from bmfm_methylation.llama.finetune_llama import load_wced_llama_checkpoint

    print(f"[1/6] Loading checkpoint: {a.checkpoint}", flush=True)
    module = load_wced_llama_checkpoint(a.checkpoint).to(a.device).eval()
    encoder, decoder = module.encoder, module.decoder

    print(f"[2/6] Loading dataset: {a.data} (split={a.split})", flush=True)
    ds = MethylationDataset(h5ad_path=a.data, split=a.split, normalize_age=False)
    n_eval = len(ds)
    if n_eval == 0:
        raise RuntimeError(f"split='{a.split}' selected 0 rows")
    if a.expected_n and n_eval != a.expected_n:
        raise RuntimeError(f"split has {n_eval} profiles, expected {a.expected_n}; refusing to run on a different population")
    print(f"      evaluating ALL {n_eval} profiles of split '{a.split}' (no subsampling)", flush=True)

    n_cpgs = len(ds.cpg_sites)
    tok = MultiFieldTokenizer.from_pretrained(a.tokenizer)
    collator = WCEDCollator(tokenizer=tok, cpg_sites=ds.cpg_sites, vocab_size=n_cpgs,
                            input_ratio=a.input_ratio, contrastive=False, genomic_rank_path=a.genomic_rank)
    loader = DataLoader(ds, batch_size=a.batch_size, collate_fn=collator, shuffle=False, num_workers=0)

    emb_vocab = encoder.embeddings.cpg_sites_embeddings.weight.shape[0]
    dec_final = [m for m in decoder.modules() if isinstance(m, torch.nn.Linear)][-1]
    dec_vocab = dec_final.out_features
    n_special = emb_vocab - dec_vocab
    col_to_dec = np.array(collator.vocab_cpg_ids, dtype=np.int64) - n_special
    assert dec_vocab == n_cpgs, f"decoder vocab {dec_vocab} != n_cpgs {n_cpgs}"
    assert col_to_dec.min() >= 0 and col_to_dec.max() < dec_vocab, "decoder index out of range"
    assert len(np.unique(col_to_dec)) == n_cpgs, "column->decoder map is not a permutation"
    col_to_dec_t = torch.from_numpy(col_to_dec).to(a.device)
    print(f"[3/6] Vocab map verified: emb={emb_vocab}, decoder={dec_vocab}, n_special={n_special}", flush=True)

    ps = {k: [] for k in ["model", "b_mean", "b_shuffle", "b_random"]}
    ps_norm_model, n_withheld_per_profile = [], []
    sum_obs = torch.zeros(n_cpgs, dtype=torch.float64)
    cnt_obs = torch.zeros(n_cpgs, dtype=torch.float64)
    stash_tgt, stash_pred, stash_mask = [], [], []
    pear = PearsonAcc(a.device)
    hist = torch.zeros(N_BINS, N_BINS, dtype=torch.int64, device=a.device)
    n_seen = 0

    print(f"[4/6] Pass 1: forward passes ({n_eval} profiles)", flush=True)
    with torch.no_grad():
        for batch in loader:
            cpg_ids = batch["cpg_ids"].to(a.device)
            betas = batch["beta_values"].to(a.device)
            attn = batch["attention_mask"].to(a.device)
            pos = batch.get("position_ids")
            pos = pos.to(a.device) if pos is not None else None
            all_betas = batch["all_betas"].to(a.device)
            valid = batch["valid_mask"].to(a.device)
            inp = batch["input_mask"].to(a.device)
            withheld = valid & ~inp

            input_ids = torch.stack([cpg_ids.float(), betas], dim=1)
            cls_real = encoder(input_ids=input_ids, attention_mask=attn, position_ids=pos).pooler_output
            B = cls_real.shape[0]
            dec_gather = col_to_dec_t.unsqueeze(0).expand(B, -1)
            pred_model = decoder(cls_real).gather(1, dec_gather)
            pred_shuf = decoder(cls_real[torch.randperm(B, device=a.device)]).gather(1, dec_gather)
            pred_rand = decoder(torch.randn_like(cls_real)).gather(1, dec_gather)

            ps["model"].append(per_sample_masked_mse(pred_model, all_betas, withheld))
            ps["b_shuffle"].append(per_sample_masked_mse(pred_shuf, all_betas, withheld))
            ps["b_random"].append(per_sample_masked_mse(pred_rand, all_betas, withheld))
            ps_norm_model.append(per_sample_normalized_mse(pred_model, all_betas, withheld))
            n_withheld_per_profile.append(withheld.sum(dim=1).cpu().numpy())

            obs_w, pred_w = all_betas[withheld], pred_model[withheld]
            pear.add(obs_w, pred_w)
            hist2d_add(hist, obs_w, pred_w)

            wf = withheld.float()
            sum_obs += (all_betas * wf).sum(dim=0).double().cpu()
            cnt_obs += wf.sum(dim=0).double().cpu()

            stash_tgt.append(all_betas.cpu())
            stash_pred.append(pred_model.cpu())
            stash_mask.append(withheld.cpu())
            n_seen += B
            if n_seen % 1600 < a.batch_size:
                print(f"      {n_seen}/{n_eval} profiles", flush=True)
    assert n_seen == n_eval, f"processed {n_seen} of {n_eval} profiles"

    print("[5/6] Pass 2: per-CpG mean baseline, CpG-centred Pearson", flush=True)
    cpg_mean = (sum_obs / cnt_obs.clamp(min=1)).float()
    pear_c = PearsonAcc("cpu")
    for tgt, prd, msk in zip(stash_tgt, stash_pred, stash_mask):
        mean_pred = cpg_mean.unsqueeze(0).expand_as(tgt)
        ps["b_mean"].append(per_sample_masked_mse(mean_pred, tgt, msk))
        pear_c.add((tgt - mean_pred)[msk], (prd - mean_pred)[msk])
    r_all, n_pos = pear.r()
    r_centred, n_pos_c = pear_c.r()
    assert n_pos == n_pos_c, "position counts differ between passes"

    results = {k: np.concatenate(v) for k, v in ps.items()}
    norm_model = np.concatenate(ps_norm_model)
    summary = {
        "script": "reconstruction_withheld_eval.py",
        "checkpoint": Path(a.checkpoint).name, "data": Path(a.data).name, "split": a.split,
        "n_profiles": int(n_eval), "subsampled": False,
        "input_ratio": a.input_ratio, "seed": a.seed,
        "n_withheld_positions_total": n_pos,
        "loss_mask": "valid & ~input (measured CpGs withheld from encoder input only)",
        "b_mean_source": "per-CpG mean over the evaluated cohort's withheld positions (advantages the baseline)",
        "pearson_withheld_all_positions": r_all,
        "pearson_withheld_cpg_centred_all_positions": r_centred,
        "normalized_mse_model": {"mean": float(np.nanmean(norm_model)), "median": float(np.nanmedian(norm_model))},
        "hist2d": {"bins": N_BINS, "range": [0, 1], "total_count": int(hist.sum().item())},
    }
    for k, arr in results.items():
        summary[f"raw_mse_{k}"] = {"mean": float(np.mean(arr)), "median": float(np.median(arr)),
                                   "std": float(np.std(arr)), "p05": float(np.percentile(arr, 5)),
                                   "p10": float(np.percentile(arr, 10)), "p25": float(np.percentile(arr, 25)),
                                   "p75": float(np.percentile(arr, 75)), "p90": float(np.percentile(arr, 90)),
                                   "p95": float(np.percentile(arr, 95))}
    for b in ["b_mean", "b_shuffle", "b_random"]:
        summary[f"ratio_model_vs_{b}"] = float(np.mean(results["model"]) / np.mean(results[b]))
    assert summary["hist2d"]["total_count"] == n_pos, "2-D histogram does not contain every withheld position"

    print("[6/6] Writing outputs", flush=True)
    with open(outdir / "reconstruction_withheld_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame({"dataset_index": np.arange(n_eval),
                  "n_withheld": np.concatenate(n_withheld_per_profile),
                  "model_mse": results["model"], "b_mean_mse": results["b_mean"],
                  "b_shuffle_mse": results["b_shuffle"], "b_random_mse": results["b_random"],
                  "model_normalized_mse": norm_model}).to_csv(outdir / "per_sample_metrics.csv", index=False)
    np.savez_compressed(outdir / "obs_pred_hist2d.npz", counts=hist.cpu().numpy(),
                        edges=np.linspace(0, 1, N_BINS + 1), x="observed beta", y="predicted beta")

    print(f"  profiles               {n_eval}")
    print(f"  withheld positions     {n_pos:,}")
    for k in ["model", "b_mean", "b_shuffle", "b_random"]:
        print(f"  raw MSE {k:10s}     mean={summary[f'raw_mse_{k}']['mean']:.6f}  median={summary[f'raw_mse_{k}']['median']:.6f}")
    print(f"  Pearson, all positions {r_all:.5f}")
    print(f"  Pearson, CpG-centred   {r_centred:.5f}")
    print(f"\nSaved -> {outdir}/", flush=True)


if __name__ == "__main__":
    main()
