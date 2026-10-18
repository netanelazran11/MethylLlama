"""
MethylLlama FINE-TUNED CLS extraction + representation analysis.

Sibling of extract_pretrain_cls.py — identical extraction pipeline (same h5ad,
same tokenizer, same genomic_rank_path -> same CpG site set, same sample set,
no masking), but loads a FINE-TUNED checkpoint (MethylationAgeRegressorLlama)
instead of the raw WCED pretrain encoder. This is what makes the before/after
comparison valid: the only thing that differs between this and
extract_pretrain_cls.py's output is the encoder weights (pretrain vs
fine-tuned) — same samples, same CpGs, same forward pass, no masking/dropout.

Also runs the ACTUAL trained age_head on the extracted CLS (not the linear-
probe proxy) and reports test MedAE/R² as a correctness check: this must match
the fold's known WandB test metrics, or the extraction is wrong.

Outputs (in --outdir, default results/finetuned_cls/) — same schema as
extract_pretrain_cls.py so visualize_cls.py / visualize_cls_publication.py
work unchanged:
  embeddings_cls.npy, embeddings_mean.npy, metadata.csv, cpg_alignment.csv,
  analysis_summary.json, report.txt

Usage (cluster):
  python scripts/evaluation/extract_finetune_cls.py \
    --checkpoint checkpoints/finetuned_age/fold4.ckpt \
    --data ${METHYLLAMA_DATA}/data_methyl_21k_h5ad/altumage_21k_3way.h5ad \
    --tokenizer tokenizer \
    --genomic_rank resources/cpg_genomic_rank_finetune.npy \
    --outdir results/finetuned_cls
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("finetuned_cls")

# reuse the encoder-agnostic extraction + analysis functions
from extract_pretrain_cls import extract, geometry, probe_age, probe_clf


def parse_args():
    p = argparse.ArgumentParser(description="MethylLlama fine-tuned CLS extraction + analysis")
    p.add_argument("--checkpoint", required=True, help="fine-tuned .ckpt (MethylationAgeRegressorLlama)")
    p.add_argument("--data", required=True, help="SAME labeled h5ad used for extract_pretrain_cls.py")
    p.add_argument("--tokenizer", required=True, help="SAME tokenizer used for extract_pretrain_cls.py")
    p.add_argument("--genomic_rank", required=True, help="SAME cpg_genomic_rank_finetune.npy")
    p.add_argument("--outdir", default="results/finetuned_cls")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--age_col", default="age")
    p.add_argument("--label_cols", nargs="+", default=["tissue_type", "sex", "dataset"])
    p.add_argument("--split_col", default="split")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def sanity_check_age_head(module, cls, meta, age_col, split_col, device):
    """Run the ACTUAL trained age_head (not a proxy probe) on extracted CLS.
    Must reproduce the fold's known WandB test MedAE/R2 — if it doesn't,
    extraction/pooling/normalization is wrong."""
    from sklearn.metrics import r2_score

    split = np.asarray(meta[split_col].astype(str)) if split_col in meta else None
    age = np.asarray(meta[age_col], dtype=float) if age_col in meta else None
    if split is None or age is None:
        return None
    te = (split == "test") & ~np.isnan(age)

    head = module.age_head.to(device).eval()
    with torch.no_grad():
        x = torch.tensor(cls[te], dtype=torch.float32, device=device)
        pred_norm = head(x).squeeze(-1).cpu().numpy()
    pred_years = pred_norm * float(module.age_std) + float(module.age_mean)
    label_years = age[te]

    return {
        "n_test": int(te.sum()),
        "medae": round(float(np.median(np.abs(pred_years - label_years))), 4),
        "mae": round(float(np.mean(np.abs(pred_years - label_years))), 4),
        "r2": round(float(r2_score(label_years, pred_years)), 4),
        "note": "compare against this fold's WandB test/medae, test/mae, test/r2 — "
                "should match closely (extraction correctness gate)",
    }


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    from bmfm_methylation.llama.finetune_llama import load_finetune_llama_checkpoint
    log.info(f"Loading fine-tuned checkpoint: {args.checkpoint}")
    module = load_finetune_llama_checkpoint(args.checkpoint)
    encoder = module.encoder
    log.info(f"Encoder: {encoder.config.num_hidden_layers}L x {encoder.config.hidden_size}D "
             f"| pooling={module.pooling} | age_mean={module.age_mean:.3f} age_std={module.age_std:.3f}")

    cpg_emb = encoder.embeddings.cpg_sites_embeddings.weight.detach().cpu().float().numpy()
    np.save(outdir / "cpg_embedding_matrix.npy", cpg_emb)
    log.info(f"Saved cpg_embedding_matrix.npy {cpg_emb.shape}")

    cls, mean, meta, align = extract(
        encoder, args.data, args.tokenizer, args.genomic_rank, args.batch_size, args.device
    )
    np.save(outdir / "embeddings_cls.npy", cls)
    np.save(outdir / "embeddings_mean.npy", mean)
    meta.to_csv(outdir / "metadata.csv", index=False)
    align.to_csv(outdir / "cpg_alignment.csv", index=False)

    split = np.asarray(meta[args.split_col].astype(str)) if args.split_col in meta else np.full(len(cls), "train")
    age = np.asarray(meta[args.age_col], dtype=float) if args.age_col in meta else None

    summary = {
        "checkpoint": args.checkpoint,
        "n_samples": int(len(cls)),
        "cpg_embedding_matrix_shape": list(cpg_emb.shape),
        "geometry_cls": geometry(cls),
        "geometry_mean": geometry(mean),
    }
    if age is not None:
        summary["age_probe_cls"] = probe_age(cls, age, split)
        summary["age_probe_mean"] = probe_age(mean, age, split)
        summary["age_head_actual"] = sanity_check_age_head(
            module, cls, meta, args.age_col, args.split_col, args.device
        )
    summary["class_probes_cls"] = {
        c: probe_clf(cls, meta[c], split) for c in args.label_cols if c in meta
    }

    with open(outdir / "analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    lines = ["MethylLlama FINE-TUNED — CLS representation analysis", "=" * 55,
             f"checkpoint: {args.checkpoint}", f"samples: {summary['n_samples']}", ""]
    for tag in ("cls", "mean"):
        g = summary[f"geometry_{tag}"]
        lines += [f"[{tag.upper()}] geometry:",
                  f"  eff_rank={g['effective_rank']}/{cls.shape[1]}  top1_sv={g['top1_sv_frac']}  "
                  f"anisotropy={g['anisotropy_mean_cos']}  dead_dims={g['dead_dims_lt1pct']}"]
    if age is not None:
        a = summary["age_probe_cls"]
        lines += ["", "[CLS] age probe (linear/replica proxy):",
                  f"  linear R2={a['linear_ridge_r2']}  replica-head R2={a['replica_head_r2']}  "
                  f"MedAE={a['replica_head_medae']}yr"]
        ah = summary["age_head_actual"]
        if ah:
            lines += ["", "[CLS] ACTUAL trained age_head (correctness gate vs WandB test metrics):",
                      f"  test MedAE={ah['medae']}yr  MAE={ah['mae']}yr  R2={ah['r2']}  n={ah['n_test']}"]
    lines += ["", "[CLS] biological structure (balanced-acc vs chance):"]
    for c, r in summary["class_probes_cls"].items():
        if r:
            lines += [f"  {c}: {r['balanced_acc']} (chance {r['chance']}, {r['n_classes']} classes)"]
    (outdir / "report.txt").write_text("\n".join(lines))
    log.info("\n" + "\n".join(lines))
    log.info(f"All outputs -> {outdir}/")


if __name__ == "__main__":
    main()
