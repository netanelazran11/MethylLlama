"""
extract_kfold_test_predictions.py
===================================
Per-sample MethylLlama test-set predictions, for a paired subject-level
bootstrap comparison against MethylGPT (predictions extracted separately in
the MethylGPT-thesis repo). Isolated, inference-only script -- does not
modify any existing training/eval code.

Why a new script instead of reusing test_step(): the existing pipeline
(bmfm_methylation/llama/finetune_llama.py) only computes AGGREGATE test
metrics via torchmetrics -- it never emits per-sample predictions, and the
dataset's __getitem__ sets metadata["cell_name"] = str(idx) (a positional
index, not the real GSM ID), which is never used downstream. This script
replicates test_step's exact prediction logic (see _shared_step in
finetune_llama.py: CLS pooling -> age_head -> z-score -> denormalize by the
checkpoint's own saved age_mean/age_std) while independently pulling real
GSM IDs from dataset.adata.obs_names, aligned to batch order because
shuffle=False (verified in MethylationDataModule.test_dataloader).

Determinism: model.eval() + torch.no_grad() (no dropout, no stochastic
masking -- wced_input_ratio=1.0 means the full CpG profile is used, nothing
is held out), shuffle=False, fixed CPU-side data order. Optional
--determinism_check runs the full extraction twice in the same process and
asserts numerical equality before writing anything.

Usage (cluster, GPU):
  python scripts/finetune/extract_kfold_test_predictions.py \
      --fold 0 \
      --checkpoint checkpoints/finetuned_age/fold0.ckpt \
      --data ${METHYLLAMA_DATA}/data_methyl_21k_h5ad/altumage_21k_3way.h5ad \
      --tokenizer tokenizer \
      --genomic_rank resources/cpg_genomic_rank_finetune.npy \
      --test_ids outputs/kfold_splits/test_ids.npy \
      --duplicate_pairs_csv dataset_fingerprint_outputs/duplicate_pairs.csv \
      --outdir outputs/bootstrap_predictions/methyllama \
      --determinism_check
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--genomic_rank", required=True)
    p.add_argument("--test_ids", required=True, help="outputs/kfold_splits/test_ids.npy (fixed, shared across all folds)")
    p.add_argument("--duplicate_pairs_csv", default=None)
    p.add_argument("--subset_k", type=int, default=49156)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--outdir", default="outputs/bootstrap_predictions/methyllama")
    p.add_argument("--model_name", default="MethylLlama")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--determinism_check", action="store_true",
                    help="Run extraction twice in-process, assert bit-for-bit-tolerance match before writing output.")
    return p.parse_args()


def sha256_of_sorted_ids(ids) -> str:
    joined = "\n".join(sorted(ids)).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


def run_extraction(a):
    from bmfm_targets.tokenization import MultiFieldTokenizer
    from bmfm_methylation.shared.data_module import MethylationDataset, WCEDCollator
    from bmfm_methylation.llama.finetune_llama import load_finetune_llama_checkpoint

    torch.manual_seed(0)  # no stochastic ops in eval, but fixed for good measure

    print(f"[1/6] Loading checkpoint: {a.checkpoint}", flush=True)
    module = load_finetune_llama_checkpoint(a.checkpoint)
    module = module.to(a.device).eval()
    encoder = module.encoder
    age_head = module.age_head
    age_mean = float(module.age_mean)
    age_std = float(module.age_std)
    pooling = module.pooling
    assert pooling == "cls", f"Expected cls pooling (matches official k-fold recipe), got '{pooling}'"
    print(f"[2/6] Model on {a.device}. pooling={pooling} age_mean={age_mean:.3f} age_std={age_std:.3f}", flush=True)

    test_ids = np.load(a.test_ids, allow_pickle=True).astype(str)
    print(f"[3/6] Loaded {len(test_ids)} test IDs from {a.test_ids}", flush=True)

    tok = MultiFieldTokenizer.from_pretrained(a.tokenizer)

    exclude_ids = set()
    if a.duplicate_pairs_csv:
        from bmfm_methylation.shared.data_module import _compute_dedup_exclusions
        exclude_ids = _compute_dedup_exclusions(a.duplicate_pairs_csv)
    print(f"[4/6] Loading dataset: {a.data} (this can take a minute)", flush=True)

    ds = MethylationDataset(
        h5ad_path=a.data,
        split=None,
        normalize_age=False,           # we denormalize manually with THIS checkpoint's own age_mean/std
        filter_age_outliers=True,      # matches official k-fold pipeline
        exclude_ids=exclude_ids,
        override_obs_names=test_ids,   # exact match to the official fixed test set
    )
    n_found = len(ds)
    assert n_found == len(test_ids), (
        f"Test-set override selected {n_found} rows, expected {len(test_ids)} "
        f"(test_ids.npy) -- filters removed rows that should already be excluded "
        f"from test_ids.npy itself. Investigate before trusting predictions."
    )
    print(f"[5/6] Dataset ready: {n_found} samples. Building collator + starting inference loop...", flush=True)

    # Real GSM IDs in dataset order (batch order, since shuffle=False below)
    gsm_ids_in_order = np.array(ds.adata.obs_names, dtype=str)

    collator = WCEDCollator(
        tokenizer=tok, cpg_sites=ds.cpg_sites, vocab_size=a.subset_k,
        input_ratio=1.0,               # full profile, nothing held out -- matches official test eval
        contrastive=False,
        genomic_rank_path=a.genomic_rank,
    )
    loader = DataLoader(ds, batch_size=a.batch_size, collate_fn=collator,
                        shuffle=False, num_workers=0)

    raw_ages = np.asarray(ds.ages, dtype=np.float64)   # true ages, real years, dataset order

    preds_years = []
    n_seen = 0
    with torch.no_grad():
        for batch in loader:
            cpg_ids = batch["cpg_ids"].to(a.device)
            beta_values = batch["beta_values"].to(a.device)
            attn_mask = batch["attention_mask"].to(a.device)
            position_ids = batch.get("position_ids")
            position_ids = position_ids.to(a.device) if position_ids is not None else None

            input_ids = torch.stack([cpg_ids.float(), beta_values], dim=1)
            out = encoder(input_ids=input_ids, attention_mask=attn_mask, position_ids=position_ids)
            cls = out.pooler_output  # matches module._encode_cls with pooling="cls"

            pred_norm = age_head(cls).squeeze(-1)          # z-score space, same as training
            pred_years = pred_norm.double() * age_std + age_mean
            preds_years.append(pred_years.cpu().numpy())
            n_seen += cpg_ids.shape[0]
            if n_seen % 320 < a.batch_size:
                print(f"    {n_seen}/{n_found} samples", flush=True)

    preds_years = np.concatenate(preds_years)
    assert len(preds_years) == n_found, f"{len(preds_years)} predictions for {n_found} samples"
    print(f"[6/6] Inference complete: {n_found} predictions", flush=True)
    return gsm_ids_in_order, raw_ages, preds_years


def main():
    a = parse_args()
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ids1, true1, pred1 = run_extraction(a)

    if a.determinism_check:
        print("Determinism check: re-running extraction in the same process...")
        ids2, true2, pred2 = run_extraction(a)
        assert list(ids1) == list(ids2), "Sample ID order differs between the two runs -- non-deterministic dataset ordering!"
        assert np.allclose(true1, true2, atol=1e-8), "true_age differs between runs (should be exactly reproducible)"
        max_diff = float(np.max(np.abs(pred1 - pred2)))
        assert max_diff < 1e-5, f"Predictions differ between runs by up to {max_diff} -- NOT deterministic, investigate before trusting results"
        print(f"  PASSED: max prediction difference between the two runs = {max_diff:.2e}")

    df = pd.DataFrame({
        "sample_id": ids1,
        "true_age": true1,
        "predicted_age": pred1,
        "fold": a.fold,
        "model": a.model_name,
        "checkpoint": a.checkpoint,
    })

    n_dupe_ids = int(df["sample_id"].duplicated().sum())
    n_missing = int(df["predicted_age"].isna().sum())
    if n_dupe_ids or n_missing:
        raise RuntimeError(
            f"FAILING LOUDLY: duplicate_ids={n_dupe_ids}, missing_predictions={n_missing} "
            f"-- refusing to write a corrupted prediction file."
        )

    out_csv = outdir / f"fold_{a.fold}_predictions.csv"
    df.to_csv(out_csv, index=False)

    from sklearn.metrics import r2_score
    medae = float(np.median(np.abs(pred1 - true1)))
    mae = float(np.mean(np.abs(pred1 - true1)))
    r2 = float(r2_score(true1, pred1))

    verification = {
        "fold": a.fold,
        "checkpoint": a.checkpoint,
        "n_rows": int(len(df)),
        "n_unique_sample_ids": int(df["sample_id"].nunique()),
        "medae": round(medae, 6),
        "mae": round(mae, 6),
        "r2": round(r2, 6),
        "true_age_min": float(true1.min()),
        "true_age_max": float(true1.max()),
        "duplicate_id_count": n_dupe_ids,
        "missing_prediction_count": n_missing,
        "sorted_sample_id_sha256": sha256_of_sorted_ids(df["sample_id"].tolist()),
        "determinism_check_run": bool(a.determinism_check),
    }
    out_json = outdir / f"fold_{a.fold}_verification.json"
    with open(out_json, "w") as f:
        json.dump(verification, f, indent=2)

    print(json.dumps(verification, indent=2))
    print(f"\nSaved -> {out_csv}")
    print(f"Saved -> {out_json}")


if __name__ == "__main__":
    main()
