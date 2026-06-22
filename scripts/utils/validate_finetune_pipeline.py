"""
Deep validation of the fine-tuning pipeline before launching training.

Checks (in order):
  1. h5ad structure — splits, age column, shape
  2. Age statistics per split
  3. Beta value distribution (non-NaN)
  4. NaN pattern — per-sample and per-CpG
  5. Tokenizer — vocab size, special tokens
  6. CpG vocab alignment — all fine-tune CpGs in pretrain tokenizer?
  7. WCEDCollator dry run — build one real batch, inspect all fields
  8. Checkpoint loading — architecture inferred correctly?
  9. Forward pass smoke test — one batch through the encoder, no crash

Usage:
  python scripts/utils/validate_finetune_pipeline.py
"""

import sys
import numpy as np
import torch

REPO        = "/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
H5AD        = "/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_finetune_49k_h5ad/finetuning_49k.h5ad"
TOKENIZER   = f"{REPO}/tokenizer_llama_pretrain49k"
CHECKPOINT  = f"{REPO}/outputs/pretrain-llama-wced/llama-small-all49k-r0.5-w0.0-44450919/checkpoints/epoch=98-val_loss=0.0059.ckpt"
VOCAB_SIZE  = 49156
INPUT_RATIO = 0.5

PASS = "✓"
FAIL = "✗"
WARN = "⚠"

errors   = []
warnings = []

def ok(msg):    print(f"  {PASS} {msg}")
def fail(msg):  print(f"  {FAIL} {msg}"); errors.append(msg)
def warn(msg):  print(f"  {WARN} {msg}"); warnings.append(msg)

# ── 1. h5ad structure ──────────────────────────────────────────────────────────
print("\n" + "="*60)
print("[1] h5ad structure")
print("="*60)
import anndata as ad
adata = ad.read_h5ad(H5AD)
print(f"  Shape: {adata.shape}")

if adata.shape[0] > 0 and adata.shape[1] > 0:
    ok(f"Shape: {adata.shape[0]} samples × {adata.shape[1]} CpGs")
else:
    fail(f"Empty h5ad: {adata.shape}")

if adata.shape[1] == VOCAB_SIZE:
    ok(f"CpG count matches VOCAB_SIZE ({VOCAB_SIZE})")
else:
    fail(f"CpG count {adata.shape[1]} ≠ VOCAB_SIZE {VOCAB_SIZE}")

if "split" in adata.obs.columns:
    ok("'split' column present")
    counts = adata.obs["split"].value_counts()
    for s, n in counts.items():
        print(f"    {s}: {n} samples")
    for s in ["train", "valid", "test"]:
        if s not in counts:
            fail(f"Split '{s}' missing")
else:
    fail("No 'split' column in obs")

if "age" in adata.obs.columns:
    ok("'age' column present")
else:
    fail("No 'age' column in obs")

if "id" in adata.obs.columns:
    ok("'id' column present")

print(f"  obs columns: {list(adata.obs.columns)}")
print(f"  var_names[:5]: {list(adata.var_names[:5])}")
print(f"  var_names[-5:]: {list(adata.var_names[-5:])}")

# ── 2. Age statistics ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("[2] Age statistics per split")
print("="*60)
for split in ["train", "valid", "test"]:
    if "split" not in adata.obs.columns:
        break
    mask = adata.obs["split"] == split
    ages = adata.obs.loc[mask, "age"].astype(float)
    nan_age = ages.isna().sum()
    valid_ages = ages.dropna()
    if len(valid_ages) == 0:
        fail(f"{split}: no valid ages")
        continue
    print(f"  {split:5s}: n={len(ages)}, age=[{valid_ages.min():.1f}, {valid_ages.max():.1f}], "
          f"mean={valid_ages.mean():.1f}, std={valid_ages.std():.1f}, NaN={nan_age}")
    if valid_ages.std() < 5:
        warn(f"{split}: very low age std={valid_ages.std():.1f} — may be hard to learn")
    else:
        ok(f"{split}: age distribution looks healthy")

# ── 3. Beta value distribution ────────────────────────────────────────────────
print("\n" + "="*60)
print("[3] Beta value distribution (train split, sample of 200 rows)")
print("="*60)
if "split" in adata.obs.columns:
    train_mask = adata.obs["split"] == "train"
    train_idx  = np.where(train_mask)[0]
    sample_idx = train_idx[:200]
    X_sample   = adata.X[sample_idx]
    if hasattr(X_sample, "toarray"):
        X_sample = X_sample.toarray()
    X_sample = X_sample.astype(np.float32)

    nan_pct  = np.isnan(X_sample).mean() * 100
    zero_pct = (X_sample == 0).sum() / (~np.isnan(X_sample)).sum() * 100
    valid    = X_sample[~np.isnan(X_sample)]

    print(f"  NaN:   {nan_pct:.1f}%")
    print(f"  Zero (of valid): {zero_pct:.1f}%")
    print(f"  Valid β range:   [{valid.min():.4f}, {valid.max():.4f}]")
    print(f"  Valid β mean:    {valid.mean():.4f}")
    print(f"  Valid β std:     {valid.std():.4f}")
    print(f"  Valid per sample (avg): {(~np.isnan(X_sample)).sum(1).mean():.0f} / {X_sample.shape[1]}")

    if nan_pct > 80:
        warn(f"NaN rate {nan_pct:.1f}% is very high — few valid CpGs per sample")
    elif nan_pct > 40:
        ok(f"NaN {nan_pct:.1f}% — expected for multi-platform data, handled by WCEDCollator")
    else:
        ok(f"NaN {nan_pct:.1f}% — low (similar to pretrain)")

    if valid.min() < 0 or valid.max() > 1:
        fail(f"Beta values outside [0,1]: min={valid.min():.4f}, max={valid.max():.4f}")
    else:
        ok("All valid beta values in [0, 1]")

# ── 4. NaN pattern ────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("[4] NaN pattern — scattered vs structured?")
print("="*60)
if "split" in adata.obs.columns:
    # Check first 5 samples — are NaN positions the same across samples?
    first5 = adata.X[:5]
    if hasattr(first5, "toarray"):
        first5 = first5.toarray()
    first5 = first5.astype(np.float32)
    nan_masks = np.isnan(first5)
    pairwise_agree = []
    for i in range(5):
        for j in range(i+1, 5):
            agree = (nan_masks[i] == nan_masks[j]).mean()
            pairwise_agree.append(agree)
    mean_agree = np.mean(pairwise_agree)
    print(f"  NaN mask agreement across first 5 samples: {mean_agree:.3f}")
    if mean_agree > 0.95:
        warn(f"NaN patterns very similar ({mean_agree:.3f}) — may be platform-specific, not sample-specific")
    elif mean_agree > 0.7:
        ok(f"NaN patterns moderately consistent ({mean_agree:.3f}) — expected for same platform")
    else:
        ok(f"NaN patterns scattered ({mean_agree:.3f}) — diverse platforms, good")

    # Check if NaN positions are concentrated at start/end or scattered
    cpg_nan_rate = np.isnan(adata.X[:100].toarray() if hasattr(adata.X[:100], "toarray") else adata.X[:100]).mean(0)
    first_half_nan  = cpg_nan_rate[:VOCAB_SIZE//2].mean()
    second_half_nan = cpg_nan_rate[VOCAB_SIZE//2:].mean()
    print(f"  NaN rate — first half CpGs: {first_half_nan*100:.1f}%, second half: {second_half_nan*100:.1f}%")
    if abs(first_half_nan - second_half_nan) > 0.2:
        warn(f"Uneven NaN distribution — second half has more NaN (structured missingness)")
    else:
        ok("NaN distributed evenly across CpG positions")

# ── 5. Tokenizer ──────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("[5] Tokenizer")
print("="*60)
import os
sys.path.insert(0, REPO)
os.chdir(REPO)

try:
    from bmfm_targets.tokenization import MultiFieldTokenizer
    tok = MultiFieldTokenizer.from_pretrained(TOKENIZER)
    ok(f"Tokenizer loaded from {TOKENIZER}")

    cpg_tok = tok.tokenizers["cpg_sites"]
    vocab   = cpg_tok.get_vocab()
    print(f"  Vocab size:    {len(vocab)}")
    print(f"  CLS token id:  {cpg_tok.cls_token_id}")
    print(f"  PAD token id:  {cpg_tok.pad_token_id}")
    print(f"  UNK token id:  {cpg_tok.unk_token_id}")

    if len(vocab) >= VOCAB_SIZE:
        ok(f"Vocab size {len(vocab)} ≥ expected {VOCAB_SIZE}")
    else:
        warn(f"Vocab size {len(vocab)} < expected {VOCAB_SIZE}")

except Exception as e:
    fail(f"Could not load tokenizer: {e}")
    tok = None

# ── 6. CpG vocab alignment ────────────────────────────────────────────────────
print("\n" + "="*60)
print("[6] CpG vocab alignment")
print("="*60)
if tok is not None:
    pretrain_vocab = set(tok.tokenizers["cpg_sites"].get_vocab().keys())
    ft_cpgs        = set(adata.var_names)

    in_both  = ft_cpgs & pretrain_vocab
    missing  = ft_cpgs - pretrain_vocab
    extra    = pretrain_vocab - ft_cpgs  # in tokenizer but not in fine-tune data

    print(f"  Fine-tune CpGs:        {len(ft_cpgs)}")
    print(f"  Pretrain tokenizer:    {len(pretrain_vocab)}")
    print(f"  In both:               {len(in_both)}")
    print(f"  In fine-tune only:     {len(missing)}  ← these map to unk_id!")
    print(f"  In tokenizer only:     {len(extra)}   ← unused pretrain probes")

    if len(missing) == 0:
        ok("All fine-tune CpGs are in the pretrain tokenizer — no UNK mappings")
    elif len(missing) / len(ft_cpgs) < 0.01:
        warn(f"{len(missing)} CpGs ({100*len(missing)/len(ft_cpgs):.2f}%) will map to unk_id — minor")
        print(f"  Missing samples: {list(missing)[:10]}")
    else:
        fail(f"{len(missing)} CpGs ({100*len(missing)/len(ft_cpgs):.1f}%) NOT in tokenizer — vocab mismatch!")
        print(f"  First 10 missing: {list(missing)[:10]}")

    # Spot-check a few CpG lookups
    print("\n  Spot-check token IDs for first 5 CpGs:")
    for cpg in list(adata.var_names[:5]):
        tid = tok.tokenizers["cpg_sites"].get_vocab().get(cpg, -1)
        unk = tok.tokenizers["cpg_sites"].unk_token_id
        status = PASS if tid != unk else FAIL
        print(f"    {status} {cpg} → token_id={tid}")

# ── 7. WCEDCollator dry run ───────────────────────────────────────────────────
print("\n" + "="*60)
print("[7] WCEDCollator dry run (1 real batch of 4 samples)")
print("="*60)
if tok is not None and "split" in adata.obs.columns:
    try:
        from bmfm_methylation.shared.data_module import MethylationDataset, WCEDCollator
        from bmfm_targets.config import FieldInfo

        ds = MethylationDataset(
            h5ad_path=H5AD,
            split="train",
            age_column="age",
            split_column="split",
            normalize_age=True,
        )

        print(f"  Dataset: {len(ds)} train samples")
        print(f"  age_mean={ds.age_mean:.2f}, age_std={ds.age_std:.2f}")
        ok(f"Age normalization auto-computed: mean={ds.age_mean:.2f}, std={ds.age_std:.2f}")

        collator = WCEDCollator(
            tokenizer=tok,
            cpg_sites=ds.cpg_sites,
            vocab_size=VOCAB_SIZE,
            input_ratio=INPUT_RATIO,
            contrastive=False,
            fixed_subset_seed=42,
        )
        print(f"  Collator vocab_size={collator.actual_vocab_size}, "
              f"max_seq_len={collator.max_seq_len}, "
              f"input_ratio={collator.input_ratio}")

        examples = [ds[i] for i in range(4)]
        batch    = collator(examples)

        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                nan_count = torch.isnan(val.float()).sum().item()
                print(f"  {key:20s}: shape={tuple(val.shape)}, dtype={val.dtype}, NaN={nan_count}")

        # Verify input_mask + valid_mask interaction
        im = batch["input_mask"]    # [4, 49156]
        vm = batch["valid_mask"]    # [4, 49156]
        ab = batch["all_betas"]     # [4, 49156]
        recon_mask = (~im) & vm
        n_recon = recon_mask.float().sum(1)
        print(f"\n  Reconstruction targets per sample (non-input & non-NaN):")
        for i in range(4):
            print(f"    sample {i}: input={im[i].sum()}, valid={vm[i].sum()}, "
                  f"recon={recon_mask[i].sum()}, age={batch['age'][i]:.3f}")

        if recon_mask.float().sum() == 0:
            fail("recon_mask is all-zero — no valid reconstruction targets!")
        else:
            ok(f"recon_mask has valid targets (avg {n_recon.mean():.0f} per sample)")

        if torch.isnan(batch["age"]).all():
            fail("All ages are NaN in batch")
        else:
            ok(f"Age labels present: {batch['age'].tolist()}")

    except Exception as e:
        import traceback
        fail(f"WCEDCollator dry run failed: {e}")
        traceback.print_exc()

# ── 8. Checkpoint loading ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("[8] Checkpoint loading")
print("="*60)
try:
    import torch
    torch.load.__func__ if hasattr(torch.load, '__func__') else None

    sys.path.insert(0, REPO)
    from bmfm_methylation.llama.finetune_llama import load_wced_llama_checkpoint

    pretrained = load_wced_llama_checkpoint(CHECKPOINT)
    encoder = pretrained.encoder
    decoder = pretrained.decoder

    cfg = encoder.config
    print(f"  Encoder config: hidden={cfg.hidden_size}, layers={cfg.num_hidden_layers}, "
          f"heads={cfg.num_attention_heads}, ffn={cfg.intermediate_size}")
    print(f"  Vocab size: {cfg.vocab_size}")

    n_enc = sum(p.numel() for p in encoder.parameters())
    n_dec = sum(p.numel() for p in decoder.parameters())
    print(f"  Encoder params: {n_enc:,}")
    print(f"  Decoder params: {n_dec:,}")

    if cfg.vocab_size == VOCAB_SIZE + 5:   # 49156 + 5 special tokens
        ok(f"Encoder vocab_size={cfg.vocab_size} = {VOCAB_SIZE} CpGs + 5 special tokens")
    elif cfg.vocab_size >= VOCAB_SIZE:
        ok(f"Encoder vocab_size={cfg.vocab_size} ≥ {VOCAB_SIZE}")
    else:
        fail(f"Encoder vocab_size={cfg.vocab_size} < fine-tune CpGs {VOCAB_SIZE}")

    if decoder.decoder[-2].out_features == VOCAB_SIZE:
        ok(f"Decoder output size={decoder.decoder[-2].out_features} matches VOCAB_SIZE")
    else:
        fail(f"Decoder output={decoder.decoder[-2].out_features} ≠ VOCAB_SIZE={VOCAB_SIZE}")

except Exception as e:
    import traceback
    fail(f"Checkpoint loading failed: {e}")
    traceback.print_exc()

# ── 9. Forward pass smoke test ────────────────────────────────────────────────
print("\n" + "="*60)
print("[9] Forward pass smoke test (CPU, batch=2)")
print("="*60)
try:
    from bmfm_methylation.llama.finetune_llama import MethylationAgeRegressorLlama

    module = MethylationAgeRegressorLlama(
        encoder=encoder,
        decoder=decoder,
        hidden_size=encoder.config.hidden_size,
        head_hidden_size=128,
        age_mean=ds.age_mean,
        age_std=ds.age_std,
        freeze_encoder=True,
        recon_weight=0.1,
    ).eval()

    examples2 = [ds[i] for i in range(2)]
    batch2    = collator(examples2)

    with torch.no_grad():
        out = module._shared_step(batch2, "val")

    print(f"  loss={out['loss'].item():.4f}, "
          f"age_loss={out['age_loss'].item():.4f}, "
          f"recon_loss={out['recon_loss'].item():.4f}, "
          f"mae={out['mae'].item():.4f}")

    if torch.isfinite(out["loss"]):
        ok(f"Forward pass successful — loss={out['loss'].item():.4f}")
    else:
        fail(f"Non-finite loss: {out['loss'].item()}")

    if out["recon_loss"].item() > 0:
        ok(f"Reconstruction loss is non-zero: {out['recon_loss'].item():.4f}")
    else:
        warn("Reconstruction loss=0 — no valid non-input CpGs (check INPUT_RATIO vs NaN rate)")

    # MAE in years (mae is computed in _shared_step in z-score space * std)
    mae_years = out["mae"].item() * ds.age_std
    print(f"  MAE (years, untrained): {mae_years:.1f}")

except Exception as e:
    import traceback
    fail(f"Forward pass failed: {e}")
    traceback.print_exc()

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("VALIDATION SUMMARY")
print("="*60)
if errors:
    print(f"\n  {FAIL} {len(errors)} ERROR(S) — fix before running fine-tuning:")
    for e in errors:
        print(f"    - {e}")
else:
    print(f"\n  {PASS} No errors — pipeline is ready to run")

if warnings:
    print(f"\n  {WARN} {len(warnings)} WARNING(S):")
    for w in warnings:
        print(f"    - {w}")

print()
