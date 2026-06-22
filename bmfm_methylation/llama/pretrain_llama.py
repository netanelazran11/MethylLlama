#!/usr/bin/env python3
"""
Pretraining script for MethylLlama — LLaMA-style methylation encoder.

Supports both WCED and MLM pretraining modes (same data module as SCBert).

Architecture differences from standard pretrain.py:
  - MethylLlamaModel replaces SCBertModel
  - No bmfm_targets model dependency (pure PyTorch)
  - No monkey-patching required (cpg_scale + ScaleAdapt are built-in)
  - RMSNorm, Pre-LN, SwiGLU, RoPE used throughout
  - ScaleAdaptEncoder for beta values (sinusoidal basis, trainable freqs)

Usage:
  # WCED pretraining (recommended):
  python -m bmfm_methylation.llama_methyl.pretrain_llama \\
      data_path=/path/to/methylation.h5ad \\
      output_directory=./outputs/llama_wced

  # MLM pretraining:
  python -m bmfm_methylation.llama_methyl.pretrain_llama \\
      data_path=/path/to/methylation.h5ad \\
      pretraining_mode=mlm \\
      output_directory=./outputs/llama_mlm
"""

# =============================================================================
# CRITICAL: Patch torch.load BEFORE any other imports!
# PyTorch 2.6+ changed default weights_only=True which breaks Lightning checkpoints
# =============================================================================
import torch
import torch.serialization

_orig_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_load(*args, **kwargs)
torch.load = _patched_load
torch.serialization.load = _patched_load
# =============================================================================

import logging
import os
import sys
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

# Ensure parent package is importable

from bmfm_methylation.shared.tokenizer import (
    create_indexed_tokenizer,
    extract_cpg_sites_from_h5ad,
    create_methylation_multifield_tokenizer,
)
from bmfm_methylation.shared.data_module import MethylationDataModule, WCEDCollator, BMFMWCEDCollator

from .model import MethylLlamaConfig, MethylLlamaModel, init_cpg_embeddings_from_dna
from .wced_llama import WCEDLlamaModule

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tokenizer helpers (reuse from parent package)
# ---------------------------------------------------------------------------

def setup_tokenizer(cfg: DictConfig):
    """Create or load MultiField tokenizer.

    For pretrain data where var_names are integers (not CpG names),
    pass probe_ids_csv in config to use the explicit probe ID list.
    """
    tokenizer_path = Path(cfg.tokenizer_path)
    if tokenizer_path.exists() and (tokenizer_path / "tokenizers").exists():
        logger.info(f"Loading tokenizer from {tokenizer_path}")
        from bmfm_targets.tokenization import MultiFieldTokenizer
        return MultiFieldTokenizer.from_pretrained(str(tokenizer_path))
    else:
        probe_ids_csv = cfg.get("probe_ids_csv", None)
        logger.info(
            f"Creating tokenizer from {cfg.data_path}"
            + (f" + probe_ids CSV: {probe_ids_csv}" if probe_ids_csv else "")
        )
        cpg_sites = extract_cpg_sites_from_h5ad(cfg.data_path, probe_ids_csv=probe_ids_csv)
        logger.info(f"  {len(cpg_sites)} CpG sites found")
        tokenizer = create_methylation_multifield_tokenizer(
            cpg_sites=cpg_sites,
            output_dir=str(tokenizer_path),
        )
        logger.info(f"Tokenizer saved to {tokenizer_path}")
        return tokenizer


def setup_wandb(cfg: DictConfig):
    """Setup WandB or TensorBoard logger."""
    if hasattr(cfg, "track_wandb") and cfg.track_wandb.get("enabled", False):
        try:
            from pytorch_lightning.loggers import WandbLogger
            import os
            # Force rank-0-only init to avoid DDP multi-process WandB conflicts.
            # Without this, validation metrics synced via sync_dist=True are lost.
            return WandbLogger(
                project=cfg.track_wandb.get("project", "methylation-llama-pretrain"),
                entity=cfg.track_wandb.get("entity"),
                name=cfg.track_wandb.get("name", "llama_wced_pretrain"),
                save_dir=cfg.output_directory,
                log_model=False,
            )
        except Exception as e:
            logger.warning(f"WandB failed: {e} — falling back to TensorBoard")
    from pytorch_lightning.loggers import TensorBoardLogger
    return TensorBoardLogger(cfg.output_directory, name="pretrain_llama")


# ---------------------------------------------------------------------------
# Model config builder
# ---------------------------------------------------------------------------

def build_model_config(cfg: DictConfig, vocab_size: int) -> MethylLlamaConfig:
    """Build MethylLlamaConfig from Hydra config.

    vocab_size must be passed explicitly — derived from the actual tokenizer,
    not hardcoded. Formula: n_cpg_sites + 5 (UNK, SEP, PAD, CLS, MASK).
    For 8000 CpGs: vocab_size = 8005.
    """
    mc = cfg.get("model", {})
    return MethylLlamaConfig(
        hidden_size=mc.get("hidden_size", 512),
        num_hidden_layers=mc.get("num_hidden_layers", 6),
        num_attention_heads=mc.get("num_attention_heads", 8),
        intermediate_size=mc.get("intermediate_size", 1408),
        vocab_size=vocab_size,  # always tokenizer-derived, never from cfg
        rope_theta=mc.get("rope_theta", 10000.0),
        rms_norm_eps=mc.get("rms_norm_eps", 1e-6),
        hidden_dropout_prob=mc.get("hidden_dropout_prob", 0.1),
        n_sin_basis=mc.get("n_sin_basis", 48),
        basis_scale=mc.get("basis_scale", 2.0),
        scale_adapt_trainable=mc.get("scale_adapt_trainable", True),
        cpg_scale_init=mc.get("cpg_scale_init", 0.1),
        add_pooling_layer=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(
    config_path="configs",
    config_name="pretrain_llama",
    version_base="1.2",
)
def main(cfg: DictConfig):
    pretraining_mode = cfg.get("pretraining_mode", "wced").lower()

    logger.info("=" * 70)
    logger.info(f"METHYLLAMA PRETRAINING ({pretraining_mode.upper()})")
    logger.info("=" * 70)
    logger.info(f"\n{OmegaConf.to_yaml(cfg)}")

    # Seed
    if hasattr(cfg, "seed") and cfg.seed:
        pl.seed_everything(cfg.seed.get("seed_value", 42), workers=True)

    output_dir = Path(cfg.output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tokenizer
    tokenizer = setup_tokenizer(cfg)

    # Build FieldInfo for MultiFieldCollator.
    # Note: in WCED mode the collator is immediately replaced by WCEDCollator,
    # so this only needs to be valid enough to not crash MultiFieldCollator.__init__.
    from bmfm_targets.config import FieldInfo
    fields = [
        FieldInfo(
            field_name="cpg_sites",
            is_input=True,
            is_masked=False,
            tokenization_strategy="tokenize",
        ),
    ]

    # Data module settings
    dm_cfg = cfg.get("data_module", {})
    subset_k          = dm_cfg.get("subset_k", 8000)
    fixed_subset      = dm_cfg.get("fixed_subset", True)
    fixed_subset_seed = dm_cfg.get("fixed_subset_seed", 42)
    bmfm_style        = dm_cfg.get("bmfm_style", False)

    # wced_vocab_size: how many CpGs the decoder outputs per batch
    # = subset_k (the random subset selected per training step from all available CpGs)
    # This is SEPARATE from model_vocab_size (the CpG embedding table size)
    wced_vocab_size = subset_k

    use_mlm  = (pretraining_mode == "mlm")
    mask_ratio = dm_cfg.get("mask_ratio", 0.3) if use_mlm else 0.0

    data_module = MethylationDataModule(
        tokenizer=tokenizer,
        fields=fields,
        h5ad_path=cfg.data_path,
        train_split="train",
        val_split="valid",
        test_split="test",
        batch_size=dm_cfg.get("batch_size", 16),
        num_workers=dm_cfg.get("num_workers", 4),
        max_length=dm_cfg.get("max_length", 8002),
        mlm=use_mlm,
        change_ratio=dm_cfg.get("change_ratio", 0.1) if use_mlm else 0.0,
        mask_ratio=mask_ratio,
        switch_ratio=dm_cfg.get("switch_ratio", 0.0) if use_mlm else 0.0,
        collation_strategy="language_modeling",
        subset_k=subset_k,
        fixed_subset=fixed_subset,
        fixed_subset_seed=fixed_subset_seed,
        bmfm_style=bmfm_style,
    )

    # WCED settings
    wced_input_ratio     = cfg.get("wced_input_ratio", 0.5)
    wced_contrastive     = cfg.get("wced_contrastive", False)
    wced_contrastive_wt  = cfg.get("wced_contrastive_weight", 0.0)
    wced_contrastive_temp= cfg.get("wced_contrastive_temp", 0.1)
    wced_normalize_loss  = cfg.get("wced_normalize_loss", False)
    wced_age_weight      = cfg.get("wced_age_weight", 1.0)
    wced_decoder_dropout = cfg.get("wced_decoder_dropout", 0.1)

    def _wrap_collator():
        base_collator = data_module.collator

        if pretraining_mode == "wced":
            # Get CpG site list from dataset
            cpg_sites = None
            for ds in [data_module.train_dataset, data_module.val_dataset, data_module.test_dataset]:
                if ds is not None:
                    cpg_sites = ds.cpg_sites
                    break
            if cpg_sites is None:
                raise ValueError("No CpG site list found for WCED collator")

            if bmfm_style:
                # BMFM-style: NaN excluded from input; -100 labels for loss masking.
                # Requires data_module datasets created with bmfm_style=True so that
                # MFIs carry full_betas in metadata for label construction.
                wced_collator = BMFMWCEDCollator(
                    tokenizer=data_module.tokenizer,
                    cpg_sites=cpg_sites,
                    vocab_size=wced_vocab_size,
                    input_ratio=wced_input_ratio,
                    contrastive=wced_contrastive,
                    fixed_subset_seed=fixed_subset_seed,
                )
            else:
                wced_collator = WCEDCollator(
                    tokenizer=data_module.tokenizer,
                    cpg_sites=cpg_sites,
                    vocab_size=wced_vocab_size,
                    input_ratio=wced_input_ratio,
                    fixed_subset_seed=fixed_subset_seed,
                    contrastive=wced_contrastive,
                )

            data_module.collator = lambda examples: wced_collator(examples)

        else:
            # MLM: wrap batch to match WCEDLlamaModule / MLMTrainingModule interface
            def _collate_mlm(examples):
                batch = base_collator(examples)
                input_ids = torch.stack(
                    [batch["cpg_ids"].float(), batch["beta_values"]], dim=1
                )  # [B, 2, L]
                labels_beta = batch["labels_beta"].clone()
                loss_mask   = batch["loss_mask_beta"]
                labels_beta[loss_mask == 0] = -100.0
                return {
                    "input_ids":      input_ids,
                    "attention_mask": batch["attention_mask"],
                    "labels":         {"beta_values": labels_beta},
                }
            data_module.collator = _collate_mlm

    original_setup = data_module.setup

    def _setup_with_wrap(stage=None):
        original_setup(stage)
        _wrap_collator()

    data_module.setup = _setup_with_wrap
    data_module.setup()

    # Build model config
    # model_vocab_size = size of the CpG embedding lookup table
    # = ALL CpG sites in the h5ad file + 5 special tokens
    # DIFFERENT from wced_vocab_size (decoder output = subset_k per batch)
    #
    # Example with 49k pretrain data:
    #   model_vocab_size = 49156 + 5 = 49161  (embedding table covers all CpGs)
    #   wced_vocab_size  = 8000               (decoder outputs 8k CpGs per batch)
    #
    # Example with 8k AltumAge data:
    #   model_vocab_size = 8000 + 5 = 8005    (embedding table)
    #   wced_vocab_size  = 8000               (decoder = all CpGs)
    probe_ids_csv = cfg.get("probe_ids_csv", None)
    cpg_sites_all = extract_cpg_sites_from_h5ad(cfg.data_path, probe_ids_csv=probe_ids_csv)
    n_special_tokens = 5  # UNK=0, SEP=1, PAD=2, CLS=3, MASK=4
    model_vocab_size = len(cpg_sites_all) + n_special_tokens
    logger.info(
        f"Tokenizer vocab: {len(cpg_sites_all)} CpG sites + {n_special_tokens} special = {model_vocab_size}"
    )
    model_config = build_model_config(cfg, vocab_size=model_vocab_size)
    logger.info(
        f"Model: {model_config.num_hidden_layers}L × {model_config.hidden_size}D, "
        f"SwiGLU intermediate={model_config.intermediate_size}, "
        f"RoPE theta={model_config.rope_theta}, "
        f"ScaleAdapt basis={model_config.n_sin_basis} × {model_config.basis_scale}"
    )

    # Create training module
    if pretraining_mode == "wced":
        logger.info(
            f"WCED: model_vocab={model_vocab_size}, decoder_vocab={wced_vocab_size}, "
            f"input_ratio={wced_input_ratio}, age_weight={wced_age_weight}, "
            f"contrastive_weight={wced_contrastive_wt}"
        )

        tr_cfg = cfg.get("trainer", {})
        module = WCEDLlamaModule(
            model_config=model_config,
            learning_rate=tr_cfg.get("learning_rate", 5e-4),
            weight_decay=tr_cfg.get("weight_decay", 0.01),
            warmup_steps=tr_cfg.get("warmup_steps", 100),
            lr_decay_steps=tr_cfg.get("lr_decay_steps", 0),
            vocab_size=wced_vocab_size,   # decoder output size = subset_k
            contrastive_weight=wced_contrastive_wt,
            contrastive_temp=wced_contrastive_temp,
            normalize_loss=wced_normalize_loss,
            age_weight=wced_age_weight,
            decoder_dropout=wced_decoder_dropout,
            betas=tuple(tr_cfg.get("betas", [0.9, 0.99])),
            epsilon=tr_cfg.get("epsilon", 1e-8),
        )

    else:
        raise NotImplementedError(
            "MLM pretraining for MethylLlama not yet implemented. "
            "Use pretraining_mode=wced (recommended) or use pretrain.py for SCBert MLM."
        )

    # Optionally initialize CpG embedding table with BMFM-DNA embeddings
    cpg_emb_npy = cfg.get("cpg_embeddings_npy", None)
    cpg_emb_ids = cfg.get("cpg_embeddings_ids", None)
    dna_init_stats = {}
    if cpg_emb_npy and cpg_emb_ids:
        logger.info(f"Initializing CpG embeddings from BMFM-DNA: {cpg_emb_npy}")
        n = init_cpg_embeddings_from_dna(
            model=module.encoder,
            tokenizer=tokenizer,
            npy_path=cpg_emb_npy,
            ids_path=cpg_emb_ids,
        )
        logger.info(f"BMFM-DNA init complete: {n} CpGs initialized")
        # Collect embedding stats for WandB
        _w = module.encoder.embeddings.cpg_sites_embeddings.weight.data[5:].float()
        _norms = _w.norm(dim=1)
        dna_init_stats = {
            "dna_init/cpgs_initialized": n,
            "dna_init/emb_norm_mean": _norms.mean().item(),
            "dna_init/emb_norm_std":  _norms.std().item(),
            "dna_init/emb_norm_min":  _norms.min().item(),
            "dna_init/emb_norm_max":  _norms.max().item(),
        }

    # ── Startup sanity checks ──────────────────────────────────────────────────
    _sep = "=" * 70
    print(_sep)
    print("STARTUP SANITY CHECK")
    print(_sep)

    # Model
    _total_params = sum(p.numel() for p in module.encoder.parameters())
    _trainable    = sum(p.numel() for p in module.encoder.parameters() if p.requires_grad)
    print(f"[Model]  total={_total_params/1e6:.1f}M  trainable={_trainable/1e6:.1f}M")
    print(f"[Model]  vocab_size={module.encoder.config.vocab_size}  "
          f"hidden={module.encoder.config.hidden_size}  "
          f"layers={module.encoder.config.num_hidden_layers}  "
          f"heads={module.encoder.config.num_attention_heads}")
    print(f"[Model]  cpg_emb shape: {list(module.encoder.embeddings.cpg_sites_embeddings.weight.shape)}")
    _emb_w = module.encoder.embeddings.cpg_sites_embeddings.weight.data[5:].float()
    _emb_norms = _emb_w.norm(dim=1)
    _norm_mean = _emb_norms.mean().item()
    _norm_std  = _emb_norms.std().item()
    print(f"[Emb norms] mean={_norm_mean:.4f}  std={_norm_std:.4f}  (random init)")
    if dna_init_stats:
        dna_init_stats["dna_init/emb_norm_mean"] = _norm_mean
        dna_init_stats["dna_init/emb_norm_std"]  = _norm_std

    # Data splits
    for split_name, ds in [("train", data_module.train_dataset),
                            ("val",   data_module.val_dataset),
                            ("test",  data_module.test_dataset)]:
        if ds is not None:
            print(f"[Data]   {split_name}: {len(ds)} samples  "
                  f"CpGs={ds.num_cpg_sites}  subset_k={subset_k}")
        else:
            print(f"[Data]   {split_name}: None")

    # Batch smoke-test — run one batch through model to check shapes
    try:
        _loader = data_module.train_dataloader()
        _batch  = next(iter(_loader))
        print(f"[Batch]  keys: {list(_batch.keys())}")
        _cpg_ids    = _batch["cpg_ids"]       # [B, L]
        _betas      = _batch["beta_values"]   # [B, L]
        _mask       = _batch.get("attention_mask")
        print(f"[Batch]  cpg_ids shape:     {list(_cpg_ids.shape)}")
        print(f"[Batch]  beta_values shape: {list(_betas.shape)}")
        if _mask is not None:
            print(f"[Batch]  attention_mask:   {list(_mask.shape)}")
        if "all_betas" in _batch:
            print(f"[Batch]  all_betas shape:  {list(_batch['all_betas'].shape)}")
        with torch.no_grad():
            _dev = next(module.encoder.parameters()).device
            _out = module.encoder(
                input_ids=torch.stack([_cpg_ids[:2].float(), _betas[:2]], dim=1).to(_dev),
                attention_mask=_mask[:2].to(_dev) if _mask is not None else None,
            )
        print(f"[Batch]  last_hidden_state: {list(_out.last_hidden_state.shape)}")
        print(f"[Batch]  pooler_output:     {list(_out.pooler_output.shape)}")
        print("[Batch]  Smoke-test PASSED")
        _sanity_stats = {
            "sanity/train_samples": len(data_module.train_dataset),
            "sanity/val_samples":   len(data_module.val_dataset) if data_module.val_dataset else 0,
            "sanity/seq_len": _cpg_ids.shape[-1],
            "sanity/model_total_params_M": round(_total_params / 1e6, 1),
        }
    except Exception as _e:
        print(f"[Batch]  Smoke-test FAILED: {_e}")
        _sanity_stats = {}

    print(_sep)

    # Logger
    exp_logger = setup_wandb(cfg)

    # Log sanity + DNA-init stats to WandB at step 0
    _wandb_stats = {**dna_init_stats, **_sanity_stats}
    if _wandb_stats:
        try:
            import wandb
            if wandb.run is not None:
                wandb.log(_wandb_stats, step=0)
                logger.info(f"WandB sanity stats logged: {list(_wandb_stats.keys())}")
        except Exception as _we:
            logger.warning(f"WandB sanity log failed (non-fatal): {_we}")

    # Callbacks
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=str(output_dir / "checkpoints"),
            filename="epoch={epoch}-val_loss={validation/loss:.4f}",
            monitor="validation/loss",
            mode="min",
            save_top_k=3,
            auto_insert_metric_name=False,
        ),
        pl.callbacks.EarlyStopping(
            monitor="validation/loss",
            patience=cfg.get("early_stop_patience", 60),
            mode="min",
            check_on_train_epoch_end=False,  # check after validation, not before
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
    ]

    # Trainer
    # find_unused_parameters=True: projection_head and age_head are disabled
    # (contrastive_weight=0, age_weight=0) so DDP must not error on unused params.
    from pytorch_lightning.strategies import DDPStrategy
    trainer = pl.Trainer(
        max_epochs=cfg.get("pretrain_epochs", 300),
        accelerator="auto",
        devices="auto",
        strategy=DDPStrategy(find_unused_parameters=True),
        precision=cfg.get("precision", "16-mixed"),
        accumulate_grad_batches=cfg.get("accumulate_grad_batches", 8),
        gradient_clip_val=cfg.get("gradient_clip_val", 1.0),
        logger=exp_logger,
        callbacks=callbacks,
        log_every_n_steps=10,
        val_check_interval=1.0,
        enable_progress_bar=True,
    )

    # Resume from checkpoint if provided
    resume_ckpt = cfg.get("resume_checkpoint", None)
    if resume_ckpt:
        logger.info(f"Resuming from checkpoint: {resume_ckpt}")
    else:
        logger.info("Starting pretraining from scratch...")

    trainer.fit(module, datamodule=data_module, ckpt_path=resume_ckpt)

    # Test
    if data_module.test_dataset is not None:
        logger.info("Running test evaluation...")
        trainer.test(module, datamodule=data_module)

    logger.info(f"Pretraining complete. Checkpoints saved to {output_dir / 'checkpoints'}")


if __name__ == "__main__":
    main()
