#!/usr/bin/env python3
"""
Fine-tuning script for Methylation Age Prediction

This script fine-tunes a pretrained BMFM SCBertModel for age prediction
from methylation data.

Usage:
    python -m bmfm_methylation.finetune \
        data_path=/path/to/methylation.h5ad \
        checkpoint_path=/path/to/pretrained.ckpt \
        output_directory=./outputs

Or without pretraining (train from scratch):
    python -m bmfm_methylation.finetune \
        data_path=/path/to/methylation.h5ad \
        checkpoint_path=null \
        output_directory=./outputs
"""

# =============================================================================
# CRITICAL: This patch MUST be BEFORE any other imports!
# PyTorch 2.6 changed default weights_only=True which breaks Lightning checkpoints
# We monkey-patch torch.load BEFORE pytorch_lightning imports it
# =============================================================================
import torch
import torch.serialization

_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load
torch.serialization.load = _patched_torch_load  # Patch both locations
# =============================================================================

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import hydra
import pytorch_lightning as pl
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
torch.load = _patched_torch_load
# =============================================================================

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from bmfm_methylation.shared.tokenizer import (
    extract_cpg_sites_from_h5ad,
    create_methylation_multifield_tokenizer,
)
from bmfm_methylation.shared.data_module import MethylationDataModule

logger = logging.getLogger(__name__)


class AttentionPooling(nn.Module):
    """Learned weighted average over token representations.

    Computes a scalar attention weight per token, applies softmax (masked),
    then returns the weighted sum. This lets the model focus on age-relevant
    CpGs rather than treating all 8k tokens equally (mean pooling).
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # hidden_states: [batch, seq, hidden]
        # attention_mask: [batch, seq]  (1=real token, 0=pad)
        scores = self.attn(hidden_states).squeeze(-1)          # [batch, seq]
        if attention_mask.dim() == 3:
            attention_mask = attention_mask[:, 0, :]
        scores = scores.masked_fill(attention_mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # [batch, seq, 1]
        return (weights * hidden_states).sum(dim=1)            # [batch, hidden]


class MethylationAgeRegressor(pl.LightningModule):
    """
    Lightning module for methylation age regression.

    Uses the pretrained BMFM SCBert encoder to produce per-token representations
    from the multi-field input (CpG IDs + beta values), then pools and
    feeds through an MLP head for age prediction.

    Uses mean pooling over content tokens (skip CLS) since MLM pretraining
    doesn't train CLS to aggregate information.

    Pipeline:
        [CpG IDs + β-values] → Pretrained Encoder → Mean Pool → MLP head → age
    """

    def __init__(
        self,
        encoder,
        num_cpg_sites: int = 8000,
        hidden_size: int = 512,
        head_hidden_size: int = 256,
        head_dropout: float = 0.1,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_steps: int = 10000,
        age_mean: float = 0.0,
        age_std: float = 1.0,
        freeze_encoder: bool = True,
        unfreeze_encoder_epoch: int = 5,
        encoder_lr_multiplier: float = 0.1,
        use_huber_loss: bool = False,
        huber_delta: float = 2.0,
        pearson_weight: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['encoder'])

        self.encoder = encoder
        self.age_mean = age_mean
        self.age_std = age_std

        # Optionally freeze encoder (will be unfrozen at unfreeze_encoder_epoch)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            logger.info(f"Encoder frozen (will unfreeze at epoch {unfreeze_encoder_epoch})")

        # Attention pooling: learns a scalar weight per token position.
        # Softmax over all positions → weighted sum → single vector.
        # This focuses on age-relevant CpGs rather than averaging all 8k equally.
        self.attn_pool = AttentionPooling(hidden_size)

        # MLP head for age prediction: 512 -> 64 -> 1
        # The encoder does the heavy lifting — head only needs to extract age.
        # Simpler head = less overfitting, faster convergence.
        # Input LayerNorm stabilizes the pooled representation before the linear.
        self.age_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, head_hidden_size),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden_size, 1),
        )

        if use_huber_loss:
            self.loss_fn = nn.HuberLoss(delta=huber_delta)
            logger.info(f"Using HuberLoss with delta={huber_delta}")
        else:
            self.loss_fn = nn.MSELoss()
            logger.info("Using MSELoss")

        # Accumulate predictions for epoch-level R² computation
        self._val_preds = []
        self._val_labels = []
        self._test_preds = []
        self._test_labels = []

        head_params = sum(p.numel() for p in self.age_head.parameters())
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Encoder params: {encoder_params:,}")
        logger.info(f"Age head params: {head_params:,}")
        logger.info(f"Trainable params: {trainable:,}")
        logger.info(f"Freeze encoder: {freeze_encoder}, unfreeze at epoch {unfreeze_encoder_epoch}")

    def forward(self, cpg_ids, beta_values, attention_mask=None):
        # Build BMFM-style input_ids: [batch, 2, seq_len]
        input_ids = torch.stack([cpg_ids.float(), beta_values], dim=1)
        batch_size = input_ids.size(0)
        seq_length = input_ids.size(2)

        # Ensure attention_mask is 2D [batch, seq_len]
        if attention_mask is not None and attention_mask.dim() == 3:
            attention_mask = attention_mask[:, 0, :]
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length), device=input_ids.device
            )

        # Pass through pretrained encoder (uses CpG IDs + beta values)
        encoder_output = self.encoder(input_ids, attention_mask=attention_mask)
        sequence_output = encoder_output.last_hidden_state  # [batch, seq_len, hidden]

        # Attention pooling — learned weighted average over all tokens.
        # Mean pooling equally weights all 8k CpGs (dilutes age signal).
        # CLS pooling is wrong for MLM (CLS was never trained to aggregate).
        # Attention pooling lets the model learn which CpGs matter for age.
        pooled = self.attn_pool(sequence_output, attention_mask)

        # Age prediction head
        age_pred = self.age_head(pooled)

        return age_pred

    def on_train_epoch_start(self):
        """Unfreeze encoder after N epochs."""
        epoch = self.current_epoch
        if (self.hparams.freeze_encoder and
                epoch == self.hparams.unfreeze_encoder_epoch):
            logger.info("=" * 70)
            logger.info(f"[EPOCH {epoch}] UNFREEZING encoder")
            logger.info("=" * 70)
            for param in self.encoder.parameters():
                param.requires_grad = True
            # Encoder params are already in the optimizer (added at init with
            # lower LR). While frozen, they had requires_grad=False so the
            # optimizer skipped them (grad=None). Now they will get gradients.
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            logger.info(f"[EPOCH {epoch}] Trainable params after unfreeze: {trainable:,}")
            logger.info("=" * 70)

    def _shared_step(self, batch, stage: str):
        cpg_ids = batch["cpg_ids"]
        beta_values = batch["beta_values"]
        attention_mask = batch.get("attention_mask")
        labels = batch["labels"].float().view(-1, 1)

        # DEBUG: Print batch info on first step to verify data pipeline
        if not hasattr(self, '_debug_printed') or not self._debug_printed:
            self._debug_printed = True
            logger.info("=" * 70)
            logger.info("DEBUG: BATCH INSPECTION")
            logger.info(f"  cpg_ids shape: {cpg_ids.shape}")
            logger.info(f"  beta_values shape: {beta_values.shape}")
            logger.info(f"  cpg_ids dtype: {cpg_ids.dtype}")
            logger.info(f"  batch keys: {list(batch.keys())}")
            logger.info(f"  cpg_ids first 10: {cpg_ids[0, :10].tolist()}")
            logger.info(f"  beta_values first 10: {beta_values[0, :10].tolist()}")
            if beta_values.shape[0] > 1:
                b0 = beta_values[0, :10].tolist()
                b1 = beta_values[1, :10].tolist()
                logger.info(f"  beta_values sample0: {b0}")
                logger.info(f"  beta_values sample1: {b1}")
                logger.info(f"  beta_values same across samples? {b0 == b1}")
            logger.info(f"  beta_values min={beta_values.min():.4f}, max={beta_values.max():.4f}, std={beta_values.std():.4f}")
            if attention_mask is not None:
                logger.info(f"  attention_mask shape: {attention_mask.shape}")
                logger.info(f"  attention_mask sum (non-pad tokens): {attention_mask[0].sum().item()}")
            logger.info(f"  labels (first 5): {labels[:5, 0].tolist()}")
            logger.info(f"  labels std: {labels.std():.4f}")
            logger.info(f"  age_mean={self.age_mean:.2f}, age_std={self.age_std:.2f}")
            logger.info("=" * 70)

        # DEBUG: Verify Option-B subset changes across consecutive batches
        if not hasattr(self, "_debug_batch_count"):
            self._debug_batch_count = 0
            self._prev_cpg_ids = None
        if self._debug_batch_count < 2:
            self._debug_batch_count += 1
            if self._prev_cpg_ids is not None:
                prev = self._prev_cpg_ids[0, 1:]
                curr = cpg_ids[0, 1:]
                diff = int((prev != curr).sum().item())
                logger.info(f"DEBUG subset diff (batch-1 vs batch): {diff}")
            self._prev_cpg_ids = cpg_ids.detach().clone()

        # Forward pass
        predictions = self(cpg_ids, beta_values, attention_mask)

        # DEBUG: Check predictions and encoder output on first few steps
        if not hasattr(self, '_debug_pred_count'):
            self._debug_pred_count = 0
        if self._debug_pred_count < 3:
            self._debug_pred_count += 1
            preds_flat = predictions.detach()[:5, 0].tolist()
            labels_flat = labels[:5, 0].tolist()
            logger.info(f"DEBUG step {self._debug_pred_count}: preds={preds_flat}, labels={labels_flat}")

            # Also check encoder output statistics
            if self._debug_pred_count == 1:
                # Do a forward pass inspection
                with torch.no_grad():
                    input_ids = torch.stack([cpg_ids.float(), beta_values], dim=1)
                    enc_out = self.encoder(input_ids, attention_mask=attention_mask)
                    hidden = enc_out.last_hidden_state
                    logger.info("=" * 70)
                    logger.info("DEBUG: ENCODER OUTPUT INSPECTION")
                    logger.info(f"  Hidden state shape: {hidden.shape}")
                    logger.info(f"  Hidden state stats: mean={hidden.mean():.6f}, std={hidden.std():.6f}")
                    logger.info(f"  Hidden state min/max: {hidden.min():.4f} / {hidden.max():.4f}")
                    logger.info(f"  Hidden state sample[0,0,:10]: {hidden[0, 0, :10].tolist()}")
                    logger.info(f"  Hidden state sample[0,100,:10]: {hidden[0, 100, :10].tolist()}")
                    # Check if hidden states vary across positions
                    pos_variance = hidden[0].var(dim=0).mean()
                    logger.info(f"  Variance across positions: {pos_variance:.6f}")
                    # Check if hidden states vary across samples
                    if hidden.shape[0] > 1:
                        sample_variance = hidden[:, 100, :].var(dim=0).mean()
                        logger.info(f"  Variance across samples (pos 100): {sample_variance:.6f}")
                    logger.info("=" * 70)

        # Loss (on normalized values)
        mse_loss = self.loss_fn(predictions, labels)

        # Pearson correlation component — directly optimizes R²
        # loss = MSE + pearson_weight * (1 - PCC)
        pred_m  = predictions - predictions.mean()
        label_m = labels - labels.mean()
        pcc = (pred_m * label_m).sum() / (
            (pred_m.pow(2).sum() * label_m.pow(2).sum()).sqrt().clamp(min=1e-8)
        )
        loss = mse_loss + self.hparams.pearson_weight * (1.0 - pcc)

        # Denormalize for metrics
        preds_denorm = predictions * self.age_std + self.age_mean
        labels_denorm = labels * self.age_std + self.age_mean

        # Compute MAE
        mae = torch.abs(preds_denorm - labels_denorm).mean()

        return loss, mae, preds_denorm, labels_denorm

    def training_step(self, batch, batch_idx):
        loss, mae, _, _ = self._shared_step(batch, "train")

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/mae", mae, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, mae, preds, labels = self._shared_step(batch, "val")

        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/mae", mae, on_epoch=True, prog_bar=True)

        # Accumulate for epoch-level R² (per-batch R² is unreliable)
        self._val_preds.append(preds.detach())
        self._val_labels.append(labels.detach())

        return loss

    def on_validation_epoch_end(self):
        if self._val_preds:
            all_preds = torch.cat(self._val_preds, dim=0)
            all_labels = torch.cat(self._val_labels, dim=0)

            ss_res = torch.sum((all_labels - all_preds) ** 2)
            ss_tot = torch.sum((all_labels - all_labels.mean()) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-8)
            self.log("val/r2", r2, prog_bar=True)

            # Also log epoch-level MAE for accuracy
            epoch_mae = torch.abs(all_preds - all_labels).mean()
            self.log("val/mae_epoch", epoch_mae)

        self._val_preds.clear()
        self._val_labels.clear()

    def test_step(self, batch, batch_idx):
        loss, mae, preds, labels = self._shared_step(batch, "test")

        self.log("test/mae", mae, on_epoch=True)

        # Accumulate for epoch-level R²
        self._test_preds.append(preds.detach())
        self._test_labels.append(labels.detach())

        return loss

    def on_test_epoch_end(self):
        if self._test_preds:
            all_preds = torch.cat(self._test_preds, dim=0)
            all_labels = torch.cat(self._test_labels, dim=0)

            ss_res = torch.sum((all_labels - all_preds) ** 2)
            ss_tot = torch.sum((all_labels - all_labels.mean()) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-8)
            self.log("test/r2", r2)

            epoch_mae = torch.abs(all_preds - all_labels).mean()
            self.log("test/mae_epoch", epoch_mae)

            logger.info(f"Test MAE: {epoch_mae:.2f} years, R2: {r2:.4f}")

        self._test_preds.clear()
        self._test_labels.clear()

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
        encoder_lr = self.hparams.learning_rate * self.hparams.encoder_lr_multiplier

        # Include ALL params from the start (head + encoder).
        # Encoder params start frozen (requires_grad=False), so optimizer
        # skips them (grad=None). When unfrozen at unfreeze_encoder_epoch,
        # gradients flow and optimizer updates them. This avoids LR scheduler
        # mismatch when adding param groups later.
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.age_head.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
                "lr": self.hparams.learning_rate,
            },
            {
                "params": [p for n, p in self.age_head.named_parameters()
                           if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": self.hparams.learning_rate,
            },
            {
                "params": [p for n, p in self.encoder.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
                "lr": encoder_lr,
            },
            {
                "params": [p for n, p in self.encoder.named_parameters()
                           if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": encoder_lr,
            },
        ]

        # Filter out empty groups
        optimizer_grouped_parameters = [
            g for g in optimizer_grouped_parameters if len(g["params"]) > 0
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,  # Default LR (will be overridden by group LRs)
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # Learning rate scheduler with warmup + cosine decay
        def lr_lambda(current_step):
            if current_step < self.hparams.warmup_steps:
                return float(current_step) / float(max(1, self.hparams.warmup_steps))
            progress = float(current_step - self.hparams.warmup_steps) / \
                       float(max(1, self.hparams.max_steps - self.hparams.warmup_steps))
            return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * progress)).item()))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


def setup_tokenizer(cfg: DictConfig):
    """Create or load tokenizer."""
    tokenizer_path = Path(cfg.tokenizer_path)

    if tokenizer_path.exists() and (tokenizer_path / "tokenizers").exists():
        logger.info(f"Loading existing tokenizer from {tokenizer_path}")
        from bmfm_targets.tokenization import MultiFieldTokenizer
        tokenizer = MultiFieldTokenizer.from_pretrained(str(tokenizer_path))
    else:
        logger.info(f"Creating new tokenizer from {cfg.data_path}")
        cpg_sites = extract_cpg_sites_from_h5ad(cfg.data_path)
        tokenizer = create_methylation_multifield_tokenizer(
            cpg_sites=cpg_sites,
            output_dir=str(tokenizer_path),
        )
        logger.info(f"Tokenizer saved to {tokenizer_path}")

    return tokenizer


def setup_wandb(cfg: DictConfig):
    """Setup WandB logging if enabled."""
    # Check if WandB is enabled (support both nested and flat config)
    wandb_enabled = False
    if hasattr(cfg, 'track_wandb') and cfg.track_wandb.get('enabled', False):
        wandb_enabled = True
    elif cfg.get('wandb_enabled', False):
        wandb_enabled = True

    if wandb_enabled:
        try:
            import wandb
            from pytorch_lightning.loggers import WandbLogger

            # Get WandB settings from nested or flat config
            if hasattr(cfg, 'track_wandb'):
                project = cfg.track_wandb.get('project', 'methylation-age')
                entity = cfg.track_wandb.get('entity', None)
                run_name = cfg.track_wandb.get('name', None)
            else:
                project = cfg.get('wandb_project', 'methylation-age')
                entity = cfg.get('wandb_entity', None)
                run_name = cfg.get('wandb_name', None)

            # Create WandB logger
            wandb_logger = WandbLogger(
                project=project,
                entity=entity,
                name=run_name,
                save_dir=cfg.output_directory,
                log_model=True,  # Log model checkpoints
            )

            # Log all hyperparameters
            wandb_logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True))

            logger.info(f"WandB logging enabled - Project: {project}")
            return wandb_logger
        except ImportError:
            logger.warning("WandB not installed, using TensorBoard")
        except Exception as e:
            logger.warning(f"WandB setup failed: {e}, using TensorBoard")

    from pytorch_lightning.loggers import TensorBoardLogger
    logger.info("Using TensorBoard logger")
    return TensorBoardLogger(cfg.output_directory, name="finetune")


@hydra.main(
    config_path="configs",
    config_name="finetune_config",
    version_base="1.2"
)
def main(cfg: DictConfig):
    """Main fine-tuning function."""
    # Print config
    logger.info("=" * 70)
    logger.info("METHYLATION AGE FINE-TUNING")
    logger.info("=" * 70)
    logger.info(f"\nConfiguration:\n{OmegaConf.to_yaml(cfg)}")

    # Set seed
    if hasattr(cfg, 'seed') and cfg.seed:
        pl.seed_everything(cfg.seed.seed_value, workers=True)

    # Setup output directory
    output_dir = Path(cfg.output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup tokenizer
    tokenizer = setup_tokenizer(cfg)

    # Instantiate fields from config and convert to actual FieldInfo dataclass instances
    from bmfm_targets.config import FieldInfo
    fields = []
    for field_cfg in cfg.fields:
        # Convert OmegaConf to dict, remove _target_, and create FieldInfo
        field_dict = OmegaConf.to_container(field_cfg)
        field_dict.pop('_target_', None)
        fields.append(FieldInfo(**field_dict))

    # Setup data module
    data_module = MethylationDataModule(
        tokenizer=tokenizer,
        fields=fields,
        h5ad_path=cfg.data_path,
        train_split="train",
        val_split="valid",
        test_split="test",
        batch_size=cfg.data_module.batch_size,
        num_workers=cfg.data_module.num_workers,
        max_length=cfg.data_module.max_length,
        mlm=False,  # Disable MLM for fine-tuning
        collation_strategy="sequence_classification",
        # Fixed subset settings - MUST match pretraining!
        subset_k=cfg.data_module.get('subset_k', 2048),
        fixed_subset=cfg.data_module.get('fixed_subset', True),
        fixed_subset_seed=cfg.data_module.get('fixed_subset_seed', 42),
    )
    data_module.setup()

    # Setup model config
    # Hydra returns a partial when _partial_: true, so we need to call it with fields
    model_config_partial = hydra.utils.instantiate(cfg.model)
    model_config = model_config_partial(fields=fields)

    # Load pretrained encoder or create new one
    from bmfm_targets.models.predictive.scbert.modeling_scbert import SCBertModel

    model_config.checkpoint = None  # prevent SCBertModel from self-loading

    if cfg.checkpoint_path and cfg.checkpoint_path != "null":
        logger.info(f"Loading pretrained checkpoint: {cfg.checkpoint_path}")

        # Lightning checkpoints store weights under 'state_dict'.
        # Keys are like 'model.scbert.embeddings...', 'model.scbert.encoder...'
        # Strip the 'model.scbert.' prefix to get bare SCBertModel state dict.
        ckpt = torch.load(cfg.checkpoint_path, weights_only=False)
        prefix = "model.scbert."
        encoder_state = {
            k[len(prefix):]: v
            for k, v in ckpt["state_dict"].items()
            if k.startswith(prefix)
        }

        encoder = SCBertModel(model_config)
        missing, unexpected = encoder.load_state_dict(encoder_state, strict=True)
        logger.info(
            f"Loaded {len(encoder_state)} tensors. "
            f"Missing: {len(missing)}, Unexpected: {len(unexpected)}"
        )
        if missing:
            logger.warning(f"Missing keys: {missing[:5]}")
    else:
        logger.info("Training from scratch (no pretraining)")
        encoder = SCBertModel(model_config)

    # Create regression model
    freeze_encoder = cfg.get('freeze_encoder', True)
    unfreeze_encoder_epoch = cfg.get('unfreeze_encoder_epoch', 5)
    encoder_lr_multiplier = cfg.get('encoder_lr_multiplier', 0.1)
    use_huber_loss = cfg.get('use_huber_loss', False)
    huber_delta = cfg.get('huber_delta', 2.0)
    pearson_weight = cfg.get('pearson_weight', 0.5)

    effective_batch = cfg.data_module.batch_size * cfg.accumulate_grad_batches
    steps_per_epoch = len(data_module.train_dataset) // effective_batch
    total_steps = cfg.finetune_epochs * steps_per_epoch

    logger.info(f"Dataset size: {len(data_module.train_dataset)} train samples")
    logger.info(f"Effective batch size: {effective_batch}")
    logger.info(f"Steps per epoch: {steps_per_epoch}")
    logger.info(f"Total steps: {total_steps}")
    logger.info(f"Age stats: mean={data_module.age_mean:.2f}, std={data_module.age_std:.2f}")
    logger.info(f"Freeze encoder: {freeze_encoder}, unfreeze at epoch {unfreeze_encoder_epoch}")
    logger.info(f"Encoder LR multiplier: {encoder_lr_multiplier} → encoder_lr={cfg.trainer.learning_rate * encoder_lr_multiplier:.2e}")
    logger.info(f"Loss: {'Huber(delta=' + str(huber_delta) + ')' if use_huber_loss else 'MSE'}")

    num_cpg_sites = cfg.data_module.max_length - 2
    subset_k = cfg.data_module.get('subset_k', 2048)
    fixed_subset = cfg.data_module.get('fixed_subset', True)
    logger.info(f"Num CpG sites: {num_cpg_sites}")
    logger.info(f"Subset settings: k={subset_k}, fixed={fixed_subset}")
    logger.info(f"Pipeline: [CpG IDs + beta values] -> Encoder ({model_config.hidden_size}d) -> CLS -> LayerNorm -> MLP head -> age")
    logger.info(f"Loss: MSE + {pearson_weight} * (1 - PCC)")

    model = MethylationAgeRegressor(
        encoder=encoder,
        num_cpg_sites=num_cpg_sites,
        hidden_size=model_config.hidden_size,
        head_hidden_size=cfg.regression_head.hidden_size,
        head_dropout=cfg.regression_head.dropout,
        learning_rate=cfg.trainer.learning_rate,
        weight_decay=cfg.trainer.weight_decay,
        warmup_steps=cfg.trainer.warmup_steps,
        max_steps=total_steps,
        age_mean=data_module.age_mean,
        age_std=data_module.age_std,
        freeze_encoder=freeze_encoder,
        unfreeze_encoder_epoch=unfreeze_encoder_epoch,
        encoder_lr_multiplier=encoder_lr_multiplier,
        use_huber_loss=use_huber_loss,
        huber_delta=huber_delta,
        pearson_weight=pearson_weight,
    )

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup trainer
    wandb_logger = setup_wandb(cfg)

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=output_dir / "finetune_age" / "checkpoints",
            filename="epoch={epoch}-val_mae={val/mae:.4f}",
            monitor="val/mae",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        pl.callbacks.EarlyStopping(
            monitor="val/mae",
            patience=cfg.early_stopping.patience,
            mode="min",
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
    ]

    trainer = pl.Trainer(
        max_epochs=cfg.finetune_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed",
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        gradient_clip_val=1.0,
        logger=wandb_logger,
        callbacks=callbacks,
        default_root_dir=str(output_dir / "finetune_age"),
        log_every_n_steps=10,
    )

    # Train
    logger.info("Starting fine-tuning...")
    trainer.fit(model, data_module)

    # Test
    logger.info("Running test evaluation...")
    trainer.test(model, data_module)

    # Save best checkpoint path
    best_ckpt = trainer.checkpoint_callback.best_model_path
    logger.info(f"\nFine-tuning complete!")
    logger.info(f"Best checkpoint: {best_ckpt}")
    logger.info(f"Best val/mae: {trainer.checkpoint_callback.best_model_score:.4f}")

    return best_ckpt


if __name__ == "__main__":
    main()
