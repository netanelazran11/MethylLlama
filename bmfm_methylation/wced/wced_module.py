"""
Multi-task WCED Training Module - Whole Cell Expression Decoder for Methylation

Key insight: Standard WCED fails because CLS doesn't encode sample-specific info.
The decoder just learns per-CpG averages (PCC stuck at ~0.94, CLS collapses).

Solution: Add age prediction as auxiliary task to directly supervise CLS.
- CLS → Age prediction forces CLS to encode age-relevant information
- This prevents CLS collapse and enables sample-specific predictions

Architecture:
    Input:  Subset of CpGs
    Encoder: Transformer → CLS
    Decoder: Linear(CLS) → ALL CpG betas (reconstruction)
    Age Head: Linear(CLS) → age (auxiliary supervision)
    Loss:   Reconstruction + λ_age * MSE(age_pred, age_true)
"""

import logging
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from scipy.stats import pearsonr
import numpy as np

from bmfm_targets.config import SCBertConfig
from bmfm_targets.models.predictive.scbert.modeling_scbert import SCBertModel

from ..shared.config import PretrainingConfig

logger = logging.getLogger(__name__)


class WCEDDecoder(nn.Module):
    """
    WCED Decoder: Linear layer from CLS to entire vocabulary.
    """

    def __init__(
        self,
        hidden_size: int = 512,
        vocab_size: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, vocab_size),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.decoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, cls_hidden: torch.Tensor) -> torch.Tensor:
        return self.decoder(cls_hidden)


class ProjectionHead(nn.Module):
    """
    Projection head for contrastive learning (SimCLR style).
    Maps CLS embedding to a lower-dimensional space where contrastive loss is computed.
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(x), dim=-1)


class WCEDTrainingModule(pl.LightningModule):
    """
    Contrastive WCED Training Module.

    Architecture:
        Input:  Two views of each sample (different random CpG subsets)
        Encoder: Transformer → CLS embeddings
        Projection: MLP → normalized embeddings for contrastive loss
        Decoder: Linear(CLS) → ALL vocab_size beta predictions
        Loss:   Reconstruction (MSE) + λ * Contrastive (InfoNCE)

    Key insight: Contrastive loss forces CLS to encode sample identity,
    which enables sample-specific predictions instead of per-CpG averages.
    """

    def __init__(
        self,
        model_config: SCBertConfig,
        pretrain_config: Optional[PretrainingConfig] = None,
        learning_rate: float = 5e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        lr_decay_steps: int = 10000,
        vocab_size: int = 2048,
        contrastive_weight: float = 0.0,  # λ for contrastive loss (disabled by default)
        contrastive_temp: float = 0.1,    # Temperature for InfoNCE
        normalize_loss: bool = False,     # Per-sample normalize before loss
        age_weight: float = 1.0,          # λ for age prediction loss
        betas: tuple = (0.9, 0.999),
        epsilon: float = 1e-8,
        use_scale_adapt: bool = False,    # Replace MLP beta encoder with ScaleAdaptEncoder
        scale_adapt_n_sin_basis: int = 48,
        scale_adapt_basis_scale: float = 2.0,  # Higher than upstream 1.5 — beta [0,1] needs finer resolution
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model_config', 'pretrain_config'])

        self.model_config = model_config
        if pretrain_config is None:
            pretrain_config = PretrainingConfig(mode="wced")
        self.pretrain_config = pretrain_config
        self.vocab_size = vocab_size
        self.contrastive_weight = contrastive_weight
        self.contrastive_temp = contrastive_temp
        self.normalize_loss = normalize_loss
        self.age_weight = age_weight

        # Encoder
        self.encoder = SCBertModel(model_config, add_pooling_layer=True)

        # Apply ADD fusion stabilization (cpg_scale * cpg_embed + beta_embed)
        self._patch_embeddings_add_stabilized()

        # Optionally replace MLP beta encoder with ScaleAdaptEncoder
        # Must be applied AFTER _patch_embeddings_add_stabilized so the
        # add_forward patch picks up the new beta_values_embeddings automatically
        if use_scale_adapt:
            from bmfm_methylation.llama.scale_adapt import patch_scale_adapt_encoder
            ok = patch_scale_adapt_encoder(
                self.encoder.embeddings,
                hidden_size=model_config.hidden_size,
                n_sin_basis=scale_adapt_n_sin_basis,
                basis_scale=scale_adapt_basis_scale,
                trainable=True,
                zero_as_special_token=True,
            )
            if not ok:
                logger.warning("ScaleAdaptEncoder patch failed — using original MLP encoder")

        # WCED Decoder
        self.decoder = WCEDDecoder(
            hidden_size=model_config.hidden_size,
            vocab_size=vocab_size,
            dropout=pretrain_config.decoder_dropout,
        )

        # Projection head for contrastive learning
        self.projection_head = ProjectionHead(
            input_dim=model_config.hidden_size,
            hidden_dim=model_config.hidden_size // 2,
            output_dim=128,
        )

        # Age prediction head - directly supervises CLS to encode sample info
        self.age_head = nn.Sequential(
            nn.Linear(model_config.hidden_size, model_config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(model_config.hidden_size // 2, 1),
        )

        # Loss functions
        self.recon_loss_fn = nn.MSELoss(reduction='none')
        self.age_loss_fn = nn.MSELoss()

        # Log model info
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        proj_params = sum(p.numel() for p in self.projection_head.parameters())
        age_params = sum(p.numel() for p in self.age_head.parameters())

        logger.info(f"Multi-task WCED Training Module initialized:")
        logger.info(f"  Encoder params: {encoder_params:,}")
        logger.info(f"  Decoder params: {decoder_params:,}")
        logger.info(f"  Projection params: {proj_params:,}")
        logger.info(f"  Age head params: {age_params:,}")
        logger.info(f"  Total params: {encoder_params + decoder_params + proj_params + age_params:,}")
        logger.info(f"  Vocab size: {vocab_size}")
        logger.info(f"  Age weight: {age_weight}")
        logger.info(f"  Contrastive weight: {contrastive_weight}")
        logger.info(f"  Normalize loss: {normalize_loss}")

    def _patch_embeddings_add_stabilized(self, initial_cpg_scale: float = 0.1):
        """Patch embeddings to use ADD fusion with learnable CpG scaling."""
        embeddings_layer = self.encoder.embeddings
        embeddings_layer.cpg_scale = nn.Parameter(torch.tensor(float(initial_cpg_scale)))

        def add_forward(input_ids, position_ids=None, inputs_embeds=None):
            if inputs_embeds is not None:
                return inputs_embeds

            batch_size, num_fields, seq_length = input_ids.shape

            cpg_ids = input_ids[:, 0, :].long()
            cpg_embeds = embeddings_layer.cpg_sites_embeddings(cpg_ids)

            beta_values = input_ids[:, 1, :].float()
            beta_values_clean = beta_values.clone()
            beta_values_clean[beta_values_clean < 0] = 0.0
            beta_embeds = embeddings_layer.beta_values_embeddings(beta_values_clean)

            hidden_states = embeddings_layer.cpg_scale * cpg_embeds + beta_embeds

            if embeddings_layer.position_embedding_type is not None:
                if position_ids is None:
                    position_ids = embeddings_layer.position_ids[:, :seq_length]
                position_embeddings = embeddings_layer.position_embeddings(position_ids)
                hidden_states = hidden_states + position_embeddings

            hidden_states = embeddings_layer.LayerNorm(hidden_states)
            hidden_states = embeddings_layer.dropout(hidden_states)
            return hidden_states

        embeddings_layer.forward = add_forward

    def encode(
        self,
        cpg_ids: torch.Tensor,
        beta_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Encode a view and return CLS embedding, predictions, and age."""
        input_ids = torch.stack([cpg_ids.float(), beta_values], dim=1)

        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        cls_embedding = encoder_output.pooler_output
        projection = self.projection_head(cls_embedding)
        predicted_betas = self.decoder(cls_embedding)
        predicted_age = self.age_head(cls_embedding).squeeze(-1)  # [batch]

        return {
            "cls_embedding": cls_embedding,
            "projection": projection,
            "predicted_betas": predicted_betas,
            "predicted_age": predicted_age,
        }

    def forward(
        self,
        cpg_ids: torch.Tensor,
        beta_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for single view (used during inference)."""
        return self.encode(cpg_ids, beta_values, attention_mask)

    def _normalize_per_sample(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Normalize each sample to zero-mean, unit-variance.

        This removes the "predict per-CpG averages" shortcut, forcing the model
        to predict the relative pattern within each sample.
        """
        if mask is not None:
            # Normalize only on valid (non-input) positions
            # x: [batch, vocab_size], mask: [batch, vocab_size] (True = use)
            mask_float = mask.float()
            n = mask_float.sum(dim=1, keepdim=True).clamp(min=1)
            mean = (x * mask_float).sum(dim=1, keepdim=True) / n
            var = ((x - mean) ** 2 * mask_float).sum(dim=1, keepdim=True) / n
            std = torch.sqrt(var + 1e-8)
            normalized = (x - mean) / std
            return normalized * mask_float  # Zero out non-masked positions
        else:
            mean = x.mean(dim=1, keepdim=True)
            std = x.std(dim=1, keepdim=True) + 1e-8
            return (x - mean) / std

    def info_nce_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Compute InfoNCE contrastive loss.

        z1, z2: [batch, dim] - normalized projections from two views of same samples

        Positive pairs: (z1[i], z2[i]) - same sample
        Negative pairs: (z1[i], z2[j]) for j != i - different samples
        """
        batch_size = z1.shape[0]

        # Similarity matrix: [batch, batch]
        # sim[i,j] = cosine_similarity(z1[i], z2[j])
        sim = torch.matmul(z1, z2.T) / self.contrastive_temp

        # Labels: diagonal elements are positive pairs
        labels = torch.arange(batch_size, device=z1.device)

        # Cross-entropy loss (InfoNCE)
        # For each z1[i], we want it to be most similar to z2[i]
        loss_12 = F.cross_entropy(sim, labels)
        loss_21 = F.cross_entropy(sim.T, labels)

        return (loss_12 + loss_21) / 2

    def _shared_step(self, batch: Dict[str, torch.Tensor], stage: str) -> Dict[str, torch.Tensor]:
        """
        Shared step supporting two batch formats:

        BMFM-style (bmfm_style=True, BMFMWCEDCollator):
          batch["labels"]  — [B, vocab_size], -100 at input/unmeasured positions,
                             real β at held-out measured positions.
          No valid_mask or input_mask needed in the forward pass.

        Original format (WCEDCollator):
          batch["all_betas"]   — [B, vocab_size], NaN→0 at unmeasured positions
          batch["input_mask"]  — [B, vocab_size], True = in encoder input
          batch["valid_mask"]  — [B, vocab_size], True = non-NaN  (optional but used
                                 here to fix a latent bug where NaN→0 positions were
                                 incorrectly included in the reconstruction loss)
        """
        cpg_ids_v1 = batch["cpg_ids"]
        beta_values_v1 = batch["beta_values"]
        attention_mask_v1 = batch.get("attention_mask")
        age_labels = batch.get("age")

        # ── Resolve reconstruction mask and targets (format-agnostic) ──────────
        if "labels" in batch:
            # BMFM-style: -100 encodes all three categories in one tensor.
            labels_v1 = batch["labels"]
            recon_mask_v1 = (labels_v1 != -100.0)          # True = held-out measured
            target_v1 = labels_v1.clamp(min=0.0)            # -100 → 0 (masked out anyway)
        else:
            # Original format: reconstruct separately computed masks.
            all_betas = batch["all_betas"]
            input_mask_v1 = batch["input_mask"]
            valid_mask = batch.get("valid_mask")            # True = non-NaN
            non_input_v1 = ~input_mask_v1
            # Apply valid_mask so NaN→0 positions (valid_mask=False) are excluded.
            recon_mask_v1 = non_input_v1 & valid_mask if valid_mask is not None else non_input_v1
            target_v1 = all_betas

        # ── Encode view 1 ───────────────────────────────────────────────────────
        out_v1 = self.encode(cpg_ids_v1, beta_values_v1, attention_mask_v1)
        pred_v1 = out_v1["predicted_betas"]
        z1 = out_v1["projection"]
        age_pred_v1 = out_v1["predicted_age"]

        # ── Reconstruction loss — view 1 ────────────────────────────────────────
        if self.normalize_loss:
            pred_norm_v1 = self._normalize_per_sample(pred_v1, recon_mask_v1)
            target_norm_v1 = self._normalize_per_sample(target_v1, recon_mask_v1)
            loss_per_cpg_v1 = self.recon_loss_fn(pred_norm_v1, target_norm_v1)
        else:
            loss_per_cpg_v1 = self.recon_loss_fn(pred_v1, target_v1)

        recon_loss_v1 = (
            (loss_per_cpg_v1 * recon_mask_v1.float()).sum()
            / recon_mask_v1.float().sum().clamp(min=1)
        )

        # ── Age prediction loss ─────────────────────────────────────────────────
        if age_labels is not None and self.age_weight > 0:
            valid_age_mask = ~torch.isnan(age_labels)
            if valid_age_mask.any():
                age_loss = self.age_loss_fn(
                    age_pred_v1[valid_age_mask],
                    age_labels[valid_age_mask].float(),
                )
            else:
                age_loss = torch.tensor(0.0, device=recon_loss_v1.device)
        else:
            age_loss = torch.tensor(0.0, device=recon_loss_v1.device)

        # ── Contrastive mode (view 2) ───────────────────────────────────────────
        if "cpg_ids_v2" in batch and self.contrastive_weight > 0:
            cpg_ids_v2 = batch["cpg_ids_v2"]
            beta_values_v2 = batch["beta_values_v2"]
            attention_mask_v2 = batch.get("attention_mask_v2")

            if "labels_v2" in batch:
                labels_v2 = batch["labels_v2"]
                recon_mask_v2 = (labels_v2 != -100.0)
                target_v2 = labels_v2.clamp(min=0.0)
            else:
                input_mask_v2 = batch["input_mask_v2"]
                valid_mask = batch.get("valid_mask")
                non_input_v2 = ~input_mask_v2
                recon_mask_v2 = non_input_v2 & valid_mask if valid_mask is not None else non_input_v2
                target_v2 = target_v1  # same all_betas target for both views

            out_v2 = self.encode(cpg_ids_v2, beta_values_v2, attention_mask_v2)
            pred_v2 = out_v2["predicted_betas"]
            z2 = out_v2["projection"]

            if self.normalize_loss:
                pred_norm_v2 = self._normalize_per_sample(pred_v2, recon_mask_v2)
                target_norm_v2 = self._normalize_per_sample(target_v2, recon_mask_v2)
                loss_per_cpg_v2 = self.recon_loss_fn(pred_norm_v2, target_norm_v2)
            else:
                loss_per_cpg_v2 = self.recon_loss_fn(pred_v2, target_v2)

            recon_loss_v2 = (
                (loss_per_cpg_v2 * recon_mask_v2.float()).sum()
                / recon_mask_v2.float().sum().clamp(min=1)
            )

            recon_loss = (recon_loss_v1 + recon_loss_v2) / 2

            # InfoNCE on normalized CLS embeddings (not projection head)
            cls1_norm = F.normalize(out_v1["cls_embedding"], dim=-1)
            cls2_norm = F.normalize(out_v2["cls_embedding"], dim=-1)
            contrastive_loss = self.info_nce_loss(cls1_norm, cls2_norm)

            loss = recon_loss + self.contrastive_weight * contrastive_loss + self.age_weight * age_loss
            predicted_betas = pred_v1
            recon_mask = recon_mask_v1
        else:
            recon_loss = recon_loss_v1
            contrastive_loss = torch.tensor(0.0, device=recon_loss_v1.device)
            loss = recon_loss + self.age_weight * age_loss
            predicted_betas = pred_v1
            recon_mask = recon_mask_v1

        # ── Metrics ─────────────────────────────────────────────────────────────
        with torch.no_grad():
            recon_pred = predicted_betas[recon_mask]
            recon_target = target_v1[recon_mask]
            mae = torch.abs(recon_pred - recon_target).mean()
            mse = ((recon_pred - recon_target) ** 2).mean()

            # all_mae: MAE over ALL vocab positions (only meaningful for old format
            # where target_v1 = all_betas; in BMFM-style, fall back to recon MAE)
            if "labels" not in batch:
                all_mae = torch.abs(predicted_betas - target_v1).mean()
                all_mse = ((predicted_betas - target_v1) ** 2).mean()
            else:
                all_mae = mae
                all_mse = mse

            if age_labels is not None:
                age_mae = torch.abs(age_pred_v1 - age_labels).mean()
            else:
                age_mae = torch.tensor(0.0, device=mae.device)

        return {
            "loss": loss,
            "recon_loss": recon_loss,
            "contrastive_loss": contrastive_loss,
            "age_loss": age_loss,
            "age_mae": age_mae,
            "mae": mae,
            "mse": mse,
            "all_mae": all_mae,
            "all_mse": all_mse,
            "predicted_betas": predicted_betas,
            "target_betas": target_v1,
            "recon_mask": recon_mask,
            "z1": z1,
            "cls_embedding": out_v1["cls_embedding"],
            "predicted_age": age_pred_v1,
        }

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self._shared_step(batch, "train")

        self.log("train/loss", outputs["loss"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/recon_loss", outputs["recon_loss"], on_step=False, on_epoch=True)
        self.log("train/age_loss", outputs["age_loss"], on_step=False, on_epoch=True)
        self.log("train/age_mae", outputs["age_mae"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/contrastive_loss", outputs["contrastive_loss"], on_step=False, on_epoch=True)
        self.log("train/mae", outputs["mae"], on_step=False, on_epoch=True)
        self.log("train/mse", outputs["mse"], on_step=False, on_epoch=True)
        self.log("train/all_mae", outputs["all_mae"], on_step=False, on_epoch=True)

        return outputs["loss"]

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self._shared_step(batch, "val")

        self.log("validation/loss", outputs["loss"], on_epoch=True, prog_bar=True)
        self.log("validation/recon_loss", outputs["recon_loss"], on_epoch=True)
        self.log("validation/age_loss", outputs["age_loss"], on_epoch=True)
        self.log("validation/age_mae", outputs["age_mae"], on_epoch=True, prog_bar=True)
        self.log("validation/contrastive_loss", outputs["contrastive_loss"], on_epoch=True)
        self.log("validation/mae", outputs["mae"], on_epoch=True)
        self.log("validation/mse", outputs["mse"], on_epoch=True)
        self.log("validation/all_mae", outputs["all_mae"], on_epoch=True)

        # CLS diagnostic metrics (log on first batch of each epoch)
        if batch_idx == 0:
            with torch.no_grad():
                # CLS embedding variance across batch
                cls_emb = outputs.get("cls_embedding", outputs.get("z1", None))
                if cls_emb is not None:
                    # Variance across samples
                    cls_var = cls_emb.var(dim=0).mean()
                    self.log("validation/cls_variance", cls_var, on_epoch=True)

                    # Mean pairwise cosine similarity
                    cls_norm = F.normalize(cls_emb, dim=-1)
                    sim_matrix = torch.matmul(cls_norm, cls_norm.T)
                    # Upper triangle (excluding diagonal)
                    mask = torch.triu(torch.ones_like(sim_matrix), diagonal=1).bool()
                    mean_sim = sim_matrix[mask].mean()
                    self.log("validation/cls_similarity", mean_sim, on_epoch=True)

                # Prediction variance analysis
                pred = outputs["predicted_betas"]
                target = outputs["target_betas"]

                # Variance of predictions across samples (should match target variance)
                pred_var = pred.var(dim=0).mean()
                target_var = target.var(dim=0).mean()
                var_ratio = pred_var / (target_var + 1e-8)
                self.log("validation/pred_var_ratio", var_ratio, on_epoch=True)

        return outputs["loss"]

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self._shared_step(batch, "test")

        self.log("test/loss", outputs["loss"], on_epoch=True)
        self.log("test/recon_loss", outputs["recon_loss"], on_epoch=True)
        self.log("test/age_loss", outputs["age_loss"], on_epoch=True)
        self.log("test/age_mae", outputs["age_mae"], on_epoch=True)
        self.log("test/contrastive_loss", outputs["contrastive_loss"], on_epoch=True)
        self.log("test/mae", outputs["mae"], on_epoch=True)
        self.log("test/mse", outputs["mse"], on_epoch=True)
        self.log("test/all_mae", outputs["all_mae"], on_epoch=True)

        # PCC over held-out measured positions (the canonical reconstruction metric)
        recon_mask = outputs["recon_mask"]
        pred = outputs["predicted_betas"][recon_mask].detach().cpu().numpy()
        target = outputs["target_betas"][recon_mask].detach().cpu().numpy()
        if len(pred) > 1:
            pcc, _ = pearsonr(pred, target)
            self.log("test/pcc", pcc, on_epoch=True)
            self.log("test/all_pcc", pcc, on_epoch=True)  # same metric; kept for backward compat

        return outputs["loss"]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=self.hparams.betas,
            eps=self.hparams.epsilon,
        )

        lr_decay_steps = self.hparams.lr_decay_steps
        if lr_decay_steps <= 0:
            if self.trainer is not None and self.trainer.estimated_stepping_batches is not None:
                lr_decay_steps = int(self.trainer.estimated_stepping_batches)
            else:
                lr_decay_steps = 300 * 45

        warmup_steps = self.hparams.warmup_steps

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, lr_decay_steps - warmup_steps))
            progress = min(progress, 1.0)
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def get_encoder(self) -> SCBertModel:
        """Get the pretrained encoder for downstream tasks."""
        return self.encoder
