"""
PyTorch Lightning module for training methylation age prediction

This module wraps the MethylationAgeModel (which uses the original BMFM SCBertModel)
with PyTorch Lightning for easy training with metrics, logging, and checkpointing.
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union

from ..shared.config import create_methylation_config, SCBertConfig
from .model import MethylationAgeModel

# Try to import PyTorch Lightning
try:
    import pytorch_lightning as pl
    HAS_LIGHTNING = True
except ImportError:
    HAS_LIGHTNING = False
    pl = None

# Try to import torchmetrics
try:
    from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score, PearsonCorrCoef
    HAS_TORCHMETRICS = True
except ImportError:
    HAS_TORCHMETRICS = False


if HAS_LIGHTNING:

    class MethylationAgeLightningModule(pl.LightningModule):
        """
        PyTorch Lightning module for training methylation age prediction.

        Features:
            - Configurable optimizer and scheduler
            - Multiple metrics (MAE, MSE, R², Pearson)
            - Automatic logging to TensorBoard/WandB
            - Built-in validation and test steps

        Example:
            >>> config = create_methylation_config(num_cpg_sites=8000)
            >>> module = MethylationAgeLightningModule(config, learning_rate=1e-4)
            >>> trainer = pl.Trainer(max_epochs=100)
            >>> trainer.fit(module, train_loader, val_loader)
        """

        def __init__(
            self,
            config: SCBertConfig,
            learning_rate: float = 1e-4,
            weight_decay: float = 0.01,
            warmup_steps: int = 1000,
            max_steps: int = 100000,
            scheduler_type: str = "cosine",  # "cosine", "linear", "constant"
            head_hidden_size: int = 256,
            head_dropout: float = 0.1,
            normalize_targets: bool = True,
            age_mean: Optional[float] = None,
            age_std: Optional[float] = None
        ):
            """
            Args:
                config: SCBertConfig configured for methylation (use create_methylation_config)
                learning_rate: Peak learning rate
                weight_decay: L2 regularization
                warmup_steps: Number of warmup steps
                max_steps: Total training steps for scheduler
                scheduler_type: Type of LR scheduler
                head_hidden_size: Hidden size for prediction head
                head_dropout: Dropout in prediction head
                normalize_targets: Whether targets are normalized
                age_mean: Mean age for denormalization
                age_std: Std age for denormalization
            """
            super().__init__()
            self.save_hyperparameters(ignore=['config'])
            self.config = config

            # Model
            self.model = MethylationAgeModel(
                config,
                head_hidden_size=head_hidden_size,
                head_dropout=head_dropout
            )

            # Loss
            self.loss_fn = nn.MSELoss()

            # Hyperparameters
            self.learning_rate = learning_rate
            self.weight_decay = weight_decay
            self.warmup_steps = warmup_steps
            self.max_steps = max_steps
            self.scheduler_type = scheduler_type

            # Target normalization
            self.normalize_targets = normalize_targets
            self.age_mean = age_mean
            self.age_std = age_std

            # Metrics
            self._setup_metrics()

        def _setup_metrics(self):
            """Initialize metrics for train/val/test."""
            if HAS_TORCHMETRICS:
                # Training metrics
                self.train_mae = MeanAbsoluteError()
                self.train_mse = MeanSquaredError()
                self.train_rmse = MeanSquaredError(squared=False)

                # Validation metrics
                self.val_mae = MeanAbsoluteError()
                self.val_mse = MeanSquaredError()
                self.val_rmse = MeanSquaredError(squared=False)
                self.val_r2 = R2Score()
                self.val_pearson = PearsonCorrCoef()

                # Test metrics
                self.test_mae = MeanAbsoluteError()
                self.test_mse = MeanSquaredError()
                self.test_rmse = MeanSquaredError(squared=False)
                self.test_r2 = R2Score()
                self.test_pearson = PearsonCorrCoef()

        def forward(self, cpg_ids, beta_values, attention_mask=None):
            """Forward pass."""
            return self.model(cpg_ids, beta_values, attention_mask)

        def _shared_step(self, batch, stage: str):
            """Shared logic for train/val/test steps."""
            cpg_ids = batch["cpg_ids"]
            beta_values = batch["beta_values"]
            attention_mask = batch.get("attention_mask")

            # Use normalized age if available, otherwise raw age
            if self.normalize_targets and "age_normalized" in batch:
                targets = batch["age_normalized"]
            else:
                targets = batch["age"]

            # Forward
            predictions = self(cpg_ids, beta_values, attention_mask)
            predictions = predictions.squeeze(-1)

            # Loss
            loss = self.loss_fn(predictions, targets)

            # Denormalize for metrics (if applicable)
            if self.normalize_targets and self.age_mean is not None:
                preds_denorm = predictions * self.age_std + self.age_mean
                targets_denorm = targets * self.age_std + self.age_mean
            else:
                preds_denorm = predictions
                targets_denorm = targets

            return loss, preds_denorm, targets_denorm

        def training_step(self, batch, batch_idx):
            """Training step."""
            loss, preds, targets = self._shared_step(batch, "train")

            # Log loss
            self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

            # Log metrics
            if HAS_TORCHMETRICS:
                self.train_mae(preds, targets)
                self.train_mse(preds, targets)
                self.train_rmse(preds, targets)
                self.log("train/mae", self.train_mae, on_step=False, on_epoch=True)
                self.log("train/mse", self.train_mse, on_step=False, on_epoch=True)
                self.log("train/rmse", self.train_rmse, on_step=False, on_epoch=True)

            return loss

        def validation_step(self, batch, batch_idx):
            """Validation step."""
            loss, preds, targets = self._shared_step(batch, "val")

            # Log loss
            self.log("val/loss", loss, on_epoch=True, prog_bar=True)

            # Log metrics
            if HAS_TORCHMETRICS:
                self.val_mae(preds, targets)
                self.val_mse(preds, targets)
                self.val_rmse(preds, targets)
                self.val_r2(preds, targets)
                self.val_pearson(preds, targets)

                self.log("val/mae", self.val_mae, on_epoch=True, prog_bar=True)
                self.log("val/mse", self.val_mse, on_epoch=True)
                self.log("val/rmse", self.val_rmse, on_epoch=True)
                self.log("val/r2", self.val_r2, on_epoch=True)
                self.log("val/pearson", self.val_pearson, on_epoch=True)

            return loss

        def test_step(self, batch, batch_idx):
            """Test step."""
            loss, preds, targets = self._shared_step(batch, "test")

            # Log metrics
            if HAS_TORCHMETRICS:
                self.test_mae(preds, targets)
                self.test_mse(preds, targets)
                self.test_rmse(preds, targets)
                self.test_r2(preds, targets)
                self.test_pearson(preds, targets)

                self.log("test/mae", self.test_mae, on_epoch=True)
                self.log("test/mse", self.test_mse, on_epoch=True)
                self.log("test/rmse", self.test_rmse, on_epoch=True)
                self.log("test/r2", self.test_r2, on_epoch=True)
                self.log("test/pearson", self.test_pearson, on_epoch=True)

            return loss

        def configure_optimizers(self):
            """Configure optimizer and learning rate scheduler."""
            # Separate weight decay for different parameter groups
            no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]

            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters()
                               if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters()
                               if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]

            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8
            )

            # Learning rate scheduler
            if self.scheduler_type == "constant":
                return optimizer

            def lr_lambda(current_step):
                if current_step < self.warmup_steps:
                    # Linear warmup
                    return float(current_step) / float(max(1, self.warmup_steps))
                elif self.scheduler_type == "linear":
                    # Linear decay
                    progress = float(current_step - self.warmup_steps) / \
                               float(max(1, self.max_steps - self.warmup_steps))
                    return max(0.0, 1.0 - progress)
                else:  # cosine
                    # Cosine decay
                    progress = float(current_step - self.warmup_steps) / \
                               float(max(1, self.max_steps - self.warmup_steps))
                    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1
                }
            }

else:
    # Placeholder when Lightning is not available
    MethylationAgeLightningModule = None
