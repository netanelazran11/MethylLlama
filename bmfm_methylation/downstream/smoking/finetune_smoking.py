#!/usr/bin/env python3
"""
Task A — Smoking Status Classification (current / former / never).

Architecture:
    Random 4k CpGs → WCED Encoder → CLS → Linear(512, 3) → CrossEntropy

Novelty vs EpiSmokEr:
  - EpiSmokEr uses 121 hand-selected CpGs + logistic regression
  - We use all 4000 randomly sampled CpGs + pretrained transformer encoder
  - Shows foundation model representations transfer to classification tasks

Run via scripts/downstream/finetune_smoking.sh
"""

import torch
import torch.serialization

_orig = torch.load
def _patched(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig(*args, **kwargs)
torch.load = _patched
torch.serialization.load = _patched

import logging
import sys
from pathlib import Path
from typing import Optional

import hydra
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

logger = logging.getLogger(__name__)

LABEL_MAP = {"current": 0, "former": 1, "never": 2}
CLASS_NAMES = ["current", "former", "never"]


# ─────────────────────────────────────────────────────────────────────────────
# Lightning Module
# ─────────────────────────────────────────────────────────────────────────────

class SmokingClassifier(pl.LightningModule):
    """
    3-class smoking status classifier on top of a frozen/fine-tuned WCED encoder.

    Head: Linear(hidden_size → n_classes), trained with CrossEntropy.
    Metrics: accuracy, macro-F1, per-class AUC.
    """

    def __init__(
        self,
        encoder,
        hidden_size: int = 512,
        n_classes: int = 3,
        head_dropout: float = 0.1,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 200,
        max_steps: int = 5000,
        freeze_encoder: bool = True,
        unfreeze_encoder_epoch: int = 5,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder", "class_weights"])

        self.encoder = encoder
        self.n_classes = n_classes

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            logger.info(f"Encoder frozen (unfreeze at epoch {unfreeze_encoder_epoch})")

        self.head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(hidden_size, n_classes),
        )

        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        # Buffers for epoch-level metrics
        self._val_logits, self._val_labels = [], []
        self._test_logits, self._test_labels = [], []

    # ── encoding ──────────────────────────────────────────────────────────────

    def _encode(self, batch):
        cpg_ids = batch["cpg_ids"]
        beta_values = batch["beta_values"]
        attention_mask = batch.get("attention_mask")

        input_ids = torch.stack([cpg_ids.float(), beta_values], dim=1)
        bs, _, seq = input_ids.shape
        if attention_mask is not None and attention_mask.dim() == 3:
            attention_mask = attention_mask[:, 0, :]
        if attention_mask is None:
            attention_mask = torch.ones(bs, seq, device=input_ids.device)

        out = self.encoder(input_ids, attention_mask=attention_mask)
        return out.pooler_output  # [batch, hidden_size]

    def forward(self, batch):
        cls = self._encode(batch)
        return self.head(cls)  # [batch, n_classes]

    # ── lifecycle hooks ────────────────────────────────────────────────────────

    def on_train_epoch_start(self):
        if (self.hparams.freeze_encoder and
                self.current_epoch == self.hparams.unfreeze_encoder_epoch):
            logger.info(f"Unfreezing encoder at epoch {self.current_epoch}")
            for p in self.encoder.parameters():
                p.requires_grad = True

    # ── steps ─────────────────────────────────────────────────────────────────

    def training_step(self, batch, _):
        logits = self(batch)
        loss = self.loss_fn(logits, batch["class_label"])
        acc = (logits.argmax(1) == batch["class_label"]).float().mean()
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        logits = self(batch)
        loss = self.loss_fn(logits, batch["class_label"])
        acc = (logits.argmax(1) == batch["class_label"]).float().mean()
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_epoch=True, prog_bar=True)
        self._val_logits.append(logits.detach().cpu())
        self._val_labels.append(batch["class_label"].detach().cpu())
        return loss

    def on_validation_epoch_end(self):
        if not self._val_logits:
            return
        self._log_epoch_metrics(
            torch.cat(self._val_logits), torch.cat(self._val_labels), "val"
        )
        self._val_logits.clear()
        self._val_labels.clear()

    def test_step(self, batch, _):
        logits = self(batch)
        self._test_logits.append(logits.detach().cpu())
        self._test_labels.append(batch["class_label"].detach().cpu())

    def on_test_epoch_end(self):
        if not self._test_logits:
            return
        self._log_epoch_metrics(
            torch.cat(self._test_logits), torch.cat(self._test_labels), "test"
        )
        self._test_logits.clear()
        self._test_labels.clear()

    def _log_epoch_metrics(self, logits, labels, stage):
        probs = F.softmax(logits, dim=-1).numpy()
        preds = logits.argmax(1).numpy()
        labels_np = labels.numpy()

        try:
            from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
            acc = accuracy_score(labels_np, preds)
            f1 = f1_score(labels_np, preds, average="macro", zero_division=0)
            self.log(f"{stage}/acc_epoch", acc, prog_bar=(stage == "val"))
            self.log(f"{stage}/f1_macro", f1, prog_bar=(stage == "val"))

            # Per-class AUC (one-vs-rest)
            if self.n_classes == 2:
                auc = roc_auc_score(labels_np, probs[:, 1])
                self.log(f"{stage}/auc", auc)
            else:
                for i, name in enumerate(CLASS_NAMES[:self.n_classes]):
                    bin_labels = (labels_np == i).astype(int)
                    if bin_labels.sum() > 0 and (1 - bin_labels).sum() > 0:
                        auc = roc_auc_score(bin_labels, probs[:, i])
                        self.log(f"{stage}/auc_{name}", auc)

            if stage == "test":
                logger.info(
                    f"TEST — acc={acc:.4f} | macro-F1={f1:.4f}"
                )
                for i, name in enumerate(CLASS_NAMES[:self.n_classes]):
                    count = int((labels_np == i).sum())
                    logger.info(f"  class={name}: n={count}")
        except ImportError:
            logger.warning("sklearn not available, skipping F1/AUC metrics")

    # ── optimizer ─────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
        encoder_lr = self.hparams.learning_rate * 0.01

        groups = [
            {"params": [p for n, p in self.head.named_parameters()
                        if not any(nd in n for nd in no_decay)],
             "lr": self.hparams.learning_rate, "weight_decay": self.hparams.weight_decay},
            {"params": [p for n, p in self.head.named_parameters()
                        if any(nd in n for nd in no_decay)],
             "lr": self.hparams.learning_rate, "weight_decay": 0.0},
            {"params": [p for n, p in self.encoder.named_parameters()
                        if not any(nd in n for nd in no_decay)],
             "lr": encoder_lr, "weight_decay": self.hparams.weight_decay},
            {"params": [p for n, p in self.encoder.named_parameters()
                        if any(nd in n for nd in no_decay)],
             "lr": encoder_lr, "weight_decay": 0.0},
        ]
        groups = [g for g in groups if g["params"]]

        opt = torch.optim.AdamW(groups, lr=self.hparams.learning_rate, betas=(0.9, 0.999), eps=1e-8)

        def lr_lambda(step):
            if step < self.hparams.warmup_steps:
                return step / max(1, self.hparams.warmup_steps)
            progress = (step - self.hparams.warmup_steps) / max(1, self.hparams.max_steps - self.hparams.warmup_steps)
            return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * progress)).item()))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _build_encoder_config():
    """Build SCBertConfig directly from known pretraining architecture (no Hydra needed)."""
    from bmfm_targets.config import SCBertConfig, FieldInfo
    fields = [
        FieldInfo(
            field_name="cpg_sites",
            vocab_size=8005,
            is_input=True,
            is_masked=False,
            tokenization_strategy="tokenize",
        ),
        FieldInfo(
            field_name="beta_values",
            is_input=True,
            is_masked=True,
            tokenization_strategy="continuous_value_encoder",
            num_special_tokens=5,
            encoder_kwargs={"kind": "mlp_with_special_token_embedding"},
            decode_modes={"regression": {}},
        ),
    ]
    return SCBertConfig(
        fields=fields,
        num_hidden_layers=6,
        num_attention_heads=8,
        hidden_size=512,
        intermediate_size=2048,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        classifier_dropout=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        use_cache=True,
        max_position_embeddings=8002,
        attention="torch",
        label_columns=None,
        checkpoint=None,
    )


def _load_wced_encoder(checkpoint_path: str, model_config, cfg: DictConfig):
    from bmfm_methylation.wced.wced_module import WCEDTrainingModule
    from bmfm_methylation.shared.config import PretrainingConfig
    from bmfm_targets.config import SCBertConfig, TrainerConfig, FieldInfo

    torch.serialization.add_safe_globals([SCBertConfig, TrainerConfig, FieldInfo])
    wced_config = PretrainingConfig(mode="wced")
    model_config.checkpoint = None

    module = WCEDTrainingModule.load_from_checkpoint(
        checkpoint_path,
        model_config=model_config,
        pretrain_config=wced_config,
    )
    logger.info(f"WCED encoder loaded — {sum(p.numel() for p in module.encoder.parameters()):,} params")
    return module.encoder


@hydra.main(config_path="configs", config_name="finetune_smoking_config", version_base="1.2")
def main(cfg: DictConfig):
    logger.info("=" * 60)
    logger.info("TASK A — SMOKING STATUS CLASSIFICATION")
    logger.info("=" * 60)
    logger.info(OmegaConf.to_yaml(cfg))

    if hasattr(cfg, "seed"):
        pl.seed_everything(cfg.seed, workers=True)

    output_dir = Path(cfg.output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    from bmfm_methylation.downstream.shared.classification_data_module import (
        ClassificationDataModule, SMOKING_LABEL_MAP, EVER_NEVER_LABEL_MAP,
    )

    binary = cfg.get("binary", False)
    n_classes = 2 if binary else 3
    label_map = EVER_NEVER_LABEL_MAP if binary else SMOKING_LABEL_MAP
    label_col = cfg.get("label_col", "smoking_status")

    data = ClassificationDataModule(
        h5ad_path=cfg.data_path,
        label_col=label_col,
        label_map=label_map,
        n_classes=n_classes,
        split_col=cfg.get("split_col", "split"),
        subset_k=cfg.get("subset_k", 4000),
        batch_size=cfg.get("batch_size", 32),
        num_workers=cfg.get("num_workers", 4),
    )
    data.setup()

    # ── Class weights (inverse frequency) ────────────────────────────────────
    import collections
    label_counts = collections.Counter(data.train_dataset.labels.tolist())
    total = sum(label_counts.values())
    class_weights = torch.tensor(
        [total / (n_classes * label_counts.get(i, 1)) for i in range(n_classes)],
        dtype=torch.float32,
    )
    logger.info(f"Class weights: {class_weights.tolist()}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model_config = _build_encoder_config()
    encoder = _load_wced_encoder(cfg.checkpoint_path, model_config, cfg)

    steps_per_epoch = max(1, len(data.train_dataset) // (cfg.get("batch_size", 32) * cfg.get("accumulate_grad_batches", 1)))
    total_steps = cfg.get("finetune_epochs", 100) * steps_per_epoch

    model = SmokingClassifier(
        encoder=encoder,
        hidden_size=model_config.hidden_size,
        n_classes=n_classes,
        head_dropout=cfg.get("head_dropout", 0.1),
        learning_rate=cfg.get("learning_rate", 1e-4),
        weight_decay=cfg.get("weight_decay", 0.01),
        warmup_steps=cfg.get("warmup_steps", 200),
        max_steps=total_steps,
        freeze_encoder=cfg.get("freeze_encoder", True),
        unfreeze_encoder_epoch=cfg.get("unfreeze_encoder_epoch", 5),
        class_weights=class_weights,
    )

    # ── Logger ────────────────────────────────────────────────────────────────
    if cfg.get("use_wandb", False):
        from pytorch_lightning.loggers import WandbLogger
        pl_logger = WandbLogger(
            project=cfg.get("wandb_project", "methyl-downstream-smoking"),
            entity=cfg.get("wandb_entity", None),
            name=cfg.get("wandb_run_name", "smoking-cls"),
            save_dir=str(output_dir),
        )
    else:
        from pytorch_lightning.loggers import TensorBoardLogger
        pl_logger = TensorBoardLogger(str(output_dir), name="smoking")

    # ── Trainer ───────────────────────────────────────────────────────────────
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="epoch={epoch}-val_f1={val/f1_macro:.4f}",
            monitor="val/f1_macro",
            mode="max",
            save_top_k=3,
            save_last=True,
        ),
        pl.callbacks.EarlyStopping(monitor="val/f1_macro", patience=cfg.get("early_stop_patience", 20), mode="max"),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
    ]

    trainer = pl.Trainer(
        max_epochs=cfg.get("finetune_epochs", 100),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed",
        accumulate_grad_batches=cfg.get("accumulate_grad_batches", 1),
        gradient_clip_val=1.0,
        logger=pl_logger,
        callbacks=callbacks,
        default_root_dir=str(output_dir),
        log_every_n_steps=10,
    )

    trainer.fit(model, data)
    trainer.test(model, data)
    logger.info(f"Best checkpoint: {trainer.checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
