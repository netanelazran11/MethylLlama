#!/usr/bin/env python3
"""
Task B — Multi-Task Probing: one encoder, three heads simultaneously.

Architecture:
    CpGs → WCED Encoder → CLS ─┬─► Age Head     → MSE     (regression)
                                ├─► Smoking Head → CrossEntropy (3-class)
                                └─► Sex Head     → CrossEntropy (2-class)

Novelty claim:
  The WCED+contrastive CLS embedding encodes multiple phenotypes simultaneously.
  Fitting all three heads at once with a shared frozen/fine-tuned encoder shows
  the representation richness is not task-specific — a key advantage over
  MethylGPT (MLM-only) and EpiSmokEr (task-specific hand-picked CpGs).

Data requirement:
  One h5ad file with obs columns: age, smoking_status, sex, split.
  Samples missing a label for one task contribute 0 loss for that task.
  → You can merge your age h5ad and smoking h5ad on sample overlap.

Run via scripts/downstream/finetune_multitask.sh
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

import hydra
import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

logger = logging.getLogger(__name__)

SMOKING_MAP = {"current": 0, "former": 1, "never": 2}
SEX_MAP = {"M": 0, "F": 1, "male": 0, "female": 1}


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class MultiTaskDataset(torch.utils.data.Dataset):
    """
    Reads one h5ad with age + smoking_status + sex columns.
    Missing labels are returned as -1 (masked out in loss).
    """

    def __init__(self, h5ad_path, split, subset_k=4000, fixed_subset=False,
                 age_mean=None, age_std=None, split_col="split"):
        from bmfm_methylation.shared.data_module import _read_h5ad_robust
        import pandas as pd

        adata = _read_h5ad_robust(h5ad_path)
        if split_col in adata.obs.columns:
            adata = adata[adata.obs[split_col] == split].copy()
        logger.info(f"[{split}] {len(adata)} samples")

        self.X = adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray()
        self.n_cpg = self.X.shape[1]
        self.subset_k = subset_k

        # Age labels (normalized)
        if "age" in adata.obs.columns:
            ages = pd.to_numeric(adata.obs["age"], errors="coerce").values.astype(np.float32)
        else:
            ages = np.full(len(adata), np.nan, dtype=np.float32)

        if age_mean is None:
            age_mean = float(np.nanmean(ages))
        if age_std is None:
            age_std = float(np.nanstd(ages))
        self.age_mean = age_mean
        self.age_std = age_std

        self.ages = np.where(np.isnan(ages), np.nan, (ages - age_mean) / (age_std + 1e-8))

        # Smoking labels
        if "smoking_status" in adata.obs.columns:
            self.smoking = np.array(
                [SMOKING_MAP.get(str(s), -1) for s in adata.obs["smoking_status"].values],
                dtype=np.int64,
            )
        else:
            self.smoking = np.full(len(adata), -1, dtype=np.int64)

        # Sex labels
        if "sex" in adata.obs.columns:
            self.sex = np.array(
                [SEX_MAP.get(str(s), -1) for s in adata.obs["sex"].values],
                dtype=np.int64,
            )
        else:
            self.sex = np.full(len(adata), -1, dtype=np.int64)

        # CpG vocab
        try:
            self.cpg_vocab = np.array(adata.var["cpg_id"].values, dtype=np.int64)
        except (KeyError, AttributeError):
            self.cpg_vocab = np.arange(5, self.n_cpg + 5, dtype=np.int64)

        # Fixed subset for val/test
        self._fixed_indices = None
        if fixed_subset:
            rng = np.random.default_rng(42)
            self._fixed_indices = rng.choice(self.n_cpg, min(subset_k, self.n_cpg), replace=False)
            self._fixed_indices.sort()

        present = {
            "age": int((~np.isnan(self.ages)).sum()),
            "smoking": int((self.smoking >= 0).sum()),
            "sex": int((self.sex >= 0).sum()),
        }
        logger.info(f"[{split}] label coverage: {present}")

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        row = self.X[idx]
        valid = np.where(~np.isnan(row))[0]

        if self._fixed_indices is not None:
            chosen = self._fixed_indices[~np.isnan(row[self._fixed_indices])]
        else:
            k = min(self.subset_k, len(valid))
            chosen = np.random.choice(valid, k, replace=False)
            chosen.sort()

        age_val = self.ages[idx]
        return {
            "cpg_ids": torch.from_numpy(self.cpg_vocab[chosen].astype(np.float32)),
            "beta_values": torch.from_numpy(row[chosen].astype(np.float32)),
            "age": torch.tensor(float("nan") if np.isnan(age_val) else age_val, dtype=torch.float32),
            "smoking": torch.tensor(self.smoking[idx], dtype=torch.long),
            "sex": torch.tensor(self.sex[idx], dtype=torch.long),
        }


def _collate_multitask(batch):
    max_len = max(b["cpg_ids"].shape[0] for b in batch)
    cpg_ids = torch.zeros(len(batch), max_len)
    beta_values = torch.zeros(len(batch), max_len)
    attn = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, b in enumerate(batch):
        n = b["cpg_ids"].shape[0]
        cpg_ids[i, :n] = b["cpg_ids"]
        beta_values[i, :n] = b["beta_values"]
        attn[i, :n] = 1
    return {
        "cpg_ids": cpg_ids,
        "beta_values": beta_values,
        "attention_mask": attn,
        "age": torch.stack([b["age"] for b in batch]),
        "smoking": torch.stack([b["smoking"] for b in batch]),
        "sex": torch.stack([b["sex"] for b in batch]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Lightning Module
# ─────────────────────────────────────────────────────────────────────────────

class MultiTaskModule(pl.LightningModule):
    """
    Three-task module on shared WCED encoder.

    Loss = age_weight * age_MSE  +  smoking_weight * CE  +  sex_weight * CE
    Samples with missing labels are masked (contribute 0 to loss).
    """

    def __init__(
        self,
        encoder,
        hidden_size: int = 512,
        age_mean: float = 0.0,
        age_std: float = 1.0,
        age_weight: float = 1.0,
        smoking_weight: float = 1.0,
        sex_weight: float = 0.5,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 200,
        max_steps: int = 5000,
        freeze_encoder: bool = True,
        unfreeze_encoder_epoch: int = 5,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder"])

        self.encoder = encoder
        self.age_mean = age_mean
        self.age_std = age_std

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # Task heads
        self.age_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )
        self.smoking_head = nn.Linear(hidden_size, 3)
        self.sex_head = nn.Linear(hidden_size, 2)

        self._val_buf = {k: [] for k in ["age_pred", "age_true", "smk_logit", "smk_true", "sex_logit", "sex_true"]}
        self._test_buf = {k: [] for k in self._val_buf}

    def _encode(self, batch):
        cpg_ids = batch["cpg_ids"]
        beta_values = batch["beta_values"]
        attn = batch.get("attention_mask")
        input_ids = torch.stack([cpg_ids.float(), beta_values], dim=1)
        bs, _, seq = input_ids.shape
        if attn is not None and attn.dim() == 3:
            attn = attn[:, 0, :]
        if attn is None:
            attn = torch.ones(bs, seq, device=input_ids.device)
        return self.encoder(input_ids, attention_mask=attn).pooler_output

    def _compute_loss(self, cls, batch, buf=None):
        losses = {}

        # Age (regression, masked on NaN)
        age_labels = batch["age"]               # [B]  may contain NaN
        age_mask = ~torch.isnan(age_labels)
        if age_mask.any():
            age_pred = self.age_head(cls[age_mask]).squeeze(-1)
            losses["age"] = F.mse_loss(age_pred, age_labels[age_mask])
            if buf is not None:
                pred_yr = age_pred.detach() * self.age_std + self.age_mean
                true_yr = age_labels[age_mask].detach() * self.age_std + self.age_mean
                buf["age_pred"].append(pred_yr.cpu())
                buf["age_true"].append(true_yr.cpu())
        else:
            losses["age"] = torch.tensor(0.0, device=cls.device)

        # Smoking (3-class, masked on -1)
        smk = batch["smoking"]
        smk_mask = smk >= 0
        if smk_mask.any():
            logits = self.smoking_head(cls[smk_mask])
            losses["smoking"] = F.cross_entropy(logits, smk[smk_mask])
            if buf is not None:
                buf["smk_logit"].append(logits.detach().cpu())
                buf["smk_true"].append(smk[smk_mask].detach().cpu())
        else:
            losses["smoking"] = torch.tensor(0.0, device=cls.device)

        # Sex (2-class, masked on -1)
        sex = batch["sex"]
        sex_mask = sex >= 0
        if sex_mask.any():
            logits = self.sex_head(cls[sex_mask])
            losses["sex"] = F.cross_entropy(logits, sex[sex_mask])
            if buf is not None:
                buf["sex_logit"].append(logits.detach().cpu())
                buf["sex_true"].append(sex[sex_mask].detach().cpu())
        else:
            losses["sex"] = torch.tensor(0.0, device=cls.device)

        total = (
            self.hparams.age_weight * losses["age"]
            + self.hparams.smoking_weight * losses["smoking"]
            + self.hparams.sex_weight * losses["sex"]
        )
        return total, losses

    def on_train_epoch_start(self):
        if self.hparams.freeze_encoder and self.current_epoch == self.hparams.unfreeze_encoder_epoch:
            logger.info(f"Unfreezing encoder at epoch {self.current_epoch}")
            for p in self.encoder.parameters():
                p.requires_grad = True

    def training_step(self, batch, _):
        cls = self._encode(batch)
        total, losses = self._compute_loss(cls, batch)
        self.log("train/loss", total, on_step=True, on_epoch=True, prog_bar=True)
        for k, v in losses.items():
            self.log(f"train/loss_{k}", v, on_step=False, on_epoch=True)
        return total

    def validation_step(self, batch, _):
        cls = self._encode(batch)
        total, losses = self._compute_loss(cls, batch, buf=self._val_buf)
        self.log("val/loss", total, on_epoch=True, prog_bar=True)
        for k, v in losses.items():
            self.log(f"val/loss_{k}", v, on_epoch=True)
        return total

    def on_validation_epoch_end(self):
        self._log_epoch(self._val_buf, "val")
        for v in self._val_buf.values():
            v.clear()

    def test_step(self, batch, _):
        cls = self._encode(batch)
        self._compute_loss(cls, batch, buf=self._test_buf)

    def on_test_epoch_end(self):
        self._log_epoch(self._test_buf, "test")
        for v in self._test_buf.values():
            v.clear()

    def _log_epoch(self, buf, stage):
        try:
            from sklearn.metrics import f1_score, r2_score, accuracy_score
        except ImportError:
            return

        if buf["age_pred"]:
            p = torch.cat(buf["age_pred"]).numpy()
            t = torch.cat(buf["age_true"]).numpy()
            self.log(f"{stage}/age_r2", r2_score(t, p), prog_bar=(stage == "val"))
            self.log(f"{stage}/age_mae", float(np.abs(p - t).mean()))

        if buf["smk_logit"]:
            logits = torch.cat(buf["smk_logit"])
            labels = torch.cat(buf["smk_true"]).numpy()
            preds = logits.argmax(1).numpy()
            self.log(f"{stage}/smk_acc", accuracy_score(labels, preds), prog_bar=(stage == "val"))
            self.log(f"{stage}/smk_f1", f1_score(labels, preds, average="macro", zero_division=0))

        if buf["sex_logit"]:
            logits = torch.cat(buf["sex_logit"])
            labels = torch.cat(buf["sex_true"]).numpy()
            preds = logits.argmax(1).numpy()
            self.log(f"{stage}/sex_acc", accuracy_score(labels, preds))

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        enc_lr = self.hparams.learning_rate * 0.01
        head_params = list(self.age_head.parameters()) + list(self.smoking_head.parameters()) + list(self.sex_head.parameters())

        groups = [
            {"params": head_params, "lr": self.hparams.learning_rate, "weight_decay": self.hparams.weight_decay},
            {"params": [p for n, p in self.encoder.named_parameters() if not any(nd in n for nd in no_decay)],
             "lr": enc_lr, "weight_decay": self.hparams.weight_decay},
            {"params": [p for n, p in self.encoder.named_parameters() if any(nd in n for nd in no_decay)],
             "lr": enc_lr, "weight_decay": 0.0},
        ]
        groups = [g for g in groups if g["params"]]

        opt = torch.optim.AdamW(groups, lr=self.hparams.learning_rate)

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

@hydra.main(config_path="configs", config_name="finetune_multitask_config", version_base="1.2")
def main(cfg: DictConfig):
    logger.info("=" * 60)
    logger.info("TASK B — MULTI-TASK PROBING (age + smoking + sex)")
    logger.info("=" * 60)
    logger.info(OmegaConf.to_yaml(cfg))

    if hasattr(cfg, "seed"):
        pl.seed_everything(cfg.seed, workers=True)

    output_dir = Path(cfg.output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    train_ds = MultiTaskDataset(cfg.data_path, "train", subset_k=cfg.get("subset_k", 4000))
    val_ds = MultiTaskDataset(cfg.data_path, "valid", subset_k=cfg.get("subset_k", 4000),
                              fixed_subset=True, age_mean=train_ds.age_mean, age_std=train_ds.age_std)
    test_ds = MultiTaskDataset(cfg.data_path, "test", subset_k=cfg.get("subset_k", 4000),
                               fixed_subset=True, age_mean=train_ds.age_mean, age_std=train_ds.age_std)

    bs = cfg.get("batch_size", 32)
    nw = cfg.get("num_workers", 4)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=bs, shuffle=True,
                                           num_workers=nw, collate_fn=_collate_multitask, pin_memory=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=bs, shuffle=False,
                                         num_workers=nw, collate_fn=_collate_multitask, pin_memory=True)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=bs, shuffle=False,
                                          num_workers=nw, collate_fn=_collate_multitask, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    from bmfm_targets.config import SCBertConfig, TrainerConfig, FieldInfo
    from bmfm_methylation.wced.wced_module import WCEDTrainingModule
    from bmfm_methylation.shared.config import PretrainingConfig

    fields = [
        FieldInfo(field_name="cpg_sites", vocab_size=8005, is_input=True, is_masked=False,
                  tokenization_strategy="tokenize"),
        FieldInfo(field_name="beta_values", is_input=True, is_masked=True,
                  tokenization_strategy="continuous_value_encoder", num_special_tokens=5,
                  encoder_kwargs={"kind": "mlp_with_special_token_embedding"},
                  decode_modes={"regression": {}}),
    ]
    model_config = SCBertConfig(
        fields=fields, num_hidden_layers=6, num_attention_heads=8, hidden_size=512,
        intermediate_size=2048, hidden_act="gelu", hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1, classifier_dropout=0.1, initializer_range=0.02,
        layer_norm_eps=1e-12, pad_token_id=0, use_cache=True, max_position_embeddings=8002,
        attention="torch", label_columns=None, checkpoint=None,
    )

    torch.serialization.add_safe_globals([SCBertConfig, TrainerConfig, FieldInfo])
    wced_config = PretrainingConfig(mode="wced")
    model_config.checkpoint = None
    pt_module = WCEDTrainingModule.load_from_checkpoint(
        cfg.checkpoint_path, model_config=model_config, pretrain_config=wced_config,
    )
    encoder = pt_module.encoder

    steps_per_epoch = len(train_ds) // (bs * cfg.get("accumulate_grad_batches", 1))
    total_steps = cfg.get("finetune_epochs", 100) * steps_per_epoch

    model = MultiTaskModule(
        encoder=encoder,
        hidden_size=model_config.hidden_size,
        age_mean=train_ds.age_mean,
        age_std=train_ds.age_std,
        age_weight=cfg.get("age_weight", 1.0),
        smoking_weight=cfg.get("smoking_weight", 1.0),
        sex_weight=cfg.get("sex_weight", 0.5),
        learning_rate=cfg.get("learning_rate", 1e-4),
        weight_decay=cfg.get("weight_decay", 0.01),
        warmup_steps=cfg.get("warmup_steps", 200),
        max_steps=total_steps,
        freeze_encoder=cfg.get("freeze_encoder", True),
        unfreeze_encoder_epoch=cfg.get("unfreeze_encoder_epoch", 5),
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    if cfg.get("use_wandb", False):
        from pytorch_lightning.loggers import WandbLogger
        pl_logger = WandbLogger(
            project=cfg.get("wandb_project", "methyl-downstream-multitask"),
            entity=cfg.get("wandb_entity", None),
            name=cfg.get("wandb_run_name", "multitask"),
            save_dir=str(output_dir),
        )
    else:
        from pytorch_lightning.loggers import TensorBoardLogger
        pl_logger = TensorBoardLogger(str(output_dir), name="multitask")

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="epoch={epoch}-val_loss={val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
        ),
        pl.callbacks.EarlyStopping(monitor="val/loss", patience=cfg.get("early_stop_patience", 20), mode="min"),
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

    trainer.fit(model, train_dl, val_dl)
    trainer.test(model, test_dl)


if __name__ == "__main__":
    main()
