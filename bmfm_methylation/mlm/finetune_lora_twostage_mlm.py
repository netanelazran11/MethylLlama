#!/usr/bin/env python3
"""
Two-Stage LoRA Fine-tuning for Methylation Age Prediction.

Solves the "moving target" problem of joint head+LoRA training:

  Stage 1 (epochs 0 → lora_warmup_epochs):
    Head trains freely, LoRA LR = 0 (adapters frozen at zero delta).
    Head converges to the frozen-encoder ceiling (~R²=0.93) quickly.

  Stage 2 (epochs lora_warmup_epochs → end):
    LoRA unlocked at lora_lr=1e-4. Head gradient signal is now stable,
    so LoRA gets clean supervision to adapt attention toward age.

Why this beats joint training:
  Joint:     head and LoRA move simultaneously → interference → slow convergence
  Two-stage: head converges first → LoRA gets stable signal → cleaner adaptation

Architecture identical to finetune_lora.py, only the LR schedule differs.
"""

# =============================================================================
# CRITICAL: weights_only patch MUST be before any other imports
# =============================================================================
import torch
import torch.serialization

_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load
torch.serialization.load = _patched_torch_load
# =============================================================================

import logging
import sys
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

from bmfm_methylation.shared.tokenizer import (
    extract_cpg_sites_from_h5ad,
    create_methylation_multifield_tokenizer,
)
from bmfm_methylation.shared.data_module import MethylationDataModule
from bmfm_methylation.mlm.lora import inject_lora, get_lora_parameters

logger = logging.getLogger(__name__)

# Index of LoRA param group in optimizer (0=head_decay, 1=head_nodecay, 2=lora)
LORA_PARAM_GROUP_IDX = 2


class LoRAUnlockCallback(pl.Callback):
    """
    At epoch `warmup_epochs`, set LoRA param group LR from 0 → lora_lr.

    During Stage 1 (0 to warmup_epochs-1): LoRA LR = 0, adapters unchanged.
    During Stage 2 (warmup_epochs onwards):  LoRA LR = lora_lr, adapters adapt.
    """

    def __init__(self, warmup_epochs: int, lora_lr: float):
        self.warmup_epochs = warmup_epochs
        self.lora_lr = lora_lr
        self.unlocked = False

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if not self.unlocked and trainer.current_epoch >= self.warmup_epochs:
            optimizer = trainer.optimizers[0]

            # Set the current LR for this param group
            optimizer.param_groups[LORA_PARAM_GROUP_IDX]['lr'] = self.lora_lr
            # Some PyTorch versions track initial_lr separately
            optimizer.param_groups[LORA_PARAM_GROUP_IDX]['initial_lr'] = self.lora_lr

            # CRITICAL: LambdaLR stores base_lrs and computes lr = base_lr * lambda.
            # If base_lr=0, scheduler overrides our change every step → LoRA stays frozen.
            # Fix: update base_lrs so the scheduler uses the correct base.
            for sch_config in trainer.lr_scheduler_configs:
                sch = sch_config.scheduler
                if hasattr(sch, 'base_lrs') and len(sch.base_lrs) > LORA_PARAM_GROUP_IDX:
                    sch.base_lrs[LORA_PARAM_GROUP_IDX] = self.lora_lr

            self.unlocked = True
            logger.info(
                f"[TwoStage] Epoch {trainer.current_epoch}: "
                f"LoRA UNLOCKED → lr={self.lora_lr}"
            )
            trainer.logger.log_metrics(
                {"lora_unlocked_epoch": float(trainer.current_epoch)},
                step=trainer.global_step,
            )


class MethylationAgeRegressorTwoStage(pl.LightningModule):
    """
    Two-stage LoRA fine-tuning module.

    Identical to MethylationAgeRegressorLoRA except LoRA starts with lr=0
    and the LoRAUnlockCallback handles the stage transition.
    """

    def __init__(
        self,
        encoder,
        hidden_size: int = 512,
        head_hidden_size: int = 256,
        head_dropout: float = 0.1,
        learning_rate: float = 1e-3,
        lora_lr: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        max_steps: int = 10000,
        age_mean: float = 0.0,
        age_std: float = 1.0,
        use_huber_loss: bool = False,
        huber_delta: float = 2.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['encoder'])

        self.encoder = encoder
        self.age_mean = age_mean
        self.age_std = age_std

        # MLP head: hidden_size → head_hidden_size → head_hidden_size//2 → 1
        self.age_head = nn.Sequential(
            nn.Linear(hidden_size, head_hidden_size),
            nn.LayerNorm(head_hidden_size),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden_size, head_hidden_size // 2),
            nn.LayerNorm(head_hidden_size // 2),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden_size // 2, 1),
        )

        self.loss_fn = nn.HuberLoss(delta=huber_delta) if use_huber_loss else nn.MSELoss()

        self._val_preds = []
        self._val_labels = []
        self._test_preds = []
        self._test_labels = []

        lora_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        frozen_params = sum(p.numel() for p in encoder.parameters() if not p.requires_grad)
        head_params = sum(p.numel() for p in self.age_head.parameters())

        logger.info("Two-Stage LoRA Fine-tuning module:")
        logger.info(f"  Encoder frozen:    {frozen_params:,}")
        logger.info(f"  Encoder LoRA:      {lora_params:,}  (trainable, starts at lr=0)")
        logger.info(f"  Age head:          {head_params:,}  (trainable from epoch 0)")
        logger.info(f"  Head LR:           {learning_rate}  |  LoRA LR (stage 2): {lora_lr}")

    def _encode(self, cpg_ids, beta_values, attention_mask=None):
        input_ids = torch.stack([cpg_ids.float(), beta_values], dim=1)
        batch_size = input_ids.size(0)
        seq_length = input_ids.size(2)

        if attention_mask is not None and attention_mask.dim() == 3:
            attention_mask = attention_mask[:, 0, :]
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length), device=input_ids.device
            )

        encoder_output = self.encoder(input_ids, attention_mask=attention_mask)
        return encoder_output.pooler_output  # CLS [batch, hidden_size]

    def forward(self, cpg_ids, beta_values, attention_mask=None):
        cls = self._encode(cpg_ids, beta_values, attention_mask)
        return self.age_head(cls)

    def _shared_step(self, batch, stage: str):
        cpg_ids = batch["cpg_ids"]
        beta_values = batch["beta_values"]
        attention_mask = batch.get("attention_mask")
        labels = batch["labels"].float().view(-1, 1)

        cls_embedding = self._encode(cpg_ids, beta_values, attention_mask)
        predictions = self.age_head(cls_embedding)
        loss = self.loss_fn(predictions, labels)

        preds_denorm = predictions * self.age_std + self.age_mean
        labels_denorm = labels * self.age_std + self.age_mean
        mae = torch.abs(preds_denorm - labels_denorm).mean()

        return loss, preds_denorm, labels_denorm, mae

    def training_step(self, batch, batch_idx):
        loss, _, _, mae = self._shared_step(batch, "train")
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/mae", mae, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, labels, mae = self._shared_step(batch, "val")
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/mae", mae, on_epoch=True, prog_bar=True)
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
            self.log("val/mae_epoch", torch.abs(all_preds - all_labels).mean())
        self._val_preds.clear()
        self._val_labels.clear()

    def test_step(self, batch, batch_idx):
        loss, preds, labels, mae = self._shared_step(batch, "test")
        self.log("test/mae", mae, on_epoch=True)
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
            logger.info(f"Test MAE: {epoch_mae:.2f} years, R²: {r2:.4f}")
        self._test_preds.clear()
        self._test_labels.clear()

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]

        # Head — fast learning from epoch 0
        head_params_decay = [p for n, p in self.age_head.named_parameters()
                             if not any(nd in n for nd in no_decay)]
        head_params_no_decay = [p for n, p in self.age_head.named_parameters()
                                if any(nd in n for nd in no_decay)]

        # LoRA — lr=0 in Stage 1; LoRAUnlockCallback sets it to lora_lr at Stage 2
        lora_params = get_lora_parameters(self.encoder)

        optimizer_grouped_parameters = [
            # group 0: head with weight decay
            {"params": head_params_decay,    "lr": self.hparams.learning_rate,
             "weight_decay": self.hparams.weight_decay},
            # group 1: head without weight decay (bias, norms)
            {"params": head_params_no_decay, "lr": self.hparams.learning_rate,
             "weight_decay": 0.0},
            # group 2 (LORA_PARAM_GROUP_IDX): LoRA — starts frozen at lr=0
            {"params": lora_params,          "lr": 0.0,
             "weight_decay": 0.0},
        ]
        optimizer_grouped_parameters = [
            g for g in optimizer_grouped_parameters if len(g["params"]) > 0
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        def lr_lambda(current_step):
            if current_step < self.hparams.warmup_steps:
                return float(current_step) / float(max(1, self.hparams.warmup_steps))
            progress = float(current_step - self.hparams.warmup_steps) / \
                       float(max(1, self.hparams.max_steps - self.hparams.warmup_steps))
            return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * progress)).item()))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }


def setup_tokenizer(cfg: DictConfig):
    tokenizer_path = Path(cfg.tokenizer_path)
    if tokenizer_path.exists() and (tokenizer_path / "tokenizers").exists():
        from bmfm_targets.tokenization import MultiFieldTokenizer
        return MultiFieldTokenizer.from_pretrained(str(tokenizer_path))
    cpg_sites = extract_cpg_sites_from_h5ad(cfg.data_path)
    return create_methylation_multifield_tokenizer(
        cpg_sites=cpg_sites,
        output_dir=str(tokenizer_path),
    )


def setup_wandb(cfg: DictConfig):
    if hasattr(cfg, 'track_wandb') and cfg.track_wandb.get('enabled', False):
        try:
            from pytorch_lightning.loggers import WandbLogger
            return WandbLogger(
                project=cfg.track_wandb.get('project', 'methylation-age-lora-twostage'),
                entity=cfg.track_wandb.get('entity', None),
                name=cfg.track_wandb.get('name', None),
                save_dir=cfg.output_directory,
                log_model=True,
            )
        except Exception as e:
            logger.warning(f"WandB setup failed: {e}, using TensorBoard")
    from pytorch_lightning.loggers import TensorBoardLogger
    return TensorBoardLogger(cfg.output_directory, name="finetune_lora_twostage")


@hydra.main(
    config_path="configs",
    config_name="finetune_config",
    version_base="1.2"
)
def main(cfg: DictConfig):
    logger.info("=" * 70)
    logger.info("METHYLATION AGE FINE-TUNING — Two-Stage LoRA")
    logger.info("  Stage 1: head only (LoRA LR=0, frozen adapters)")
    logger.info("  Stage 2: LoRA unlocked after lora_warmup_epochs")
    logger.info("=" * 70)
    logger.info(f"\nConfiguration:\n{OmegaConf.to_yaml(cfg)}")

    if hasattr(cfg, 'seed') and cfg.seed:
        pl.seed_everything(cfg.seed.seed_value, workers=True)

    output_dir = Path(cfg.output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = setup_tokenizer(cfg)

    from bmfm_targets.config import FieldInfo
    fields = []
    for field_cfg in cfg.fields:
        field_dict = OmegaConf.to_container(field_cfg)
        field_dict.pop('_target_', None)
        fields.append(FieldInfo(**field_dict))

    # -------------------------------------------------------------------------
    # Data module — MethylationCollator, 8k fixed subset (maximum information)
    # -------------------------------------------------------------------------
    subset_k = cfg.data_module.get('subset_k', 8000)
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
        mlm=False,
        collation_strategy="sequence_classification",
        subset_k=subset_k,
        fixed_subset=True,
        fixed_subset_seed=cfg.data_module.get('fixed_subset_seed', 42),
        use_wced_collator=False,
    )
    data_module.setup()

    model_config_partial = hydra.utils.instantiate(cfg.model)
    model_config = model_config_partial(fields=fields)

    # -------------------------------------------------------------------------
    # Load WCED pretrained encoder
    # -------------------------------------------------------------------------
    if not cfg.checkpoint_path or cfg.checkpoint_path == "null":
        raise ValueError("checkpoint_path is required.")

    logger.info(f"Loading WCED checkpoint: {cfg.checkpoint_path}")

    from bmfm_methylation.wced.wced_module import WCEDTrainingModule
    from bmfm_methylation.shared.config import PretrainingConfig
    from bmfm_targets.config import SCBertConfig, TrainerConfig

    torch.serialization.add_safe_globals([SCBertConfig, TrainerConfig, FieldInfo])

    wced_config = PretrainingConfig(mode="wced")
    model_config.checkpoint = None

    pretrained_module = WCEDTrainingModule.load_from_checkpoint(
        cfg.checkpoint_path,
        model_config=model_config,
        pretrain_config=wced_config,
    )

    encoder = pretrained_module.encoder
    logger.info(f"Encoder loaded: {sum(p.numel() for p in encoder.parameters()):,} params")
    if hasattr(encoder.embeddings, 'cpg_scale'):
        logger.info(f"  cpg_scale: {encoder.embeddings.cpg_scale.item():.4f}")

    # -------------------------------------------------------------------------
    # Inject LoRA (adapters start at B=0, LR=0 until Stage 2)
    # -------------------------------------------------------------------------
    lora_rank = cfg.get('lora_rank', 8)
    lora_alpha = cfg.get('lora_alpha', 16.0)
    lora_lr = cfg.get('lora_lr', 1e-4)
    lora_warmup_epochs = cfg.get('lora_warmup_epochs', 50)

    lora_param_count = inject_lora(
        encoder=encoder,
        rank=lora_rank,
        alpha=lora_alpha,
        target_modules=("query", "value"),
        dropout=0.0,
    )
    logger.info(f"LoRA injected: rank={lora_rank}, alpha={lora_alpha}")
    logger.info(f"  Stage 1 (epochs 0-{lora_warmup_epochs-1}): LoRA LR=0 (head only)")
    logger.info(f"  Stage 2 (epoch {lora_warmup_epochs}+):      LoRA LR={lora_lr}")
    logger.info(f"  Trainable LoRA params: {lora_param_count:,}")

    # -------------------------------------------------------------------------
    # Fine-tuning model
    # -------------------------------------------------------------------------
    effective_batch = cfg.data_module.batch_size * cfg.accumulate_grad_batches
    steps_per_epoch = len(data_module.train_dataset) // effective_batch
    total_steps = cfg.finetune_epochs * steps_per_epoch

    logger.info(f"Train samples: {len(data_module.train_dataset)}")
    logger.info(f"Effective batch: {effective_batch}, total steps: {total_steps}")
    logger.info(f"Age stats: mean={data_module.age_mean:.2f}, std={data_module.age_std:.2f}")

    model = MethylationAgeRegressorTwoStage(
        encoder=encoder,
        hidden_size=model_config.hidden_size,
        head_hidden_size=cfg.regression_head.hidden_size,
        head_dropout=cfg.regression_head.dropout,
        learning_rate=cfg.trainer.learning_rate,
        lora_lr=lora_lr,
        weight_decay=cfg.trainer.weight_decay,
        warmup_steps=cfg.trainer.warmup_steps,
        max_steps=total_steps,
        age_mean=data_module.age_mean,
        age_std=data_module.age_std,
        use_huber_loss=cfg.get('use_huber_loss', False),
        huber_delta=cfg.get('huber_delta', 2.0),
    )

    wandb_logger = setup_wandb(cfg)

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=output_dir / "finetune_lora_twostage" / "checkpoints",
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
        LoRAUnlockCallback(
            warmup_epochs=lora_warmup_epochs,
            lora_lr=lora_lr,
        ),
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
        default_root_dir=str(output_dir / "finetune_lora_twostage"),
        log_every_n_steps=10,
    )

    logger.info("Starting Two-Stage LoRA fine-tuning...")
    trainer.fit(model, data_module)

    logger.info("Running test evaluation...")
    trainer.test(model, data_module)

    best_ckpt = trainer.checkpoint_callback.best_model_path
    logger.info(f"Two-Stage LoRA complete. Best checkpoint: {best_ckpt}")
    logger.info(f"Best val/mae: {trainer.checkpoint_callback.best_model_score:.4f}")

    return best_ckpt


if __name__ == "__main__":
    main()
