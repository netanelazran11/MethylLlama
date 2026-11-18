"""
Reconstruction-only arm of the InfoNCE ablation.

WCEDLlamaModule (bmfm_methylation/llama/wced_llama.py) encodes view 2 only when
contrastive_weight > 0. Setting the weight to 0 there therefore also drops the
second reconstruction view, so a "λ = 0" run would differ from the published
run in two ways: no InfoNCE AND half the reconstruction signal per step.

This subclass removes that confound. Per step it does exactly what the
published objective does (encode both views, average the two reconstruction
losses), except that InfoNCE is not part of the loss:

    published:  L = (L_recon(v1) + L_recon(v2)) / 2 + 0.05 * InfoNCE(head(cls1), head(cls2))
    this arm:   L = (L_recon(v1) + L_recon(v2)) / 2

The projection head is still built (same parameter init, same RNG stream) but
receives no gradient. Same data order, same views, same dropout calls, so the
two arms differ only in the gradient.

Monitoring (no gradient, does not touch the loss): cross-view identity on the
L2-normalised CLS (pooler output, the representation used downstream) within
the per-GPU batch — InfoNCE at the run's temperature, top-1, matched and
unmatched cosine. Logged as */monitor_cls_*. Compute the same quantities on the
published checkpoints post hoc for the comparison.

Pilot (objective sweep): WCEDLlamaAblationModule runs every arm through the same
code — two-view reconstruction + λ · InfoNCE, with λ = wced_contrastive_weight.
At λ > 0 it reproduces the published objective exactly (loss and gradients,
checked in test_recon_only.py); at λ = 0 it is the reconstruction-only arm.
WCEDLlamaReconOnlyModule is the same class with λ forced to 0 (full-corpus run).

The parent file is not modified.
"""

import logging
import os
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F

from bmfm_methylation.llama.wced_llama import WCEDLlamaModule

logger = logging.getLogger(__name__)


def _parse_epochs(s: str) -> set:
    return {int(x) for x in s.split(",") if x.strip()}


class WCEDLlamaAblationModule(WCEDLlamaModule):
    """Two-view reconstruction + contrastive_weight * InfoNCE, for every weight including 0."""

    # Set by the entry point from environment variables (Hydra config is struct).
    analysis_epochs: set = set()
    stop_after_epoch: int = -1

    # Set from ABLATION_GRAD_CKPT=1 by configure_from_env(). Recomputes each encoder layer in the
    # backward pass instead of storing its activations: same loss and gradients (dropout RNG is
    # replayed; checked in test_recon_only.py), less GPU memory, ~30 % slower. Needed for batch 32
    # (31 InfoNCE negatives, as published) on a 48 GB L40S; not used on H200.
    grad_checkpointing: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._monitor: Dict[str, torch.Tensor] = {}
        if self.grad_checkpointing:
            self.enable_grad_checkpointing()

    def enable_grad_checkpointing(self):
        """Wrap each encoder layer's forward (this instance only; model.py is untouched)."""
        from torch.utils.checkpoint import checkpoint

        for layer in self.encoder.encoder.layers:
            plain = layer.forward

            def wrapped(hidden_states, attention_mask=None, position_ids=None,
                        output_attentions=False, _plain=plain):
                if torch.is_grad_enabled() and not output_attentions:
                    return checkpoint(_plain, hidden_states, attention_mask, position_ids, False,
                                      use_reentrant=False, preserve_rng_state=True)
                return _plain(hidden_states, attention_mask=attention_mask,
                              position_ids=position_ids, output_attentions=output_attentions)

            layer.forward = wrapped
        logger.info("gradient checkpointing ON for %d encoder layers", len(self.encoder.encoder.layers))
        logger.info(f"{type(self).__name__}: two-view reconstruction + "
                    f"{self.contrastive_weight} x InfoNCE (CLS identity monitor logged)")

    # -------------------------------------------------------------------------
    # Identity monitor on CLS (no gradient)
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def _cls_identity_monitor(self, cls1: torch.Tensor, cls2: torch.Tensor) -> Dict[str, torch.Tensor]:
        c1 = F.normalize(cls1.float(), dim=-1)
        c2 = F.normalize(cls2.float(), dim=-1)
        B = c1.shape[0]
        sim = c1 @ c2.T
        labels = torch.arange(B, device=c1.device)
        logits = sim / self.contrastive_temp
        infonce = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        top1 = ((sim.argmax(dim=1) == labels).float().mean()
                + (sim.argmax(dim=0) == labels).float().mean()) / 2
        pos = sim.diagonal().mean()
        off = ~torch.eye(B, dtype=torch.bool, device=c1.device)
        neg = sim[off].mean() if B > 1 else torch.tensor(float("nan"), device=c1.device)
        return {"monitor_cls_infonce": infonce, "monitor_cls_top1": top1,
                "monitor_cls_pos_cos": pos, "monitor_cls_neg_cos": neg}

    # -------------------------------------------------------------------------
    # Shared step: same as the parent's two-view path, without InfoNCE in the loss
    # -------------------------------------------------------------------------

    def _recon_loss(self, pred, target, mask):
        if self.normalize_loss:
            pred = self._normalize_per_sample(pred, mask)
            target = self._normalize_per_sample(target, mask)
        lpc = self.recon_loss_fn(pred, target)
        return (lpc * mask.float()).sum() / mask.float().sum().clamp(min=1)

    def _shared_step(self, batch: Dict[str, torch.Tensor], stage: str) -> Dict:
        if "cpg_ids_v2" not in batch:
            raise ValueError("Recon-only arm needs view 2: run with wced_contrastive=true "
                             "(the collator builds view 2 only in contrastive mode).")
        if "labels" in batch:
            raise ValueError("Recon-only arm supports the WCEDCollator format only (bmfm_style=false).")

        all_betas  = batch["all_betas"]
        valid_mask = batch.get("valid_mask")
        recon_mask_v1 = ~batch["input_mask"]
        recon_mask_v2 = ~batch["input_mask_v2"]
        if valid_mask is not None:
            recon_mask_v1 = recon_mask_v1 & valid_mask
            recon_mask_v2 = recon_mask_v2 & valid_mask

        out_v1 = self.encode(batch["cpg_ids"], batch["beta_values"], batch.get("attention_mask"),
                             position_ids=batch.get("position_ids"))
        out_v2 = self.encode(batch["cpg_ids_v2"], batch["beta_values_v2"], batch.get("attention_mask_v2"),
                             position_ids=batch.get("position_ids_v2"))

        recon_loss_v1 = self._recon_loss(out_v1["predicted_betas"], all_betas, recon_mask_v1)
        recon_loss_v2 = self._recon_loss(out_v2["predicted_betas"], all_betas, recon_mask_v2)
        recon_loss = (recon_loss_v1 + recon_loss_v2) / 2
        zero = torch.tensor(0.0, device=recon_loss.device)
        if self.contrastive_weight > 0:
            # Same computation as the parent's two-view path.
            z1 = F.normalize(self.projection_head(out_v1["cls_embedding"]), dim=-1)
            z2 = F.normalize(self.projection_head(out_v2["cls_embedding"]), dim=-1)
            contrastive_loss = self.info_nce_loss(z1, z2)
        else:
            contrastive_loss = zero
        loss = recon_loss + self.contrastive_weight * contrastive_loss  # no age term (age_weight 0)

        if not loss.isfinite():
            raise ValueError(f"Non-finite loss={loss.item():.6f} at global_step={self.global_step}")

        self._monitor = self._cls_identity_monitor(out_v1["cls_embedding"], out_v2["cls_embedding"])

        predicted_betas = out_v1["predicted_betas"]
        with torch.no_grad():
            ni_pred   = predicted_betas[recon_mask_v1]
            ni_target = all_betas[recon_mask_v1]
            mae = torch.abs(ni_pred - ni_target).mean()
            metrics = {
                "mae": mae,
                "mse": ((ni_pred - ni_target) ** 2).mean(),
                "all_mae": torch.abs(predicted_betas - all_betas).mean(),
                "pcc": self._pearson_corr(ni_pred, ni_target),
                "valid_pct": recon_mask_v1.float().mean() * 100.0,
                "pred_mean": ni_pred.mean(),
                "pred_std": ni_pred.std().clamp(min=1e-8),
                "cpg_scale": self.encoder.embeddings.cpg_scale.abs().detach(),
            }

        return {
            "loss": loss,
            "recon_loss": recon_loss,
            "contrastive_loss": contrastive_loss,
            "age_loss": zero,
            "age_mae": zero,
            **metrics,
            "predicted_betas": predicted_betas,
            "target_betas": all_betas,
            "non_input_mask": recon_mask_v1,
            "cls_embedding": out_v1["cls_embedding"],
            "predicted_age": out_v1["predicted_age"],
        }

    # -------------------------------------------------------------------------
    # Logging of the monitor (parent hooks unchanged otherwise)
    # -------------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        for k, v in self._monitor.items():
            self.log(f"train/{k}", v, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = super().validation_step(batch, batch_idx)
        for k, v in self._monitor.items():
            self.log(f"validation/{k}", v, on_epoch=True, sync_dist=True)
        return loss

    # -------------------------------------------------------------------------
    # Analysis checkpoints at fixed epochs + optional stop (schedule untouched)
    # -------------------------------------------------------------------------

    def on_fit_start(self):
        # Arms that share a seed must start from identical weights: compare these lines across logs.
        with torch.no_grad():
            enc = sum(p.double().sum().item() for p in self.encoder.parameters())
            allp = sum(p.double().sum().item() for p in self.parameters())
        if self.trainer.is_global_zero:
            print(f"[objective_pilot] weights at fit start: encoder_sum={enc:.10e} all_sum={allp:.10e} "
                  f"lambda={self.contrastive_weight} global_step={self.global_step}", flush=True)

    def setup(self, stage=None):
        # Smoke runs only: cap the post-fit test pass (pretrain_llama runs it on the full held-out split).
        n = os.environ.get("LIMIT_TEST_BATCHES")
        if stage == "test" and n:
            self.trainer.limit_test_batches = int(n)

    def on_train_epoch_end(self):
        ep = self.current_epoch
        if torch.cuda.is_available() and self.trainer.is_global_zero:
            print(f"[objective_pilot] epoch {ep}: peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB "
                  f"of {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB "
                  f"(grad_checkpointing={self.grad_checkpointing})", flush=True)
        if ep in self.analysis_epochs and self.trainer.checkpoint_callback is not None:
            d = Path(self.trainer.checkpoint_callback.dirpath) / "analysis"
            if self.trainer.is_global_zero:
                d.mkdir(parents=True, exist_ok=True)
            self.trainer.strategy.barrier()
            self.trainer.save_checkpoint(str(d / f"epoch={ep}-analysis.ckpt"))  # all ranks call; rank 0 writes
        if 0 <= self.stop_after_epoch <= ep:
            logger.info(f"stop_after_epoch={self.stop_after_epoch} reached at epoch {ep}: stopping")
            self.trainer.should_stop = True


class WCEDLlamaReconOnlyModule(WCEDLlamaAblationModule):
    """The λ = 0 arm, with the weight forced to 0 whatever the config says."""

    def __init__(self, *args, **kwargs):
        kwargs["contrastive_weight"] = 0.0
        super().__init__(*args, **kwargs)


def configure_from_env() -> None:
    """Read ANALYSIS_EPOCHS ("0,1,2,...") and STOP_AFTER_EPOCH from the environment."""
    WCEDLlamaAblationModule.analysis_epochs = _parse_epochs(os.environ.get("ANALYSIS_EPOCHS", ""))
    WCEDLlamaAblationModule.stop_after_epoch = int(os.environ.get("STOP_AFTER_EPOCH", "-1"))
    WCEDLlamaAblationModule.grad_checkpointing = os.environ.get("ABLATION_GRAD_CKPT", "0") == "1"
