#!/usr/bin/env python3
"""
Task C-1 — Data Efficiency: few-shot learning curves.

Question: How many labeled samples does the WCED-pretrained model need
          to reach near-SOTA performance vs. random initialization?

Experiment:
  For N in [10, 25, 50, 100, 250, 500, 1000, "all"]:
    Train SmokingClassifier (or AgeRegressor) with N labeled training samples.
    Evaluate on the FULL validation/test set (always fixed).
  Repeat for WCED-pretrained vs. random-init encoder.
  → Plot val accuracy / MAE vs N — the gap IS your pretraining contribution.

Usage:
  python -m bmfm_methylation.downstream.probing.data_efficiency \
      --checkpoint_path /path/to/wced.ckpt \
      --data_path /path/to/smoking.h5ad \
      --task smoking \
      --output_dir ./outputs/downstream/probing/data_efficiency
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

N_SAMPLES_LIST = [10, 25, 50, 100, 250, 500, 1000, None]  # None = all
N_SEEDS = 3          # average over seeds to reduce variance at small N
TARGET_STEPS = 3000  # fixed gradient steps regardless of N — ensures small N gets enough updates
VALIDATE_EVERY = 500 # validate every N steps (6 checkpoints per run)
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
SUBSET_K = 4000


# ─────────────────────────────────────────────────────────────────────────────
# Tiny training loop (no Lightning overhead for speed)
# ─────────────────────────────────────────────────────────────────────────────

def _make_loader(dataset, shuffle, batch_size=BATCH_SIZE):
    from bmfm_methylation.downstream.shared.classification_data_module import _collate_classification
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        collate_fn=_collate_classification, pin_memory=False, num_workers=0,
    )


def _encode(encoder, batch, device):
    cpg_ids = batch["cpg_ids"].to(device)
    beta_values = batch["beta_values"].to(device)
    attn = batch.get("attention_mask", None)
    input_ids = torch.stack([cpg_ids.float(), beta_values], dim=1)
    bs, _, seq = input_ids.shape
    if attn is not None:
        attn = attn.to(device)
        if attn.dim() == 3:
            attn = attn[:, 0, :]
    else:
        attn = torch.ones(bs, seq, device=device)
    return encoder(input_ids, attention_mask=attn).pooler_output


def _validate(head, encoder, val_dl, task, device, age_mean=0.0, age_std=1.0):
    head.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_dl:
            cls = _encode(encoder, batch, device)
            all_preds.append(head(cls).cpu())
            all_labels.append(batch["class_label"].cpu())
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    if task == "smoking":
        return (preds.argmax(1) == labels).float().mean().item()
    else:
        # MAE in original years
        pred_years = preds.squeeze(-1) * age_std + age_mean
        return (pred_years - labels.float()).abs().mean().item()


def _train_one_run(encoder, head, train_ds, val_ds, task, device,
                   n_steps=TARGET_STEPS, age_mean=0.0, age_std=1.0):
    """Train head (encoder frozen) for exactly n_steps gradient steps.

    Using fixed steps (not fixed epochs) ensures small-N experiments get
    enough gradient updates: with N=10 and epoch-based training, 50 epochs
    = 50 steps, which is far too few for the MLP head to converge.
    """
    head = head.to(device)
    encoder = encoder.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    opt = torch.optim.AdamW(head.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    train_dl = _make_loader(train_ds, shuffle=True)
    val_dl = _make_loader(val_ds, shuffle=False)

    best_metric = -np.inf if task == "smoking" else np.inf
    validate_every = max(100, n_steps // 6)

    step = 0
    train_iter = iter(train_dl)

    while step < n_steps:
        head.train()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dl)
            batch = next(train_iter)

        opt.zero_grad()
        with torch.no_grad():
            cls = _encode(encoder, batch, device)

        if task == "smoking":
            loss = F.cross_entropy(head(cls), batch["class_label"].to(device))
        else:
            raw = batch["class_label"].float().to(device)
            normalized = (raw - age_mean) / age_std
            loss = F.mse_loss(head(cls).squeeze(-1), normalized)

        loss.backward()
        opt.step()
        step += 1

        if step % validate_every == 0 or step == n_steps:
            metric = _validate(head, encoder, val_dl, task, device, age_mean, age_std)
            if task == "smoking":
                if metric > best_metric:
                    best_metric = metric
            else:
                if metric < best_metric:
                    best_metric = metric

    return best_metric


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


def _load_encoder(checkpoint_path, random_init=False):
    """Load WCED encoder or create a random-init copy."""
    import torch.serialization
    from bmfm_targets.config import SCBertConfig, TrainerConfig, FieldInfo
    from bmfm_methylation.wced.wced_module import WCEDTrainingModule
    from bmfm_methylation.shared.config import PretrainingConfig

    torch.serialization.add_safe_globals([SCBertConfig, TrainerConfig, FieldInfo])
    model_config = _build_encoder_config()

    if random_init:
        from bmfm_targets.models.predictive.scbert.modeling_scbert import SCBertModel
        encoder = SCBertModel(model_config)
        logger.info("Random-init encoder created")
        return encoder

    pt = WCEDTrainingModule.load_from_checkpoint(
        checkpoint_path,
        model_config=model_config,
        pretrain_config=PretrainingConfig(mode="wced"),
    )
    logger.info(f"WCED encoder loaded from {checkpoint_path}")
    return pt.encoder


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import torch
    import torch.serialization
    _orig = torch.load
    def _p(*a, **kw):
        kw["weights_only"] = False
        return _orig(*a, **kw)
    torch.load = _p
    torch.serialization.load = _p

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--task", default="smoking", choices=["smoking", "age"])
    parser.add_argument("--output_dir", default="./outputs/downstream/probing/data_efficiency")
    parser.add_argument("--n_steps", type=int, default=TARGET_STEPS,
                        help="Fixed gradient steps per run (replaces epoch-based training)")
    parser.add_argument("--n_epochs", type=int, default=None,
                        help="Deprecated: use --n_steps instead")
    parser.add_argument("--n_seeds", type=int, default=N_SEEDS)
    parser.add_argument("--init_type", default="both", choices=["wced_pretrained", "random_init", "both"],
                        help="Which encoder init to evaluate (split jobs to fit time limits)")
    args = parser.parse_args()

    # n_epochs is kept for backward compat but n_steps takes precedence
    n_steps = args.n_steps

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Training: {n_steps} fixed steps per run, {args.n_seeds} seeds")

    # ── Build datasets ────────────────────────────────────────────────────────
    from bmfm_methylation.downstream.shared.classification_data_module import (
        ClassificationDataset, SMOKING_LABEL_MAP,
    )

    label_col = "smoking_status" if args.task == "smoking" else "age"
    label_map = SMOKING_LABEL_MAP if args.task == "smoking" else None

    full_train = ClassificationDataset(
        args.data_path, "train", label_col=label_col, label_map=label_map,
        subset_k=SUBSET_K, fixed_subset=False,
    )
    val_ds = ClassificationDataset(
        args.data_path, "valid", label_col=label_col, label_map=label_map,
        subset_k=SUBSET_K, fixed_subset=True,
    )
    test_ds = ClassificationDataset(
        args.data_path, "test", label_col=label_col, label_map=label_map,
        subset_k=SUBSET_K, fixed_subset=True,
    )

    logger.info(f"Full train: {len(full_train)}, val: {len(val_ds)}, test: {len(test_ds)}")

    # ── Age normalization (z-score from full training set) ────────────────────
    age_mean, age_std = 0.0, 1.0
    if args.task == "age":
        all_ages = full_train.labels.astype(np.float32)
        age_mean = float(all_ages.mean())
        age_std = float(all_ages.std()) + 1e-8
        logger.info(f"Age normalization: mean={age_mean:.1f}yr, std={age_std:.1f}yr")

    n_classes = 3 if args.task == "smoking" else 1
    hidden = 512  # matches pretrain config

    results = []
    init_types = ["wced_pretrained", "random_init"] if args.init_type == "both" else [args.init_type]

    for init_type in init_types:
        logger.info(f"\n{'='*50}")
        logger.info(f"Init: {init_type}")

        for n_train in N_SAMPLES_LIST:
            label = "all" if n_train is None else str(n_train)
            actual_n = len(full_train) if n_train is None else min(n_train, len(full_train))
            logger.info(f"  N={label} ({actual_n} samples)")

            seed_metrics = []
            for seed in range(args.n_seeds):
                # Subsample training set
                rng = np.random.default_rng(seed)
                if n_train is None or actual_n == len(full_train):
                    train_ds = full_train
                else:
                    # Per-class balanced subsampling for small N
                    indices = []
                    per_class = max(1, n_train // n_classes)
                    for cls_id in range(n_classes):
                        cls_idx = np.where(full_train.labels == cls_id)[0]
                        chosen = rng.choice(cls_idx, min(per_class, len(cls_idx)), replace=False)
                        indices.extend(chosen.tolist())
                    indices = sorted(indices)
                    from torch.utils.data import Subset
                    train_ds = Subset(full_train, indices)

                encoder = _load_encoder(
                    args.checkpoint_path,
                    random_init=(init_type == "random_init"),
                )
                encoder.eval()

                if args.task == "smoking":
                    head = nn.Linear(hidden, n_classes)
                else:
                    head = nn.Sequential(nn.Linear(hidden, 64), nn.GELU(), nn.Linear(64, 1))

                metric = _train_one_run(encoder, head, train_ds, val_ds, args.task, device,
                                       n_steps=n_steps, age_mean=age_mean, age_std=age_std)
                seed_metrics.append(metric)
                logger.info(f"    seed={seed} → val_metric={metric:.4f}")

            mean_m = float(np.mean(seed_metrics))
            std_m = float(np.std(seed_metrics))
            results.append({
                "init": init_type,
                "n_train": label,
                "n_train_int": actual_n,
                "val_metric_mean": mean_m,
                "val_metric_std": std_m,
                "task": args.task,
                "metric_name": "accuracy" if args.task == "smoking" else "mae",
            })
            logger.info(f"  N={label} | {init_type} | val={mean_m:.4f} ± {std_m:.4f}")

    df = pd.DataFrame(results)
    csv_path = output_dir / "data_efficiency_results.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Results saved to {csv_path}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = {"wced_pretrained": "#2196F3", "random_init": "#FF5722"}

        for init_type in ["wced_pretrained", "random_init"]:
            sub = df[df["init"] == init_type].sort_values("n_train_int")
            xs = sub["n_train_int"].values
            ys = sub["val_metric_mean"].values
            errs = sub["val_metric_std"].values
            label = "WCED pretrained" if init_type == "wced_pretrained" else "Random init"
            ax.plot(xs, ys, "o-", label=label, color=colors[init_type])
            ax.fill_between(xs, ys - errs, ys + errs, alpha=0.2, color=colors[init_type])

        ax.set_xscale("log")
        ax.set_xlabel("Number of labeled training samples")
        metric_name = "Accuracy" if args.task == "smoking" else "MAE (years)"
        ax.set_ylabel(f"Validation {metric_name}")
        ax.set_title(f"Data Efficiency — {args.task} prediction")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig_path = output_dir / "data_efficiency_curve.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        logger.info(f"Plot saved to {fig_path}")
    except ImportError:
        logger.warning("matplotlib not available, skipping plot")


if __name__ == "__main__":
    main()
