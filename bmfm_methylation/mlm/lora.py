"""
LoRA (Low-Rank Adaptation) for WCED fine-tuning.

Injects trainable low-rank adapter matrices into the frozen encoder's
attention projections. Only A and B matrices train (~0.4% of encoder params).

    W_out = W_original(x) + scaling * B(A(x))
    A: (in_features × rank)  — initialized kaiming
    B: (rank × out_features) — initialized zeros  → starts as identity

Reference: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models"
"""

import math
import logging
from typing import List, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with a frozen original weight
    and a trainable low-rank adapter.

        out = W(x) + (alpha/rank) * B(A(x))
    """

    def __init__(
        self,
        original: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.original = original
        self.rank = rank
        self.scaling = alpha / rank

        # Freeze original weights — LoRA never changes them
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

        in_features = original.in_features
        out_features = original.out_features

        # Low-rank adapter: A projects down, B projects back up
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Init: A ~ kaiming, B = 0 → LoRA output is zero at start (safe init)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_out = self.original(x)
        lora_out = self.lora_B(self.lora_dropout(self.lora_A(x)))
        return original_out + self.scaling * lora_out


def inject_lora(
    encoder: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    target_modules: Tuple[str, ...] = ("query", "value"),
    dropout: float = 0.0,
) -> int:
    """
    Inject LoRA adapters into all target Linear layers in the encoder.

    Steps:
      1. Find all Linear layers whose leaf name matches target_modules
      2. Replace each with a LoRALinear (original frozen + trainable A, B)
      3. Freeze ALL encoder params
      4. Unfreeze only LoRA A and B matrices

    Returns:
        Number of trainable LoRA parameters added.
    """
    # Collect replacements first (avoid modifying dict during iteration)
    replacements = []
    for name, module in encoder.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        leaf_name = name.split(".")[-1]
        if leaf_name not in target_modules:
            continue

        # Navigate to parent
        parts = name.split(".")
        parent = encoder
        for part in parts[:-1]:
            parent = getattr(parent, part)

        replacements.append((parent, parts[-1], module))

    if not replacements:
        raise ValueError(
            f"No Linear layers found matching target_modules={target_modules}. "
            f"Check module names with encoder.named_modules()."
        )

    # Apply replacements
    for parent, attr_name, original_linear in replacements:
        lora_layer = LoRALinear(original_linear, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent, attr_name, lora_layer)

    logger.info(f"Injected LoRA into {len(replacements)} layers: {[r[1] for r in replacements[:4]]}...")

    # Freeze ALL encoder params (including newly-added LoRA A, B)
    for param in encoder.parameters():
        param.requires_grad = False

    # Unfreeze only LoRA A and B
    lora_param_count = 0
    for module in encoder.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.weight.requires_grad = True
            module.lora_B.weight.requires_grad = True
            lora_param_count += (
                module.lora_A.weight.numel() + module.lora_B.weight.numel()
            )

    encoder_total = sum(p.numel() for p in encoder.parameters())
    logger.info(f"LoRA injection complete:")
    logger.info(f"  Layers with LoRA:  {len(replacements)}")
    logger.info(f"  LoRA params:       {lora_param_count:,} ({100*lora_param_count/encoder_total:.2f}% of encoder)")
    logger.info(f"  Frozen params:     {encoder_total - lora_param_count:,}")

    return lora_param_count


def get_lora_parameters(encoder: nn.Module) -> List[nn.Parameter]:
    """Return only the trainable LoRA parameters from the encoder."""
    params = []
    for module in encoder.modules():
        if isinstance(module, LoRALinear):
            params.append(module.lora_A.weight)
            params.append(module.lora_B.weight)
    return params
