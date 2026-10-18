"""
ScaleAdaptEncoder — Sinusoidal continuous value encoder.

Replaces the default 2-layer MLP beta encoder in SCBertModel embeddings.

Why: Beta values [0, 1] need high-resolution continuous encoding.
A 2-layer MLP can miss subtle differences (0.85 vs 0.87) that are
biologically meaningful for age prediction.

ScaleAdapt formula:
    features_k = [sin(x * freq_k), cos(x * freq_k)]  for k = 1..n_sin_basis
    output = Linear(2*n_sin_basis → hidden_size)(concat(features))

Trainable frequencies start from N(0, 2π * basis_scale) and are learned
during pretraining — the model decides which "resolutions" of beta variation matter.

Special tokens (x <= 0) are routed to a separate learned nn.Embedding table:
    x = -1  (MASK)  → index 0
    x = -2  (CLS)   → index 1
    x = -3  (SEP)   → index 2
    x = -4  (PAD)   → index 3
    x =  0  (zero)  → index 4  (when zero_as_special_token=True)

Usage (apply after model creation):
    from bmfm_methylation.scale_adapt import patch_scale_adapt_encoder
    patch_scale_adapt_encoder(embeddings_layer, hidden_size=512)
"""

import math
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ScaleAdaptEncoder(nn.Module):
    """
    Sinusoidal basis continuous value encoder.

    For each beta value x in [0, 1]:
        features = [sin(x*f_k), cos(x*f_k)] for k=1..n_sin_basis
        output   = Linear(2*n_sin_basis → hidden_size)(features)

    Special tokens (x <= 0 when zero_as_special_token=True, or x < 0 otherwise)
    are routed to a learned nn.Embedding table instead.

    Input:  [batch, seq_len]          — raw beta values + special token markers
    Output: [batch, seq_len, hidden_size]
    """

    def __init__(
        self,
        hidden_size: int,
        n_sin_basis: int = 48,
        basis_scale: float = 2.0,
        trainable: bool = True,
        zero_as_special_token: bool = False,  # False=MLM (beta=0 is real), True=WCED (0 means special)
        n_special_tokens: int = 8,
    ):
        """
        Args:
            hidden_size:           Output embedding dimension (must match encoder hidden_size)
            n_sin_basis:           Number of sinusoidal basis pairs. Total features = 2*n_sin_basis.
            basis_scale:           Frequency init scale. Frequencies ~ N(0, 2π * basis_scale).
                                   Upstream uses 1.5 for scRNA (range 0–10+).
                                   Use 2.0 for methylation beta values (range [0,1]) — tighter
                                   range needs higher initial frequencies to resolve fine differences.
                                   Frequencies are trainable so this only affects initialization.
            trainable:             If True, frequencies are learned nn.Parameters.
                                   If False, fixed geometric progression (2^k).
            zero_as_special_token: Controls how beta=0 is handled.
                                   False (MLM):  beta=0 is a real value (fully unmethylated CpG)
                                                 → encode with sinusoidal basis like any other value
                                   True  (WCED): WCED add_forward zeroes out CLS/special tokens
                                                 → beta=0 means "was a special token"
                                                 → route to learned special token embedding
            n_special_tokens:      Size of the special token embedding table.
                                   Must cover all negative token markers used.
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.n_sin_basis = n_sin_basis
        self.zero_as_special_token = zero_as_special_token
        self.n_special_tokens = n_special_tokens

        # Special token embedding table
        # Negative values are mapped: x=-1→0, x=-2→1, x=-3→2, x=-4→3
        # Zero is mapped: x=0→n_special_tokens-1 (when zero_as_special_token=True)
        self.special_token_embeddings = nn.Embedding(n_special_tokens, hidden_size)

        # Sinusoidal basis frequencies
        if trainable:
            init_freqs = torch.randn(n_sin_basis) * (2.0 * math.pi * basis_scale)
            self.basis = nn.Parameter(init_freqs)
        else:
            # Fixed geometric progression: 1, 2, 4, 8, ... (similar to Fourier features)
            fixed_freqs = torch.tensor([2.0 ** k for k in range(n_sin_basis)])
            self.register_buffer("basis", fixed_freqs)

        # Project [sin, cos] features → hidden_size (no bias — matches upstream)
        self.dense = nn.Linear(2 * n_sin_basis, hidden_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.dense.weight)
        nn.init.normal_(self.special_token_embeddings.weight, std=0.02)

    def _encode_continuous(self, x_flat: torch.Tensor) -> torch.Tensor:
        """
        x_flat: [N] — flat tensor of continuous beta values
        returns: [N, hidden_size]
        """
        # x_flat: [N] → [N, 1] * [n_sin_basis] → [N, n_sin_basis]
        angles = x_flat.unsqueeze(-1) * self.basis        # [N, n_sin_basis]
        features = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # [N, 2*n_sin_basis]
        return self.dense(features)                        # [N, hidden_size]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:       [batch, seq_len]          — beta values in [0,1] + special markers
        returns: [batch, seq_len, hidden_size]
        """
        batch_size, seq_length = x.shape
        x_flat = x.reshape(-1).float()                    # [batch*seq_len]

        # Identify special token positions
        if self.zero_as_special_token:
            special_mask = x_flat <= 0.0
        else:
            special_mask = x_flat < 0.0

        # Encode all positions with sinusoidal basis
        out = self._encode_continuous(x_flat)              # [N, hidden_size]

        # Overwrite special token positions with learned embeddings
        if special_mask.any():
            special_vals = x_flat[special_mask]

            # Map: -1→0 (MASK), -2→1 (CLS), -3→2 (SEP), -4→3 (PAD)
            indices = -(special_vals.long() + 1)

            # Map zero → last slot in the embedding table
            if self.zero_as_special_token:
                indices[special_vals == 0.0] = self.n_special_tokens - 1

            # Safety clamp — prevents index out of bounds
            indices = indices.clamp(0, self.n_special_tokens - 1)

            out[special_mask] = self.special_token_embeddings(indices).to(out.dtype)

        return out.reshape(batch_size, seq_length, self.hidden_size)


# ---------------------------------------------------------------------------
# Patch helpers — call these after creating the model
# ---------------------------------------------------------------------------

def patch_scale_adapt_encoder(
    embeddings_layer,
    hidden_size: int,
    n_sin_basis: int = 48,
    basis_scale: float = 2.0,
    trainable: bool = True,
    zero_as_special_token: bool = False,  # caller must set True for WCED, False for MLM
) -> bool:
    """
    Replace the MLP beta encoder on an SCBert embeddings layer with ScaleAdaptEncoder.

    Works for both MLM and WCED (the WCED add_forward patch uses
    embeddings_layer.beta_values_embeddings with the same interface).

    Args:
        embeddings_layer: model.scbert.embeddings  (or model.model.scbert.embeddings)
        hidden_size:       must match the model's hidden_size

    Returns:
        True if patched successfully, False if beta_values_embeddings not found.

    Example (MLM):
        patch_scale_adapt_encoder(model.model.scbert.embeddings, hidden_size=512)

    Example (WCED):
        patch_scale_adapt_encoder(wced_module.encoder.embeddings, hidden_size=512)
    """
    if not hasattr(embeddings_layer, "beta_values_embeddings"):
        logger.error("embeddings_layer has no beta_values_embeddings — cannot patch")
        return False

    scale_adapt = ScaleAdaptEncoder(
        hidden_size=hidden_size,
        n_sin_basis=n_sin_basis,
        basis_scale=basis_scale,
        trainable=trainable,
        zero_as_special_token=zero_as_special_token,
    )

    # Replace in-place — same attribute name so all existing forward calls work
    embeddings_layer.beta_values_embeddings = scale_adapt

    n_params = sum(p.numel() for p in scale_adapt.parameters())
    logger.info(
        f"ScaleAdaptEncoder applied: n_sin_basis={n_sin_basis}, "
        f"basis_scale={basis_scale}, trainable={trainable}, "
        f"zero_as_special_token={zero_as_special_token}, "
        f"params={n_params:,}"
    )
    return True
