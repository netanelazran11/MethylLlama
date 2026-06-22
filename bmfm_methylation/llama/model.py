"""
MethylLlama — LLaMA-style encoder for methylation (dual-field) data.

Key improvements over SCBertModel (BERT):
  1. RMSNorm: ~40% cheaper than LayerNorm (no mean subtraction, just RMS scale)
  2. Pre-LN: normalize BEFORE sublayers → stable training from step 1
  3. SwiGLU MLP: SiLU(gate) * up → down (gated activation, more expressive)
  4. RoPE: relative position encoding inside attention heads,
     NO absolute position embedding table (saves ~4M params for seq_len=8002)
  5. ScaleAdaptEncoder: trainable sinusoidal basis for beta [0,1]
     → finer resolution than 2-layer MLP; handles special tokens natively

Interface — identical to SCBertModel (drop-in replacement):
  model(input_ids=[B,2,L], attention_mask=[B,L])
    → BaseModelOutputWithPoolingAndCrossAttentions
         .last_hidden_state: [B, L, D]
         .pooler_output:     [B, D]   (CLS after linear+tanh)

Embedding fusion (built-in; no monkey-patching needed):
  h = cpg_scale * cpg_embed(cpg_ids) + ScaleAdapt(beta_values)
  (no additive position embedding — RoPE handles positions in attention)

ScaleAdaptEncoder special-token routing (zero_as_special_token=False):
  beta <  0   → learned nn.Embedding (index = -(beta+1))
                 -1 → idx 0 (MASK), -2 → idx 1 (CLS), -3 → idx 2 (SEP), -4 → idx 3 (PAD)
  beta >= 0   → sinusoidal basis: [sin(beta*f_k), cos(beta*f_k)] → Linear
                 beta=0 is real (unmethylated CpG, not a special token)

Parameter count comparison (6 layers, 512D):
  SCBertModel: ~23M (includes ~4M absolute pos embedding table for 8002 positions)
  MethylLlama: ~19M (RoPE removes pos table; SwiGLU 3-matrix FFN similar to 2-matrix GELU)
"""

import math
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

# ScaleAdaptEncoder lives in the parent package (bmfm_methylation/scale_adapt.py)
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from .scale_adapt import ScaleAdaptEncoder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MethylLlamaConfig:
    """
    Configuration for MethylLlamaModel.

    Design notes:
    - intermediate_size = 1408: SwiGLU uses 3 linear projections (gate, up, down)
        BERT FFN: 2 × 512 × 2048 = 2.1M params per layer
        SwiGLU:   3 × 512 × 1408 = 2.2M params per layer (≈ same)
        Formula: round(2/3 × 4 × hidden_size / 64) × 64
                 = round(1365/64) × 64 = 22 × 64 = 1408

    - basis_scale=2.0 vs upstream 1.5:
        Upstream (biomed-multi-omic) uses 1.5 for scRNA (expression range 0–10+).
        Methylation beta values are [0,1] — a tighter range needs higher initial
        frequencies to resolve fine differences (e.g. 0.85 vs 0.87).
        Frequencies are trainable so this only affects initialization.

    - cpg_scale_init=0.1:
        Start with beta encoder dominating (cpg_scale small) so the model
        focuses on methylation level first. Scale is learned.
    """
    # Core dimensions
    hidden_size: int = 512
    num_hidden_layers: int = 6
    num_attention_heads: int = 8
    intermediate_size: int = 1408     # SwiGLU: 3 × 512 × 1408 ≈ BERT 2 × 512 × 2048

    # Vocabulary (CpG sites + special tokens)
    # = n_cpg_sites + 5 special tokens (UNK=0, SEP=1, PAD=2, CLS=3, MASK=4)
    # For 8000 CpGs: vocab_size = 8005
    # Must match the tokenizer created by create_methylation_multifield_tokenizer
    # NOTE: max_position_embeddings=8002 in SCBert YAML is sequence length, not vocab!
    vocab_size: int = 8005

    # Rotary Position Embedding
    rope_theta: float = 10000.0
    max_seq_len: int = 16384          # Upper bound for RoPE cache (covers 49k×0.25+1=12290)

    # Normalization / Regularization
    rms_norm_eps: float = 1e-6
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.0   # SDPA manages attention dropout

    # ScaleAdaptEncoder (beta value encoder)
    n_sin_basis: int = 48             # Basis pairs → 96 total features → Linear(96→512)
    basis_scale: float = 2.0         # Freq init: N(0, 2π × basis_scale); 2.0 for [0,1] range
    scale_adapt_trainable: bool = True

    # CpG-scale initial value
    cpg_scale_init: float = 0.1      # Small init; learned during training

    # CLS pooling
    add_pooling_layer: bool = True

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Simpler than LayerNorm: no mean subtraction, just normalize by RMS.
    ~40% faster in practice (used in LLaMA, Mistral, Gemma).

    Formula: x / RMS(x) * weight
    """
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Work in float32 for numerical stability, then cast back
        x_float = x.float()
        rms = x_float.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return (x_float / rms * self.weight.float()).to(x.dtype)


# ---------------------------------------------------------------------------
# Rotary Position Embedding (RoPE)
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (Su et al., 2021).

    Encodes position by rotating Q and K vectors in 2D frequency planes.
    No learned parameters — purely deterministic frequency table.

    Advantage for 8k methylation sequences:
    - No position embedding lookup table (saves vocab_size × hidden_size ≈ 4M params)
    - Relative position signal generalizes to unseen sequence lengths
    - Applies to every head independently (each head learns different position sensitivity)
    """
    def __init__(self, dim: int, max_seq_len: int = 8192, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        # Inverse frequencies: [dim/2]  (one freq per rotation plane)
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._precompute_table(max_seq_len)

    def _precompute_table(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)   # [seq_len, dim/2]
        emb = torch.cat([freqs, freqs], dim=-1) # [seq_len, dim] — duplicate for sin and cos
        self.register_buffer("cos_table", emb.cos(), persistent=False)
        self.register_buffer("sin_table", emb.sin(), persistent=False)

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) tables for positions 0..seq_len-1."""
        if seq_len > self.cos_table.shape[0]:
            self._precompute_table(seq_len)
        return self.cos_table[:seq_len], self.sin_table[:seq_len]  # [seq_len, dim]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate vector by splitting in half: [a, b] → [-b, a]."""
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE to query and key tensors.

    Args:
        q, k:     [B, H, L, Dh]
        cos, sin: [L, Dh]         (from RotaryEmbedding)
    Returns:
        q_rot, k_rot: [B, H, L, Dh]
    """
    # [L, Dh] → [1, 1, L, Dh] for broadcasting
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot


# ---------------------------------------------------------------------------
# Multi-Head Self-Attention with RoPE
# ---------------------------------------------------------------------------

class MethylLlamaAttention(nn.Module):
    """
    Multi-head self-attention with Rotary Position Embedding.

    Uses PyTorch F.scaled_dot_product_attention which automatically uses:
    - Flash Attention 2 when flash_attn is installed and on CUDA
    - Efficient attention (xFormers) on CUDA when available
    - Standard SDPA otherwise (still O(n²) but with fused operations)

    No bias terms in Q/K/V/O projections (LLaMA convention, saves params).
    """
    def __init__(self, config: MethylLlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.attn_dropout = config.attention_dropout_prob

        assert self.hidden_size % self.num_heads == 0, (
            f"hidden_size ({self.hidden_size}) must be divisible by "
            f"num_attention_heads ({self.num_heads})"
        )

        # QKV projections (no bias — matches LLaMA convention)
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # RoPE (applied to Q and K)
        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,            # [B, L, D]
        attention_mask: Optional[torch.Tensor] = None,  # [B, L] int 1=attend, 0=pad
    ) -> torch.Tensor:
        B, L, D = hidden_states.shape

        # Project to Q, K, V
        q = self.q_proj(hidden_states)  # [B, L, D]
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape: [B, L, D] → [B, H, L, Dh]
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K (encodes position without lookup table)
        cos, sin = self.rotary_emb(L)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Prepare SDPA attention mask
        # Input:  [B, L] with 1=attend, 0=pad
        # SDPA boolean: True=attend, False=block (mask out)
        sdpa_mask = None
        if attention_mask is not None:
            # [B, L] → [B, 1, 1, L] broadcasts to [B, H, Lq, Lk]
            sdpa_mask = attention_mask.bool().unsqueeze(1).unsqueeze(2)

        # Scaled dot-product attention (automatically uses Flash Attention when available)
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=sdpa_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )  # [B, H, L, Dh]

        # Reshape back: [B, H, L, Dh] → [B, L, D]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(attn_out)


# ---------------------------------------------------------------------------
# SwiGLU Feed-Forward Network
# ---------------------------------------------------------------------------

class MethylLlamaMLP(nn.Module):
    """
    SwiGLU Feed-Forward Network (used in LLaMA, PaLM, Gemma).

    Formula: FFN(x) = W_down(SiLU(W_gate(x)) ⊙ W_up(x))

    Why better than GELU-FFN:
    - Gated mechanism: gate controls information flow per-dimension
    - SiLU (Sigmoid Linear Unit) is smooth and avoids dying-ReLU
    - Same parameter count as BERT FFN when intermediate_size ≈ 2/3 × 4 × hidden_size

    No bias terms (LLaMA convention).
    """
    def __init__(self, config: MethylLlamaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj   = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: gate the up-projection with SiLU-activated gate-projection
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Transformer Layer (Pre-LN)
# ---------------------------------------------------------------------------

class MethylLlamaLayer(nn.Module):
    """
    Single Transformer layer with Pre-LN (Layer Norm before each sublayer).

    Pre-LN layout:
        h = h + Attn(RMSNorm(h))   ← attention block
        h = h + MLP(RMSNorm(h))    ← FFN block

    Pre-LN vs Post-LN (standard BERT):
    - Pre-LN: gradient magnitude is stable from step 1 → no warmup needed
    - Post-LN: gradient can explode/vanish early → needs careful LR warmup
    - Pre-LN: normalized activations throughout → faster convergence
    """
    def __init__(self, config: MethylLlamaConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn      = MethylLlamaAttention(config)
        self.mlp_norm  = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp       = MethylLlamaMLP(config)
        self.dropout   = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Attention block (Pre-LN)
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states = self.attn(hidden_states, attention_mask=attention_mask)
        hidden_states = self.dropout(hidden_states) + residual

        # MLP block (Pre-LN)
        residual = hidden_states
        hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.dropout(hidden_states) + residual

        return hidden_states


# ---------------------------------------------------------------------------
# Dual-Field Embeddings (no absolute position table)
# ---------------------------------------------------------------------------

class MethylLlamaEmbeddings(nn.Module):
    """
    Dual-field embedding layer for methylation data.

    Input:  input_ids [B, 2, L]
              Field 0: CpG token IDs   [B, L]  (long int, site identity)
              Field 1: Beta values     [B, L]  (float, methylation level + special markers)

    Output: [B, L, hidden_size]

    Fusion:
        h = cpg_scale * cpg_embed(cpg_ids) + ScaleAdapt(beta_values)

    No absolute position embedding table. Positions encoded via RoPE in attention.
    This saves ~4M parameters for max_seq_len=8002, hidden_size=512.

    ScaleAdaptEncoder handles all token types:
        beta  < 0 → special token (MASK/CLS/SEP/PAD): learned nn.Embedding
        beta >= 0 → real methylation value: sinusoidal basis → Linear(96→512)
        (zero_as_special_token=False so beta=0.0 is real unmethylated CpG)
    """
    def __init__(self, config: MethylLlamaConfig):
        super().__init__()

        # CpG site identity embeddings
        self.cpg_sites_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)

        # Beta value encoder: ScaleAdaptEncoder (from bmfm_methylation/scale_adapt.py)
        # zero_as_special_token=False: beta=0 is real (unmethylated CpG), beta<0 is special
        self.beta_values_embeddings = ScaleAdaptEncoder(
            hidden_size=config.hidden_size,
            n_sin_basis=config.n_sin_basis,
            basis_scale=config.basis_scale,
            trainable=config.scale_adapt_trainable,
            zero_as_special_token=False,
            n_special_tokens=8,  # Covers MASK(-1), CLS(-2), SEP(-3), PAD(-4) + spares
        )

        # Learnable scale for CpG embeddings
        # Small init: beta encoder dominates early, learned during training
        self.cpg_scale = nn.Parameter(torch.tensor(float(config.cpg_scale_init)))

        # Post-embedding normalization (RMSNorm — consistent with the rest of the model)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cpg_sites_embeddings.weight, std=0.02)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,        # [B, 2, L]
        inputs_embeds: Optional[torch.Tensor] = None,    # pre-computed embeddings
    ) -> torch.Tensor:
        # Allow bypassing embedding computation (e.g., for probing)
        if inputs_embeds is not None:
            return inputs_embeds

        B, num_fields, L = input_ids.shape
        assert num_fields == 2, (
            f"Expected dual-field input [B, 2, L], got [B, {num_fields}, L]. "
            f"Field 0 = cpg_ids, Field 1 = beta_values."
        )

        # Split dual-field input
        cpg_ids    = input_ids[:, 0, :].long()   # [B, L]
        beta_values = input_ids[:, 1, :].float()  # [B, L]

        # CpG ID bounds check (catches tokenization bugs before silent embedding lookup errors)
        max_id = cpg_ids.max()
        if max_id >= self.cpg_sites_embeddings.num_embeddings:
            raise ValueError(
                f"CpG ID out of vocabulary range: max_id={max_id.item()}, "
                f"vocab_size={self.cpg_sites_embeddings.num_embeddings}. "
                f"Check probe_ids_csv or tokenizer."
            )

        # CpG site identity embeddings: [B, L] → [B, L, D]
        cpg_embeds = self.cpg_sites_embeddings(cpg_ids)

        # Beta value embeddings: [B, L] → [B, L, D]
        # ScaleAdaptEncoder handles special markers (negative values) internally
        beta_embeds = self.beta_values_embeddings(beta_values)

        # Fusion: cpg_scale * site_embed + beta_embed
        # cpg_scale learned from 0.1 init → balances identity vs. methylation level
        hidden_states = self.cpg_scale * cpg_embeds + beta_embeds  # [B, L, D]

        # Normalize and apply dropout
        hidden_states = self.norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states  # [B, L, D]


# ---------------------------------------------------------------------------
# Encoder (stack of layers + final norm)
# ---------------------------------------------------------------------------

class MethylLlamaEncoder(nn.Module):
    """Stack of MethylLlamaLayers with final RMSNorm."""
    def __init__(self, config: MethylLlamaConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            MethylLlamaLayer(config) for _ in range(config.num_hidden_layers)
        ])
        # Final norm after all layers (LLaMA-style: output is well-normalized)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)
        return self.norm(hidden_states)


# ---------------------------------------------------------------------------
# Pooler (CLS token → global representation)
# ---------------------------------------------------------------------------

class MethylLlamaPooler(nn.Module):
    """
    CLS-token pooler: Linear(CLS) → Tanh.

    Identical interface to BERT's BertPooler.
    Used for WCED: pooler_output → decoder → all_betas reconstruction.

    During WCED pretraining, the reconstruction + age losses force CLS to
    encode a global summary of the full methylation profile.
    """
    def __init__(self, config: MethylLlamaConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # CLS token is always at position 0
        cls_token = hidden_states[:, 0, :]  # [B, D]
        return self.activation(self.dense(cls_token))  # [B, D]


# ---------------------------------------------------------------------------
# Complete Model (SCBertModel-compatible interface)
# ---------------------------------------------------------------------------

class MethylLlamaModel(nn.Module):
    """
    LLaMA-style encoder for methylation data.

    Drop-in replacement for SCBertModel in WCED and MLM pipelines.

    Architecture:
        MethylLlamaEmbeddings → MethylLlamaEncoder (6× layers) → MethylLlamaPooler
        Each layer: RMSNorm + MHA(RoPE) + RMSNorm + SwiGLU (Pre-LN)

    Input:
        input_ids:      [B, 2, L]  (field 0: cpg_ids, field 1: beta_values)
        attention_mask: [B, L]     (1=attend, 0=pad)

    Output: BaseModelOutputWithPoolingAndCrossAttentions
        .last_hidden_state: [B, L, D]
        .pooler_output:     [B, D]  (CLS after linear+tanh; None if add_pooling_layer=False)
    """
    def __init__(self, config: MethylLlamaConfig):
        super().__init__()
        self.config = config
        self.embeddings = MethylLlamaEmbeddings(config)
        self.encoder    = MethylLlamaEncoder(config)
        self.pooler     = MethylLlamaPooler(config) if config.add_pooling_layer else None

        # Log parameter count
        n_params = sum(p.numel() for p in self.parameters())
        n_embed  = sum(p.numel() for p in self.embeddings.parameters())
        n_enc    = sum(p.numel() for p in self.encoder.parameters())
        logger.info(
            f"MethylLlamaModel initialized: "
            f"{config.num_hidden_layers}L × {config.hidden_size}D × {config.num_attention_heads}H, "
            f"intermediate={config.intermediate_size}, "
            f"total_params={n_params:,} "
            f"(embed={n_embed:,}, encoder={n_enc:,})"
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,        # [B, 2, L]
        attention_mask: Optional[torch.Tensor] = None,   # [B, L]
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,   # Accept extra BERT kwargs (token_type_ids, etc.) silently
    ) -> BaseModelOutputWithPoolingAndCrossAttentions:

        # Embed the dual-field input
        hidden_states = self.embeddings(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
        )  # [B, L, D]

        # Encode through transformer layers
        hidden_states = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
        )  # [B, L, D]

        # Pool CLS token
        pooler_output = (
            self.pooler(hidden_states) if self.pooler is not None else None
        )  # [B, D] or None

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=hidden_states,
            pooler_output=pooler_output,
        )


# ---------------------------------------------------------------------------
# Convenience: build model from common presets
# ---------------------------------------------------------------------------

def init_cpg_embeddings_from_dna(
    model: "MethylLlamaModel",
    tokenizer,
    npy_path: str,
    ids_path: str,
) -> int:
    """
    Initialize MethylLlamaModel's CpG embedding table with BMFM-DNA embeddings.

    Maps each CpG name in ids_path → token ID via tokenizer → overwrites the
    corresponding row in cpg_sites_embeddings.weight with the BMFM-DNA vector.

    Special tokens (rows 0-4) are never touched.

    Args:
        model:      MethylLlamaModel (must be on CPU or have matching dtype)
        tokenizer:  MultiFieldTokenizer (must have "cpg_sites" sub-tokenizer)
        npy_path:   Path to cpg_embeddings_bmfdna_21k.npy  [N, hidden_size]
        ids_path:   Path to cpg_ids_order.txt              (N CpG names, one per line)

    Returns:
        Number of CpGs successfully initialized.
    """
    import numpy as np

    emb = np.load(npy_path).astype(np.float32)          # [N, hidden_size]
    cpg_names = open(ids_path).read().splitlines()       # N CpG names

    assert emb.shape[0] == len(cpg_names), (
        f"Shape mismatch: npy has {emb.shape[0]} rows but ids_path has {len(cpg_names)} names"
    )
    assert emb.shape[1] == model.config.hidden_size, (
        f"Embedding dim mismatch: BMFM-DNA={emb.shape[1]}, model hidden_size={model.config.hidden_size}"
    )

    cpg_tokenizer = tokenizer.tokenizers["cpg_sites"]
    vocab = cpg_tokenizer.get_vocab()

    weight = model.embeddings.cpg_sites_embeddings.weight.data
    n_init = 0
    for i, name in enumerate(cpg_names):
        token_id = vocab.get(name)
        if token_id is not None and token_id >= 5:   # never overwrite special tokens
            weight[token_id] = torch.tensor(emb[i], dtype=weight.dtype)
            n_init += 1

    logger.info(
        f"BMFM-DNA init: {n_init}/{len(cpg_names)} CpG embeddings loaded "
        f"({len(cpg_names) - n_init} skipped — not in tokenizer vocab)"
    )
    return n_init


def build_methyl_llama(
    vocab_size: int = 8002,
    hidden_size: int = 512,
    num_hidden_layers: int = 6,
    num_attention_heads: int = 8,
    intermediate_size: int = 1408,
    n_sin_basis: int = 48,
    basis_scale: float = 2.0,
    add_pooling_layer: bool = True,
    **kwargs,
) -> MethylLlamaModel:
    """Build a MethylLlamaModel with explicit parameters."""
    config = MethylLlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        n_sin_basis=n_sin_basis,
        basis_scale=basis_scale,
        add_pooling_layer=add_pooling_layer,
        **kwargs,
    )
    return MethylLlamaModel(config)
