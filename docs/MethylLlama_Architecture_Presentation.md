# MethylLlama: LLaMA-Style Encoder for DNA Methylation Age Prediction

---

## 1. Data Overview

### Pretraining Dataset

| Property | Value |
|---|---|
| Samples | ~169,000 |
| CpG sites per sample | 49,156 |
| Total tokens | ~8.3 billion |
| Source | Public methylation arrays (GSE datasets) |
| Input format | Dual-field: (CpG ID, beta value) per position |
| Sequence length | 24,580 (CLS + 50% of 49,156 CpGs) |

**No data leakage in pretraining.** The pretraining set contains completely different samples from the downstream fine-tuning set. The model never sees AltumAge samples during pretraining.

### Fine-tuning Dataset (AltumAge)

| Property | Value |
|---|---|
| Samples | ~11,500 |
| CpG sites measured | 21,368 |
| CpG sites in pretrain vocab | 19,608 |
| CpG sites NOT in pretrain | 1,760 (excluded) |
| Final CpGs used for fine-tuning | **19,608** |
| Target | Chronological age (years) |

**Why 19,608 CpGs?**
The pretrain tokenizer was built on 49,156 CpG sites.
AltumAge measures 21,368 of these, but 1,760 are NOT in the pretrain vocabulary.
We use all 19,608 that are in both — **no feature selection needed, no data leakage**.

**Why not the original 8,000 CpG subset?**
The 8k CpGs were selected by age correlation across the entire AltumAge dataset including the test set → **data leakage**. Using all 19,608 in-vocab CpGs eliminates this bias entirely.

---

## 2. Architecture Overview

```
Input: [B, 2, L]  ← (CpG_IDs, Beta_values) dual-field
         │
         ▼
┌─────────────────────────────────────────┐
│  MethylLlamaEmbeddings                  │
│  h = cpg_scale × CpgEmbed(ids)          │
│    + ScaleAdaptEncoder(betas)           │
│  → RMSNorm → Dropout                   │
└─────────────────────────────────────────┘
         │  [B, L, 768]
         ▼
┌─────────────────────────────────────────┐
│  MethylLlamaLayer × 8                   │
│  ┌──────────────────────────────────┐   │
│  │ h = h + Attn(RMSNorm(h))  [RoPE] │   │  Pre-LN
│  │ h = h + MLP(RMSNorm(h)) [SwiGLU] │   │
│  └──────────────────────────────────┘   │
└─────────────────────────────────────────┘
         │  [B, L, 768]
         ▼
    RMSNorm (final)
         │
         ▼
    Pooler: Linear(768→768) → Tanh
         │  [B, 768]  ← CLS embedding
         ▼
┌─────────────────────────────────────────┐
│  WCED Heads (pretraining only)          │
│  ├── WCEDDecoder → [B, 49,156] (betas)  │
│  ├── ProjectionHead → [B, 128] (InfoNCE)│
│  └── AgeHead → [B, 1] (auxiliary age)  │
└─────────────────────────────────────────┘
```

---

## 3. Model Configuration (Actual Run)

```yaml
# pretrain_llama.yaml — what actually runs
hidden_size:          768      # Embedding dimension D
num_hidden_layers:    8        # Transformer depth
num_attention_heads:  12       # Attention heads H
head_dim:             64       # D / H = 768 / 12

intermediate_size:    2048     # SwiGLU hidden dim
# Formula: round(2/3 × 4 × 768 / 64) × 64 = 32 × 64 = 2048
# Same parameter count as BERT FFN (2 × 768 × 3072)

vocab_size:           49,161   # 49,156 CpG sites + 5 special tokens
max_seq_len:          24,580   # CLS + 50% of 49,156 CpGs

n_sin_basis:          48       # ScaleAdapt basis pairs (96 features total)
basis_scale:          2.0      # Freq init N(0, 2π × 2.0) for [0,1] range
cpg_scale_init:       0.1      # Small init: beta encoder dominates first

rope_theta:           10,000   # RoPE base frequency
rms_norm_eps:         1e-6
hidden_dropout_prob:  0.1

# Training
batch_size:           8        # per GPU
accumulate_grad_batches: 8
# Effective batch: 8 × 8 × 2 GPUs = 128 samples
learning_rate:        3e-4
warmup_steps:         1,000
betas:                [0.9, 0.99]
precision:            16-mixed
pretrain_epochs:      300
```

**Parameter count estimate:**

| Component | Approx Params |
|---|---|
| CpG ID embeddings (49,161 × 768) | ~37.8M |
| ScaleAdaptEncoder (48 basis → Linear 96→768) | ~0.075M |
| 8 × Transformer layers | ~56.6M |
| Pooler | ~0.6M |
| **Total encoder** | **~95M** |

---

## 4. Architecture Components

### 4.1 RMSNorm

**BERT uses LayerNorm:**
$$\text{LN}(x) = \frac{x - \mu}{\sigma} \cdot \gamma + \beta$$
- Requires computing mean AND variance
- 2 passes over the vector

**MethylLlama uses RMSNorm:**
$$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot \gamma \quad \text{where} \quad \text{RMS}(x) = \sqrt{\frac{1}{d}\sum_i x_i^2}$$
- No mean subtraction
- ~40% faster in practice (used in LLaMA, Mistral, Gemma)
- Works at float32 for numerical stability, casts back to fp16

**Effect on training:** numerically cleaner activations, less overhead per step.

---

### 4.2 Pre-LN vs Post-LN

**BERT (Post-LN) — normalize AFTER residual:**
```
h = LayerNorm(h + Attn(h))   ← standard BERT
h = LayerNorm(h + MLP(h))
```

**MethylLlama (Pre-LN) — normalize BEFORE sublayer:**
```
h = h + Attn(RMSNorm(h))     ← Pre-LN
h = h + MLP(RMSNorm(h))
```

**Why Pre-LN is better for this task:**

| Property | Post-LN (BERT) | Pre-LN (LLaMA) |
|---|---|---|
| Gradient flow at step 0 | Can explode/vanish | Stable from the start |
| LR warmup requirement | Mandatory, sensitive | Less critical |
| Training stability | Needs careful tuning | Stable default |
| Convergence speed | Slower (needs warmup) | Faster |

For a model training on 24k-token sequences with 49k-site reconstruction, training instability is a real risk. Pre-LN eliminates this.

---

### 4.3 SwiGLU Feed-Forward Network

**BERT FFN (GELU activation, 2 matrices):**
$$\text{FFN}(x) = W_2 \cdot \text{GELU}(W_1 x)$$
Parameters per layer: `2 × D × 4D = 8D²`

**MethylLlama SwiGLU (3 matrices):**
$$\text{FFN}(x) = W_{\text{down}} \big( \text{SiLU}(W_{\text{gate}} x) \odot W_{\text{up}} x \big)$$
Parameters per layer: `3 × D × (2/3 × 4D) = 8D²`  (same count, different shape)

**Why SwiGLU:**
- **Gated mechanism**: each dimension has a learned gate controlling information flow
- **SiLU** (Sigmoid Linear Unit) = `x × sigmoid(x)` — smooth, avoids dying-ReLU
- Consistently outperforms GELU-FFN in practice (used in PaLM, LLaMA, Gemma)
- Same parameter budget: intermediate_size = `round(2/3 × 4 × 768 / 64) × 64 = 2048`

---

### 4.4 Rotary Position Embedding (RoPE)

**BERT absolute positions:**
```
embedding_table = nn.Embedding(max_seq_len, hidden_size)
# For 8002 positions × 512D = ~4M parameters
# Fixed: position 5000 was never seen at test time if max_seq=4096 during training
```

**MethylLlama RoPE:**
```
# No learned parameters — deterministic frequency table
inv_freq[k] = 1 / (rope_theta ^ (2k / head_dim))
# Positions encoded by rotating Q and K in 2D frequency planes
q_rot = q * cos(pos × inv_freq) + rotate_half(q) * sin(pos × inv_freq)
```

**Advantages for methylation sequences:**

| Property | Absolute (BERT) | RoPE |
|---|---|---|
| Parameters | ~4M for L=8002 | 0 |
| New sequence lengths | Fails (untrained) | Generalizes |
| Position signal | Additive (global) | Relative (per-head) |
| Memory | Extra embedding table | Just cached cos/sin |

For 24,580-token sequences (the actual input length), RoPE saves ~14M parameters compared to absolute embeddings at hidden_size=768. Each attention head independently learns how position-sensitive to be.

---

### 4.5 ScaleAdaptEncoder (Beta Value Encoding)

**Problem:** Beta values are continuous in [0, 1]. A 2-layer MLP may not resolve fine differences like 0.85 vs 0.87, which matter for age prediction.

**Solution: Sinusoidal basis encoding:**
```
features_k = [sin(β × f_k), cos(β × f_k)]   for k = 1..48
output = Linear(96 → 768)(concat(all features))
```

Where frequencies `f_k ~ N(0, 2π × basis_scale)` are **trainable** — the model learns which "resolutions" of beta variation matter biologically.

**Special token routing:**
```
β < 0  → learned nn.Embedding (MASK→0, CLS→1, SEP→2, PAD→3)
β ≥ 0  → sinusoidal basis (β=0 is real: unmethylated CpG)
```

**basis_scale = 2.0 (not 1.5):**
The upstream BMFM model (scRNA) uses 1.5 because expression values range 0–10+.
Methylation [0, 1] is a tighter range — higher initial frequencies (2.0) needed to resolve fine differences. Since frequencies are trainable, this only affects initialization.

**Why better than 2-layer MLP (SCBert approach):**
- MLP can learn only smooth global mappings; sine basis explicitly covers multiple frequency bands
- Trainable frequencies adapt to which beta differences are predictive
- Special tokens handled natively without separate logic

---

### 4.6 cpg_scale Initialization

**Embedding fusion:**
```python
h = cpg_scale × CpgEmbed(cpg_ids) + ScaleAdaptEncoder(betas)
```

`cpg_scale` is a single learned scalar, initialized at **0.1**.

**Why 0.1 (small init):**
- At step 0, beta embedding dominates → model focuses on methylation levels first
- CpG identity (which site) is slowly learned as the model discovers which sites matter
- If `cpg_scale` were initialized large (e.g. 1.0), the model might collapse to predicting by site identity alone (mean beta per site), ignoring individual sample variation
- `cpg_scale` is logged during training — a healthy run shows it growing from 0.1 toward ~0.3–0.5

**Effect on convergence:** prevents the common failure mode where the decoder learns to predict per-site averages instead of per-sample values.

---

## 5. WCED Pretraining Objective

### What is WCED?

**Whole-Context Encoder Distillation** — the model sees half the CpGs and must reconstruct all CpGs from the CLS token alone.

```
Full methylation profile: [c₁, c₂, c₃, ..., c₄₉₁₅₆]
                                  ↓ random 50% split
View: [CLS, c₃, c₇, c₁₂, ...]   ← 24,579 tokens seen
                                  ↓ encoder
CLS embedding: [B, 768]           ← must encode everything
                                  ↓ decoder
Predictions: [ĉ₁, ĉ₂, ..., ĉ₄₉₁₅₆]   ← all 49,156 sites
                                  ↓ loss
MSE only on NON-input sites       ← must generalize beyond input
```

### Loss Function

$$\mathcal{L} = \mathcal{L}_{\text{recon}} + \lambda_{\text{age}} \cdot \mathcal{L}_{\text{age}}$$

| Term | Formula | Purpose |
|---|---|---|
| Reconstruction | MSE(predicted, true) over non-input, non-NaN CpGs | Compress full profile into CLS |
| Age MSE | MSE(age_head(CLS), chronological_age) | Prevent CLS collapse; encode age |
| InfoNCE | (disabled in first run, λ=0) | Align two views of same sample |

**Why age auxiliary loss:**
Without it, the CLS token can collapse to encoding mean methylation level (easy shortcut).
The age prediction head forces CLS to capture biological age signal even during pretraining.

### Metrics Logged

| Metric | Meaning |
|---|---|
| `train/pcc` | Pearson correlation on hidden (non-input) CpGs |
| `validation/pcc` | Same on validation set |
| `train/cpg_scale` | Learned embedding balance (should grow from 0.1) |
| `validation/cls_variance` | CLS diversity (collapse = near 0) |
| `validation/cls_similarity` | Mean cosine sim between CLS (collapse = near 1) |
| `validation/pred_var_ratio` | pred_var / target_var (collapse = near 0) |

---

## 6. Why MethylLlama Should Outperform SCBert + WCED

### Comparison Table

| Property | SCBert + WCED | MethylLlama + WCED |
|---|---|---|
| Normalization | LayerNorm (mean+variance) | RMSNorm (RMS only, ~40% faster) |
| Layer order | Post-LN (BERT style) | Pre-LN (stable from step 0) |
| FFN | GELU (2 matrices) | SwiGLU (3 matrices, gated) |
| Position encoding | Absolute table (~4M params) | RoPE (0 params, generalizes) |
| Beta encoder | 2-layer MLP (monkey-patched) | ScaleAdaptEncoder (built-in) |
| Embedding patch | `add_forward` monkey-patch | Native in `__init__` |
| cpg_scale | Learned, patched post-init | Built-in scalar Parameter |
| Code complexity | High (patches + hacks) | Clean, no patching needed |

### Why Larger Pretraining Corpus Matters

**169,000 × 49,156 CpGs** compared to typical clock models that train on 500–5,000 samples:

1. **Biological variation coverage:** 169k samples span diverse tissues, ages, conditions — the model learns which CpG patterns are universal vs tissue-specific

2. **CpG co-methylation patterns:** With 49k CpGs per sample, the encoder learns long-range correlations between distant sites (e.g., methylation at one gene promoter predicts another)

3. **CLS quality:** A CLS trained to reconstruct all 49k CpGs from 24.5k inputs must compress far more information than a model trained on 8k CpGs — richer representations for downstream tasks

4. **Transfer to 19,608 CpGs:** The fine-tuning dataset covers 19,608 CpGs all seen during pretraining — the encoder already knows these sites' methylation dynamics from 169k samples

5. **Age signal without labels:** The age auxiliary head during pretraining means the CLS already encodes chronological age before any labeled age data is shown

---

## 7. No Data Leakage — Clean Experimental Setup

### Original Problem (8k CpG selection)
```
AltumAge (21k CpGs) → select top 8k by |corr(CpG, age)|
                           ↑
                   computed on ALL samples including test set
                           ↑
                   LEAKAGE: test age signal informed feature selection
```

### Fixed Setup (19,608 CpGs)
```
Step 1: Build pretrain vocab on pretrain-only samples (no AltumAge)
        → 49,156 CpG sites in vocabulary

Step 2: AltumAge fine-tune data = CpGs in (AltumAge ∩ pretrain vocab)
        → 21,368 ∩ 49,156 = 19,608 CpGs

Step 3: Train/val/test split first, then fine-tune
        → No test-set information ever touches feature selection
```

**Overlap numbers:**
- Pretrain vocab: 49,156 CpGs
- AltumAge measured: 21,368 CpGs
- Overlap (in both): **19,608 CpGs** ← these are used for fine-tuning
- AltumAge-only (not in pretrain): 1,760 CpGs ← excluded (no pretrain knowledge)

**Conclusion:** By using the full intersection, we eliminate feature selection bias entirely. The 1,760 excluded CpGs have no pretrain representation anyway — including them would require random initialization for 8.2% of the fine-tuning input, which hurts more than excluding them.

---

## 8. Expected Results and Why

### Pretraining Quality

The run `llama-wced-all49k-r0.5-w0.0-44248557` trains on full 49k CpGs with:
- `wced_input_ratio = 0.5`: sees 24,579 CpGs, must reconstruct all 49,156
- `wced_age_weight = 1.0`: age MSE forces CLS to encode biological age
- `wced_contrastive = false`: pure reconstruction first run

Expected PCC convergence: → 0.90–0.95 on hidden CpGs (similar to SCBert WCED which reached 0.9866 on 8k)

### Fine-tuning Quality

After pretraining on 169k samples, fine-tuning on 11.5k AltumAge samples with 19,608 CpGs:

**Why better than training from scratch:**
1. CLS already encodes age from pretraining (age auxiliary loss)
2. All 19,608 fine-tune CpGs have rich pretrain representations
3. 169k samples → encoder understands methylation variation across biological conditions
4. Fine-tuning just calibrates CLS → age regression head

**Expected improvement over SCBert WCED:**
- Pre-LN = more stable fine-tuning (less sensitivity to LR)
- SwiGLU = more expressive representations
- RoPE = handles variable input lengths (different number of available CpGs per sample)
- No monkey-patching = reproducible, no hidden initialization bugs

---

## 9. Summary: Key Contributions of MethylLlama

| Contribution | Technical Detail | Benefit |
|---|---|---|
| **RMSNorm** | No mean subtraction | 40% faster norm, numerically cleaner |
| **Pre-LN** | Norm before sublayers | Stable training from step 1 |
| **SwiGLU** | Gated FFN (3 matrices) | More expressive same-param-budget FFN |
| **RoPE** | Rotary position encoding | 0 params, generalizes to unseen lengths |
| **ScaleAdaptEncoder** | Trainable sinusoidal beta encoding | Fine-grained [0,1] methylation resolution |
| **cpg_scale_init=0.1** | Small CpG identity weight | Prevents per-site average shortcuts |
| **WCED objective** | Reconstruct all from half | Compresses global methylation profile into CLS |
| **Age auxiliary loss** | MSE(CLS→age) during pretrain | CLS encodes biological age before fine-tune |
| **No leakage** | 19,608 vocab-intersection CpGs | Clean test-set evaluation |
| **169k samples** | Large pretrain corpus | Rich co-methylation pattern learning |
