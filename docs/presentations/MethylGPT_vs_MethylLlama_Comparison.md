# MethylGPT vs MethylLlama-Small — Full Comparison Prompt

---

## Context

I am doing a thesis project comparing two DNA methylation foundation models for age prediction from 450k/EPIC array data.

The task: given a sample's CpG methylation beta values (450k/EPIC array), predict the biological age of the sample in years.

Dataset: 10,656 samples × 19,608 CpGs, spanning 80 tissue types, age range 0–114 years.

---

## Model 1 — MethylGPT (baseline)

MethylGPT is an existing published model trained on DNA methylation data using a GPT-style transformer.

### Architecture
| Component | Value |
|-----------|-------|
| Model family | GPT-2 style transformer |
| Embedding dimension (hidden size) | 64 |
| Transformer layers | 6 |
| Attention heads | 4 |
| Head dimension | 64 / 4 = 16 |
| FFN intermediate dimension | 64 |
| FFN expansion ratio | 64 / 64 = 1.0× (no expansion) |
| Positional encoding | Learned / absolute |
| Normalization | LayerNorm |
| Activation | GELU |
| Total transformer parameters | ~150k (excl. embedding) |

### Training
- Pre-trained on methylation data in a generative (autoregressive) fashion
- CpG sites tokenized and treated as a sequence
- Fine-tuned for age prediction by attaching a regression head

### Limitations
- 64-dimensional hidden state is very narrow for 19,608 CpG tokens
- Each attention head has only 16 dimensions — limited capacity to model CpG-CpG relationships
- FFN ratio of 1.0× means no nonlinear expansion — effectively a linear mapping per layer
- GPT-2 style absolute positional encoding does not generalize to variable sequence lengths
- ~150k transformer parameters may underfit the complexity of multi-tissue methylation

---

## Model 2 — MethylLlama-Small (our model)

MethylLlama-Small is a LLaMA-style transformer we pretrained on 49,156 CpG sites using a Weighted Correlation Encoder-Decoder (WCED) contrastive objective.

### Architecture
| Component | Value |
|-----------|-------|
| Model family | LLaMA style (modern transformer) |
| Embedding dimension (hidden size) | 256 |
| Transformer layers | 4 |
| Attention heads | 4 |
| Head dimension | 256 / 4 = 64 |
| FFN intermediate dimension | 320 |
| FFN expansion ratio | 320 / 256 = 1.25× |
| Positional encoding | RoPE (Rotary Position Embedding) |
| Normalization | RMSNorm (Pre-LN) |
| Activation | SwiGLU (gated linear unit) |
| Total transformer parameters | ~5M (excl. embedding) |
| Embedding table | 49,156 CpGs × 256D = ~12.5M (frozen during fine-tuning) |

### Pretraining
- Pretrained on 49,156 CpGs using WCED (contrastive + reconstruction objective)
- Input: random 50% of CpGs → reconstruct remaining 50% + contrastive InfoNCE on CLS
- Best checkpoint: epoch 98, val_loss = 0.0059
- Pretraining objective forces the model to learn global methylation representations

### Fine-tuning
- Input: all 19,608 CpGs (input_ratio = 1.0)
- Pooling: mean pooling over all CpG token representations (not CLS)
- Head: Linear(256→256) → LayerNorm → GELU → Linear(256→128) → LayerNorm → GELU → Linear(128→1)
- Loss: MSE on z-scored age labels
- Optimizer: AdamW with 4 parameter groups (separate decay/no-decay for head and encoder)
- Scheduler: LambdaLR with linear warmup + cosine decay

---

## Architecture Comparison Table

| Dimension | MethylGPT | MethylLlama-Small | Difference |
|-----------|-----------|-------------------|------------|
| Hidden dim | 64 | 256 | 4× wider |
| Layers | 6 | 4 | MethylGPT deeper |
| Attention heads | 4 | 4 | same |
| Head dimension | 16 | 64 | 4× richer per head |
| FFN intermediate | 64 | 320 | 5× larger |
| FFN expansion ratio | 1.0× | 1.25× | MethylGPT has no expansion |
| Positional encoding | Learned/absolute | RoPE | RoPE generalizes to any length |
| Normalization | LayerNorm (Post-LN) | RMSNorm (Pre-LN) | Pre-LN more stable |
| Activation | GELU | SwiGLU | SwiGLU more expressive |
| Architecture family | GPT-2 | LLaMA | MethylLlama is modern |
| Transformer params | ~150k | ~5M | 33× more capacity |
| Pretraining objective | Autoregressive | WCED contrastive + recon | Bidirectional vs unidirectional |

---

## Key Design Differences and Why They Matter

### 1. Width vs Depth
MethylGPT is narrow (64D) and deep (6 layers).
MethylLlama-Small is wider (256D) and shallower (4 layers).

For methylation data, width matters more than depth because:
- Each CpG token must carry enough information in its embedding to be meaningful
- At 16D per attention head (MethylGPT), the model cannot express complex CpG-CpG relationships
- At 64D per attention head (MethylLlama), there is sufficient capacity per head

### 2. FFN Expansion
MethylGPT's FFN ratio of 1.0× means the feed-forward network is a bottleneck — input and output have the same dimension. Standard transformers use 4.0× expansion; LLaMA uses ~2.67× with SwiGLU. MethylLlama-Small at 1.25× is still conservative but has more nonlinear capacity than MethylGPT.

### 3. Positional Encoding
MethylGPT uses absolute learned positional embeddings — fixed to the training sequence length. RoPE in MethylLlama-Small encodes relative position, naturally handles variable-length inputs, and generalizes across sequence lengths.

### 4. Pretraining Objective
MethylGPT pretrains autoregressively (left-to-right). This means each CpG can only attend to previous CpGs — half the context is unavailable. MethylLlama-Small pretrains with WCED (bidirectional attention), meaning every CpG can attend to every other CpG — better suited for methylation pattern modeling.

### 5. Parameter Efficiency
MethylLlama-Small has 33× more transformer parameters. For a task with 19,608 input tokens across 80 tissue types and 114 years of age range, more capacity is appropriate.

---

## Fine-tuning Results

| Run | Model | test/r2 | test/mae | Notes |
|-----|-------|---------|----------|-------|
| V1 | MethylLlama-Small | 0.862 | 6.81 yr | best so far |
| V2 | MethylLlama-Small | 0.823 | 7.50 yr | CLS pooling (wrong) |
| V3 | MethylLlama-Small | 0.745 | 9.01 yr | biased split + warmup bug |
| V4 | MethylLlama-Small | pending | target <6.81yr | 7 root causes fixed |
| MethylGPT | MethylGPT | ? | ? | baseline to compare |

---

## What to Compare

When comparing MethylGPT and MethylLlama-Small, the key metrics are:
1. **test/mae** — Mean Absolute Error in years (primary metric)
2. **test/r2** — R² correlation between predicted and true age
3. **test/medae** — Median Absolute Error (robust to outliers)
4. Both models should be evaluated on the **same stratified test split** for a fair comparison
5. The test split uses age_bin × tissue_type stratification (0.26yr val/test age gap)

---

## Open Questions for Comparison

1. Is the 33× parameter advantage of MethylLlama-Small sufficient to overcome MethylGPT's 6-layer depth?
2. Does bidirectional pretraining (WCED) vs autoregressive (MethylGPT) matter for age prediction?
3. Can MethylLlama-Small beat MethylGPT's age prediction despite a more conservative FFN ratio (1.25× vs standard 2.67×)?
4. What is the effect of tissue type on per-tissue MAE for each model?
