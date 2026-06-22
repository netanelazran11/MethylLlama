# BMFM vs MethylGPT: Architecture Comparison for Methylation Age Prediction

## 1. Overview

Comparison of two transformer-based architectures for DNA methylation age prediction:

| Model | Description |
|-------|-------------|
| **BMFM** | BioMedical Foundation Model - Originally designed for RNA/gene expression data with variable gene sets |
| **MethylGPT** | Based on scGPT Architecture - Specifically adapted for DNA methylation data with fixed CpG sites |

---

## 2. The Fundamental Problem: Constant vs Variable Features

### Why BMFM Architecture Needs Adaptation for Methylation Data

BMFM was designed for **gene expression data** where different cells express different genes.
But **methylation data** has the same CpG sites for ALL samples. This fundamental difference requires architectural changes.

### 2.1 Gene Expression Data (What BMFM Was Designed For)

**Example: Gene Expression - Different Genes Per Sample**

| Sample | Cell Type | Gene IDs | Expression | Sequence Length |
|--------|-----------|----------|------------|-----------------|
| Sample 1 | Blood Cell | HBA1, HBB, CD45, ... | 150, 200, 80, ... | ~1,200 genes |
| Sample 2 | Liver Cell | ALB, CYP3A4, HNF4A, ... | 500, 120, 90, ... | ~1,500 genes |
| Sample 3 | Neuron | SYN1, NEFL, GRIN1, ... | 180, 95, 220, ... | ~1,800 genes |
| Sample 4 | Muscle Cell | MYH1, ACTA1, TTN, ... | 400, 350, 280, ... | ~1,100 genes |

**Key Point:** Each sample has DIFFERENT genes with DIFFERENT sequence lengths. The Gene ID embedding learns which genes are important.

### 2.2 Methylation Data (What We Need to Adapt For)

**Example: Methylation - SAME CpG Sites For ALL Samples**

| Sample | Age | CpG IDs | Beta Values | Sequence Length |
|--------|-----|---------|-------------|-----------------|
| Sample 1 | 25 years | cg00000029, cg00000108, cg00000109, ... | 0.85, 0.12, 0.67, ... | **8,000 CpG sites** |
| Sample 2 | 45 years | cg00000029, cg00000108, cg00000109, ... | 0.72, 0.25, 0.58, ... | **8,000 CpG sites** |
| Sample 3 | 65 years | cg00000029, cg00000108, cg00000109, ... | 0.55, 0.42, 0.48, ... | **8,000 CpG sites** |
| Sample 4 | 80 years | cg00000029, cg00000108, cg00000109, ... | 0.38, 0.55, 0.35, ... | **8,000 CpG sites** |

**Key Point:** ALL samples have the SAME 8,000 CpG sites in the SAME order. Only the beta values change. CpG IDs act like POSITION information.

### 2.3 Summary: The Core Difference

| Aspect | Gene Expression (BMFM) | Methylation (Needed) |
|--------|------------------------|----------------------|
| **Feature IDs** | VARIABLE per sample | CONSTANT for all samples |
| **Sequence Length** | VARIABLE (500 - 5000 genes) | FIXED (8,000 CpG sites) |
| **Feature Order** | VARIABLE (different genes) | SAME order for all |
| **ID Embedding Role** | Learns gene identity & importance | Acts like POSITION encoding |
| **Value Range** | 0 - 10,000+ (expression counts) | 0 - 1 (beta values) |
| **Model Requirement** | Large model for diverse patterns | Smaller model sufficient |

### Why This Matters for Architecture

- **Hidden Size:** With fixed 8K sites, a smaller model (64-dim) works better than large (512-dim)
- **CpG Embedding:** Acts like position encoding - simpler treatment possible
- **Pooling:** Full sequence processing preferred - position matters!
- **Regression:** ResNet1D can capture patterns across fixed positions

---

## 3. Architecture Diagrams

### 3.1 MethylGPT Architecture

```
Input: CpG IDs [batch, seq], Beta Values [batch, seq]
                ↓                        ↓
      ┌─────────────────┐      ┌─────────────────────┐
      │  GeneEncoder    │      │ ContinuousValueEnc  │
      │ Embed(64)       │      │  Linear(1→64)→ReLU  │
      │ + LayerNorm     │      │  Linear(64→64)      │
      └────────┬────────┘      │  + LayerNorm        │
               │               └──────────┬──────────┘
               │                          │
               └──────────┬───────────────┘
                          ↓
                   ADD: gene + value
                          ↓
                 TransformerEncoder
                 (6 layers, 4 heads, d=64)
                          ↓
                    [batch, seq, 64]
                          ↓
                 Remove CLS [:, 1:, :]
                          ↓
                Permute [batch, 64, seq-1]
                          ↓
                      ResNet1D
                          ↓
                 Sigmoid → Age [0, 1]
```

### 3.2 BMFM Architecture

```
Input: input_ids [batch, 2, seq]
       (Field 0: CpG IDs, Field 1: Beta Values)
                          ↓
      ┌─────────────────────┐    ┌─────────────────────┐
      │   CpG Embedding     │    │  Beta Value Encoder │
      │   Embed(512)        │    │  MLP(1→128→512)     │
      └──────────┬──────────┘    └──────────┬──────────┘
                 │                          │
                 └──────────┬───────────────┘
                            ↓
                    Combine (ADD or MULTIPLY)
                            ↓
                        LayerNorm
                            ↓
                   TransformerEncoder
                   (6 layers, 8 heads, d=512)
                            ↓
                      [batch, seq, 512]
                            ↓
                       Mean Pooling
                            ↓
                       [batch, 512]
                            ↓
                         MLP Head
                   (512→256→128→1)
                            ↓
                  Linear → Age (z-score)
```

---

## 4. Model Size Comparison

| Model | Parameters | Hidden Size |
|-------|------------|-------------|
| **MethylGPT** | ~500K | 64 |
| **BMFM** | ~13M | 512 |
| **Ratio** | 26× | 8× |

### Why Smaller is Better for Methylation

- **Fixed 8K sites:** Less variability = simpler patterns to learn
- **Overfitting risk:** Large model memorizes training data
- **Generalization:** Smaller model generalizes better to new samples
- **Training speed:** Faster convergence with fewer parameters

---

## 5. Detailed Architecture Comparison

### 5.1 Core Model Parameters

| Component | MethylGPT | BMFM |
|-----------|-----------|------|
| **Hidden Dimension** | 64 | 512 |
| **Number of Layers** | 6 | 6 |
| **Attention Heads** | 4 | 8 |
| **Feed-Forward Dimension** | 64 | 2048 |
| **Total Parameters** | ~500K | ~13M |

### 5.2 Embedding Layer Comparison

| Component | MethylGPT | BMFM |
|-----------|-----------|------|
| **CpG Embedding Size** | 64 | 512 |
| **CpG LayerNorm** | After embedding | After combination |
| **Value Encoder** | Linear(1→64)→ReLU→Linear(64→64) | MLP(1→128→512) |
| **Value LayerNorm** | After encoder | After combination |
| **Combination Mode** | ADD (gene + value) | ADD or MULTIPLY |

### 5.3 Pooling and Regression Head

| Component | MethylGPT | BMFM |
|-----------|-----------|------|
| **Pooling Strategy** | Full sequence (remove CLS) | Mean pooling |
| **Pooling Output Shape** | [batch, hidden, seq_len] | [batch, hidden] |
| **Regression Head Type** | ResNet1D | MLP |
| **Head Architecture** | Conv1D + Residual Blocks | Linear→LayerNorm→GELU |
| **Output Activation** | Sigmoid | Linear |

### 5.4 Training Configuration

| Component | MethylGPT | BMFM |
|-----------|-----------|------|
| **Age Normalization** | MinMax [0, 1] | Z-score (μ=0, σ=1) |
| **Output Range** | [0, 1] (bounded) | Unbounded |
| **Loss Function** | MSE (normalized) | MSE (normalized) |
| **Optimizer** | Adam | AdamW |

### 5.5 Data Format

| Component | MethylGPT | BMFM |
|-----------|-----------|------|
| **Input Format** | Two separate tensors | Single 3D tensor |
| **CpG IDs Shape** | [batch, seq_len] | [batch, 0, seq_len] |
| **Beta Values Shape** | [batch, seq_len] | [batch, 1, seq_len] |
| **Padding Value** | -2 | -4 |
| **Mask Value** | -1 | -1 |
| **Special Tokens** | `<pad>`, `<cls>`, `<eoc>` | `<pad>`, `<cls>`, `<sep>`, `<unk>`, `<mask>` |

---

## 6. Required Adaptations for BMFM with Methylation Data

### 6.1 Tokenizer Adaptation

| Component | Original BMFM | Adapted for Methylation |
|-----------|---------------|-------------------------|
| **Vocabulary** | ~30,000 genes (variable) | ~8,000 CpG sites (fixed) |
| **Token IDs** | Gene symbols → IDs | CpG probe IDs → IDs |
| **Value Range** | 0 - 10,000+ (counts) | 0 - 1 (beta values) |
| **Special Tokens** | `<pad>`, `<cls>`, `<sep>`, `<unk>`, `<mask>` | `<pad>`, `<cls>`, `<mask>` (simplified) |

### 6.2 Model Architecture Adaptation

| Parameter | Original BMFM | Adapted | Reason |
|-----------|---------------|---------|--------|
| **Hidden Size** | 512 | 64 - 128 | Fixed 8K sites need smaller model |
| **Attention Heads** | 8 | 4 | Proportional to hidden size |
| **FFN Size** | 2048 | 64 - 256 | Reduce overfitting |
| **Total Params** | ~13M | ~500K - 1M | Better generalization |

### 6.3 Embedding Layer Adaptation

**BMFM Original:**
```
CpG_embed ────┐
              ├→ Combine → LayerNorm → Transformer
Value_embed ──┘
```

**Adapted:**
```
CpG_embed → LayerNorm ──┐
                        ├→ ADD → Transformer
Value_embed → LayerNorm ┘
```

### 6.4 Embedding Combination Mode

| Mode | Formula | For Methylation |
|------|---------|-----------------|
| **ADD** | `h = CpG_embed + Value_embed` | Recommended |
| **MULTIPLY** | `h = CpG_embed × β_value` | Not recommended (β is 0-1) |

### 6.5 Pooling Strategy Adaptation

**BMFM Original:**
```
Transformer Output [batch, seq, 512]
              ↓
         Mean Pooling [batch, 512]
              ↓
          MLP → Age
```

**Adapted:**
```
Transformer Output [batch, seq, 64]
              ↓
     Remove CLS [:, 1:, :] [batch, seq-1, 64]
              ↓
     Permute [batch, 64, seq-1]
              ↓
         ResNet1D → Age
```

**Why:** CpG positions matter for age prediction. Mean pooling loses this positional information.

### 6.6 Regression Head Adaptation

| Component | Original BMFM | Adapted |
|-----------|---------------|---------|
| **Head Type** | MLP | ResNet1D |
| **Input Shape** | [batch, 512] | [batch, 64, seq_len] |
| **Architecture** | Linear → LN → GELU → Linear | Conv1D → ResBlocks → Pool → Linear |
| **Advantage** | Simple, fast | Captures positional patterns |

### 6.7 Age Normalization Adaptation

| Component | Original BMFM | Adapted |
|-----------|---------------|---------|
| **Normalization** | Z-score: (age - μ) / σ | MinMax: (age - min) / (max - min) |
| **Output Range** | Unbounded (-∞, +∞) | [0, 1] |
| **Activation** | Linear (none) | Sigmoid |
| **Inverse Transform** | pred × σ + μ | pred × (max - min) + min |

---

## 7. Complete Adaptation Summary

| # | Component | Original BMFM | Adapted for Methylation |
|---|-----------|---------------|-------------------------|
| 1 | **Tokenizer Vocabulary** | ~30,000 genes | ~8,000 CpG sites |
| 2 | **Hidden Dimension** | 512 | 64 - 128 |
| 3 | **Attention Heads** | 8 | 4 |
| 4 | **FFN Dimension** | 2048 | 64 - 256 |
| 5 | **LayerNorm Placement** | After combination | After each embedding |
| 6 | **Embedding Mode** | ADD or MULTIPLY | ADD only |
| 7 | **Pooling** | Mean pooling | Full sequence |
| 8 | **Regression Head** | MLP | ResNet1D |
| 9 | **Age Normalization** | Z-score | MinMax [0, 1] |
| 10 | **Output Activation** | Linear | Sigmoid |

---

## 8. Final Summary

### Key Problem
BMFM was designed for **variable gene expression** data where different samples have different genes. Methylation data has **fixed CpG sites** that are the same for all samples.

### Solution
Adapt BMFM architecture to match MethylGPT's approach: smaller model, ADD mode embeddings, full sequence processing with ResNet1D, and bounded output with Sigmoid.

| Data Type | Feature IDs | Sequence Length | Ideal Model Size | Pooling |
|-----------|-------------|-----------------|------------------|---------|
| **Gene Expression** | Variable per sample | 500 - 5,000 | Large (512-dim) | Mean/CLS pooling |
| **Methylation** | Fixed for all samples | Fixed ~8,000 | Small (64-dim) | Full sequence |

---

*Document generated for thesis presentation on BMFM vs MethylGPT architecture comparison.*
