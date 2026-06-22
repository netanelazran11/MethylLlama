# WCED Methylation Biological Age Prediction
## Implementation & Results Summary

---

## 1. The Goal

Predict a person's **biological age** (in years) from their DNA methylation profile.

- **Input:** Beta values (0–1) across **8,000 CpG sites** per person
- **Output:** Predicted age in years
- **Metrics:** R² (max=1.0, higher=better) · MAE in years (lower=better)

---

## 2. The Model

```
8,000 CpG sites
      ↓
┌─────────────────────────────────────────┐
│  Embedding Layer                        │
│  token = cpg_scale × CpG_embed         │
│          + beta_embed + position_embed  │
│  cpg_scale = learned scalar (init=0.1) │
└─────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────┐
│  SCBert Transformer Encoder             │
│  6 layers · 8 heads · hidden=512        │
│  Flash Attention (PyTorch SDPA)         │
│  ~27M parameters                        │
└─────────────────────────────────────────┘
      ↓
   [CLS]  ← single 512-dim vector
   (summary of the whole person)
      ↓
┌──────────────────┐
│  MLP Age Head    │
│  512→256→128→1   │
│  ~165k params    │
└──────────────────┘
      ↓
  Predicted Age
```

---

## 3. Phase 1 — WCED Pretraining

### What is WCED?

**Whole Cell Expression Decoder** — a self-supervised pretraining method that forces the CLS token to become a rich global summary of the entire methylation profile.

Standard BERT (MLM) predicts masked tokens locally.
WCED forces CLS to encode the full profile → better downstream representation.

### Architecture

```
One person's 8k CpG sites
         ↓
  Sample 2 random subsets of 4k CpGs each
         ↓              ↓
    View 1 (4k)    View 2 (4k)   ← same person, different CpGs
         ↓              ↓
      Encoder        Encoder     (shared weights)
         ↓              ↓
       CLS₁           CLS₂
    ┌────┴─────┐    ┌────┴─────┐
    ↓          ↓    ↓          ↓
 Decoder  Proj.Head  Decoder  Proj.Head
(recon)  (contrast) (recon)  (contrast)
    ↓
 Age Head
(auxiliary)
```

### Three Training Objectives

| Objective | What it does |
|-----------|-------------|
| **Reconstruction** | CLS → Decoder → predict all 8k beta values. Forces CLS to encode the full methylation profile. Loss computed only on CpGs NOT given as input. |
| **Contrastive (InfoNCE)** | CLS of View 1 vs CLS of View 2 of same person should be similar; vs other people should be different. Forces CLS to encode sample identity. |
| **Age auxiliary** | CLS → small head → age. Directly injects age signal into CLS. Prevents CLS from collapsing. |

```
Total loss = Reconstruction + 0.1 × InfoNCE + Age_MSE
```

### Pretraining Setup

```
Data:       8k CpG sites per person
Input/view: 4,000 random CpGs (50% of 8k)
Views/sample: 2 (different random subsets)
Epochs:     300 (best at epoch 190)
Batch:      32 × 4 accumulation = 128 effective
```

### Pretraining Result ✅

```
Best checkpoint: epoch 190
test/pcc        = 0.9866
Age linear probe R² ≈ 0.90
```

> **Interpretation:** The frozen CLS representation already predicts age with R²=0.90 using just a linear layer. Fine-tuning extracts more.

---

## 4. Phase 2 — Fine-tuning Experiments

All experiments start from the same WCED checkpoint (epoch 190).

---

### Experiment A — Frozen Encoder, 8k Fixed Input ⭐ BEST

**Idea:** Keep the encoder completely frozen. Train only the MLP head to map CLS → age.
Use all 8,000 CpG sites, same selection every batch.

```
8k CpGs (fixed, same every batch)
      ↓
Encoder [FROZEN — no gradients]
      ↓
    CLS (512-dim)
      ↓
MLP Head [trainable, lr=1e-3]
      ↓
  Predicted Age

Loss: MSE(predicted_age, true_age)
```

**Key settings:**
```python
fixed_subset = True      # same 8k CpGs every batch → stable gradients
freeze_encoder = True    # all 27M params frozen
learning_rate = 1e-3     # head only
```

**Result:**
```
test/R²  = 0.9327  ✅ BEST
test/MAE = 4.61 years
```

**Why it works:** Fixed input → stable gradients → head converges cleanly.
8k features → maximum information per step.

---

### Experiment B — Full Encoder Unfreeze ❌ CATASTROPHIC FORGETTING

**Idea:** Allow the entire encoder to adapt to age prediction.

```python
freeze_encoder = False
learning_rate = 1e-4  # same LR for head AND encoder
```

**What happened:**
```
WCED pretraining:  3 objectives → stable, rich CLS representation
Fine-tuning:       1 objective (age only) → encoder forgets everything else

R² before:  0.896
R² after:   0.417   ← destroyed
```

**Lesson:** 27M parameters cannot be trained on age signal alone.
Without reconstruction + contrastive loss, the encoder forgets the global representation.

---

### Experiment C — WCED-Correct Fine-tuning (Decoder Regularizer)

**Idea:** Keep the WCED decoder alive during fine-tuning.
The decoder forces CLS to stay globally informative (same as pretraining).
This prevents forgetting while allowing gentle encoder adaptation.

```
4k random CpGs (matches pretraining distribution)
      ↓
Encoder [trainable, lr=1e-5]
      ↓
    CLS (512-dim)
   ┌────┴────┐
   ↓         ↓
Age Head  Decoder [lr=1e-5]
(lr=1e-3)  ↓
   ↓      All 8k beta predictions
Pred. Age

Loss = age_MSE + 0.1 × recon_MSE(non-input CpGs only)
```

**Key settings:**
```python
head_lr    = 1e-3    # fast
encoder_lr = 1e-5    # 100× slower → gentle adaptation
decoder_lr = 1e-5    # same as encoder
recon_weight = 0.1   # regularizer weight
input = 4k random    # must be partial for decoder to work
```

**Fundamental constraint discovered:**
```
With 8k input → all CpGs in input → no non-input CpGs → reconstruction loss = 0
Cannot have full 8k input AND decoder regularizer simultaneously.
```

**Result:**
```
test/R²  = 0.9156
test/MAE = 5.22 years
```

**Why worse than Experiment A:**
1. Only 4k features (half the information)
2. Decoder competes with age head for CLS capacity
3. Two competing objectives slow convergence

---

### Experiment D — LoRA Fine-tuning (Two Variants)

**Idea:** Inject tiny trainable adapter matrices into frozen attention layers.
Original weights never change → no forgetting. Only 0.4% of encoder params train.

**LoRA math:**
```
Output = W_original × input  +  (α/r) × B(A(input))
                                ↑ LoRA delta (starts at zero)
A: (512×8)  initialized Kaiming
B: (8×512)  initialized ZEROS → delta=0 at start → safe
```

**Where injected:** Query + Value projections in all 6 attention layers
```
6 layers × 2 projections × 2 matrices (A+B)
= 24 matrices × ~4,096 params each = ~98,304 LoRA params (0.4% of encoder)
```

**Attempt 1 — Joint training (head + LoRA simultaneously):**
```
Problem: Moving target
  LoRA changes CLS → head's learned mapping invalid → head re-learns
  Head changes → LoRA gets different gradient signal
  → interference → slow convergence
```

**Attempt 2 — Two-stage (head first, then LoRA):**
```
Stage 1 (epochs 0-49):   LoRA LR = 0 → head trains alone → converges to ~R²=0.93
Stage 2 (epoch 50+):     LoRA LR unlocked → adapts attention with stable head signal
```

**Result (both attempts):**
```
test/R²  = 0.87  (worse than frozen 0.9327)
test/MAE = 6.6 years
```

**Why LoRA hurts:**
The WCED pretraining shaped attention patterns over 190 epochs with 3 objectives.
Even 0.4% parameter changes to query/value projections disrupts the CLS representation
that makes age prediction work. The frozen CLS is more valuable than any adaptation.

---

## 5. Results Comparison

```
Method                    test/R²   test/MAE   Encoder State
─────────────────────────────────────────────────────────────
WCED pretrain (linear)    ~0.90     —          Frozen
─────────────────────────────────────────────────────────────
★ Frozen + 8k fixed        0.9327   4.61 yr   Frozen (best)
  WCED + decoder reg.      0.9156   5.22 yr   Trainable 1e-5
  LoRA two-stage           0.8731   6.64 yr   Frozen + LoRA
  Full unfreeze            0.4170   —          Trainable 1e-4
─────────────────────────────────────────────────────────────
```

```
R²
1.00 ┤
0.93 ┤  ████  ← Frozen 8k (BEST)
0.92 ┤  ████
0.90 ┤  ████  ████ ← WCED linear probe (pretrain)
     ┤  ████  ████
0.87 ┤  ████  ████  ████ ← LoRA
     ┤  ████  ████  ████
0.50 ┤  ████  ████  ████
0.42 ┤  ████  ████  ████  ████ ← Full unfreeze
     └──────────────────────────
      Frozen  WCED  LoRA  Unfrz
```

---

## 6. Key Insights

| Finding | Evidence |
|---------|----------|
| **8k > 4k input** | More CpG sites = more information per step |
| **Fixed > random subset** | Stable gradients → faster, cleaner head convergence |
| **Any encoder modification hurts** | Frozen (0.9327) > LoRA (0.873) > WCED FT (0.916) > Unfreeze (0.417) |
| **WCED CLS is the key asset** | 190 epochs of 3-objective training produced a near-optimal representation |
| **The ceiling is the frozen encoder** | Head capacity is not the bottleneck — CLS quality is |

---

## 7. Final Answer

> **Best method: Frozen WCED Encoder + Fixed 8k Input + MLP Head**
>
> test/R² = **0.9327** · test/MAE = **4.61 years**
>
> The WCED pretraining produced a CLS representation so well-calibrated
> that the optimal fine-tuning strategy is to leave the encoder completely
> untouched and train only a lightweight MLP head on top.
