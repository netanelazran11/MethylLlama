# DNA Methylation Age Prediction with Multi-Field Tokenization
## Adapting BMFM-RNA for Epigenetic Data

**Author:** Netanel Azran
**Institution:** Hebrew University of Jerusalem
**Date:** February 2026

**Repository:** [https://github.com/netanelazran11/BiomedSciAI_ACL_Project](https://github.com/netanelazran11/BiomedSciAI_ACL_Project)
**Methylation Adaptation:** [methyl/bmfm_methylation](https://github.com/netanelazran11/BiomedSciAI_ACL_Project/tree/main/methyl)

---

# Part 1: Introduction & Motivation

## 1.1 The Problem: Age Prediction from DNA Methylation

DNA methylation is an epigenetic modification where methyl groups attach to CpG dinucleotides. This pattern changes predictably with age, creating an "epigenetic clock."

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    THE EPIGENETIC CLOCK                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   DNA Methylation Pattern Changes with Age:                                │
│                                                                             │
│   Young (β ≈ 0.2)        Middle Age (β ≈ 0.5)        Old (β ≈ 0.8)       │
│   ─────────────────      ─────────────────────      ─────────────────     │
│   CpG Site:  ○○○○○       CpG Site:  ●○●○●           CpG Site:  ●●●●●      │
│                                                                             │
│   β-value: Methylation fraction (0 = unmethylated, 1 = methylated)        │
│                                                                             │
│   Goal: Learn mapping f(β₁, β₂, ..., βₙ) → Age                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 1.2 Why Adapt BMFM-RNA?

| Aspect | scRNA-seq (Original) | DNA Methylation (Adaptation) |
|--------|---------------------|------------------------------|
| **Data Type** | Gene expression counts | Methylation β-values |
| **Value Range** | 0 to ~10,000+ (integers) | 0.0 to 1.0 (continuous) |
| **Features** | ~20,000 genes | ~850,000 CpG sites |
| **Token IDs** | Gene symbols (BRCA1, TP53, ...) | CpG probe IDs (cg00000029, ...) |
| **Task** | Cell type classification | Age regression |

**Key Insight:** Both share the same structure:
- **Discrete tokens** (gene IDs / CpG IDs)
- **Continuous values** (expression / β-values)
- **Sample-level output** (cell type / age)

→ BMFM's Multi-Field architecture is directly applicable!

---

# Part 2: Files Created/Modified

## 2.1 New Files in `methyl/bmfm_methylation/`

| File | Purpose | Lines |
|------|---------|-------|
| **`tokenizer.py`** | Creates CpG vocabulary and MultiFieldTokenizer | 229 |
| **`data_module.py`** | PyTorch Lightning DataModule for h5ad files | 529 |
| **`model.py`** | MethylationEncoder wrapper around SCBertModel | 239 |
| **`pretrain.py`** | MLM pretraining script | 271 |
| **`finetune.py`** | Age regression fine-tuning | 495 |
| **`config.py`** | Configuration dataclasses | 115 |
| **`lightning_module.py`** | Training modules | 267 |

## 2.2 Configuration Files in `configs/`

```
configs/
├── fields/
│   └── methylation.yaml       # Field definitions (CpG IDs + β-values)
├── model/
│   └── scbert_methylation.yaml # Model architecture
├── data_module/
│   └── methylation.yaml       # Data loading parameters
├── trainer/
│   └── methylation.yaml       # Training hyperparameters
├── pretrain_config.yaml       # Pretraining main config
└── finetune_config.yaml       # Fine-tuning main config
```

---

# Part 3: Tokenizer Adaptation

## 3.1 The Challenge

**scRNA-seq (Original):**
```python
# Gene names are well-known vocabulary
genes = ["BRCA1", "TP53", "EGFR", ...]  # ~20K genes
expressions = [150, 0, 2345, ...]  # Count values
```

**DNA Methylation (Adaptation):**
```python
# CpG probe IDs are Illumina array identifiers
cpg_sites = ["cg00000029", "cg00000108", ...]  # ~21K CpGs
beta_values = [0.85, 0.12, 0.67, ...]  # Continuous 0-1
```

## 3.2 Solution: `tokenizer.py`

```python
# tokenizer.py - Key Implementation

SPECIAL_TOKENS = {
    "pad_token": "[PAD]",   # ID 2
    "unk_token": "[UNK]",   # ID 0
    "cls_token": "[CLS]",   # ID 3
    "sep_token": "[SEP]",   # ID 1
    "mask_token": "[MASK]", # ID 4
}

def create_cpg_vocabulary(cpg_sites: List[str], output_dir: str):
    """
    Create vocabulary file mapping CpG names to token IDs.

    vocab.txt:
        [UNK]        # 0
        [SEP]        # 1
        [PAD]        # 2
        [CLS]        # 3
        [MASK]       # 4
        cg00000029   # 5
        cg00000108   # 6
        ...
    """
    vocab = [
        SPECIAL_TOKENS["unk_token"],   # Must be ID 0
        SPECIAL_TOKENS["sep_token"],   # Must be ID 1
        SPECIAL_TOKENS["pad_token"],   # Must be ID 2
        SPECIAL_TOKENS["cls_token"],   # Must be ID 3
        SPECIAL_TOKENS["mask_token"],  # Must be ID 4
    ]
    vocab.extend(cpg_sites)  # CpG IDs start at 5

    with open(output_dir / "vocab.txt", "w") as f:
        f.write("\n".join(vocab))

def create_methylation_multifield_tokenizer(cpg_sites, output_dir):
    """
    Create MultiFieldTokenizer for methylation data.

    Only creates tokenizer for cpg_sites field.
    beta_values are continuous - handled by ContinuousValueEncoder.
    """
    # Create BertTokenizerFast for CpG IDs
    create_methylation_tokenizer(cpg_sites, output_dir, "cpg_sites")

    # Load as MultiFieldTokenizer
    return MultiFieldTokenizer.from_pretrained(output_dir)
```

## 3.3 Key Design Decision: Two-Field Approach

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MULTI-FIELD TOKENIZATION                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Raw Sample:                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  {"cg00000029": 0.85, "cg00000108": 0.12, "cg00000165": 0.67}      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                           │                                                │
│           ┌───────────────┴───────────────┐                               │
│           │                               │                               │
│           ▼                               ▼                               │
│  ┌─────────────────┐           ┌─────────────────────────┐               │
│  │  Field 1:       │           │  Field 2:               │               │
│  │  cpg_sites      │           │  beta_values            │               │
│  │  (DISCRETE)     │           │  (CONTINUOUS)           │               │
│  │                 │           │                         │               │
│  │  Tokenization:  │           │  Encoding:              │               │
│  │  BertTokenizer  │           │  ContinuousValueEncoder │               │
│  │                 │           │  (MLP: 1→128→512)       │               │
│  │  cg00000029 → 5 │           │  0.85 → [e₁, e₂, ..., e₅₁₂] │          │
│  │  cg00000108 → 6 │           │  0.12 → [e₁, e₂, ..., e₅₁₂] │          │
│  │  cg00000165 → 7 │           │  0.67 → [e₁, e₂, ..., e₅₁₂] │          │
│  └─────────────────┘           └─────────────────────────┘               │
│           │                               │                               │
│           ▼                               ▼                               │
│  ┌─────────────────┐           ┌─────────────────────────┐               │
│  │  nn.Embedding   │           │  MLP Embeddings         │               │
│  │  (8005, 512)    │           │  (batch, seq, 512)      │               │
│  └─────────────────┘           └─────────────────────────┘               │
│           │                               │                               │
│           └───────────────┬───────────────┘                               │
│                           │                                                │
│                           ▼                                                │
│                  ┌─────────────────┐                                      │
│                  │  ELEMENT-WISE   │                                      │
│                  │  ADDITION       │                                      │
│                  │  CpG + β        │                                      │
│                  └─────────────────┘                                      │
│                           │                                                │
│                           ▼                                                │
│                  Combined Embeddings                                      │
│                  (batch, seq, 512)                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

# Part 4: Field Configuration

## 4.1 `configs/fields/methylation.yaml`

```yaml
# Field 1: CpG Site IDs (Discrete Tokens)
- _target_: bmfm_targets.config.FieldInfo
  field_name: "cpg_sites"
  vocab_size: 8005           # 8000 CpG sites + 5 special tokens
  is_input: true             # Used as model input
  is_masked: false           # NOT masked during MLM (fixed identifiers)
  tokenization_strategy: "tokenize"  # Standard vocabulary lookup

# Field 2: Beta Values (Continuous)
- _target_: bmfm_targets.config.FieldInfo
  field_name: "beta_values"
  is_input: true             # Used as model input
  is_masked: true            # MASKED during MLM (prediction target)
  tokenization_strategy: "continuous_value_encoder"
  num_special_tokens: 5      # PAD, UNK, CLS, SEP, MASK
  encoder_kwargs:
    kind: "mlp_with_special_token_embedding"
  decode_modes:
    regression: {}           # Predict continuous values with MSE
```

## 4.2 Why This Design?

| Aspect | CpG IDs | Beta Values |
|--------|---------|-------------|
| **Type** | Categorical | Continuous |
| **Masking** | NO (fixed identifiers) | YES (prediction target) |
| **Encoder** | nn.Embedding | MLP (ContinuousValueEncoder) |
| **Reasoning** | We always know WHICH site we're looking at | We want to predict the VALUE |

**This differs from scGPT/MethylGPT** which mask both gene IDs and values. BMFM's approach is cleaner:
- Gene/CpG IDs are structural information (always known)
- Expression/methylation values are the biological signal (prediction target)

---

# Part 5: Architecture Adaptation

## 5.1 Model Configuration: `configs/model/scbert_methylation.yaml`

```yaml
_target_: bmfm_targets.config.SCBertConfig
_partial_: true

# Architecture (smaller than BERT-base for efficiency)
num_hidden_layers: 6         # 6 transformer layers
num_attention_heads: 8       # 8 attention heads
hidden_size: 512             # 512 hidden dimension
intermediate_size: 2048      # FFN dimension (4x hidden)
hidden_act: "gelu"           # GELU activation

# Regularization
hidden_dropout_prob: 0.1
attention_probs_dropout_prob: 0.1
classifier_dropout: 0.1

# Sequence length
max_position_embeddings: 8002  # 8000 CpG + [CLS] + [SEP]

# Flash Attention for memory efficiency
attention: "torch"           # Uses F.scaled_dot_product_attention
```

## 5.2 Architecture Comparison

| Component | BMFM-RNA (Original) | BMFM-Methylation |
|-----------|---------------------|------------------|
| **Layers** | 12 | **6** |
| **Heads** | 12 | **8** |
| **Hidden** | 768 | **512** |
| **FFN** | 3072 | **2048** |
| **Vocab** | ~60K genes | **8005** CpGs |
| **Max Seq** | 4096 | **8002** |
| **Parameters** | ~110M | **~25M** |

**Why smaller?**
- Methylation data is more homogeneous than scRNA-seq
- Fewer samples available (11K vs 33M cells in BMFM)
- Faster iteration for research

## 5.3 Model Wrapper: `model.py`

```python
class MethylationEncoder(nn.Module):
    """
    Wrapper around the original BMFM SCBertModel.
    Converts methylation inputs to BMFM's expected format.
    """

    def __init__(self, config: SCBertConfig):
        super().__init__()
        # Use the EXACT SAME SCBertModel from BMFM
        self.encoder = SCBertModel(config, add_pooling_layer=True)

        # Optionally stabilize embedding fusion
        patch_embeddings_add_stabilized(self.encoder)

    def forward(self, cpg_ids, beta_values, attention_mask=None):
        # Convert to BMFM format: [batch, num_fields, seq_len]
        input_ids = torch.zeros(batch_size, 2, seq_len)
        input_ids[:, 0, :] = cpg_ids.float()   # Field 0: CpG IDs
        input_ids[:, 1, :] = beta_values       # Field 1: β-values

        # Forward through original BMFM encoder
        outputs = self.encoder(input_ids, attention_mask)

        return {
            "last_hidden_state": outputs.last_hidden_state,
            "pooler_output": outputs.pooler_output,  # [CLS] embedding
        }


class MethylationAgeModel(nn.Module):
    """Complete model for age prediction."""

    def __init__(self, config):
        self.encoder = MethylationEncoder(config)

        # Age regression head (MLP on [CLS])
        self.age_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

    def forward(self, cpg_ids, beta_values, attention_mask):
        outputs = self.encoder(cpg_ids, beta_values, attention_mask)
        pooled = outputs["pooler_output"]  # [batch, 512]
        age = self.age_head(pooled)        # [batch, 1]
        return age
```

---

# Part 6: Data Pipeline

## 6.1 Data Module: `data_module.py`

```python
class MethylationDataset(Dataset):
    """Loads methylation data from h5ad (AnnData) format."""

    def __init__(self, h5ad_path, split="train"):
        self.adata = sc.read_h5ad(h5ad_path)

        # Filter by split
        if split:
            mask = self.adata.obs["split"] == split
            self.adata = self.adata[mask]

        # Get CpG site names
        self.cpg_sites = list(self.adata.var_names)

        # Get ages
        self.ages = self.adata.obs["age"].values

    def __getitem__(self, idx):
        beta_values = self.adata.X[idx]  # [8000] β-values
        age = self.ages[idx]

        return MultiFieldInstance(
            data={"beta_values": beta_values},
            metadata={"labels": age}
        )


class MethylationCollator:
    """
    Collates samples into batches.

    Key feature: FIXED CpG subset selection.
    """

    def __init__(self, tokenizer, cpg_sites, k=2048, fixed_subset=True):
        self.k = k  # Number of CpGs to use per sample
        self.fixed_subset = fixed_subset

        if fixed_subset:
            # Select K CpGs ONCE (same for all samples)
            rng = np.random.default_rng(42)
            self.fixed_indices = np.sort(
                rng.choice(len(cpg_sites), size=k, replace=False)
            )

    def __call__(self, examples):
        for example in examples:
            betas = example.data["beta_values"]

            # Use FIXED subset of CpGs
            subset_idx = self.fixed_indices

            # Build sequence: [CLS] + selected CpGs
            cpg_ids[i] = [CLS_ID] + [vocab[cpg_sites[j]] for j in subset_idx]
            beta_values[i] = [CLS_BETA] + [betas[j] for j in subset_idx]

            # Apply masking for MLM (optional)
            if self.mask_ratio > 0:
                mask_pos = random.choice(positions, int(0.3 * len))
                beta_values[i, mask_pos] = MASK_BETA  # -1.0

        return {
            "cpg_ids": cpg_ids,
            "beta_values": beta_values,
            "attention_mask": attention_mask,
        }
```

## 6.2 Why Fixed CpG Subset?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              RANDOM vs FIXED CpG SUBSET SELECTION                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  RANDOM SUBSET (Original Option-B):                                        │
│  ────────────────────────────────                                          │
│                                                                             │
│  Sample 1: CpG[102, 4521, 892, ...]   (random 2048 each time)             │
│  Sample 2: CpG[7, 3302, 5001, ...]    (different 2048)                    │
│  Sample 3: CpG[456, 12, 8834, ...]    (different again)                   │
│                                                                             │
│  Problem: Model sees different CpGs each epoch                             │
│           → Cannot learn specific CpG-age associations                     │
│                                                                             │
│  ═══════════════════════════════════════════════════════════════════════  │
│                                                                             │
│  FIXED SUBSET (Our Approach):                                              │
│  ────────────────────────────                                              │
│                                                                             │
│  Sample 1: CpG[5, 12, 89, 102, ...]   (same 2048 always)                  │
│  Sample 2: CpG[5, 12, 89, 102, ...]   (same 2048)                         │
│  Sample 3: CpG[5, 12, 89, 102, ...]   (same 2048)                         │
│                                                                             │
│  Advantage: Model learns which CpGs are informative for age              │
│             → Better generalization                                        │
│             → Reproducibility                                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

# Part 7: Pretraining Pipeline

## 7.1 Pretraining Configuration: `pretrain_config.yaml`

```yaml
defaults:
  - data_module: methylation
  - fields: methylation
  - model: scbert_methylation

# Paths
data_path: ???              # Required: h5ad file
output_directory: ./outputs
tokenizer_path: ./tokenizer

# Training
pretrain_epochs: 300
accumulate_grad_batches: 4  # Effective batch = 32 × 4 = 128
early_stop_patience: 15

# MLM Settings
mlm_enabled: true
collation_strategy: "language_modeling"

# Embedding combination
combine_style: add          # h = CpG_embed + β_embed
```

## 7.2 Pretraining Script: `pretrain.py`

```python
@hydra.main(config_path="configs", config_name="pretrain_config")
def main(cfg):
    # 1. Setup tokenizer
    tokenizer = setup_tokenizer(cfg)

    # 2. Setup data module with MLM
    data_module = MethylationDataModule(
        tokenizer=tokenizer,
        fields=cfg.fields,
        h5ad_path=cfg.data_path,
        mlm=True,                          # Enable masking
        change_ratio=0.15,                 # Mask 15% of tokens
        mask_ratio=0.8,                    # 80% of masked → [MASK]
        switch_ratio=0.1,                  # 10% → random value
        collation_strategy="language_modeling",
    )

    # 3. Create model
    model = MLMTrainingModule(
        model_config=model_config,
        trainer_config=trainer_config,
        tokenizer=tokenizer,
    )

    # 4. Train with PyTorch Lightning
    trainer = pl.Trainer(
        max_epochs=cfg.pretrain_epochs,
        accelerator="gpu",
        precision="16-mixed",
        callbacks=[
            ModelCheckpoint(monitor="validation/loss"),
            EarlyStopping(patience=cfg.early_stop_patience),
        ],
    )
    trainer.fit(model, data_module)
```

## 7.3 MLM Objective for Methylation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   MASKED LANGUAGE MODELING (MLM)                            │
│                   Adapted for Methylation β-values                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input Sample:                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  CpG IDs:  [CLS]  cg29   cg108  cg165  cg201  cg445  cg892  ...    │   │
│  │  β-values: -2.0   0.85   0.12   0.67   0.34   0.91   0.23   ...    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                    Apply 15% masking (only to β-values!)                   │
│                                    │                                        │
│                                    ▼                                        │
│  Masked Input:                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  CpG IDs:  [CLS]  cg29   cg108  cg165  cg201  cg445  cg892  ...    │   │
│  │  β-values: -2.0   [MASK] 0.12   [MASK] 0.34   0.91   [MASK] ...    │   │
│  │                   ↑             ↑                    ↑              │   │
│  │                   Mask token = -1.0                                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                          Transformer Encoder                               │
│                                    │                                        │
│                                    ▼                                        │
│  Predictions:                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Predicted β: [CLS]  0.82   0.12   0.71   0.34   0.91   0.19  ...  │   │
│  │  True β:      [CLS]  0.85   0.12   0.67   0.34   0.91   0.23  ...  │   │
│  │                       ↑             ↑                    ↑          │   │
│  │                      Loss computed only on masked positions        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Loss = MSE(predicted_masked, true_masked)                                 │
│       = MSE([0.82, 0.71, 0.19], [0.85, 0.67, 0.23])                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 7.4 Key Difference from BMFM-RNA Pretraining

| Aspect | BMFM-RNA (scRNA-seq) | BMFM-Methylation |
|--------|---------------------|------------------|
| **Masked Field** | Expression values | β-values |
| **Value Range** | 0-10,000+ | 0.0-1.0 |
| **Loss** | MSE on log-normalized | MSE on raw β |
| **CpG/Gene IDs** | NOT masked | NOT masked |
| **Mask Token** | Negative value (-1) | Negative value (-1) |

---

# Part 8: Fine-tuning Pipeline

## 8.1 Fine-tuning Configuration: `finetune_config.yaml`

```yaml
defaults:
  - data_module: methylation
  - fields: methylation
  - model: scbert_methylation

# Paths
data_path: ???
checkpoint_path: ???        # Pretrained MLM checkpoint
output_directory: ./outputs

# Training
finetune_epochs: 300
accumulate_grad_batches: 2  # Effective batch = 32 × 2 = 64

# Regression head
regression_head:
  hidden_size: 256
  dropout: 0.2

# Optimizer
trainer:
  learning_rate: 0.0005     # 5e-4
  weight_decay: 0.01
  warmup_steps: 100

# Early stopping
early_stopping:
  patience: 100
  monitor: "val/mae"

# Options
freeze_encoder: true        # Freeze pretrained encoder
use_huber_loss: false       # Use MSE (can switch to Huber)
```

## 8.2 Fine-tuning Script: `finetune.py`

```python
class MethylationAgeRegressor(pl.LightningModule):
    """Age regression from methylation using pretrained encoder."""

    def __init__(self, encoder, config):
        self.encoder = encoder

        # Freeze pretrained encoder (optional but recommended)
        if config.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

        # Loss function
        self.loss_fn = nn.HuberLoss(delta=1.0)  # or nn.MSELoss()

    def forward(self, input_ids, attention_mask):
        # Get encoder output
        outputs = self.encoder(input_ids, attention_mask)

        # Use [CLS] token (pooled output)
        pooled = outputs.pooler_output  # [batch, 512]

        # Predict age
        age_pred = self.regression_head(pooled)  # [batch, 1]
        return age_pred

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]  # Normalized ages

        predictions = self(input_ids, attention_mask)
        loss = self.loss_fn(predictions, labels)

        # Denormalize for MAE
        preds_denorm = predictions * self.age_std + self.age_mean
        labels_denorm = labels * self.age_std + self.age_mean
        mae = torch.abs(preds_denorm - labels_denorm).mean()

        self.log("train/loss", loss)
        self.log("train/mae", mae)
        return loss


@hydra.main(config_path="configs", config_name="finetune_config")
def main(cfg):
    # 1. Load pretrained encoder
    from bmfm_targets.training.modules import MLMTrainingModule

    pretrained = MLMTrainingModule.load_from_checkpoint(cfg.checkpoint_path)
    encoder = pretrained.model.scbert  # Extract SCBertModel

    # 2. Create age regression model
    model = MethylationAgeRegressor(
        encoder=encoder,
        hidden_size=cfg.model.hidden_size,
        age_mean=data_module.age_mean,
        age_std=data_module.age_std,
        freeze_encoder=cfg.freeze_encoder,
    )

    # 3. Train
    trainer = pl.Trainer(
        max_epochs=cfg.finetune_epochs,
        callbacks=[
            ModelCheckpoint(monitor="val/mae"),
            EarlyStopping(patience=cfg.early_stopping.patience),
        ],
    )
    trainer.fit(model, data_module)
```

## 8.3 Fine-tuning Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      FINE-TUNING ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input:                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  cpg_ids:     [3, 5, 6, 7, 8, ...]         (2049 tokens)           │   │
│  │  beta_values: [-2.0, 0.85, 0.12, 0.67, ...] (2049 values)          │   │
│  │  attention:   [1, 1, 1, 1, 1, ...]         (all attended)          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    PRETRAINED ENCODER                               │   │
│  │                    (FROZEN - no gradient)                           │   │
│  │                                                                      │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │  SCEmbeddingsLayer                                          │   │   │
│  │  │  h = α·CpG_embed + β_embed                                  │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  │                              │                                      │   │
│  │                              ▼                                      │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │  6× Transformer Layers                                      │   │   │
│  │  │  (8 heads, 512 hidden, 2048 FFN)                           │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  │                              │                                      │   │
│  │                              ▼                                      │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │  SCPooler                                                    │   │   │
│  │  │  pooled = Tanh(Linear(hidden_states[:, 0]))                 │   │   │
│  │  │         = [CLS] embedding                                    │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  │                                                                      │   │
│  │  Output: pooled_output (batch, 512)                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    REGRESSION HEAD                                  │   │
│  │                    (TRAINABLE)                                      │   │
│  │                                                                      │   │
│  │     Linear(512, 256) → BatchNorm → GELU → Dropout(0.2)            │   │
│  │                              │                                      │   │
│  │     Linear(256, 128) → BatchNorm → GELU → Dropout(0.2)            │   │
│  │                              │                                      │   │
│  │     Linear(128, 1)                                                 │   │
│  │                              │                                      │   │
│  │  Output: predicted_age (batch, 1)                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│                      Loss = MSE(predicted, true)                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

# Part 9: Results & Analysis

## 9.1 Dataset Statistics

| Split | Samples | Percentage |
|-------|---------|------------|
| Train | 5,483 | 47.7% |
| Validation | 1,371 | 11.9% |
| Test | 4,646 | 40.4% |
| **Total** | **11,500** | 100% |

**Age Distribution:**
- Range: 0 - 114 years
- Mean: 42.0 ± 26.5 years
- CpG Sites: 8,000 (subset of 850K)
- Fixed Subset: 2,048 CpGs used

## 9.2 Multi-Seed Performance (n=3 seeds)

| Metric | Mean ± Std | Best | Status |
|--------|-----------|------|--------|
| **MAE (years)** | **4.84 ± 0.16** | 4.67 | ✅ |
| **R²** | **0.926 ± 0.005** | 0.931 | ✅ |
| **MedAE (years)** | 3.75 ± 0.11 | 3.63 | Est. |
| **RMSE (years)** | 6.41 ± 0.21 | 6.17 | Est. |

## 9.3 Comparison with Baselines

| Model | MAE | R² | Notes |
|-------|-----|-----|-------|
| **BMFM-Methylation** | **4.84** | **0.926** | Our model |
| MethylGPT (baseline) | 4.95 | 0.920 | scGPT-based |
| Ridge Regression | 4.49 | 0.940 | Linear baseline |
| Horvath Clock | ~3.6 | ~0.960 | Pan-tissue |
| Hannum Clock | ~4.9 | ~0.910 | Blood-based |

## 9.4 Training Dynamics

```
Training Curve:

MAE │
    │ ●●●●●
 10 │      ●●●●
    │          ●●●●
  8 │              ●●●●
    │                  ●●●
  6 │                     ●●●●●●●●●●●●●●
    │                               ●●●●●●●
  4 │─────────────────────────────────────●●●●●●●●●●●
    │
  2 │
    └───────────────────────────────────────────────▶
    0           50          100         150      Epoch

    ──── Train MAE    ──── Val MAE    ──── Test MAE
```

**Key Observations:**
- Rapid convergence in first 50 epochs
- Best performance at epoch 85-120
- Moderate overfitting (train-val gap: 2.85 years)

---

# Part 10: Summary & Key Takeaways

## 10.1 What We Changed

| Component | Original BMFM-RNA | Methylation Adaptation |
|-----------|-------------------|------------------------|
| **Tokenizer** | Gene symbols | CpG probe IDs |
| **Vocabulary** | ~60K genes | 8K CpGs + 5 special |
| **Value Range** | 0-10,000+ | 0.0-1.0 |
| **Task** | Cell type classification | Age regression |
| **Model Size** | 110M params | 25M params |
| **Subset** | Random per sample | Fixed 2048 |

## 10.2 Files Summary

```
methyl/bmfm_methylation/
├── tokenizer.py      ← Creates CpG vocabulary + MultiFieldTokenizer
├── data_module.py    ← h5ad loading + MethylationCollator
├── model.py          ← MethylationEncoder + MethylationAgeModel
├── pretrain.py       ← MLM pretraining on β-values
├── finetune.py       ← Age regression from [CLS]
├── config.py         ← Configuration dataclasses
├── configs/
│   ├── fields/methylation.yaml      ← Two-field definition
│   ├── model/scbert_methylation.yaml ← Architecture config
│   └── pretrain_config.yaml          ← Training config
```

## 10.3 Key Design Decisions

1. **Two-Field Tokenization:** CpG IDs (discrete) + β-values (continuous)
2. **Mask Only Values:** CpG IDs are structural, β-values are targets
3. **Fixed CpG Subset:** Reproducibility + better learning
4. **Frozen Encoder:** Reduces overfitting in fine-tuning
5. **[CLS] Pooling:** Global sample representation for age

## 10.4 Final Results

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FINAL PERFORMANCE                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                      MAE = 4.84 years                                      │
│                      R² = 0.926                                            │
│                                                                             │
│  ✅ Successfully adapted BMFM-RNA for DNA methylation                      │
│  ✅ Comparable to established epigenetic clocks                            │
│  ✅ Multi-seed validation shows consistency (σ = 0.16)                     │
│  ✅ Explains 92.6% of variance in chronological age                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

# Appendix A: Running the Code

## A.1 Installation

```bash
git clone https://github.com/netanelazran11/BiomedSciAI_ACL_Project.git
cd BiomedSciAI_ACL_Project/methyl
pip install -r requirements.txt
```

## A.2 Pretraining

```bash
python -m bmfm_methylation.pretrain \
    data_path=/path/to/methylation.h5ad \
    output_directory=./outputs \
    pretrain_epochs=300
```

## A.3 Fine-tuning

```bash
python -m bmfm_methylation.finetune \
    data_path=/path/to/methylation.h5ad \
    checkpoint_path=./outputs/pretrain/best.ckpt \
    output_directory=./outputs \
    finetune_epochs=300
```

---

# Appendix B: References

1. **BMFM-RNA:** Dandala et al. "BMFM-RNA: An Open Framework for Building and Evaluating Transcriptomic Foundation Models" arXiv:2506.14861, 2025

2. **MethylGPT:** Transformer-based methylation clock (baseline comparison)

3. **Horvath Clock:** Horvath, S. "DNA methylation age of human tissues and cell types" Genome Biology, 2013

4. **scGPT:** Cui et al. "scGPT: Toward Building a Foundation Model for Single-Cell Multi-omics Using Generative AI" Nature Methods, 2024

---

**End of Presentation**

*Last Updated: February 2026*
