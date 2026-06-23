# MethylLlama: A Foundation Model for DNA Methylation Age Prediction

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-netanelazran1%2FMethylLlama-yellow)](https://huggingface.co/netanelazran1/MethylLlama)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)

**MSc Thesis — Hebrew University of Jerusalem**
**Author:** Netanel Azran · [netanelazran11@gmail.com](mailto:netanelazran11@gmail.com)
**Supervisors:** Prof. Michal Rozen-Zvi · Prof. Benjamin Yakir

---

![MethylLlama pipeline](docs/images/nanno%20banna%20generate%20visual%20/CATEGORY_1___Archite-418490.png)

---

## Overview

**MethylLlama** adapts a Llama-style Transformer foundation model to predict biological age from DNA methylation (DNAm) profiles. CpG sites are tokenised by identity and β-value, pretrained with a **Whole-Cell Expression Decoder (WCED)** objective, then fine-tuned for age regression across diverse human tissues.

**Main result**: WCED pretraining vs random initialisation — same architecture, same data, same fine-tuning recipe:

| Model | Test R² | MedAE | MAE |
|---|---|---|---|
| **WCED pretrained** (job 44895876) | **0.905** | **3.65 yr** | 5.46 yr |
| Random init (job 44981091) | 0.526 | 8.91 yr | 13.06 yr |

---

## Quick Start

### Tutorials (inference only — no cluster needed)

```bash
git clone https://github.com/netanelazran11/MethylLlama.git
cd MethylLlama
pip install -r requirements_tutorials.txt
pip install -e .
jupyter lab tutorials/quickstart.ipynb
```

The checkpoint and tokenizer download automatically from HuggingFace on first run (~330 MB).

### Full training environment (cluster)

```bash
pip install -r requirements.txt
pip install git+https://github.com/netanelazran11/BiomedSciAI_ACL_Project.git --no-deps
pip install -e .
```

Or with conda:

```bash
conda env create -f environment.yml
conda activate bmfm_methyl_env
pip install git+https://github.com/netanelazran11/BiomedSciAI_ACL_Project.git --no-deps
pip install -e .
```

---

## Architecture

```
Input: N CpG sites per sample
  ↓  CpG-ID embedding + β-value embedding (ScaleAdaptEncoder, sinusoidal basis)
  ↓  [CLS] + CpG token sequence
  ↓  4 × LlamaLayer (RMSNorm + MHA/RoPE + SwiGLU, Pre-LN)
  ↓  CLS pooling (pooler_output = Linear + Tanh)
  ↓  Regression head (256 → 256 → 1)
Output: Predicted biological age (years)
```

| Parameter | MethylLlama (small) | MethylGPT |
|---|---|---|
| Layers | 4 | 6 |
| Heads | 4 | 4 |
| Hidden dim | 256 | 64 |
| Norm | RMSNorm | LayerNorm |
| Position | RoPE | Absolute |
| FFN | SwiGLU | GELU |
| CpG input | 21,368 | 49,156 |

---

## Repository Structure

```
MethylLlama/
├── bmfm_methylation/          # Core model codebase
│   ├── llama/                 # MethylLlama model + WCED training + fine-tuning
│   │   ├── model.py           # MethylLlamaModel (RoPE, SwiGLU, RMSNorm, ScaleAdapt)
│   │   ├── wced_llama.py      # WCEDLlamaModule (pretraining Lightning module)
│   │   ├── finetune_llama.py  # MethylationAgeRegressorLlama (fine-tuning)
│   │   └── configs/           # Hydra configs (pretrain / finetune)
│   ├── wced/                  # SCBert WCED variant
│   ├── mlm/                   # MLM pretraining variant
│   └── shared/                # Tokenizer, data module, dataset
├── tutorials/
│   ├── data/                  # Demo dataset (500 samples × 21,368 CpGs)
│   ├── quickstart.ipynb       # Load checkpoint → CLS embeddings → age predictions
│   ├── embedding_analysis.ipynb  # UMAP by age and tissue type
│   └── age_prediction.ipynb   # Fine-tuning walkthrough + paper results
├── scripts/
│   ├── llama/                 # SLURM scripts for pretrain / finetune
│   ├── wced/                  # SCBert WCED scripts
│   └── download_checkpoint.sh # Download from HuggingFace
├── data_prep/                 # Dataset preprocessing scripts
├── tokenizer/                 # CpG-site and β-value tokenizers
├── data/                      # CpG manifests and split info
├── docs/                      # Presentations, figures, results
├── environment.yml
├── requirements.txt
└── pyproject.toml
```

---

## Pretraining

**WCED (Whole-Cell Expression Decoder)**:
- 50 % of CpG β-values as input (random subset per step)
- CLS token reconstructs the full methylation profile via a decoder
- InfoNCE contrastive loss on CLS embeddings (two views per sample)
- Auxiliary age head prevents CLS collapse
- Best checkpoint: epoch 98, val_loss=0.0059

```bash
sbatch scripts/llama/pretrain_llama_small.sh
```

Pretrained checkpoint available on [HuggingFace](https://huggingface.co/netanelazran1/MethylLlama):

```python
from huggingface_hub import hf_hub_download
ckpt_path = hf_hub_download("netanelazran1/MethylLlama",
                             "pretrain_llama_epoch98_val0.0059.ckpt")
```

---

## Fine-tuning

```bash
# WCED pretrained encoder → age regression
sbatch scripts/llama/finetune_llama_small_v5.sh

# Random-init ablation
sbatch scripts/llama/finetune_llama_random_init.sh
```

**Key hyperparameters (V5 best run)**:
- Loss: Huber (δ = 5 yr / age_std)
- LR: 1e-4 (head), 2e-5 (encoder, unfrozen at epoch 10)
- Batch: 32, grad accum 4 (eff. 128)
- Early stopping: patience 100, max epochs 300

---

## Tutorials

| Notebook | Description |
|---|---|
| [`tutorials/quickstart.ipynb`](tutorials/quickstart.ipynb) | Load checkpoint from HF, tokenise demo data, extract CLS embeddings |
| [`tutorials/embedding_analysis.ipynb`](tutorials/embedding_analysis.ipynb) | UMAP coloured by age and tissue type |
| [`tutorials/age_prediction.ipynb`](tutorials/age_prediction.ipynb) | Fine-tuning walkthrough and paper results |

The demo dataset (`tutorials/data/methylllama_demo_500samples.h5ad`) contains
500 stratified samples (0–102 yr, 15 tissue types) from the 21k CpG dataset.

---


## Running on a Cluster (SLURM)

All scripts under `scripts/` target a SLURM cluster. Update the following
variables at the top of each script before submitting:

```bash
REPO="/path/to/your/MethylLlama"   # clone location on your cluster
DATA="/path/to/your/data.h5ad"     # input dataset (altumage_21k_3way.h5ad)
```

The partition (`--partition=salmon`) and GPU resource (`--gres=gpu:l40s:1`)
lines should also be updated to match your cluster's hardware.

---

## References

1. Ying, K. et al. "MethylGPT: a foundation model for the DNA methylome." *bioRxiv*, 2024.
2. Touvron, H. et al. "Llama 2: Open foundation and fine-tuned chat models." *arXiv*, 2023.
3. Horvath, S. "DNA methylation age of human tissues and cell types." *Genome Biology*, 14(10):R115, 2013.
4. de Lima Camillo, L.P. et al. "AltumAge: A pan-tissue DNA methylation epigenetic clock based on deep learning." *npj Aging*, 8(1):1–15, 2022.

---

## Citation

```bibtex
@mastersthesis{azran2026methylllama,
  title   = {MethylLlama: A Foundation Model for DNA Methylation Age Prediction},
  author  = {Azran, Netanel},
  school  = {Hebrew University of Jerusalem},
  year    = {2026},
  url     = {https://github.com/netanelazran11/MethylLlama}
}
```
