# MethylLlama: A Foundation Model for DNA Methylation Age Prediction

**MSc Thesis — Hebrew University of Jerusalem**  
**Author:** Netanel Azran · [netanelazran11@gmail.com](mailto:netanelazran11@gmail.com)  
**Supervisors:** Prof. Michal Rozen-Zvi · Prof. Benjamin Yakir

---

## Overview

**MethylLlama** adapts a Llama-style Transformer foundation model to predict biological age from DNA methylation (DNAm) profiles. CpG sites are tokenized by identity and β-value, pretrained with a Whole-Cell Expression Decoder (WCED) objective, then fine-tuned for age regression across diverse human tissues.

The project benchmarks against [MethylGPT (Ying et al., 2024)](https://www.biorxiv.org/content/10.1101/2024.10.30.621013v2) on both datasets.

---

## Architecture

```
Input: N CpG sites per sample
  ↓  CpG-ID embedding + β-value embedding (element-wise sum)
  ↓  [CLS] + CpG token sequence
  ↓  Transformer encoder (6 layers, 8 heads, d=512)
  ↓  CLS pooling (pooler_output)
  ↓  Regression head (512 → 256 → 128 → 1)
Output: Predicted biological age (years)
```

| Parameter | MethylLlama | MethylGPT |
|-----------|-------------|-----------|
| Layers | 6 | 6 |
| Heads | 8 | 4 |
| Hidden dim | 512 | 64 |
| Params (encoder) | ~23M | ~2M |
| CpG subset | 8,000–19,608 | 49,156 |
| Pretraining | WCED (InfoNCE + reconstruction) | MLM + reconstruction |

---

## Repository Structure

```
MethylLlama/
├── bmfm_methylation/          # Core model codebase
│   ├── llama/                 # Llama-style transformer + fine-tuning
│   │   ├── configs/           # Hydra configs (pretrain / finetune)
│   │   ├── finetune_llama.py  # Fine-tuning Lightning module
│   │   └── pretrain_llama.py  # Pretraining Lightning module
│   ├── wced/                  # WCED pretraining (InfoNCE + reconstruction)
│   ├── mlm/                   # MLM pretraining variant
│   └── shared/                # Tokenizer, data module, dataset
├── scripts/
│   ├── llama/                 # SLURM scripts for Llama pretraining/finetuning
│   ├── wced/                  # SLURM scripts for WCED pretraining
│   └── utils/                 # Analysis, baseline, and inspection scripts
├── data_prep/                 # Data preprocessing and merge scripts
├── tokenizer/                 # CpG-site and β-value tokenizers
├── data/                      # Dataset metadata (CpG lists, split info)
├── docs/
│   ├── presentations/         # Slides and comparison docs
│   └── images/                # Architecture diagrams and training curves
├── requirements.txt
└── README.md
```

---

## Pretraining

**WCED (Whole-Cell Expression Decoder):**
- All 49,156 CpG β-values are input (no masking)
- [CLS] reconstructs the full methylation profile
- InfoNCE contrastive loss on CLS embeddings
- Best checkpoint: epoch 190, val_loss=0.1264, test/pcc=0.987

```bash
sbatch scripts/wced/pretrain_wced.sh
```

---

## Fine-tuning

```bash
# V4b warmstart (loads WCED pretrain checkpoint)
sbatch scripts/llama/finetune_llama_small_v4b.sh

# V4b scratch (no warmstart — ablation)
WARMSTART_WEIGHTS="" sbatch scripts/llama/finetune_llama_small_v4b.sh
```

**V4b hyperparameters:**
- Loss: Huber (δ=5yr / age_std)
- LR: 1e-4 (head), 2e-5 (encoder), weight decay 0.01
- Batch: 32, grad accum 4 (eff. 128)
- Early stopping: patience 100
- Max epochs: 300

---

## WandB Runs

| Run | WandB ID | Description |
|-----|----------|-------------|
| V4b warmstart | `1w1rk694` | Best fine-tuning run |
| V4b scratch | `8mjxsoez` | Ablation — no warmstart |
| V4 (reference) | `bvt444p3` | Earlier run (stopped ep150) |
| WCED pretrain | job 44450919 | Best pretrain checkpoint |

---

## Installation

```bash
git clone https://github.com/netanelazran11/MethylLlama.git
cd MethylLlama
python3.10 -m venv bmfm_methyl_env
source bmfm_methyl_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install git+https://github.com/netanelazran11/BiomedSciAI_ACL_Project.git --no-deps
```

---

## References

1. Ying, K. et al. "MethylGPT: a foundation model for the DNA methylome." *bioRxiv*, 2024.
2. Touvron, H. et al. "Llama 2: Open foundation and fine-tuned chat models." *arXiv*, 2023.
3. Horvath, S. "DNA methylation age of human tissues and cell types." *Genome Biology*, 14(10):R115, 2013.
4. de Lima Camillo, L.P. et al. "AltumAge: A pan-tissue DNA methylation epigenetic clock based on deep learning." *npj Aging*, 8(1):1–15, 2022.
5. Zou, H. & Hastie, T. "Regularization and variable selection via the elastic net." *J. R. Stat. Soc. B*, 67(2):301–320, 2005.

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
