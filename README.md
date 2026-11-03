<!--
RELEASE TODO (remove this block before tagging v1.0.0)
- (done) tokenizer/ and resources/ added from the cluster bundle; scripts point at them.
- (done 2026-10-18) checkpoints on Hugging Face (commit 350dd50), download script verified end to end.
- Fill in the two data-source links in "Data" and the citation once the preprint exists.
-->

# MethylLlama

MethylLlama is a Transformer encoder for DNA methylation profiles, pretrained without phenotype
labels on 169,120 array profiles and fine-tuned here for chronological-age prediction. It is
pretrained with the **Whole-CpG Encoder–Decoder (WCED)** objective: two random partial views of
each profile are encoded, the sample representation of each view must reconstruct the measured
CpGs withheld from it, and a contrastive term aligns the two views of the same profile.

This repository contains the model code, every script used to produce the results of the
manuscript, and the exact software environment of those runs.

## Model

| | |
|---|---|
| Input | one token per measured CpG: CpG identity embedding + continuous encoding of its β value |
| Encoder | 6-layer pre-norm Transformer, hidden size 256, 4 heads, SwiGLU feed-forward (512), RMSNorm |
| Positions | rotary embeddings (RoPE) on queries and keys, with the CpG's **genomic rank** (chromosome, then coordinate) as position |
| Sample representation | CLS token → linear + tanh (pooler) |
| Vocabulary | 49,156 CpGs + 5 special tokens |
| Parameters | 16.6 M (12.6 M in the CpG embedding table) |
| Pretraining | WCED: two independent 50 % views; reconstruction of withheld CpGs from the CLS representation; symmetric InfoNCE between views (weight 0.05, temperature 0.1) |
| Fine-tuning (age) | head 256 → 128 → 64 → 1, Huber loss, five folds, predictions averaged |

The encoder (`bmfm_methylation/llama/`) is implemented in plain PyTorch. Tokenization uses the
multi-field tokenizer of [BMFM-RNA](https://github.com/BiomedSciAI/biomed-multi-omic), the β encoder
follows its ScaleAdapt encoder, and the WCED objective adapts BMFM-RNA's Whole-cell Expression
Decoder to methylation (see `NOTICE`).

## Repository layout

```
bmfm_methylation/
  llama/        model.py (encoder), scale_adapt.py (β encoder), wced_llama.py (WCED objective),
                pretrain_llama.py, finetune_llama.py (Hydra entry points), configs/
  shared/       data_module.py (dataset, view sampling, genomic ordering), tokenizer.py
scripts/
  data/         genomic rank of the CpG panel, duplicate detection, k-fold splits, pipeline checks
  pretrain/     pretraining (SLURM)
  finetune/     k-fold fine-tuning, test evaluation, per-sample predictions, random initialisation
  baselines/    ElasticNetCV
  evaluation/   frozen-representation probes, leave-one-study-out, withheld-CpG reconstruction,
                cross-view consistency on the held-out partition, paired bootstrap
  rope_pilot/   genomic-rank versus sequential positions (Supplementary Note 1)
  methylgpt/    MethylGPT comparison: fold data, fine-tuning, test predictions
  figures/      the manuscript figures
configs/        paths.example.sh (copy to paths.sh)
results/        result files of the manuscript (results/README.md)
tutorials/      notebooks and a 500-profile demo dataset; reproduce_paper_results.ipynb
docs/           PROVENANCE.md (names of the original runs)
```

## Installation

The manuscript ran on Linux with Python 3.11.2. To reproduce that environment exactly:

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt
pip install --no-deps -e .
```

On other platforms, or to use the package with your own versions within the tested ranges:

```bash
pip install -e ".[analysis,wandb,tutorials]"
```

`bmfm-targets` is installed from the BMFM-RNA repository at the commit used for the manuscript
(it is not on PyPI).

## Checkpoints

Pretrained and fine-tuned weights are distributed on Hugging Face
([`netanelazran1/MethylLlama`](https://huggingface.co/netanelazran1/MethylLlama)):

| File | Content |
|---|---|
| `pretrained/methyllama_pretrained_epoch85.ckpt` | WCED-pretrained encoder (epoch 85) |
| `finetuned_age/fold{0..4}.ckpt` | the five age models of the benchmark |
| `tokenizer/cpg_sites/` | the 49,161-entry vocabulary |
| `resources/cpg_genomic_rank*.npy` | genomic-rank position ids for the pretraining and downstream panels |

## Quick start: sample representations

```python
import torch
from bmfm_targets.tokenization import MultiFieldTokenizer
from bmfm_methylation.shared.data_module import MethylationDataset, WCEDCollator
from bmfm_methylation.llama.finetune_llama import load_wced_llama_checkpoint

encoder = load_wced_llama_checkpoint("methyllama_pretrained_epoch85.ckpt").encoder.eval()
tokenizer = MultiFieldTokenizer.from_pretrained("tokenizer")
data = MethylationDataset(h5ad_path="tutorials/data/methylllama_demo_500samples.h5ad",
                          split=None, normalize_age=False)
collate = WCEDCollator(tokenizer=tokenizer, cpg_sites=data.cpg_sites, vocab_size=len(data.cpg_sites),
                       input_ratio=1.0, contrastive=False,
                       genomic_rank_path="resources/cpg_genomic_rank_finetune.npy")

batch = collate([data[0]])
x = torch.stack([batch["cpg_ids"].float(), batch["beta_values"]], dim=1)
with torch.no_grad():
    cls = encoder(input_ids=x, attention_mask=batch["attention_mask"],
                  position_ids=batch["position_ids"]).pooler_output     # (1, 256)
```

The genomic-rank file must match the CpG panel of the input (`cpg_genomic_rank.npy` for the
49,156-CpG pretraining panel, `cpg_genomic_rank_finetune.npy` for the 21,368-CpG downstream
panel). Without `position_ids` the encoder falls back to sequential positions and the
representation is not the one reported. On the first 20 demo profiles these steps reproduce the
manuscript's stored representations (cosine similarity 1.000000, CPU versus the original GPU run).

## Reproducing the manuscript

`tutorials/reproduce_paper_results.ipynb` recomputes the numbers of Table 1, Figs. 2–4 and
Supplementary Tables S1–S2 from the result files in `results/` and compares each with the
manuscript; it needs no GPU and no data download. To produce those files from the public data:

Copy `configs/paths.example.sh` to `configs/paths.sh`, set the data directory and the environment
setup for your system, and run every script from the repository root. The SLURM scripts record the
resources of the original runs in a comment; set a partition if your cluster needs one.

| Step | Script | Output |
|---|---|---|
| Genomic rank of the CpG panel | `scripts/data/sort_probes_genomic.sh`, `scripts/data/create_finetune_genomic_rank.py` | `resources/` (shipped; regenerating overwrites them) |
| Pretraining | `scripts/pretrain/pretrain_methyllama.sh` | `outputs/pretrain-llama-wced/` |
| Duplicates and k-fold splits | `scripts/data/analyze_full_dataset.py`, `scripts/data/create_kfold_splits.py` | `outputs/kfold_splits/` |
| Fine-tuning, five folds | `scripts/finetune/launch_kfold.sh` (→ `finetune_kfold.sh`), `testeval_kfold.sh` | `outputs/finetune-llama-small/` |
| Random-initialisation control | `scripts/finetune/finetune_random_init.sh`, `testeval_random_init.sh` | |
| Per-sample test predictions | `scripts/finetune/run_kfold_test_predictions.sh`, `validate_kfold_test_predictions.py` | `outputs/bootstrap_predictions/` |
| ElasticNetCV | `scripts/baselines/elasticnet_age.sh` | `outputs/baselines/` |
| MethylGPT fine-tuning and test predictions | `scripts/methylgpt/` (see its README) | `outputs/bootstrap_predictions/methylgpt/` |
| Paired bootstrap against MethylGPT | `scripts/evaluation/paired_bootstrap_comparison.py` | |
| Frozen representations, LODO | `scripts/evaluation/run_full_analysis.sh`, `leave_one_dataset_out_probe.py` | `results/pretrained_cls/` |
| Withheld-CpG reconstruction | `scripts/evaluation/run_reconstruction_withheld.sh` | `results/pretrained_cls/reconstruction_withheld/` |
| Cross-view consistency (held-out) | `scripts/evaluation/run_view_consistency_heldout_full.sh`, `run_view_design_heldout.sh` | `results/pretrained_cls/view_*_heldout*/` |
| Genomic-rank pilot | `scripts/rope_pilot/` | Supplementary Note 1 |
| Figures | `scripts/figures/` | `figures/paper/` |

The MethylGPT comparison fine-tunes the released MethylGPT checkpoint (encoder and age head
trained) with MethylGPT's fine-tuning code on the same samples and folds (`scripts/methylgpt/`);
`scripts/data/compare_splits_21k.py` and `verify_21k_comparison.py` check that both pipelines use
the same data.

`docs/PROVENANCE.md` maps the script, run and folder names used here to those of the original runs
(SLURM job ids and W&B runs).

## Data

All data come from public repositories (GEO, ArrayExpress and the open-access tier of TCGA).
Apart from the 500-profile demo subset in `tutorials/data/` and the derived result files in
`results/`, they are not redistributed here.

- **Pretraining corpus**: the processed 49,156-CpG matrix assembled for MethylGPT
  (Ying et al., 2024) from the EWAS Data Hub and ClockBase (169,120 profiles in the version used).
  <!-- TODO: link -->
- **Downstream corpus**: the AltumAge collection (de Lima Camillo et al., 2022), 10,988 profiles at
  21,368 CpGs, as organised in the [AltumAge repository](https://github.com/rsinghlab/AltumAge)
  (MIT licence).
- **CpG coordinates**: Illumina HumanMethylation450 manifest, hg38.

## Citation

<!-- TODO: preprint reference and BibTeX -->

## Licence

Apache License 2.0. See `LICENSE` and `NOTICE`.
