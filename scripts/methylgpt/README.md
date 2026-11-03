# MethylGPT comparison

The pipeline that produced the MethylGPT results of the manuscript (Table 1, Supplementary
Table S1, Fig. 4): the released MethylGPT checkpoint fine-tuned for age on the same five folds and
test set as MethylLlama, and its per-profile test predictions
(`results/age_benchmark/methylgpt/`).

## Steps

Run from the repository root, after `scripts/data/create_kfold_splits.py` has written
`outputs/kfold_splits/` (the same split files MethylLlama uses).

| Step | Script | Output |
|---|---|---|
| 1. Folds as MethylGPT parquet files | `convert_kfold.sh` | `data/methylgpt/21k_altumage/finetuning_data_kfold/` |
| 2. Fine-tuning, one job per fold | `sbatch --export=ALL,FOLD=k finetune_fold.sh` | `outputs/methylgpt/checkpoints_21k_altumage_fold{k}/` |
| 3. Test predictions, one job per fold | `sbatch --export=ALL,FOLD=k extract_predictions_fold.sh` | `outputs/bootstrap_predictions/methylgpt/fold_{k}_predictions.csv` |
| 4. Checks of the five prediction files | `python scripts/methylgpt/validate_kfold_predictions.py` | |

`fetch_kfold_test_results.py` reads the test metrics that training logged to W&B, and
`validate_kfold_predictions.py` compares them with the extracted predictions. Both need
`WANDB_API_KEY` in the environment and `WANDB_ENTITY` (`configs/paths.sh`).

## Requirements

- **Environment** (activated by `methylgpt_env`, `configs/paths.sh`): the runs used Python 3.11.2,
  `methylgpt==0.1.2` (PyPI; the package of [albert-ying/MethylGPT](https://github.com/albert-ying/MethylGPT)
  at commit `ca14cc2`, Apache License 2.0), torch 2.2.2 (CUDA 12.1), lightning 2.1.4,
  flash-attn 2.5.9.post1, torchtext 0.17.2 and numpy 1.26.4. The full list, as recorded by W&B
  for the fold-0 run, is `requirements-lock.txt`.
- **Pretrained checkpoint**: the released 256-dimensional checkpoint (6 layers, 4 heads, 15.4 M
  parameters) at `pretrained/methylgpt/medium-best_model_epoch6.pt`. It was listed as
  *methylGPT-medium* in the MethylGPT README when it was downloaded and is listed as
  *methylGPT-large* (same download folder) since the repository's March 2026 revision.
- **Data**: `METHYLLAMA_DATA` holds the downstream h5ad (as for MethylLlama). The configs read
  MethylGPT files from `data/methylgpt/` (set `METHYLGPT_DATA` to the same place, or link it):
  `cpg_ids/probe_ids_type3.csv`, the MethylGPT 49,156-CpG input list.

## Settings

As run for the manuscript (`configs/train_methylgpt_21k_altumage_fold{0..4}.yml`, identical
apart from the fold): the pretrained encoder and the age head are trained jointly
(`freeze_encoder: false`); Adam, encoder learning rate 1e-5, head learning rate 3e-5, gradient clipping 0.5, batch size 1, no CpG masking, 200 epochs, bf16 mixed precision, seed 42; the three
checkpoints with the lowest validation MedAE are kept and the best one is used for the test
predictions. Of the 21,368 downstream CpGs, 19,608 map to the MethylGPT input vocabulary.

## Changes to the MethylGPT code

The four files in `finetuning/` come from `tutorials/finetuning_age_prediction/` of MethylGPT at
commit `ca14cc2`. `fintuning_age_metrics.py` is unchanged. The others were changed as follows.

`finetuning_age_models.py`
- optional `freeze_encoder`: the parameters of the pretrained encoder do not require gradients;
- batch normalisation in the residual age head replaced by instance normalisation
  (`InstanceNorm1d(affine=True)`);
- the test set is evaluated after training, once per kept checkpoint (logged as
  `test_top1_*`, `test_top2_*`, `test_top3_*`), instead of alongside validation in every epoch;
- the padding mask is computed with `eq` instead of the in-place `eq_`, which modifies the
  token-id tensor;
- shape handling for batches of one (`view(-1)` instead of `squeeze()`), attention modules
  with either `Wqkv` or `in_proj_weight`, import fallbacks.

`finetuning_age_main.py`
- precision `bf16-mixed` instead of `bf16-true`;
- the three best checkpoints by validation MedAE are kept (was one), with optional early stopping;
- relative paths in the configuration are resolved against the repository root.

`finetuning_age_datasets.py`
- input paths resolved relative to the repository or `METHYLGPT_FINETUNE_DATA_ROOT`; imports.

Data loading, the min-max age scaling fitted on the training fold, and the loss are as in
MethylGPT.

## Provenance

The original files are in the authors' working copy of MethylGPT
(`tutorials_local/MethylGPT_tutorials/finetuning_age_prediction/` and
`scripts/finetuning_age_prediction_medium/`). The five per-fold SLURM scripts there are replaced
by `finetune_fold.sh` and `extract_predictions_fold.sh`, which take the fold as `FOLD`; paths
were made relative. W&B runs of the manuscript: see `docs/PROVENANCE.md`.
