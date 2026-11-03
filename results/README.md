# Result files of the manuscript

Outputs of the original runs, from which `tutorials/reproduce_paper_results.ipynb` recomputes the
numbers of the manuscript and `scripts/figures/` draws Figs. 2–4. Values are unchanged; cluster paths
in the run records were replaced by the released file names, the internal model label by
"MethylLlama", and the per-row checkpoint path column of the fold prediction files was dropped.
Run names and SLURM jobs are in `docs/PROVENANCE.md`.

## age_benchmark/ — Table 1, Supplementary Table S1, Fig. 4

| File | Content | Written by |
|---|---|---|
| `methyllama/fold_{0..4}_predictions.csv` | per-profile age predictions of each fold on the 2,149-profile test set | `scripts/finetune/run_kfold_test_predictions.sh` |
| `methylgpt/fold_{0..4}_predictions.csv` | the same for the five MethylGPT folds (released MethylGPT checkpoint, encoder and age head trained, same folds; settings in `scripts/methylgpt/README.md`) | MethylGPT fine-tuning pipeline (`METHYLGPT_DIR`) |
| `paired_per_subject_predictions.csv` | true age, MethylLlama ensemble, MethylGPT ensemble (same profiles) | `scripts/evaluation/paired_bootstrap_comparison.py` |
| `paired_bootstrap_summary.json` | point estimates and paired bootstrap intervals (10,000 resamples, seed 0) | same |
| `age_restricted_paired_bootstrap_summary.json` | the same within age ≥ 18 and age ≥ 40 | |
| `elasticnet_results.json` | ElasticNetCV selection and test metrics | `scripts/baselines/elasticnet_age.sh` |
| `fold0_train_history.csv`, `random_init_fold0_history.csv` | fold-0 validation metrics per epoch, pretrained and random initialisation | `scripts/finetune/kfold_full_history_analysis.py` (W&B) |

The scripts write these files to `outputs/bootstrap_predictions/` and `kfold_full_history_analysis/`.
ElasticNetCV profile-level predictions were not retained.

## splits/ — data partitions (identifiers only)

| File | Content | Written by |
|---|---|---|
| `downstream_age_splits.csv` | the 10,589 profiles of the age benchmark: `test` (the common 2,149-profile test set) or the validation `fold` (0–4) of each cross-validation profile; a fold trains on the other four | `scripts/release/export_split_ids.py` (thesis repo), from `outputs/kfold_splits/` |
| `pretraining_heldout_ids.csv` | the 16,912 profiles of the held-out pretraining partition used in Fig. 2 and Supplementary Fig. S1 | same, from the hard-negative run |

## pretrained_cls/ — Figs. 2 and 3, Supplementary Table S2

| File | Content | Written by |
|---|---|---|
| `metadata.csv` | the 10,988 downstream profiles: study, tissue, age, sex, original partition | `scripts/evaluation/extract_pretrain_cls.py` |
| `embeddings_cls.npy`, `embeddings_mean.npy` | frozen CLS and mean-pooled representations (10,988 × 256, float32), rows as in `metadata.csv` | same |
| `analysis_summary.json` | probe and geometry summary of that run | same |
| `cls_umap50_coords.npy` | UMAP coordinates of Fig. 3a,b, as stored (UMAP of the PCA-50 CLS space; UMAP output depends on the library version, so a rerun of `scripts/evaluation/visualize_cls.py` does not reproduce these coordinates exactly) | |
| `lodo_age_probe.csv`, `lodo_reference.json` | study-held-out age probe and its in-distribution reference | `scripts/evaluation/leave_one_dataset_out_probe.py` |
| `view_consistency_heldout_full/` | cross-view statistics on the 16,912 held-out pretraining profiles: per draw, matched cosine and partner rank of every profile, histogram of all unmatched pairs | `scripts/evaluation/run_view_consistency_heldout_full.sh` |
| `reconstruction_withheld/` | per-profile reconstruction MSE of the model and three controls on 5,000 held-out profiles; random sample of withheld positions (Fig. 2c) | `scripts/evaluation/run_reconstruction_withheld.sh` |

The per-draw view embeddings of the cross-view runs (31 MB each) are not included.
