# Provenance of names

The manuscript runs were made in the authors' work repository under internal names. This
release uses descriptive names. The table maps every renamed script to the name it had when
it produced the reported results, so cluster logs and W&B runs can be traced.

## Runs behind the manuscript

| Role | SLURM job | Run / output directory at the time | W&B run (training) |
|---|---|---|---|
| Pretraining (checkpoint epoch 85) | 45468861 | `llama-6L-all49k-r0.5-w0.05-genomic-45468861` | `iujgrcvs` |
| Fine-tuning fold 0 | 45586010 | `llama-v7b-kfold-fold0-ep300-45586010` | `a154lzjy` |
| Fine-tuning fold 1 | 45586011 | `llama-v7b-kfold-fold1-ep300-45586011` | `aas23t9m` |
| Fine-tuning fold 2 | 45586012 | `llama-v7b-kfold-fold2-ep300-45586012` | `vw5f9nh4` |
| Fine-tuning fold 3 | 45586013 | `llama-v7b-kfold-fold3-ep300-45586013` | `vqogel6f` |
| Fine-tuning fold 4 | 45586014 | `llama-v7b-kfold-fold4-ep300-45586014` | `6qt4ma56` |
| Random initialisation, fold 0 | 45683236 | `llama-v7b-random-init-fold0-ep300-45683236` | `axcqqgrh` |
| Genomic-rank pilot, run A (genomic positions) | 45740643 | `ropeA-genomic-5k-45740643` | |
| Genomic-rank pilot, run B (sequential positions) | 45763963 | `ropeB-nogenomic-5k-45763963` | |

Checkpoint used from each run, under the file name written by the training code (the name records
the epoch and the monitored validation metrics):

| Run | Checkpoint file at the time | Released as |
|---|---|---|
| Pretraining | `epoch=85-recon=0.0552-pcc=0.9713.ckpt` | `checkpoints/pretrained/methyllama_pretrained_epoch85.ckpt` |
| Fine-tuning fold 0 | `epoch=92-val_medae=2.6250.ckpt` | `checkpoints/finetuned_age/fold0.ckpt` |
| Fine-tuning fold 1 | `epoch=137-val_medae=2.5940.ckpt` | `checkpoints/finetuned_age/fold1.ckpt` |
| Fine-tuning fold 2 | `epoch=131-val_medae=2.6875.ckpt` | `checkpoints/finetuned_age/fold2.ckpt` |
| Fine-tuning fold 3 | `epoch=137-val_medae=2.7188.ckpt` | `checkpoints/finetuned_age/fold3.ckpt` |
| Fine-tuning fold 4 | `epoch=138-val_medae=2.6875.ckpt` | `checkpoints/finetuned_age/fold4.ckpt` (also the fine-tuned representation of Supplementary Fig. S3, `results/finetuned_cls/`) |
| Pilot run A | `epoch=60-recon=0.0906-pcc=0.9357.ckpt` | not released (Supplementary Note 1: reconstruction 0.0906, Pearson 0.9357) |
| Pilot run B | `epoch=72-recon=0.1585-pcc=0.9017.ckpt` | not released (Supplementary Note 1: reconstruction 0.1585, Pearson 0.9017) |

New runs started with this repository are named `methyllama-kfold-fold<k>-…` and
`methyllama-random-init-fold0-…`; analysis outputs go to `results/` instead of `figures/v7b_*`.

## Renamed scripts (old path in the work repository → path here)

| Then | Now |
|---|---|
| `scripts/llama/create_finetune_genomic_rank.py` | `scripts/data/create_finetune_genomic_rank.py` |
| `scripts/llama/finetune_llama_small_v7b_kfold.sh` | `scripts/finetune/finetune_kfold.sh` |
| `scripts/llama/finetune_llama_v7b_random_init.sh` | `scripts/finetune/finetune_random_init.sh` |
| `scripts/llama/launch_kfold.sh` | `scripts/finetune/launch_kfold.sh` |
| `scripts/llama/pretrain_ablation_ropeA_genomic.sh` | `scripts/rope_pilot/pretrain_ablation_ropeA_genomic.sh` |
| `scripts/llama/pretrain_ablation_ropeB_nogenomic.sh` | `scripts/rope_pilot/pretrain_ablation_ropeB_nogenomic.sh` |
| `scripts/llama/pretrain_llama_small_6L_contrastive.sh` | `scripts/pretrain/pretrain_methyllama.sh` |
| `scripts/llama/run_ablation_rope_step0_subset.sh` | `scripts/rope_pilot/run_ablation_rope_step0_subset.sh` |
| `scripts/llama/run_kfold_test_predictions.sh` | `scripts/finetune/run_kfold_test_predictions.sh` |
| `scripts/llama/testeval_kfold.sh` | `scripts/finetune/testeval_kfold.sh` |
| `scripts/llama/testeval_random_init.sh` | `scripts/finetune/testeval_random_init.sh` |
| `scripts/paper_figures/common_style.py` | `scripts/figures/common_style.py` |
| `scripts/paper_figures/fig2_hardneg_mockup.py` | `scripts/figures/fig2_retrieval_reconstruction.py` |
| `scripts/paper_figures/fig3_representation_mockup.py` | `scripts/figures/fig3_representation.py` |
| `scripts/paper_figures/fig4_v2_panels.py` (wrote `fig4_v2_versionA`, `figS_age_benchmark_supp`) | `scripts/figures/fig4_age_benchmark.py` (writes `fig4_age_benchmark`, `figS5_age_benchmark`) |
| `scripts/paper_figures/figS2_crossview_mockup.py` | `scripts/figures/figS_crossview.py` |
| `scripts/paper_figures/figS_finetuning_effect.py` | `scripts/figures/figS_finetuning_effect.py` |
| `scripts/paper_figures/figS_panel_independence.py` | `scripts/figures/figS_panel_independence.py` |
| `scripts/repr_analysis/extract_kfold_test_predictions.py` | `scripts/finetune/extract_kfold_test_predictions.py` |
| `scripts/repr_analysis/kfold_full_history_analysis.py` | `scripts/finetune/kfold_full_history_analysis.py` |
| `scripts/repr_analysis/paired_bootstrap_by_exposure_v2.py` | `scripts/evaluation/paired_bootstrap_by_exposure.py` |
| `scripts/repr_analysis/paired_bootstrap_comparison.py` | `scripts/evaluation/paired_bootstrap_comparison.py` |
| `scripts/repr_analysis/paired_bootstrap_grouped_sensitivity.py` | `scripts/evaluation/paired_bootstrap_grouped_sensitivity.py` |
| `scripts/repr_analysis/validate_kfold_test_predictions.py` | `scripts/finetune/validate_kfold_test_predictions.py` |
| `scripts/repr_analysis_v7b/analyze_cpg_embeddings.py` | `scripts/evaluation/analyze_cpg_embeddings.py` |
| `scripts/repr_analysis_v7b/analyze_rope_ablation_comparison.py` | `scripts/rope_pilot/analyze_rope_ablation_comparison.py` |
| `scripts/repr_analysis_v7b/extract_contextual_cpg.py` | `scripts/rope_pilot/extract_contextual_cpg.py` |
| `scripts/repr_analysis_v7b/extract_finetune_cls.py` | `scripts/evaluation/extract_finetune_cls.py` |
| `scripts/repr_analysis_v7b/extract_pretrain_cls.py` | `scripts/evaluation/extract_pretrain_cls.py` |
| `scripts/repr_analysis_v7b/fig3_probe_cis.py` | `scripts/evaluation/probe_cis.py` |
| `scripts/repr_analysis_v7b/frozen_probe_sensitivity.py` | `scripts/evaluation/frozen_probe_sensitivity.py` |
| `scripts/repr_analysis_v7b/hard_negative_heldout_full.py` | `scripts/evaluation/hard_negative_heldout.py` |
| `scripts/repr_analysis_v7b/leave_one_dataset_out_probe.py` | `scripts/evaluation/leave_one_dataset_out_probe.py` (with the `--exclude_study` option added; see below) |
| `scripts/repr_analysis_v7b/reconstruction_withheld_eval_full.py` | `scripts/evaluation/reconstruction_withheld_eval.py` (the 500,000-position sample of observed and predicted β is no longer written) |
| `scripts/repr_analysis_v7b/run_cpg_context_ablation_runA.sh` | `scripts/rope_pilot/run_cpg_context_ablation_runA.sh` |
| `scripts/repr_analysis_v7b/run_cpg_context_ablation_runB.sh` | `scripts/rope_pilot/run_cpg_context_ablation_runB.sh` |
| `scripts/repr_analysis_v7b/run_finetune_cls.sh` | `scripts/evaluation/run_finetune_cls.sh` |
| `scripts/repr_analysis_v7b/run_full_analysis.sh` | `scripts/evaluation/run_full_analysis.sh` |
| `scripts/repr_analysis_v7b/run_pretrain_cls.sh` | `scripts/evaluation/run_pretrain_cls.sh` |
| `scripts/repr_analysis_v7b/run_reconstruction_withheld_full.sh` | `scripts/evaluation/run_reconstruction_withheld.sh` |
| `scripts/repr_analysis_v7b/run_view_consistency_heldout_full.sh` | `scripts/evaluation/run_view_consistency_heldout_full.sh` |
| `scripts/repr_analysis_v7b/two_view_consistency.py` | `scripts/evaluation/two_view_consistency.py` |
| `scripts/repr_analysis_v7b/view_consistency_heldout_full.py` | `scripts/evaluation/view_consistency_heldout_full.py` |
| `scripts/repr_analysis_v7b/view_design_eval.py` (functions `build_views`, `encode_views` only) | `scripts/evaluation/view_encoding.py` |
| `scripts/repr_analysis_v7b/visualize_cls.py` | `scripts/evaluation/visualize_cls.py` |
| `scripts/repr_analysis_v7b/within_tissue_study_structure.py` | `scripts/evaluation/within_tissue_study_structure.py` |
| `scripts/utils/analyze_full_dataset.py` | `scripts/data/analyze_full_dataset.py` |
| `scripts/utils/compare_splits_21k.py` | `scripts/data/compare_splits_21k.py` |
| `scripts/utils/create_kfold_splits.py` | `scripts/data/create_kfold_splits.py` |
| `scripts/utils/create_pretrain_subset_h5ad.py` | `scripts/rope_pilot/create_pretrain_subset_h5ad.py` |
| `scripts/utils/data_structure_audit.py` | `scripts/data/data_structure_audit.py` |
| `scripts/utils/downstream_integrity_check.py` | `scripts/data/downstream_integrity_check.py` |
| `scripts/utils/pretrain_heldout_duplicates.py` | `scripts/data/pretrain_heldout_duplicates.py` |
| `scripts/utils/pretrain_overlap_by_id.py` | `scripts/data/pretrain_overlap_by_id.py` |
| `scripts/utils/sort_probes_genomic.py` | `scripts/data/sort_probes_genomic.py` |
| `scripts/utils/sort_probes_genomic.sh` | `scripts/data/sort_probes_genomic.sh` |
| `scripts/utils/tcga_same_participant_check.py` | `scripts/data/tcga_same_participant_check.py` |
| `scripts/utils/verify_21k_comparison.py` | `scripts/data/verify_21k_comparison.py` |
| `scripts/utils/verify_elasticnet_test_set.py` | `scripts/baselines/verify_elasticnet_test_set.py` |
| `scripts/utils/verify_elasticnet_test_set.sh` | `scripts/baselines/verify_elasticnet_test_set.sh` |

New in this release (no counterpart in the work repository):
`scripts/evaluation/reconstruction_paired_bootstrap.py` writes `paired_bootstrap_ci.json`, which was
computed interactively at the time; it reproduces the stored file exactly. The `--exclude_study`
option of `leave_one_dataset_out_probe.py` and its `lodo_reference.json` output reproduce the
study-level-QC LODO files of the work repository (`outputs/data_structure_audit/lodo_qc/`) exactly.

## Result files (work repository → this release)

| Then | Now |
|---|---|
| `figures/v7b_pretrain_cls/{embeddings_*.npy, metadata.csv, analysis_summary.json, cls_umap50_coords.npy, cls_umap_coords.npy, two_view_v1n.npy, cpg_embedding_analysis.json, within_tissue_study_cls.json}` | `results/pretrained_cls/` (same names) |
| `figures/v7b_pretrain_cls/fig3_probe_cis.json` | `results/pretrained_cls/probe_cis.json` |
| `outputs/data_structure_audit/lodo_qc/{lodo_age_probe.csv, lodo_reference.json}` | `results/pretrained_cls/` (same names) |
| `figures/v7b_pretrain_cls/reconstruction_withheld_full/{reconstruction_withheld_summary.json, per_sample_metrics.csv, obs_pred_hist2d.npz, paired_bootstrap_ci.json}` | `results/pretrained_cls/reconstruction_withheld/` |
| `figures/v7b_pretrain_cls/hard_negative_heldout_full/{summary.json, per_query.npz}` | `results/pretrained_cls/hard_negative_heldout/` |
| `figures/v7b_pretrain_cls/view_consistency_heldout_full/` | `results/pretrained_cls/view_consistency_heldout_full/` |
| `figures/v7b_finetuned_cls/{analysis_summary.json, embeddings_cls.npy, cls_umap_coords.npy}` | `results/finetuned_cls/` |
| `outputs/data_structure_audit/{audit.json, downstream_exposure_per_profile.csv, downstream_integrity_summary.json, downstream_integrity_by_study.csv, tcga_pairs_summary.json, heldout_duplicates_summary.json, frozen_probe_sensitivity.json, lodo_sensitivity.csv, paired_bootstrap_grouped_sensitivity.json, per_study_mae.csv}` | `results/data_audit/` (same names) |
| `outputs/data_structure_audit/paired_bootstrap_by_exposure_v2.json` | `results/data_audit/paired_bootstrap_by_exposure.json` |
| `outputs/data_structure_audit/heldout_duplicates_pairs.csv` (all screened pairs, 35 MB) | `results/data_audit/heldout_duplicates_flagged_pairs.csv` (flagged pairs, five columns) |
| `dataset_fingerprint_outputs/duplicate_pairs.csv` | `results/data_audit/downstream_duplicate_pairs.csv` (six columns) |
| `outputs/ablation_rope/{rope_ablation_summary.json, rope_ablation_curve_A.csv, rope_ablation_curve_B.csv}` | `results/rope_pilot/` |
| `outputs/bootstrap_predictions/`, `kfold_full_history_analysis/` | `results/age_benchmark/` |

## Other renamed identifiers

| Then | Now |
|---|---|
| `figures/v7b_pretrain_cls/` | `results/pretrained_cls/` |
| `figures/v7b_finetuned_cls/` | `results/finetuned_cls/` |
| `outputs/data_structure_audit/` | `outputs/data_audit/` (scripts in `scripts/data/`), `results/data_audit/` (shipped) |
| figure files `figures/paper/*_mockup.*`, `fig4_v2_*` | `figures/paper/<manuscript file name>.*` |
| model label `MethylLlamaV7b` (prediction files) | `MethylLlama` |
| W&B projects/run prefixes `v7b-…` | `methyllama-…` |

## MethylGPT comparison

Original location: the authors' working copy of MethylGPT on the cluster. Fine-tuning ran on
2026-08-04/05, prediction extraction on 2026-08-16 (fold 0 repeated on 2026-08-17 as a
determinism check; identical output).

| Then | Now |
|---|---|
| `tutorials_local/MethylGPT_tutorials/finetuning_age_prediction/finetuning_age_*.py`, `fintuning_age_metrics.py` | `scripts/methylgpt/finetuning/` |
| `scripts/finetuning_age_prediction_medium/train_methylgpt_21k_altumage_fold{0..4}.yml`, `args.json` | `scripts/methylgpt/configs/` |
| `scripts/finetuning_age_prediction_medium/run_medium_21k_altumage_fold{0..4}.sbatch` | `scripts/methylgpt/finetune_fold.sh` (`FOLD=0..4`) |
| `scripts/finetuning_age_prediction_medium/convert_21k_kfold.sbatch`, `convert_h5ad_to_parquet_21k_kfold.py` | `scripts/methylgpt/convert_kfold.sh`, same `.py` |
| `scripts/finetuning_age_prediction_medium/run_extract_kfold_predictions_fold{0..4}.sbatch`, `extract_kfold_predictions.py` | `scripts/methylgpt/extract_predictions_fold.sh` (`FOLD=0..4`), same `.py` |
| `validate_kfold_predictions.py`, `fetch_kfold_test_results.py` | `scripts/methylgpt/` |

| Fold | W&B project | W&B run | Selected checkpoint (validation MedAE) |
|---|---|---|---|
| 0 | `methylGPT_medium_21k_altumage_fold0` | `1ofweo18` | epoch 173 (3.3403) |
| 1 | `methylGPT_medium_21k_altumage_fold1` | `64yo6n6d` | epoch 130 (3.4043) |
| 2 | `methylGPT_medium_21k_altumage_fold2` | `0e9m6vdi` | epoch 189 (3.3343) |
| 3 | `methylGPT_medium_21k_altumage_fold3` | `pmcs6y12` | epoch 161 (3.3652) |
| 4 | `methylGPT_medium_21k_altumage_fold4` | `6ezzcivo` | epoch 126 (3.3242) |


## CNS tumour classification

Runs of 2026-10-25 to 2026-11-12 (embedding extraction and raw-β probes on the cluster, embedding probes and
figures locally). The original scripts were in `scripts/cns_task/` and wrote to `outputs/cns_task/`; here they
are `scripts/cns/` writing to `outputs/cns/`, with cluster paths replaced by `configs/paths.sh`, the released
checkpoint, `tokenizer/` and `resources/`. Computation is unchanged. Two renames:

| Then | Now |
|---|---|
| `scripts/cns_task/cns_check_cluster.py` | `scripts/cns/cns_preflight.py` |
| `open_stream`, `series_dir` of `scripts/cns_task/external_feasibility.py` (a survey of candidate external cohorts, not part of the release) | `scripts/cns/geo_stream.py` |

The result directories behind the manuscript: complete profiles `outputs/cns_task/probe_all/` (all
representations, default C grid) and `compare/`; Scenario A `sparse/probe/A/`; Scenario B
`sparse/probe_wideC/B/` (C grid to 10⁴); both scenarios `sparse/compare_wideC_all/`; external cohort
`external/probe/`, `external/sparse/probe/` and `probe_B/`. The SLURM submission scripts of these runs
(`submit_*.sh`), preliminary checks, an earlier set of figure drafts and the per-family confusion figure are not
part of the release; `scripts/cns/README.md` lists the steps in the order they were run.

## Contrastive-weight pilot

Runs of 2026-10-18 to 2026-10-21. The original package was `ablation_infonce/`; here it is `objective_pilot/`,
with cluster paths replaced by `configs/paths.sh` and the released files; computation unchanged. The pilot's
reconstruction evaluation used the 5,000-profile version of `reconstruction_withheld_eval.py`
(`scripts/repr_analysis_v7b/` at the time), shipped as `objective_pilot/reconstruction_withheld_subset.py`.
W&B project `pretrain-llama-infonce-pilot`, runs `pilot-lam<λ>_seed42-<job>`. The module also
supports a full-corpus reconstruction-only run (`WCEDLlamaReconOnlyModule`), which was not run. An age
fine-tuning of the pilot models is not reported in the manuscript and is not included.
