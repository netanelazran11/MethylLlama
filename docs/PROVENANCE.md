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
| `scripts/paper_figures/fig1_study_design.py` | `scripts/figures/fig1_study_design.py` |
| `scripts/paper_figures/fig2_wced_consistency_heldout.py` | `scripts/figures/fig2_wced_consistency_heldout.py` |
| `scripts/paper_figures/fig3_representation_encoding.py` | `scripts/figures/fig3_representation_encoding.py` |
| `scripts/paper_figures/fig4_age_benchmark_extended.py` | `scripts/figures/fig4_age_benchmark_extended.py` |
| `scripts/paper_figures/figS1_architecture.py` | `scripts/figures/figS1_architecture.py` |
| `scripts/paper_figures/figS_finetuning_effect.py` | `scripts/figures/figS_finetuning_effect.py` |
| `scripts/paper_figures/figS_panel_independence.py` | `scripts/figures/figS_panel_independence.py` |
| `scripts/paper_figures/figS_view_design.py` | `scripts/figures/figS_view_design.py` |
| `scripts/paper_figures/figS_view_design_heldout.py` | `scripts/figures/figS_view_design_heldout.py` |
| `scripts/repr_analysis/extract_kfold_test_predictions.py` | `scripts/finetune/extract_kfold_test_predictions.py` |
| `scripts/repr_analysis/kfold_full_history_analysis.py` | `scripts/finetune/kfold_full_history_analysis.py` |
| `scripts/repr_analysis/paired_bootstrap_comparison.py` | `scripts/evaluation/paired_bootstrap_comparison.py` |
| `scripts/repr_analysis/validate_kfold_test_predictions.py` | `scripts/finetune/validate_kfold_test_predictions.py` |
| `scripts/repr_analysis_v7b/analyze_rope_ablation_comparison.py` | `scripts/rope_pilot/analyze_rope_ablation_comparison.py` |
| `scripts/repr_analysis_v7b/extract_contextual_cpg.py` | `scripts/rope_pilot/extract_contextual_cpg.py` |
| `scripts/repr_analysis_v7b/extract_finetune_cls.py` | `scripts/evaluation/extract_finetune_cls.py` |
| `scripts/repr_analysis_v7b/extract_pretrain_cls.py` | `scripts/evaluation/extract_pretrain_cls.py` |
| `scripts/repr_analysis_v7b/leave_one_dataset_out_probe.py` | `scripts/evaluation/leave_one_dataset_out_probe.py` |
| `scripts/repr_analysis_v7b/reconstruction_withheld_eval.py` | `scripts/evaluation/reconstruction_withheld_eval.py` |
| `scripts/repr_analysis_v7b/run_cpg_context_ablation_runA.sh` | `scripts/rope_pilot/run_cpg_context_ablation_runA.sh` |
| `scripts/repr_analysis_v7b/run_cpg_context_ablation_runB.sh` | `scripts/rope_pilot/run_cpg_context_ablation_runB.sh` |
| `scripts/repr_analysis_v7b/run_finetune_cls.sh` | `scripts/evaluation/run_finetune_cls.sh` |
| `scripts/repr_analysis_v7b/run_full_analysis.sh` | `scripts/evaluation/run_full_analysis.sh` |
| `scripts/repr_analysis_v7b/run_pretrain_cls.sh` | `scripts/evaluation/run_pretrain_cls.sh` |
| `scripts/repr_analysis_v7b/run_reconstruction_withheld.sh` | `scripts/evaluation/run_reconstruction_withheld.sh` |
| `scripts/repr_analysis_v7b/run_view_consistency_heldout_full.sh` | `scripts/evaluation/run_view_consistency_heldout_full.sh` |
| `scripts/repr_analysis_v7b/run_view_design_heldout.sh` | `scripts/evaluation/run_view_design_heldout.sh` |
| `scripts/repr_analysis_v7b/view_consistency_heldout_full.py` | `scripts/evaluation/view_consistency_heldout_full.py` |
| `scripts/repr_analysis_v7b/view_design_eval.py` | `scripts/evaluation/view_design_eval.py` |
| `scripts/repr_analysis_v7b/view_design_eval_heldout.py` | `scripts/evaluation/view_design_eval_heldout.py` |
| `scripts/repr_analysis_v7b/visualize_cls.py` | `scripts/evaluation/visualize_cls.py` |
| `scripts/utils/analyze_full_dataset.py` | `scripts/data/analyze_full_dataset.py` |
| `scripts/utils/compare_splits_21k.py` | `scripts/data/compare_splits_21k.py` |
| `scripts/utils/create_kfold_splits.py` | `scripts/data/create_kfold_splits.py` |
| `scripts/utils/create_pretrain_subset_h5ad.py` | `scripts/rope_pilot/create_pretrain_subset_h5ad.py` |
| `scripts/utils/sort_probes_genomic.py` | `scripts/data/sort_probes_genomic.py` |
| `scripts/utils/sort_probes_genomic.sh` | `scripts/data/sort_probes_genomic.sh` |
| `scripts/utils/verify_21k_comparison.py` | `scripts/data/verify_21k_comparison.py` |
| `scripts/utils/verify_elasticnet_test_set.py` | `scripts/baselines/verify_elasticnet_test_set.py` |
| `scripts/utils/verify_elasticnet_test_set.sh` | `scripts/baselines/verify_elasticnet_test_set.sh` |

## Other renamed identifiers

| Then | Now |
|---|---|
| `figures/v7b_pretrain_cls/` | `results/pretrained_cls/` |
| `figures/v7b_finetuned_cls/` | `results/finetuned_cls/` |
| `figures/v7b_cpg_context/` | `results/cpg_context/` |
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
| `validate_kfold_predictions.py`, `fetch_kfold_test_results.py`, `test_validate_kfold_predictions_smoke.py` | `scripts/methylgpt/` |

| Fold | W&B project | W&B run | Selected checkpoint (validation MedAE) |
|---|---|---|---|
| 0 | `methylGPT_medium_21k_altumage_fold0` | `1ofweo18` | epoch 173 (3.3403) |
| 1 | `methylGPT_medium_21k_altumage_fold1` | `64yo6n6d` | epoch 130 (3.4043) |
| 2 | `methylGPT_medium_21k_altumage_fold2` | `0e9m6vdi` | epoch 189 (3.3343) |
| 3 | `methylGPT_medium_21k_altumage_fold3` | `pmcs6y12` | epoch 161 (3.3652) |
| 4 | `methylGPT_medium_21k_altumage_fold4` | `6ezzcivo` | epoch 126 (3.3242) |

