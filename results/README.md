# Result files of the manuscript

Outputs of the original runs, from which `tutorials/reproduce_paper_results.ipynb` recomputes the
numbers of the manuscript and `scripts/figures/` draws Figs. 2–4 and Supplementary Figs. S1–S4.
Values are unchanged; cluster paths in the run records were replaced by the released file names,
the internal model label by "MethylLlama", and the per-row checkpoint path column of the fold
prediction files was dropped. Run names, SLURM jobs and the original file names are in
`docs/PROVENANCE.md`. No file here contains methylation β values: the files hold identifiers,
annotations, model outputs (predictions, embeddings, cosine similarities, errors), histograms and
summary statistics.

## age_benchmark/ — Table 1, Supplementary Table S1, Fig. 4, Supplementary Fig. S4

| File | Content | Written by |
|---|---|---|
| `methyllama/fold_{0..4}_predictions.csv` | per-profile age predictions of each fold on the 2,149-profile test set | `scripts/finetune/run_kfold_test_predictions.sh` |
| `methylgpt/fold_{0..4}_predictions.csv` | the same for the five MethylGPT folds (released MethylGPT checkpoint, encoder and age head trained, same folds; settings in `scripts/methylgpt/README.md`) | MethylGPT fine-tuning pipeline (`METHYLGPT_DIR`) |
| `paired_per_subject_predictions.csv` | true age, MethylLlama ensemble, MethylGPT ensemble (same profiles) | `scripts/evaluation/paired_bootstrap_comparison.py` |
| `paired_bootstrap_summary.json` | point estimates and paired bootstrap intervals (10,000 resamples, seed 0) | same |
| `age_restricted_paired_bootstrap_summary.json` | the same within age ≥ 18 and age ≥ 40 (Fig. 4c) | `scripts/evaluation/age_restricted_paired_bootstrap.py` (added for the release; reproduces the stored values exactly) |
| `elasticnet_results.json` | ElasticNetCV selection and test metrics | `scripts/baselines/elasticnet_age.sh` |
| `elasticnet_test_predictions.csv` | per-profile ElasticNetCV predictions on the 2,149-profile test set | same |
| `paired_bootstrap_vs_elasticnet_summary.json` | paired bootstrap of the MethylLlama ensemble against ElasticNetCV (10,000 resamples, seed 0) | `scripts/evaluation/paired_bootstrap_vs_elasticnet.py` |
| `fold0_train_history.csv`, `random_init_fold0_history.csv` | fold-0 validation metrics per epoch, pretrained and random initialisation (Supplementary Fig. S4b) | `scripts/finetune/kfold_full_history_analysis.py` (W&B) |

The scripts write these files to `outputs/bootstrap_predictions/` and `kfold_full_history_analysis/`.

**Sign conventions.** The manuscript signs every paired difference so that positive values favour
MethylLlama (Δerror = comparator − MethylLlama; ΔR² = MethylLlama − comparator). Two stored files
differ for R²: `paired_bootstrap_summary.json` stores all three gaps as MethylGPT − MethylLlama
(`gap_gpt_minus_llama`), and `paired_bootstrap_vs_elasticnet_summary.json` as ElasticNetCV −
MethylLlama, so their R² gaps are negative (−0.0254 [−0.0329, −0.0183] and −0.0043 [−0.0113, 0.0029])
where the manuscript reports +0.0254 [0.0183, 0.0329] and +0.0043 [−0.0029, 0.0113]. Their error gaps
already have the manuscript's sign. `age_restricted_paired_bootstrap_summary.json` and the files in
`data_audit/` use the manuscript's convention (stated in each file).

## splits/ — data partitions (identifiers only)

| File | Content | Written by |
|---|---|---|
| `downstream_age_splits.csv` | the 10,589 profiles of the age benchmark: `test` (the common 2,149-profile test set) or the validation `fold` (0–4) of each cross-validation profile; a fold trains on the other four | exported for the release from the split files of `scripts/data/create_kfold_splits.py` (`outputs/kfold_splits/`) |
| `pretraining_heldout_ids.csv` | the 16,912 profiles of the held-out pretraining partition used in Fig. 2 and Supplementary Fig. S1 | exported for the release from the sample ids of the held-out runs (`pretrained_cls/hard_negative_heldout/per_query.npz`) |

## pretrained_cls/ — Figs. 2 and 3, Supplementary Figs. S1–S3, Supplementary Table S2

| File | Content | Written by |
|---|---|---|
| `metadata.csv` | the 10,988 downstream profiles: study, tissue, age, sex, original partition | `scripts/evaluation/extract_pretrain_cls.py` |
| `embeddings_cls.npy`, `embeddings_mean.npy` | frozen CLS and mean-pooled representations (10,988 × 256, float32), rows as in `metadata.csv` | same |
| `analysis_summary.json` | probe and geometry summary of that run (Table S2; see the note on 5.17 below) | same |
| `cls_umap50_coords.npy` | UMAP coordinates of Fig. 3a,b, as stored (UMAP of the PCA-50 CLS space). The call that wrote this file was not preserved, and UMAP output depends on the library version, so a rerun of `scripts/evaluation/visualize_cls.py` does not reproduce these coordinates exactly | interactive run (see left) |
| `cls_umap_coords.npy` | UMAP coordinates of the pretrained panel of Supplementary Fig. S3 (10,988 × 2) | `scripts/evaluation/visualize_cls.py` (step 2 of `run_full_analysis.sh`) |
| `probe_cis.json` | frozen-probe estimates with 95% bootstrap intervals (10,000 resamples of the evaluation profiles) for Fig. 3c,d | `scripts/evaluation/probe_cis.py` |
| `within_tissue_study_cls.json` | same-study fraction of the ten nearest same-tissue neighbours, overall and within sex and age decade, with chance levels (whole blood, cord blood, placenta; Results) | `scripts/evaluation/within_tissue_study_structure.py` |
| `lodo_age_probe.csv`, `lodo_reference.json` | study-held-out age probe and its in-distribution reference, E-GEOD-56105 excluded by the study-level quality check (Fig. 3e) | `scripts/evaluation/leave_one_dataset_out_probe.py --exclude_study E-GEOD-56105` |
| `view_consistency_heldout_full/` | cross-view statistics on the 16,912 held-out pretraining profiles: per draw, matched cosine and partner rank of every profile, histogram of all unmatched pairs (Results, Supplementary Fig. S1a,b) | `scripts/evaluation/run_view_consistency_heldout_full.sh` |
| `hard_negative_heldout/` | retrieval of the matched view against 31 negatives from pools of increasing similarity, overlapping and disjoint views and the pattern-only control. `summary.json`: per view draw and pool, top-1 rate, bootstrap CI, range over five negative draws, median margin. `per_query.npz`: per profile, matched-view cosine and hardest-negative cosine of the first negative draw (Fig. 2a,b) | `scripts/evaluation/hard_negative_heldout.py` |
| `reconstruction_withheld/` | reconstruction of withheld CpGs on all 16,912 held-out pretraining profiles (3.80 × 10⁸ positions). `reconstruction_withheld_summary.json`: MSE of the model and three controls, Pearson r over all positions and CpG-centred. `per_sample_metrics.csv`: per-profile MSE. `obs_pred_hist2d.npz`: 200 × 200 counts of observed versus predicted β over all positions (bin counts only). `paired_bootstrap_ci.json`: paired control − model differences (Fig. 2c, Supplementary Fig. S1c) | `scripts/evaluation/run_reconstruction_withheld.sh`, then `scripts/evaluation/reconstruction_paired_bootstrap.py` |
| `two_view_v1n.npy` | L2-normalised CLS of one random 50% CpG view of 2,000 downstream profiles (2,000 × 256, float32; rows `np.random.default_rng(0).choice(10988, 2000, replace=False)`) (Supplementary Fig. S2) | `scripts/evaluation/two_view_consistency.py` (step 4 of `run_full_analysis.sh`) |
| `cpg_embedding_analysis.json` | genomic locality of the static CpG token embeddings (adjacent − random cosine −0.0008; Supplementary Note 1) | `scripts/evaluation/analyze_cpg_embeddings.py` (step 3 of `run_full_analysis.sh`) |

The per-draw view embeddings of the cross-view runs (31 MB each), which `hard_negative_heldout.py`
reads, are not included; `run_view_consistency_heldout_full.sh` writes them.

**Note on 5.17 and 5.18.** `analysis_summary.json` stores the CLS age-probe MedAE as
`linear_ridge_medae: 5.17`, as written by the original run. The same probe recomputed from the
shipped embeddings gives 5.1756 years, the manuscript's 5.18 (`tutorials/reproduce_paper_results.ipynb`;
`data_audit/frozen_probe_sensitivity.json`, block `as_reported`). The stored file is left as written.

**Note on the LODO MedAE.** The median MedAE of the ten studies is the mean of the two middle
per-study values, which are stored with two decimals (7.31 and 7.80): 7.555, printed as 7.56 in the
manuscript (the script's `:.2f` output shows 7.55 because of binary rounding).

## finetuned_cls/ — Supplementary Fig. S3

| File | Content | Written by |
|---|---|---|
| `embeddings_cls.npy` | CLS representations of the 10,988 downstream profiles from the fine-tuned fold-4 checkpoint (10,988 × 256, float32), rows as in `pretrained_cls/metadata.csv` | `scripts/evaluation/run_finetune_cls.sh` |
| `analysis_summary.json` | probe and geometry summary of the fine-tuned representation (age R² 0.938, tissue 0.930, study 0.949) | same |
| `cls_umap_coords.npy` | UMAP coordinates of the fine-tuned panel of Supplementary Fig. S3 | `scripts/evaluation/visualize_cls.py --dir results/finetuned_cls` |

These embeddings include profiles used for fine-tuning; Supplementary Fig. S3 is a representation
diagnostic, not a generalisation test.

## data_audit/ — data-structure counts (Methods, Discussion) and sensitivity analyses (Fig. 4c,d)

Individual identifiers (`individual` column) come from public GEO and TCGA metadata (GEO
`PATIENT_ID` within a study, TCGA participant barcode, HipSci donor code). They exist for a
minority of profiles, so counts of shared individuals are lower bounds.

| File | Content | Written by |
|---|---|---|
| `audit.json` | partitions, tissue and study annotation counts, individuals shared between partitions in both corpora, pretraining exposure classes of the 2,149 test profiles (Methods "Datasets", Discussion) | `scripts/data/data_structure_audit.py` |
| `downstream_exposure_per_profile.csv` | the 10,988 downstream profiles: study, tissue, age, original partition, benchmark membership, pretraining partition of the same accession, individual identifier, pretraining exposure class | same |
| `downstream_integrity_by_study.csv`, `downstream_integrity_summary.json` | per study, correlation of each downstream profile with a reference methylation profile and with the copy of the same accession in the pretraining corpus; E-GEOD-56105: median r 0.002 with its corpus copies, corpus copies 0.985 with the reference (Methods). The `column_map` block of the summary was written by an earlier revision of the script, before unusable columns were handled; it is not used in the manuscript | `scripts/data/downstream_integrity_check.py` |
| `tcga_pairs_summary.json` | downstream TCGA profiles compared with the corpus profiles of the same participant (none is the same measurement) | `scripts/data/tcga_same_participant_check.py` |
| `heldout_duplicates_summary.json`, `heldout_duplicates_flagged_pairs.csv` | duplicate and near-duplicate profiles within the held-out pretraining partition and between held-out and training profiles. The CSV lists the flagged pairs only (comparison, id_a, id_b, duplicate, near_duplicate); it is the negative-exclusion list of the hard-negative analysis | `scripts/data/pretrain_heldout_duplicates.py` |
| `downstream_duplicate_pairs.csv` | redundant downstream profile pairs (cosine ≥ 0.9999) used by `frozen_probe_sensitivity.py` | `scripts/data/analyze_full_dataset.py` |
| `frozen_probe_sensitivity.json`, `lodo_sensitivity.csv` | frozen probes and LODO with all profiles, without E-GEOD-56105, and additionally deduplicated (Methods: excluding E-GEOD-56105 changes age R² by 0.001 and balanced accuracy by at most 0.003) | `scripts/evaluation/frozen_probe_sensitivity.py` |
| `paired_bootstrap_grouped_sensitivity.json`, `per_study_mae.csv` | paired comparison without the 33 test profiles whose individual also occurs in development (n = 2,116), study-cluster bootstrap over 77 studies, per-study MAE (Fig. 4c,d) | `scripts/evaluation/paired_bootstrap_grouped_sensitivity.py` |
| `paired_bootstrap_by_exposure.json` | paired comparison by pretraining-exposure class, including the 626 test profiles with no profile of the same individual in pretraining training (classes 4–6; Results) | `scripts/evaluation/paired_bootstrap_by_exposure.py` |

`scripts/data/*` write to `outputs/data_audit/`; the files above are copies. The
`scripts/evaluation/*` sensitivity scripts read the shipped copies and write here directly.
`frozen_probe_sensitivity.json` was regenerated for the release in the pinned environment
(`requirements-lock.txt`, scikit-learn 1.8.0). The copy from the original run was made with another
scikit-learn version and differs only in four balanced accuracies at the third decimal (block
`as_reported`: 0.937 and 0.955 there, 0.936 and 0.956 here and in the manuscript).

## rope_pilot/ — Supplementary Note 1

| File | Content | Written by |
|---|---|---|
| `rope_ablation_summary.json` | contextual CpG similarity of the two pilot models for adjacent CpGs, random pairs and distance 100 (genomic positions 0.9831 / 0.8857; sequential positions 0.9931 / 0.9621) | `scripts/rope_pilot/analyze_rope_ablation_comparison.py` |
| `rope_ablation_curve_A.csv`, `rope_ablation_curve_B.csv` | mean cosine by genomic distance, run A (genomic positions) and run B (sequential positions) | same |

The script writes these files to `outputs/ablation_rope/`. The pilot checkpoints and their
reconstruction diagnostics are listed in `docs/PROVENANCE.md`.
