# CNS tumour classification

The pipeline behind the CNS results of the manuscript: Figs. 5 and 6, Supplementary Figs. S5–S8 and
Supplementary Table S3. Frozen 256-dimensional CLS embeddings of MethylLlama and MethylGPT, and raw-β
baselines, are classified into the 91 methylation classes of the Capper et al. reference cohort with the same
linear classifier, on complete profiles and on profiles that keep only a random share of their CpGs.
Neither encoder is trained here.

The numbers of the manuscript can be checked without running anything: `tutorials/reproduce_paper_results.ipynb`
(section 7) recomputes them from `results/cns/`, and `tests_from_correctness.py` recomputes Fig. 6 and Table S3
from `results/cns/sparse_correctness.npz`.

## Inputs

| Input | Where |
|---|---|
| Pretraining corpus h5ad (the Capper profiles are read from it, on the 49,156-CpG panel) | `${METHYLLAMA_DATA}/data_methyl_pretrain_type3_h5ad/methylgpt_pretrain_type3.h5ad` |
| Capper sample sheet: GSM, set (reference GSE90496 / validation GSE109379), methylation class, material, platform, from the GEO series matrices | `resources/cns/capper_samples.csv`, `resources/cns/class_description.csv` |
| Pretraining partition of every corpus profile | `outputs/pretrain_overlap/pretrain_ids.csv.gz` (`scripts/data/pretrain_overlap_by_id.py`) |
| MethylLlama checkpoint, tokenizer, genomic rank | `checkpoints/pretrained/` (`scripts/download_checkpoint.sh`), `tokenizer/`, `resources/cpg_genomic_rank.npy` |
| MethylGPT checkpoint and input list | `pretrained/methylgpt/medium-best_model_epoch6.pt`, `${METHYLGPT_DATA}/cpg_ids/probe_ids_type3.csv` (`scripts/methylgpt/README.md`) |
| External cohort labels | Supplementary Table 1 of Sturm et al., *Nat. Med.* 2023 (`41591_2023_2255_MOESM3_ESM.xlsx`) and the two GSE215240 series matrices; read by `external_labels.py` |

MethylLlama steps run in the MethylLlama environment (`methyllama_env`), MethylGPT extraction in the MethylGPT
environment (`methylgpt_env`, with flash-attn; see `scripts/methylgpt/README.md`), all from the repository root.
Everything is written under `outputs/cns/`.

## Steps

| Step | Command | Output (`outputs/cns/`) | Manuscript |
|---|---|---|---|
| 1. Checks of the inputs: same CpG panel and order for both models and the corpus; Capper profiles in the corpus, their pretraining partition and missing-value fraction; near-duplicate pairs | `python scripts/cns/cns_preflight.py` | `preflight/` | Methods (Data) |
| 2. Capper dataset and the shared folds (5 × 5 stratified CV, few-shot draws) | `python scripts/cns/build_cns_dataset.py` | `data/cns_capper.h5ad`, `data/folds.csv`, `data/fewshot.csv` | |
| 3. Complete-profile embeddings, MethylLlama | `python scripts/evaluation/extract_pretrain_cls.py --checkpoint checkpoints/pretrained/methyllama_pretrained_epoch85.ckpt --data outputs/cns/data/cns_capper.h5ad --tokenizer tokenizer --genomic_rank resources/cpg_genomic_rank.npy --outdir outputs/cns/cls_methylllama` | `cls_methylllama/` | Fig. 5 |
| 3. Complete-profile embeddings, MethylGPT | `python scripts/cns/extract_cls_methylgpt.py` | `cls_methylgpt/` | Fig. 5 |
| 4. Classification of complete profiles, all representations | `python scripts/cns/probe_cns.py --reps emb:llama_cls=outputs/cns/cls_methylllama,emb:gpt_cls=outputs/cns/cls_methylgpt,emb:llama_mean=outputs/cns/cls_methylllama#embeddings_mean.npy,emb:gpt_mean=outputs/cns/cls_methylgpt#embeddings_mean.npy,raw_top5k,raw_top10k,raw_top32k,raw_all,raw_pca256` | `probe_all/<rep>/` | |
| 5. Comparison of complete profiles | `python scripts/cns/compare_cns.py --reps llama_cls,gpt_cls,llama_mean,gpt_mean,raw_top5k,raw_top10k,raw_top32k,raw_all,raw_pca256 --raw raw_top5k,raw_top10k,raw_top32k,raw_all,raw_pca256` | `compare/` | Supplementary Fig. S6; five raw-β variants (Methods) |
| 6. Masks: for each profile, a random order of its measured CpGs, nested across levels (3 seeds) | `python scripts/cns/make_sparse_masks.py`, then `python scripts/cns/make_sparse_masks_high.py` (60–90 %, same orders) | `sparse/masks/`, `sparse/masks_high/` | Methods (Input levels) |
| 7. Embeddings of the masked profiles | `python scripts/cns/extract_sparse_llama.py --levels 100,50,20,10,5,2,1,0.5 --seeds 0,1,2` and `--masks outputs/cns/sparse/masks_high --levels 60,70,80,90 --seeds 0,1,2`; the same for `extract_sparse_gpt.py` | `sparse/emb_llama/`, `sparse/emb_gpt/` | |
| 8. Scenario A: classifier trained on complete profiles, tested on masked ones | `python scripts/cns/probe_sparse.py --scenario A --levels 100,50,20,10,5,2,1,0.5 --seeds 0,1,2 --reps llama_cls=outputs/cns/sparse/emb_llama,gpt_cls=outputs/cns/sparse/emb_gpt,raw_top5k,raw_pca256` | `sparse/probe/A/` | Supplementary Fig. S7; Fig. 6c |
| 9. Scenario B: classifier trained and tested at the same level | `python scripts/cns/probe_sparse_wideC.py --scenario B --levels 50,20,10,5,2,1,0.5 --seeds 0,1,2 --reps <as in 8>`, then the same with `--masks outputs/cns/sparse/masks_high --levels 60,70,80,90` | `sparse/probe_wideC/B/` | Fig. 6; Supplementary Table S3 |
| 10. Comparison of both scenarios and embedding stability | `ln -s ../probe/A outputs/cns/sparse/probe_wideC/A`, then `python scripts/cns/compare_sparse.py` | `sparse/compare_wideC_all/` | Supplementary Fig. S7 |
| 11. External cohort GSE215240: labels, input, embeddings | `python scripts/cns/external_labels.py --supp <xlsx> --series <dir>`; `python scripts/cns/build_external_cns.py`; `python scripts/cns/extract_external_llama.py`; `python scripts/cns/extract_external_gpt.py` | `external/` | Supplementary Fig. S8 |
| 12. External cohort: classification (classifier trained once on all reference profiles), complete and masked profiles | `python scripts/cns/probe_external.py`; masks and embeddings with `make_sparse_masks.py` and `extract_sparse_*.py` (`--data outputs/cns/external/GSE215240.h5ad --masks outputs/cns/external/sparse/masks --outdir outputs/cns/external/sparse/emb_<model> --reference outputs/cns/external/emb_<model>`); `python scripts/cns/probe_external_sparse.py`; `python scripts/cns/probe_external_sparse_B.py` | `external/probe/`, `external/sparse/` | Supplementary Fig. S8 |
| 13. Figures | `python scripts/cns/fig_cns_landscape.py`, `fig_cns_sparse_B.py`, then `fig_cns_final.py` (Figs. 5 and 6); `fig_cns_taxonomy.py` (S5); `fig_cns_data.py`, then `fig_cns.py` (S6); `fig_cns_sparse.py` (S7); `fig_cns_external.py` (S8) | `figures/` | Figs. 5–6, S5–S8 |

Notes
- **Regularisation.** The inner cross-validation chooses C from {10⁻³, …, 10²} (`probe_cns.C_GRID`). In Scenario B
  the largest value was chosen in most folds for every representation, so Scenario B was run with the grid extended
  to 10⁴ (`probe_sparse_wideC.py`, otherwise `probe_sparse.py` unchanged). Fig. 6 and Table S3 use these runs; at
  100 % both scenarios are the same fit, taken from Scenario A.
- **MethylGPT input.** MethylGPT reads all 49,156 positions; CpGs that are missing or removed by a mask get its
  missing-value convention (−2). It is loaded strictly and run with batch size 1 (`extract_cls_methylgpt.py`
  explains why).
- **Reproducibility checks built into the scripts.** The 100 % level of `extract_sparse_*.py` must match the
  complete-profile embeddings (minimum cosine > 0.9999); the masks above 50 % must extend the existing orders
  (`make_sparse_masks_high.py`); `build_cns_dataset.py` checks the folds and the class counts.
- **Resources of the original runs.** Embedding extraction: one 48 GB GPU, 1–3 h per model and seed. Probes on
  the raw baselines: 12–16 CPU cores, up to about 1 h per level with BLAS limited to one thread per worker
  (`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`).

## Results shipped

`results/cns/` (see `results/README.md`) holds the comparison of complete profiles, the plotted values of every
CNS figure, the comparison of both incomplete-input scenarios, the aggregate results of the external cohort and
`sparse_correctness.npz`. Methylation values, masks, per-profile embeddings and the labels of the external cohort
(derived from a patient-level supplementary table) are not included.
