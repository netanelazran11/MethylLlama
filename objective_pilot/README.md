# Contrastive-weight pilot

The pilot of Supplementary Note 2 and Methods ("Contrastive-weight pilot"): what the InfoNCE term adds to
withheld-CpG reconstruction. Four arms differing only in the InfoNCE weight λ (0, 0.01, 0.05, 0.2) were pretrained
with seed 42 on a fixed subset of 9,000 training, 1,000 validation and 1,000 test profiles, with the production
architecture, genomic-rank rotary encoding, view construction, optimiser and per-GPU batch of 32, but a reduced budget
(20 epochs, 200 warm-up steps, effective batch 64). The last checkpoint of every arm was evaluated with the
manuscript's own pipelines.

Its numbers can be checked without running anything: `tutorials/reproduce_paper_results.ipynb` (section 8) reads
`results/objective_pilot/pilot_table.csv`, and `summarize_pilot.py` rebuilds that table from the per-arm files in
`results/objective_pilot/arms/`:

    python -m objective_pilot.summarize_pilot --eval-root results/objective_pilot/arms --out outputs/objective_pilot/check

## The λ = 0 arm

`WCEDLlamaModule` encodes the second view only when the contrastive weight is positive, so a run with weight 0
would also lose half of the reconstruction signal. `wced_llama_recon_only.py` removes that confound:
`WCEDLlamaAblationModule` always encodes both views and averages their reconstruction losses, and adds λ · InfoNCE.
At λ > 0 it reproduces the reported objective exactly (loss and gradients); at λ = 0 it is reconstruction only.
`test_recon_only.py` checks this on CPU without data (`python -m objective_pilot.test_recon_only`; every
pretraining job runs it first).

## Steps

Run from the repository root with `configs/paths.sh` in place.

| Step | Command | Output (`outputs/objective_pilot/`) |
|---|---|---|
| 1. Pilot subset, drawn from the train / validation / held-out partitions of the reported run | `sbatch objective_pilot/create_pilot_subset.sh` | `pilot_data/pilot_9k_seed2026.h5ad` |
| 2. Pretraining, one array task per arm | `sbatch objective_pilot/run_objective_pilot.sh` | `pilot/lam<λ>_seed42/` |
| 3. Evaluation per arm: cross-view retrieval on the 16,912 held-out profiles (`scripts/evaluation/view_consistency_heldout_full.py`), reconstruction on 5,000 held-out profiles (`reconstruction_withheld_subset.py`), frozen probes on the downstream collection (`scripts/evaluation/extract_pretrain_cls.py`); task 4 evaluates the released checkpoint as a pipeline check | `sbatch objective_pilot/evaluate_pilot.sh` | `pilot_eval/<arm>/` |
| 4. Hard-negative retrieval per arm (31 negatives from pools of increasing similarity) | `python -m objective_pilot.hard_negative_eval --emb-dir outputs/objective_pilot/pilot_eval/<arm>/view_consistency/per_run/ --out outputs/objective_pilot/pilot_eval/<arm>/hard_negative.json --per-query-out outputs/objective_pilot/pilot_eval/<arm>/hard_negative_queries.npz` | `pilot_eval/<arm>/hard_negative*.{json,npz}` |
| 5. Table and paired comparison with λ = 0 (bootstrap over query profiles, B = 2,000) | `python -m objective_pilot.summarize_pilot` | `pilot_summary/pilot_table.csv`, `pilot_verdict.json` |

`pretrain_recon_only.py` is the entry point of step 2: it runs `bmfm_methylation/llama/pretrain_llama.py` with the
module of `wced_llama_recon_only.py` in place of `WCEDLlamaModule`. Resources of the original runs: one L40S
(48 GB) per arm for about 35 h; evaluation about 12 h per arm on the same GPU.

## What the manuscript reports

From `pilot_table.csv` (arms `lam0.0_seed42_ep19` and `lam0.05_seed42_ep19`): disjoint-view retrieval against 31
negatives matched for study and tissue, 28.5 % → 55.1 % (chance 3.1 %), monotonic over the four weights;
disjoint-view retrieval among all 16,912 held-out profiles, 20.0 % → 43.7 %; reconstruction Pearson correlation
0.930 → 0.931. Higher weights increase retrieval further and lower the frozen probes (λ = 0.2 below λ = 0 for age,
tissue and study). Single seed, reduced budget: the pilot shows the direction of the effect, not its size at full
pretraining scale.
