"""
MethylLlama pretrain CLS extraction + representation analysis  (genomic-RoPE correct).

Targets the CURRENT 6L Genomic-RoPE pretrain checkpoint (ep85). Fixes the bug in
an earlier probing script, which built the collator
WITHOUT genomic_rank_path and called the encoder WITHOUT position_ids — invalid
for a model trained with Genomic RoPE.

Outputs (in --outdir, default results/pretrained_cls/):
  embeddings_cls.npy        per-sample CLS pooler_output   [N, 256]
  embeddings_mean.npy       per-sample mean-pool           [N, 256]
  cpg_embedding_matrix.npy  CpG embedding table            [vocab, 256]
  metadata.csv              aligned obs (age/tissue/sex/dataset/split)  [N]
  analysis_summary.json     geometry + probe metrics
  report.txt                human-readable summary

Correctness note: this is the PRETRAIN encoder (no age supervision, age_weight=0),
so a weak age probe is EXPECTED — pretraining is pure reconstruction+contrastive.
The health signals to check are geometry (effective rank, anisotropy) and
biological structure (tissue separability). The genomic-RoPE fix is what makes
the extracted CLS actually match the trained model.

Usage (cluster):
  python scripts/evaluation/extract_pretrain_cls.py \
    --checkpoint checkpoints/pretrained/methyllama_pretrained_epoch85.ckpt \
    --data <altumage_21k_3way.h5ad> \
    --tokenizer tokenizer \
    --genomic_rank resources/cpg_genomic_rank_finetune.npy \
    --outdir results/pretrained_cls
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("pretrained_cls")


# ── args ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="MethylLlama pretrain CLS extraction + analysis")
    p.add_argument("--checkpoint", required=True, help="WCED pretrain .ckpt (ep85)")
    p.add_argument("--data", required=True, help="labeled h5ad (e.g. altumage_21k_3way.h5ad)")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--genomic_rank", required=True, help="cpg_genomic_rank_finetune.npy")
    p.add_argument("--outdir", default="results/pretrained_cls")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--age_col", default="age")
    p.add_argument("--label_cols", nargs="+", default=["tissue_type", "sex", "dataset"])
    p.add_argument("--split_col", default="split")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ── extraction (genomic-RoPE correct) ─────────────────────────────────────────
def extract(encoder, data_path, tokenizer_path, genomic_rank, batch_size, device):
    from bmfm_targets.tokenization import MultiFieldTokenizer
    from bmfm_methylation.shared.data_module import MethylationDataset, WCEDCollator

    encoder = encoder.to(device).eval()
    tokenizer = MultiFieldTokenizer.from_pretrained(tokenizer_path)
    dataset = MethylationDataset(h5ad_path=data_path, split=None, normalize_age=False)
    cpg_sites = dataset.cpg_sites
    log.info(f"Dataset: {len(dataset)} samples × {len(cpg_sites)} CpGs")

    # THE FIX: genomic_rank_path set → collator sorts by genomic position and emits
    # position_ids; input_ratio=1.0 keeps all CpGs (no masking).
    collator = WCEDCollator(
        tokenizer=tokenizer,
        cpg_sites=cpg_sites,
        vocab_size=len(cpg_sites),
        input_ratio=1.0,
        contrastive=False,
        genomic_rank_path=genomic_rank,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=collator,
        shuffle=False, num_workers=0, pin_memory=(device == "cuda"),
    )

    cls_list, mean_list = [], []
    saw_pos = False
    with torch.no_grad():
        for i, batch in enumerate(loader):
            cpg_ids = batch["cpg_ids"].to(device)
            beta_values = batch["beta_values"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            position_ids = batch.get("position_ids")
            if position_ids is not None:
                position_ids = position_ids.to(device)
                saw_pos = True
            input_ids = torch.stack([cpg_ids.float(), beta_values], dim=1)
            out = encoder(
                input_ids=input_ids,
                attention_mask=attn_mask,
                position_ids=position_ids,   # <-- genomic RoPE positions
            )
            cls_list.append(out.pooler_output.cpu().float().numpy())
            hidden = out.last_hidden_state[:, 1:, :].cpu().float()
            m1d = attn_mask[:, 1:].cpu().float().unsqueeze(-1)
            mean_list.append(((hidden * m1d).sum(1) / m1d.sum(1).clamp(min=1)).numpy())
            if (i + 1) % 50 == 0:
                log.info(f"  batch {i+1}/{len(loader)}")

    if not saw_pos:
        raise RuntimeError(
            "position_ids missing from collator output — genomic_rank not active. "
            "Extraction would NOT match the Genomic-RoPE model. Aborting."
        )
    cls = np.concatenate(cls_list)
    mean = np.concatenate(mean_list)
    # aligned metadata in loader (shuffle=False) order
    meta = dataset.adata.obs.copy()
    meta = meta.reset_index().rename(columns={meta.index.name or "index": "sample_id"})
    # CpG alignment. genomic_rank is indexed by DATA COLUMN j (identity vocab slots),
    # but the encoder embedding-table ROW for column j is the tokenizer vocab id
    # collator.vocab_cpg_ids[j] (into the full 49,161-row table). Use that, not j.
    gr = np.load(genomic_rank)
    encoder_vocab_ids = np.asarray(collator.vocab_cpg_ids, dtype=np.int64)
    align = pd.DataFrame(
        {"column_index": np.arange(len(cpg_sites)),
         "encoder_vocab_id": encoder_vocab_ids,
         "cpg_name": list(cpg_sites),
         "genomic_rank": gr[: len(cpg_sites)]}
    )
    log.info(f"Extracted CLS {cls.shape}  Mean {mean.shape}  meta {meta.shape}")
    return cls, mean, meta, align


# ── geometry ──────────────────────────────────────────────────────────────────
def geometry(X):
    Xc = X - X.mean(0)
    s = np.linalg.svd(Xc, compute_uv=False)
    p = s**2 / (s**2).sum()
    eff_rank = float(np.exp(-(p * np.log(p + 1e-12)).sum()))
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X), min(4000, len(X)), replace=False)
    G = Xn[idx] @ Xn[idx].T
    iu = np.triu_indices(len(idx), 1)
    var = X.var(0)
    return {
        "mean_l2_norm": float(np.linalg.norm(X, axis=1).mean()),
        "effective_rank": round(eff_rank, 2),
        "top1_sv_frac": round(float(p[0]), 4),
        "top10_sv_frac": round(float(p[:10].sum()), 4),
        "anisotropy_mean_cos": round(float(G[iu].mean()), 4),
        "dead_dims_lt1pct": int((var < 0.01 * var.max()).sum()),
    }


# ── probes ────────────────────────────────────────────────────────────────────
def probe_age(X, age, split):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score
    import torch.nn as nn

    tr = (split != "test") & ~np.isnan(age)
    te = (split == "test") & ~np.isnan(age)
    sc = StandardScaler().fit(X[tr])
    Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
    ytr, yte = age[tr], age[te]

    lin = Ridge(alpha=1.0).fit(Xtr, ytr).predict(Xte)
    lin_r2, lin_med = float(r2_score(yte, lin)), float(np.median(np.abs(lin - yte)))

    # replica of the model's age head — the correctness gate for the fine-tuned model
    torch.manual_seed(0)
    mu, sd = ytr.mean(), ytr.std()
    head = nn.Sequential(nn.LayerNorm(X.shape[1]), nn.Linear(X.shape[1], 128),
                         nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 1))
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
    lossf = nn.HuberLoss(delta=5.0 / sd)
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor((ytr - mu) / sd, dtype=torch.float32)
    idx = np.arange(len(Xt))
    head.train()
    for _ in range(300):
        np.random.shuffle(idx)
        for j in range(0, len(idx), 256):
            b = idx[j:j + 256]
            opt.zero_grad()
            lossf(head(Xt[b]).squeeze(-1), yt[b]).backward()
            opt.step()
    head.eval()
    with torch.no_grad():
        p = head(torch.tensor(Xte, dtype=torch.float32)).squeeze(-1).numpy() * sd + mu
    return {
        "linear_ridge_r2": round(lin_r2, 3), "linear_ridge_medae": round(lin_med, 2),
        "replica_head_r2": round(float(r2_score(yte, p)), 3),
        "replica_head_medae": round(float(np.median(np.abs(p - yte))), 2),
        "n_train": int(tr.sum()), "n_test": int(te.sum()),
    }


def probe_clf(X, labels, split):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import balanced_accuracy_score

    labels = np.asarray(pd.Series(labels).astype(str), dtype=object)
    vc = pd.Series(labels).value_counts()
    keep = np.isin(labels, vc[vc >= 30].index) & (labels != "nan")
    tr = keep & (split != "test")
    te = keep & (split == "test")
    if tr.sum() < 50 or te.sum() < 20 or pd.Series(labels[keep]).nunique() < 2:
        return None
    le = LabelEncoder().fit(labels[keep])
    sc = StandardScaler().fit(X[tr])
    m = LogisticRegression(max_iter=2000).fit(sc.transform(X[tr]), le.transform(labels[tr]))
    pred = m.predict(sc.transform(X[te]))
    return {
        "balanced_acc": round(float(balanced_accuracy_score(le.transform(labels[te]), pred)), 3),
        "n_classes": int(pd.Series(labels[keep]).nunique()),
        "chance": round(1.0 / pd.Series(labels[keep]).nunique(), 3),
        "n_test": int(te.sum()),
    }


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    from bmfm_methylation.llama.finetune_llama import load_wced_llama_checkpoint
    log.info(f"Loading pretrain checkpoint: {args.checkpoint}")
    module = load_wced_llama_checkpoint(args.checkpoint)
    encoder = module.encoder
    log.info(f"Encoder: {encoder.config.num_hidden_layers}L × {encoder.config.hidden_size}D")

    # CpG embedding matrix (the token embedding table)
    cpg_emb = encoder.embeddings.cpg_sites_embeddings.weight.detach().cpu().float().numpy()
    np.save(outdir / "cpg_embedding_matrix.npy", cpg_emb)
    log.info(f"Saved cpg_embedding_matrix.npy {cpg_emb.shape}")

    cls, mean, meta, align = extract(
        encoder, args.data, args.tokenizer, args.genomic_rank, args.batch_size, args.device
    )
    np.save(outdir / "embeddings_cls.npy", cls)
    np.save(outdir / "embeddings_mean.npy", mean)
    meta.to_csv(outdir / "metadata.csv", index=False)
    align.to_csv(outdir / "cpg_alignment.csv", index=False)

    split = np.asarray(meta[args.split_col].astype(str)) if args.split_col in meta else np.full(len(cls), "train")
    age = np.asarray(meta[args.age_col], dtype=float) if args.age_col in meta else None

    summary = {
        "checkpoint": args.checkpoint,
        "n_samples": int(len(cls)),
        "cpg_embedding_matrix_shape": list(cpg_emb.shape),
        "geometry_cls": geometry(cls),
        "geometry_mean": geometry(mean),
    }
    if age is not None:
        summary["age_probe_cls"] = probe_age(cls, age, split)
        summary["age_probe_mean"] = probe_age(mean, age, split)
    summary["class_probes_cls"] = {
        c: probe_clf(cls, meta[c], split) for c in args.label_cols if c in meta
    }

    with open(outdir / "analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # report
    lines = ["MethylLlama PRETRAIN (ep85) — CLS representation analysis", "=" * 55,
             f"checkpoint: {args.checkpoint}", f"samples: {summary['n_samples']}", ""]
    for tag in ("cls", "mean"):
        g = summary[f"geometry_{tag}"]
        lines += [f"[{tag.upper()}] geometry:",
                  f"  eff_rank={g['effective_rank']}/{cls.shape[1]}  top1_sv={g['top1_sv_frac']}  "
                  f"anisotropy={g['anisotropy_mean_cos']}  dead_dims={g['dead_dims_lt1pct']}"]
    if age is not None:
        a = summary["age_probe_cls"]
        lines += ["", "[CLS] age probe (pretrain — weak is EXPECTED, no age supervision):",
                  f"  linear R²={a['linear_ridge_r2']}  replica-head R²={a['replica_head_r2']}  "
                  f"MedAE={a['replica_head_medae']}yr"]
    lines += ["", "[CLS] biological structure (balanced-acc vs chance):"]
    for c, r in summary["class_probes_cls"].items():
        if r:
            lines += [f"  {c}: {r['balanced_acc']} (chance {r['chance']}, {r['n_classes']} classes)"]
    (outdir / "report.txt").write_text("\n".join(lines))
    log.info("\n" + "\n".join(lines))
    log.info(f"All outputs → {outdir}/")


if __name__ == "__main__":
    main()
