"""
MethylLlama Embedding Evaluation
=================================
Extracts 256-dim mean-pooled embeddings from MethylLlama and evaluates them
on tissue_type / gender / dataset classification — mimicking bmfm-targets
SGDCallback + CziBenchmarkCallback logic.

Supports BOTH checkpoint types:
  --ckpt_type pretrain   loads WCEDLlamaModule (encoder only, no age head)
  --ckpt_type finetune   loads MethylationAgeRegressorLlama (full fine-tuned)

Running both back-to-back tells you:
  pretrain F1  → did reconstruction alone encode tissue biology?
  finetune F1  → did age fine-tuning preserve or destroy that signal?

Outputs:
  embeddings.npy    — [N, 256] float32
  eval_results.txt  — SGD F1 + 95% CI, per-class breakdown, CV F1

Usage:
    # pretrained checkpoint (before fine-tuning):
    python scripts/utils/eval_embeddings.py \\
        --checkpoint outputs/pretrain-llama-wced/.../epoch=98.ckpt \\
        --ckpt_type pretrain \\
        --data /path/to/finetuning_19608_clean.h5ad \\
        --tokenizer tokenizer_llama_pretrain49k \\
        --outdir outputs/eval_embeddings/pretrain/

    # fine-tuned checkpoint (after age training):
    python scripts/utils/eval_embeddings.py \\
        --checkpoint outputs/finetune-llama-small/.../best.ckpt \\
        --ckpt_type finetune \\
        --data /path/to/finetuning_19608_clean.h5ad \\
        --tokenizer tokenizer_llama_pretrain49k \\
        --outdir outputs/eval_embeddings/finetune/
"""

import argparse
import os

import numpy as np
import scanpy as sc
import torch
from scipy.special import binom
from sklearn.linear_model import LogisticRegressionCV, SGDClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader

# ── project imports (deferred to main to avoid circular import issues) ────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--ckpt_type",  default="pretrain",
                   choices=["pretrain", "finetune"],
                   help="pretrain=WCEDLlamaModule, finetune=MethylationAgeRegressorLlama")
    p.add_argument("--data",       required=True)
    p.add_argument("--tokenizer",  required=True)
    p.add_argument("--outdir",     default="outputs/eval_embeddings")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--target_col", default="tissue_type",
                   help="obs column for classification (tissue_type, gender, dataset)")
    p.add_argument("--split_col",  default="split")
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ── Embedding extraction ──────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(encoder, data_path, tokenizer_path, batch_size, device):
    from bmfm_targets.tokenization import MultiFieldTokenizer
    from bmfm_methylation.shared.data_module import MethylationDataset, WCEDCollator

    encoder.eval().to(device)

    # Load tokenizer
    tokenizer = MultiFieldTokenizer.from_pretrained(tokenizer_path)

    # Dataset — no split filter, loads all samples
    dataset = MethylationDataset(h5ad_path=data_path, split=None, normalize_age=False)
    cpg_sites = dataset.cpg_sites

    # Collator: input_ratio=1.0 → all CpGs as input, no held-out
    collator = WCEDCollator(
        tokenizer=tokenizer,
        cpg_sites=cpg_sites,
        vocab_size=len(cpg_sites),
        input_ratio=1.0,
        contrastive=False,
    )
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator,
                        shuffle=False, num_workers=4)

    all_embs = []
    for batch in loader:
        cpg_ids     = batch["cpg_ids"].to(device)
        beta_values = batch["beta_values"].to(device)
        attn_mask   = batch["attention_mask"].to(device)

        # Encoder forward — same as MethylationAgeRegressorLlama._encode_cls
        input_ids = torch.stack([cpg_ids.float(), beta_values], dim=1)  # [B, 2, L]
        out = encoder(input_ids=input_ids, attention_mask=attn_mask)

        # Mean pooling over non-CLS tokens (skip pos 0)
        hidden = out.last_hidden_state[:, 1:, :]         # [B, L-1, 256]
        mask   = attn_mask[:, 1:].unsqueeze(-1).float()  # [B, L-1, 1]
        emb    = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)  # [B, 256]
        all_embs.append(emb.cpu().float().numpy())

    return np.concatenate(all_embs, axis=0)  # [N, 256]


# ── SGDCallback equivalent ────────────────────────────────────────────────────

def wilson_ci(n_correct, n_total, z=1.96):
    p = n_correct / n_total
    denom = 1 + z**2 / n_total
    center = (p + z**2 / (2 * n_total)) / denom
    margin = z * np.sqrt(p*(1-p)/n_total + z**2/(4*n_total**2)) / denom
    return center - margin, center + margin


def sgd_eval(X_train, y_train, X_test, y_test, target_col):
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    clf = SGDClassifier(loss="modified_huber", max_iter=1000,
                        class_weight="balanced", random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    f1    = f1_score(y_test, y_pred, average="macro", zero_division=0)
    n_ok  = int((y_pred == y_test).sum())
    lo, hi = wilson_ci(n_ok, len(y_test))

    lines = []
    lines.append(f"── SGDClassifier on {target_col} ──")
    lines.append(f"  Macro F1 : {f1:.4f}")
    lines.append(f"  Accuracy : {n_ok}/{len(y_test)} = {n_ok/len(y_test):.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
    lines.append("")
    lines.append(classification_report(y_test, y_pred, zero_division=0))
    return "\n".join(lines), f1


# ── CziBenchmarkCallback equivalent ──────────────────────────────────────────

def cv_logreg_eval(X, y, target_col, n_folds=5):
    scaler = StandardScaler()
    X      = scaler.fit_transform(X)

    clf = LogisticRegressionCV(
        cv=n_folds, max_iter=1000,
        class_weight="balanced", scoring="f1_macro",
        random_state=42, n_jobs=-1,
    )
    clf.fit(X, y)
    best_scores = clf.scores_[clf.classes_[0]].mean(axis=0)
    f1_cv = best_scores.max()

    lines = []
    lines.append(f"── LogisticRegressionCV ({n_folds}-fold) on {target_col} ──")
    lines.append(f"  CV Macro F1 : {f1_cv:.4f}")
    return "\n".join(lines), f1_cv


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Loading data: {args.data}")
    adata = sc.read_h5ad(args.data)
    print(f"  {adata.n_obs} samples × {adata.n_vars} CpGs")
    print(f"  obs cols: {list(adata.obs.columns)}")

    print(f"\nLoading checkpoint ({args.ckpt_type}): {args.checkpoint}")
    if args.ckpt_type == "pretrain":
        from bmfm_methylation.llama.finetune_llama import load_wced_llama_checkpoint
        pretrained = load_wced_llama_checkpoint(args.checkpoint)
        encoder = pretrained.encoder
    else:
        from bmfm_methylation.llama.finetune_llama import MethylationAgeRegressorLlama
        ft_module = MethylationAgeRegressorLlama.load_from_checkpoint(args.checkpoint)
        encoder = ft_module.encoder

    print("\nExtracting embeddings...")
    embs = extract_embeddings(encoder, args.data, args.tokenizer,
                              args.batch_size, args.device)
    print(f"  Embeddings shape: {embs.shape}")
    np.save(os.path.join(args.outdir, "embeddings.npy"), embs)

    # ── Label encoding ────────────────────────────────────────────────────────
    if args.target_col not in adata.obs.columns:
        raise ValueError(f"Column '{args.target_col}' not in obs. "
                         f"Available: {list(adata.obs.columns)}")

    le = LabelEncoder()
    labels = le.fit_transform(adata.obs[args.target_col].astype(str))
    print(f"\n  Classes ({len(le.classes_)}): {list(le.classes_)}")

    split = adata.obs[args.split_col].values
    train_mask = np.isin(split, ["train", "valid"])
    test_mask  = split == "test"

    X_train, y_train = embs[train_mask], labels[train_mask]
    X_test,  y_test  = embs[test_mask],  labels[test_mask]

    print(f"  Train: {X_train.shape[0]}  Test: {X_test.shape[0]}")

    # ── Evaluation ────────────────────────────────────────────────────────────
    sgd_report, sgd_f1   = sgd_eval(X_train, y_train, X_test, y_test, args.target_col)
    cv_report,  cv_f1    = cv_logreg_eval(embs, labels, args.target_col)

    report = "\n".join([
        f"MethylLlama Embedding Evaluation",
        f"Checkpoint : {args.checkpoint}",
        f"Target     : {args.target_col}",
        f"Device     : {args.device}",
        "=" * 60,
        sgd_report,
        "=" * 60,
        cv_report,
    ])

    print("\n" + report)
    out_path = os.path.join(args.outdir, "eval_results.txt")
    with open(out_path, "w") as f:
        f.write(report)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
