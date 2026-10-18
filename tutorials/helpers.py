"""
Helpers shared by the tutorial notebooks.

The inference path is the one used for the manuscript
(scripts/evaluation/extract_pretrain_cls.py and scripts/finetune/extract_kfold_test_predictions.py):
full profiles (input_ratio=1.0), CpGs ordered by genome position and passed with
genomic-rank position ids, sample representation = pooler output of the CLS token.
"""
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
HF_REPO = "netanelazran1/MethylLlama"
DEMO = ROOT / "tutorials/data/methylllama_demo_500samples.h5ad"
TOKENIZER = ROOT / "tokenizer"
RANK_DOWNSTREAM = ROOT / "resources/cpg_genomic_rank_finetune.npy"   # 21,368-CpG panel of the demo data


def checkpoint(name):
    """Path of a released checkpoint, e.g. 'pretrained/methyllama_pretrained_epoch85.ckpt'.

    Downloaded from Hugging Face, or read from $METHYLLAMA_CKPT_DIR/<name> if that variable is set.
    """
    local = os.environ.get("METHYLLAMA_CKPT_DIR")
    if local:
        return str(Path(local) / name)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=HF_REPO, filename=name)


def load_demo():
    """The 500-profile demo data set (21,368 CpGs) with its annotations."""
    from bmfm_methylation.shared.data_module import MethylationDataset
    return MethylationDataset(h5ad_path=str(DEMO), split=None, normalize_age=False)


def _loader(dataset, rows, batch_size):
    from bmfm_targets.tokenization import MultiFieldTokenizer
    from bmfm_methylation.shared.data_module import WCEDCollator
    collate = WCEDCollator(tokenizer=MultiFieldTokenizer.from_pretrained(str(TOKENIZER)),
                           cpg_sites=dataset.cpg_sites, vocab_size=len(dataset.cpg_sites),
                           input_ratio=1.0, contrastive=False, genomic_rank_path=str(RANK_DOWNSTREAM))
    return DataLoader(Subset(dataset, list(rows)), batch_size=batch_size, collate_fn=collate, shuffle=False)


def _forward(encoder, batch, device):
    x = torch.stack([batch["cpg_ids"].float(), batch["beta_values"]], dim=1).to(device)
    return encoder(input_ids=x, attention_mask=batch["attention_mask"].to(device),
                   position_ids=batch["position_ids"].to(device)).pooler_output


def encode(encoder, dataset, rows, batch_size=8, device="cpu"):
    """256-dimensional sample representations for the given rows of the data set."""
    encoder = encoder.to(device).eval()
    out = []
    with torch.no_grad():
        for batch in _loader(dataset, rows, batch_size):
            out.append(_forward(encoder, batch, device).float().cpu().numpy())
    return np.concatenate(out)


def predict_age(module, dataset, rows, batch_size=8, device="cpu"):
    """Predicted age in years from one fine-tuned fold (head output de-standardised with the fold's own statistics)."""
    module = module.to(device).eval()
    preds = []
    with torch.no_grad():
        for batch in _loader(dataset, rows, batch_size):
            z = module.age_head(_forward(module.encoder, batch, device)).squeeze(-1)
            preds.append((z.double() * float(module.age_std) + float(module.age_mean)).cpu().numpy())
    return np.concatenate(preds)
