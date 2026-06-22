"""
Classification DataModule — shared by smoking, sex, disease, and any other
discrete-label downstream task.

Reads an h5ad that has:
  - X: beta values [n_samples, n_cpg]
  - obs[label_col]: string or int class labels
  - obs[split_col]: "train" / "valid" / "test"

Returns batches: {cpg_ids, beta_values, attention_mask, class_label (long)}
Compatible with the same WCED encoder used in finetune_wced.py.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

# Reuse the h5ad loader + cache from the shared data_module
from bmfm_methylation.shared.data_module import _read_h5ad_robust


SMOKING_LABEL_MAP = {"current": 0, "former": 1, "never": 2}
EVER_NEVER_LABEL_MAP = {"current": 1, "former": 1, "never": 0}  # binary: ever vs never
SEX_LABEL_MAP = {"M": 0, "F": 1, "male": 0, "female": 1, "0": 0, "1": 1}


class ClassificationDataset(Dataset):
    """
    Dataset for any discrete-label methylation fine-tuning task.

    Args:
        h5ad_path: Path to h5ad file.
        split: "train", "valid", or "test".
        label_col: obs column with class labels (string or int).
        label_map: Dict mapping string label → int class index.
                   If None, labels must already be integers.
        split_col: obs column with split assignments.
        subset_k: Number of CpG sites to sample per forward pass.
        fixed_subset: If True, same CpG subset every call (eval mode).
        fixed_subset_seed: RNG seed for fixed subset.
        cpg_vocab: Optional pre-built array of CpG token IDs [n_cpg].
                   If None, derived from var_names or range(n_cpg).
        max_n_samples: If set, randomly subsample this many training samples
                       (used for data-efficiency experiments).
        subsample_seed: RNG seed for subsampling.
    """

    def __init__(
        self,
        h5ad_path: str,
        split: str,
        label_col: str,
        label_map: Optional[Dict[str, int]] = None,
        split_col: str = "split",
        subset_k: int = 4000,
        fixed_subset: bool = False,
        fixed_subset_seed: int = 42,
        cpg_vocab: Optional[np.ndarray] = None,
        max_n_samples: Optional[int] = None,
        subsample_seed: int = 0,
    ):
        self.subset_k = subset_k
        self.fixed_subset = fixed_subset
        self.label_col = label_col
        self.label_map = label_map

        adata = _read_h5ad_robust(h5ad_path)

        # Filter by split
        if split_col in adata.obs.columns:
            mask = adata.obs[split_col] == split
            adata = adata[mask].copy()
        logger.info(f"[{split}] {len(adata)} samples after split filter")

        # Encode labels
        raw_labels = adata.obs[label_col].values
        if label_map is not None:
            encoded = np.array([label_map[str(l)] for l in raw_labels], dtype=np.int64)
        else:
            encoded = np.array(raw_labels, dtype=np.int64)

        # Drop samples with invalid labels
        valid = encoded >= 0
        adata = adata[valid].copy()
        encoded = encoded[valid]
        logger.info(f"[{split}] {len(adata)} samples after label filter")

        # Optional subsampling (data-efficiency experiments)
        if max_n_samples is not None and len(adata) > max_n_samples:
            rng = np.random.default_rng(subsample_seed)
            idx = rng.choice(len(adata), max_n_samples, replace=False)
            idx.sort()
            adata = adata[idx].copy()
            encoded = encoded[idx]
            logger.info(f"[{split}] subsampled to {len(adata)} samples (n={max_n_samples})")

        self.X = adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray()
        self.labels = encoded
        self.n_cpg = self.X.shape[1]

        # CpG vocabulary (token IDs): offset by 5 to match tokenizer convention
        if cpg_vocab is not None:
            self.cpg_vocab = cpg_vocab
        else:
            try:
                self.cpg_vocab = np.array(adata.var["cpg_id"].values, dtype=np.int64)
            except (KeyError, AttributeError):
                self.cpg_vocab = np.arange(5, self.n_cpg + 5, dtype=np.int64)

        # Pre-compute fixed subset if requested
        self._fixed_indices = None
        if fixed_subset:
            rng = np.random.default_rng(fixed_subset_seed)
            self._fixed_indices = rng.choice(self.n_cpg, min(subset_k, self.n_cpg), replace=False)
            self._fixed_indices.sort()

        logger.info(
            f"[{split}] ClassificationDataset: {len(self)} samples, "
            f"n_cpg={self.n_cpg}, subset_k={subset_k}, labels={np.unique(encoded)}"
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        beta_row = self.X[idx]  # [n_cpg]

        # Sample subset of CpGs (skip NaN positions)
        valid_mask = ~np.isnan(beta_row)
        valid_indices = np.where(valid_mask)[0]

        if self._fixed_indices is not None:
            # Fixed: use pre-computed subset, filter out NaNs
            chosen = self._fixed_indices[valid_mask[self._fixed_indices]]
        else:
            k = min(self.subset_k, len(valid_indices))
            chosen = np.random.choice(valid_indices, k, replace=False)
            chosen.sort()

        cpg_ids = self.cpg_vocab[chosen].astype(np.float32)
        beta_values = beta_row[chosen].astype(np.float32)

        return {
            "cpg_ids": torch.from_numpy(cpg_ids),
            "beta_values": torch.from_numpy(beta_values),
            "class_label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def _collate_classification(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Pad sequences to the longest in the batch."""
    max_len = max(b["cpg_ids"].shape[0] for b in batch)

    cpg_ids = torch.zeros(len(batch), max_len, dtype=torch.float32)
    beta_values = torch.zeros(len(batch), max_len, dtype=torch.float32)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    class_labels = torch.stack([b["class_label"] for b in batch])

    for i, b in enumerate(batch):
        n = b["cpg_ids"].shape[0]
        cpg_ids[i, :n] = b["cpg_ids"]
        beta_values[i, :n] = b["beta_values"]
        attention_mask[i, :n] = 1

    return {
        "cpg_ids": cpg_ids,
        "beta_values": beta_values,
        "attention_mask": attention_mask,
        "class_label": class_labels,
    }


class ClassificationDataModule(pl.LightningDataModule):
    """
    LightningDataModule for classification fine-tuning.

    Args:
        h5ad_path: Path to h5ad file with methylation data + labels.
        label_col: obs column with class labels.
        label_map: Dict mapping string → int class index.
        n_classes: Number of output classes.
        split_col: obs column with train/valid/test assignments.
        subset_k: CpG sites per sample.
        batch_size: Training batch size.
        num_workers: DataLoader workers.
        max_n_train: Subsample training set to this many samples (None = all).
        subsample_seed: Seed for training subsampling.
    """

    def __init__(
        self,
        h5ad_path: str,
        label_col: str,
        label_map: Optional[Dict[str, int]],
        n_classes: int,
        split_col: str = "split",
        subset_k: int = 4000,
        batch_size: int = 32,
        num_workers: int = 4,
        max_n_train: Optional[int] = None,
        subsample_seed: int = 0,
    ):
        super().__init__()
        self.h5ad_path = h5ad_path
        self.label_col = label_col
        self.label_map = label_map
        self.n_classes = n_classes
        self.split_col = split_col
        self.subset_k = subset_k
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_n_train = max_n_train
        self.subsample_seed = subsample_seed

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        common = dict(
            h5ad_path=self.h5ad_path,
            label_col=self.label_col,
            label_map=self.label_map,
            split_col=self.split_col,
            subset_k=self.subset_k,
        )
        self.train_dataset = ClassificationDataset(
            split="train", fixed_subset=False,
            max_n_samples=self.max_n_train, subsample_seed=self.subsample_seed,
            **common,
        )
        self.val_dataset = ClassificationDataset(
            split="valid", fixed_subset=True, **common,
        )
        self.test_dataset = ClassificationDataset(
            split="test", fixed_subset=True, **common,
        )

    def _loader(self, dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=_collate_classification,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self):
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._loader(self.test_dataset, shuffle=False)
