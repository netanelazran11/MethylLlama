"""
Dataset classes for methylation h5ad data
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict

# Try to import scanpy for h5ad support
try:
    import scanpy as sc
    HAS_SCANPY = True
except ImportError:
    HAS_SCANPY = False


class MethylationDataset(Dataset):
    """
    PyTorch Dataset for methylation data in h5ad format.

    Expected h5ad structure:
        - adata.X: [n_samples, n_cpg_sites] - beta values
        - adata.obs["age"]: target ages
        - adata.obs["split"]: (optional) split labels for filtering
        - adata.var_names: CpG site IDs (optional)

    Example:
        >>> adata = sc.read_h5ad("methylation.h5ad")
        >>> dataset = MethylationDataset(adata, max_cpg=8000)
        >>> loader = DataLoader(dataset, batch_size=32)

        # With split filtering:
        >>> train_dataset = MethylationDataset(adata, split="train")
        >>> val_dataset = MethylationDataset(adata, split="valid")
    """

    def __init__(
        self,
        adata_or_path,
        age_column: str = "age",
        max_cpg: Optional[int] = None,
        normalize_age: bool = False,
        age_mean: Optional[float] = None,
        age_std: Optional[float] = None,
        split: Optional[str] = None,
        split_column: str = "split"
    ):
        """
        Args:
            adata_or_path: AnnData object or path to h5ad file
            age_column: column name in adata.obs containing ages
            max_cpg: maximum number of CpG sites (truncate if needed)
            normalize_age: whether to z-score normalize ages
            age_mean: mean for normalization (computed if None)
            age_std: std for normalization (computed if None)
            split: filter to specific split ("train", "valid", "test", etc.)
            split_column: column name in adata.obs containing split labels
        """
        if not HAS_SCANPY:
            raise ImportError("scanpy is required. Install with: pip install scanpy")

        # Load data
        if isinstance(adata_or_path, str):
            adata = sc.read_h5ad(adata_or_path)
        else:
            adata = adata_or_path

        # Filter by split if specified
        if split is not None:
            if split_column not in adata.obs.columns:
                raise ValueError(f"Split column '{split_column}' not found in adata.obs. "
                               f"Available columns: {list(adata.obs.columns)}")
            mask = adata.obs[split_column] == split
            if mask.sum() == 0:
                available_splits = adata.obs[split_column].unique().tolist()
                raise ValueError(f"No samples found for split='{split}'. "
                               f"Available splits: {available_splits}")
            self.adata = adata[mask].copy()
            print(f"Filtered to split='{split}': {mask.sum()} samples")
        else:
            self.adata = adata

        # Get dimensions
        n_samples, n_cpg = self.adata.X.shape
        self.max_cpg = min(max_cpg, n_cpg) if max_cpg else n_cpg

        # Extract beta values
        X = self.adata.X[:, :self.max_cpg]
        if hasattr(X, 'toarray'):  # Handle sparse matrices
            X = X.toarray()
        self.beta_values = torch.tensor(X, dtype=torch.float32)

        # Extract ages
        self.ages = torch.tensor(
            self.adata.obs[age_column].values, dtype=torch.float32
        )

        # Normalize ages if requested
        self.normalize_age = normalize_age
        if normalize_age:
            self.age_mean = age_mean if age_mean is not None else self.ages.mean().item()
            self.age_std = age_std if age_std is not None else self.ages.std().item()
            self.ages_normalized = (self.ages - self.age_mean) / self.age_std
        else:
            self.age_mean = None
            self.age_std = None
            self.ages_normalized = self.ages

        # CpG token IDs (simple indices)
        self.cpg_ids = torch.arange(self.max_cpg, dtype=torch.long)

        print(f"Loaded {n_samples} samples with {self.max_cpg} CpG sites")
        print(f"Age range: {self.ages.min():.1f} - {self.ages.max():.1f} years")

    def __len__(self) -> int:
        return len(self.ages)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        beta_values = self.beta_values[idx]
        valid_mask = torch.isfinite(beta_values)
        return {
            "beta_values": beta_values,
            "valid_mask": valid_mask,
            "age": self.ages[idx],
            "age_normalized": self.ages_normalized[idx],
        }

    def denormalize_age(self, normalized_age: torch.Tensor) -> torch.Tensor:
        """Convert normalized age back to original scale."""
        if self.normalize_age:
            return normalized_age * self.age_std + self.age_mean
        return normalized_age


def create_data_loaders(
    train_adata,
    val_adata,
    test_adata=None,
    batch_size: int = 32,
    max_cpg: int = 8000,
    num_workers: int = 4,
    normalize_age: bool = True
) -> Dict[str, DataLoader]:
    """
    Create data loaders for training.

    Args:
        train_adata: Training AnnData or path
        val_adata: Validation AnnData or path
        test_adata: Optional test AnnData or path
        batch_size: Batch size
        max_cpg: Maximum CpG sites
        num_workers: DataLoader workers
        normalize_age: Whether to normalize ages

    Returns:
        Dictionary with 'train', 'val', and optionally 'test' DataLoaders
    """
    train_dataset = MethylationDataset(
        train_adata,
        max_cpg=max_cpg,
        normalize_age=normalize_age
    )

    val_dataset = MethylationDataset(
        val_adata,
        max_cpg=max_cpg,
        normalize_age=normalize_age,
        age_mean=train_dataset.age_mean,
        age_std=train_dataset.age_std
    )

    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    }

    if test_adata is not None:
        test_dataset = MethylationDataset(
            test_adata,
            max_cpg=max_cpg,
            normalize_age=normalize_age,
            age_mean=train_dataset.age_mean,
            age_std=train_dataset.age_std
        )
        loaders["test"] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

    return loaders
