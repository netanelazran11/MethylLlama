#!/usr/bin/env python3
"""
WCED Diagnostic Script - Simplified Version

Analyzes WHY CLS embeddings fail to encode sample-specific information
by directly analyzing the checkpoint without complex model reconstruction.

Diagnostics:
1. Decoder bias analysis: Does decoder just predict per-CpG means?
2. Prediction variance: Do predictions vary per sample?
3. Sample correlation: Are predictions correlated with actual values?

Usage:
    python scripts/diagnose_wced.py --checkpoint /path/to/checkpoint.ckpt --data /path/to/data.h5ad
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import scanpy as sc
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_decoder_weights(checkpoint_path: str):
    """Analyze the WCED decoder weights to see if it's learning per-CpG biases."""

    print("\n" + "="*70)
    print("DIAGNOSTIC 1: Decoder Weight Analysis")
    print("="*70)

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['state_dict']

    # Find decoder layers
    decoder_keys = [k for k in state_dict.keys() if 'decoder' in k]
    print(f"Decoder layers found: {decoder_keys}")

    # Get the final decoder layer (output bias)
    final_linear_bias = None
    final_linear_weight = None

    for key in decoder_keys:
        if 'bias' in key and 'decoder.decoder' in key:
            # Get the last bias layer
            final_linear_bias = state_dict[key]
            print(f"\n{key}: shape {final_linear_bias.shape}")
        if 'weight' in key and 'decoder.decoder' in key:
            final_linear_weight = state_dict[key]
            print(f"{key}: shape {final_linear_weight.shape}")

    if final_linear_bias is not None:
        bias = final_linear_bias.numpy()
        print(f"\nFinal layer bias statistics:")
        print(f"  Mean: {np.mean(bias):.6f}")
        print(f"  Std:  {np.std(bias):.6f}")
        print(f"  Min:  {np.min(bias):.6f}")
        print(f"  Max:  {np.max(bias):.6f}")

        # Check if bias values are in [0, 1] range (like beta values)
        in_range = np.sum((bias >= 0) & (bias <= 1)) / len(bias)
        print(f"  Fraction in [0,1]: {in_range:.4f}")

        # If bias is learning per-CpG means, they should vary a lot
        if np.std(bias) > 0.1:
            print("\n⚠️  Decoder bias has HIGH variance - may be learning per-CpG means!")
        else:
            print("\n✓ Decoder bias has low variance.")

    return state_dict


def analyze_data_statistics(data_path: str, vocab_size: int = 2048):
    """Analyze the per-CpG statistics in the training data."""

    print("\n" + "="*70)
    print("DIAGNOSTIC 2: Data Statistics (Per-CpG Means)")
    print("="*70)

    adata = sc.read_h5ad(data_path)

    # Get the beta values matrix
    X = adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()

    print(f"Data shape: {X.shape} (samples x CpGs)")

    # Compute per-CpG statistics
    cpg_means = np.nanmean(X, axis=0)
    cpg_stds = np.nanstd(X, axis=0)

    print(f"\nPer-CpG mean statistics:")
    print(f"  Mean of means: {np.nanmean(cpg_means):.6f}")
    print(f"  Std of means:  {np.nanstd(cpg_means):.6f}")
    print(f"  Min mean:      {np.nanmin(cpg_means):.6f}")
    print(f"  Max mean:      {np.nanmax(cpg_means):.6f}")

    print(f"\nPer-CpG std statistics:")
    print(f"  Mean std: {np.nanmean(cpg_stds):.6f}")
    print(f"  Min std:  {np.nanmin(cpg_stds):.6f}")
    print(f"  Max std:  {np.nanmax(cpg_stds):.6f}")

    # What PCC would we get if we just predicted per-CpG means?
    # Sample some test predictions
    n_samples = min(500, X.shape[0])
    sample_pccs = []

    for i in range(n_samples):
        sample = X[i, :vocab_size]
        pred_means = cpg_means[:vocab_size]

        valid_mask = ~np.isnan(sample) & ~np.isnan(pred_means)
        if valid_mask.sum() > 10:
            pcc, _ = pearsonr(sample[valid_mask], pred_means[valid_mask])
            sample_pccs.append(pcc)

    mean_pcc = np.mean(sample_pccs)
    print(f"\nBaseline PCC (predicting per-CpG means): {mean_pcc:.4f}")
    print(f"  This is what WCED achieves if it ignores sample-specific info!")

    if mean_pcc > 0.9:
        print("\n⚠️  HIGH baseline PCC!")
        print("   Per-CpG means alone explain >90% correlation.")
        print("   WCED can achieve high PCC by just learning biases.")

    return cpg_means, cpg_stds, adata


def analyze_decoder_vs_data(state_dict, cpg_means, vocab_size=2048):
    """Compare decoder bias with actual per-CpG means."""

    print("\n" + "="*70)
    print("DIAGNOSTIC 3: Decoder Bias vs Data Means")
    print("="*70)

    # Find the final decoder bias
    final_bias = None
    for key in state_dict.keys():
        if 'decoder.decoder.4.bias' in key:  # Last linear layer bias
            final_bias = state_dict[key].numpy()
            break

    if final_bias is None:
        print("Could not find decoder bias layer")
        return

    # Compare with per-CpG means
    n = min(len(final_bias), len(cpg_means), vocab_size)
    bias = final_bias[:n]
    means = cpg_means[:n]

    # Since there's a sigmoid, we need to convert bias to [0,1] range
    # Sigmoid(bias) should approximate means
    sigmoid_bias = 1 / (1 + np.exp(-bias))

    valid_mask = ~np.isnan(means)
    if valid_mask.sum() > 10:
        corr, p_val = pearsonr(sigmoid_bias[valid_mask], means[valid_mask])
        print(f"Correlation between sigmoid(decoder_bias) and per-CpG means: r={corr:.4f}")

        if corr > 0.8:
            print("\n⚠️  HIGH CORRELATION!")
            print("   The decoder bias has learned the per-CpG means.")
            print("   This confirms the model predicts averages, ignoring CLS.")
        elif corr > 0.5:
            print("\n⚠️  Moderate correlation with per-CpG means.")
        else:
            print("\n✓ Low correlation - decoder is not just predicting means.")


def analyze_sample_variance(data_path: str, vocab_size: int = 2048):
    """Analyze how much samples vary from per-CpG means."""

    print("\n" + "="*70)
    print("DIAGNOSTIC 4: Sample-Specific Variance")
    print("="*70)

    adata = sc.read_h5ad(data_path)
    X = adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()

    X = X[:, :vocab_size]

    # Per-CpG means
    cpg_means = np.nanmean(X, axis=0)

    # Per-sample deviation from means
    deviations = X - cpg_means

    # Variance explained by means vs sample-specific
    total_var = np.nanvar(X)
    mean_var = np.nanvar(cpg_means)
    residual_var = np.nanvar(deviations)

    print(f"Total variance in data: {total_var:.6f}")
    print(f"Variance from per-CpG means: {mean_var:.6f}")
    print(f"Residual (sample-specific) variance: {residual_var:.6f}")

    frac_explained = mean_var / total_var if total_var > 0 else 0
    print(f"\nFraction of variance explained by per-CpG means: {frac_explained:.4f}")

    if frac_explained > 0.8:
        print("\n⚠️  Per-CpG means explain >80% of variance!")
        print("   Sample-specific signal is weak relative to CpG patterns.")
        print("   This makes reconstruction from CLS very challenging.")

    # Check age correlation with residuals
    if 'labels' in adata.obs.columns or 'age' in adata.obs.columns:
        age_col = 'labels' if 'labels' in adata.obs.columns else 'age'
        ages = adata.obs[age_col].values.astype(float)

        # Correlate residual deviations with age
        # For each CpG, compute correlation of deviation with age
        cpg_age_corrs = []
        for j in range(min(vocab_size, X.shape[1])):
            dev = deviations[:, j]
            valid = ~np.isnan(dev)
            if valid.sum() > 10:
                corr, _ = pearsonr(dev[valid], ages[valid])
                cpg_age_corrs.append(abs(corr))

        mean_abs_corr = np.mean(cpg_age_corrs)
        max_corr = np.max(cpg_age_corrs)
        high_corr_count = sum(1 for c in cpg_age_corrs if c > 0.3)

        print(f"\nAge correlation with sample deviations:")
        print(f"  Mean |correlation|: {mean_abs_corr:.4f}")
        print(f"  Max |correlation|: {max_corr:.4f}")
        print(f"  CpGs with |r| > 0.3: {high_corr_count} / {len(cpg_age_corrs)}")

        if mean_abs_corr < 0.1:
            print("\n⚠️  Sample deviations have LOW age correlation.")
            print("   Age signal may be too weak for CLS to capture.")


def create_visualizations(cpg_means, state_dict, output_dir, vocab_size=2048):
    """Create diagnostic visualizations."""

    print("\n" + "="*70)
    print("Creating Visualizations...")
    print("="*70)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find decoder bias
    final_bias = None
    for key in state_dict.keys():
        if 'decoder.decoder.4.bias' in key:
            final_bias = state_dict[key].numpy()
            break

    if final_bias is None:
        print("Could not find decoder bias for visualization")
        return

    n = min(len(final_bias), len(cpg_means), vocab_size)
    sigmoid_bias = 1 / (1 + np.exp(-final_bias[:n]))
    means = cpg_means[:n]

    # Plot 1: Decoder bias vs per-CpG means
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    valid = ~np.isnan(means)
    axes[0].scatter(means[valid], sigmoid_bias[valid], alpha=0.3, s=5)
    axes[0].plot([0, 1], [0, 1], 'r--', label='y=x')
    axes[0].set_xlabel('Per-CpG Mean (from data)')
    axes[0].set_ylabel('Sigmoid(Decoder Bias)')
    axes[0].set_title('Decoder Bias vs Data Means')
    axes[0].legend()

    # Plot 2: Distribution of biases and means
    axes[1].hist(means[valid], bins=50, alpha=0.5, label='Data means', density=True)
    axes[1].hist(sigmoid_bias, bins=50, alpha=0.5, label='Sigmoid(bias)', density=True)
    axes[1].set_xlabel('Value')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Distribution Comparison')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'decoder_analysis.png', dpi=150)
    print(f"Saved: {output_dir / 'decoder_analysis.png'}")


def main():
    parser = argparse.ArgumentParser(description='Diagnose WCED - Simplified Version')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to WCED checkpoint')
    parser.add_argument('--data', type=str, required=True, help='Path to h5ad data file')
    parser.add_argument('--output', type=str, default='./wced_diagnosis', help='Output directory')
    parser.add_argument('--vocab_size', type=int, default=2048, help='Vocabulary size')
    parser.add_argument('--max_samples', type=int, default=500, help='Max samples to analyze')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')

    args = parser.parse_args()

    print("="*70)
    print("WCED DIAGNOSTIC ANALYSIS (Simplified)")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data: {args.data}")
    print(f"Vocab size: {args.vocab_size}")
    print("="*70)

    # Diagnostic 1: Analyze decoder weights
    state_dict = analyze_decoder_weights(args.checkpoint)

    # Diagnostic 2: Analyze data statistics
    cpg_means, cpg_stds, adata = analyze_data_statistics(args.data, args.vocab_size)

    # Diagnostic 3: Compare decoder with data
    analyze_decoder_vs_data(state_dict, cpg_means, args.vocab_size)

    # Diagnostic 4: Analyze sample variance
    analyze_sample_variance(args.data, args.vocab_size)

    # Create visualizations
    create_visualizations(cpg_means, state_dict, args.output, args.vocab_size)

    print("\n" + "="*70)
    print("DIAGNOSIS SUMMARY")
    print("="*70)
    print("""
Key Questions:
1. Has the decoder learned per-CpG means as biases?
   → If sigmoid(bias) correlates highly with data means, YES.

2. How much variance is sample-specific vs CpG-specific?
   → If CpG means explain >80% variance, sample signal is weak.

3. Is the baseline (predicting means) already high PCC?
   → If baseline PCC > 0.9, model can cheat by ignoring CLS.

If all three are problematic, WCED fundamentally struggles because:
- CpG-specific patterns dominate (high bias variance)
- Sample-specific signal is weak (low residual variance)
- Model achieves high PCC without learning sample info

Potential fixes:
1. normalize_loss=true: Forces model to predict relative patterns
2. Larger hidden_size: More capacity to encode samples
3. Different architecture: Skip-connections from encoder to decoder
4. Multi-task: Add auxiliary objectives (e.g., age prediction)
""")

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
