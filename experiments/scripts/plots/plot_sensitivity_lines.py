#!/usr/bin/env python3
"""
Generate line plots for LGTD parameter sensitivity analysis.

Creates two types of plots:
1. Window Size Sensitivity: Fixed percentile_error=50, varying window_size
2. Percentile Error Sensitivity: Fixed window_size=5, varying percentile_error
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Set academic style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 300
})

# Configuration
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

SENSITIVITY_DIR = project_root / "experiments" / "results" / "sensitivity"
WINDOW_OUTPUT_DIR = project_root / "experiments" / "results" / "sensitivity" / "window_size_plot"
PERCENTILE_OUTPUT_DIR = project_root / "experiments" / "results" / "sensitivity" / "percentile_error_plot"

WINDOW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PERCENTILE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Fixed parameters
FIXED_PERCENTILE = 50
FIXED_WINDOW_SIZE = 5


def load_data(dataset_name):
    """Load sensitivity test results for a dataset."""
    csv_path = SENSITIVITY_DIR / f"{dataset_name}_sensitivity.csv"
    return pd.read_csv(csv_path)


def plot_window_size_sensitivity(dataset_name, metric='mae'):
    """Plot MAE vs window_size with fixed percentile_error=50."""
    print(f"  Window size plot: {dataset_name} ({metric.upper()})...")

    df = load_data(dataset_name)
    df_valid = df[df['valid'] == 1].copy()

    # Filter for fixed percentile_error
    df_fixed = df_valid[df_valid['percentile_error'] == FIXED_PERCENTILE].sort_values('window_size')

    if len(df_fixed) == 0:
        print(f"    ✗ No data for percentile_error={FIXED_PERCENTILE}")
        return

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 5))

    # Line plot with markers
    ax.plot(df_fixed['window_size'], df_fixed[metric],
            marker='o', linewidth=2, markersize=6,
            color='steelblue', label=f'Percentile Error = {FIXED_PERCENTILE}')

    # Mark minimum
    min_idx = df_fixed[metric].idxmin()
    min_ws = df_fixed.loc[min_idx, 'window_size']
    min_val = df_fixed.loc[min_idx, metric]

    ax.scatter([min_ws], [min_val], c='red', s=150, marker='*',
              edgecolors='darkred', linewidths=2, zorder=5,
              label=f'Optimal: ws={min_ws:.0f}, {metric.upper()}={min_val:.4f}')

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # Labels
    ax.set_xlabel('Window Size ($W$)', fontweight='bold')
    ax.set_ylabel(f'{metric.upper()}', fontweight='bold')
    ax.set_title(f'{dataset_name.upper()}: {metric.upper()} vs Window Size\n'
                f'(Fixed Percentile Error = {FIXED_PERCENTILE})',
                fontweight='bold', pad=15)

    # Set x-axis ticks to integers
    ax.set_xticks(df_fixed['window_size'].values)

    # Legend
    ax.legend(loc='best', framealpha=0.9)

    plt.tight_layout()
    output_path = WINDOW_OUTPUT_DIR / f"{dataset_name}_{metric}_window_size.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"    ✓ Saved to: {output_path}")


def plot_percentile_sensitivity(dataset_name, metric='mae'):
    """Plot MAE vs percentile_error with fixed window_size=5."""
    print(f"  Percentile plot: {dataset_name} ({metric.upper()})...")

    df = load_data(dataset_name)
    df_valid = df[df['valid'] == 1].copy()

    # Filter for fixed window_size
    df_fixed = df_valid[df_valid['window_size'] == FIXED_WINDOW_SIZE].sort_values('percentile_error')

    if len(df_fixed) == 0:
        print(f"    ✗ No data for window_size={FIXED_WINDOW_SIZE}")
        return

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 5))

    # Line plot with markers
    ax.plot(df_fixed['percentile_error'], df_fixed[metric],
            marker='s', linewidth=2, markersize=6,
            color='forestgreen', label=f'Window Size = {FIXED_WINDOW_SIZE}')

    # Mark minimum
    min_idx = df_fixed[metric].idxmin()
    min_pe = df_fixed.loc[min_idx, 'percentile_error']
    min_val = df_fixed.loc[min_idx, metric]

    ax.scatter([min_pe], [min_val], c='red', s=150, marker='*',
              edgecolors='darkred', linewidths=2, zorder=5,
              label=f'Optimal: pe={min_pe:.0f}, {metric.upper()}={min_val:.4f}')

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # Labels
    ax.set_xlabel('Percentile Error ($\\epsilon$)', fontweight='bold')
    ax.set_ylabel(f'{metric.upper()}', fontweight='bold')
    ax.set_title(f'{dataset_name.upper()}: {metric.upper()} vs Percentile Error\n'
                f'(Fixed Window Size = {FIXED_WINDOW_SIZE})',
                fontweight='bold', pad=15)

    # Set x-axis ticks
    ax.set_xticks(df_fixed['percentile_error'].values)

    # Legend
    ax.legend(loc='best', framealpha=0.9)

    plt.tight_layout()
    output_path = PERCENTILE_OUTPUT_DIR / f"{dataset_name}_{metric}_percentile_error.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"    ✓ Saved to: {output_path}")


def plot_combined_window_size(dataset_names, metric='mae'):
    """Combined plot for all datasets: MAE vs window_size."""
    print(f"\n  Combined window size plot ({metric.upper()})...")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(dataset_names)))

    for idx, dataset_name in enumerate(dataset_names):
        try:
            df = load_data(dataset_name)
            df_valid = df[df['valid'] == 1]
            df_fixed = df_valid[df_valid['percentile_error'] == FIXED_PERCENTILE].sort_values('window_size')

            if len(df_fixed) > 0:
                ax.plot(df_fixed['window_size'], df_fixed[metric],
                       marker='o', linewidth=1.5, markersize=4,
                       color=colors[idx], label=dataset_name, alpha=0.8)
        except Exception as e:
            print(f"    ⚠ Skipping {dataset_name}: {str(e)}")

    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_xlabel('Window Size ($W$)', fontweight='bold')
    ax.set_ylabel(f'{metric.upper()}', fontweight='bold')
    ax.set_title(f'Window Size Sensitivity Across Datasets\n'
                f'(Fixed Percentile Error = {FIXED_PERCENTILE})',
                fontweight='bold', pad=15)
    ax.legend(loc='best', ncol=2, framealpha=0.9, fontsize=8)

    plt.tight_layout()
    output_path = WINDOW_OUTPUT_DIR / f"combined_{metric}_window_size.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"    ✓ Saved to: {output_path}")


def plot_combined_percentile(dataset_names, metric='mae'):
    """Combined plot for all datasets: MAE vs percentile_error."""
    print(f"\n  Combined percentile plot ({metric.upper()})...")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(dataset_names)))

    for idx, dataset_name in enumerate(dataset_names):
        try:
            df = load_data(dataset_name)
            df_valid = df[df['valid'] == 1]
            df_fixed = df_valid[df_valid['window_size'] == FIXED_WINDOW_SIZE].sort_values('percentile_error')

            if len(df_fixed) > 0:
                ax.plot(df_fixed['percentile_error'], df_fixed[metric],
                       marker='s', linewidth=1.5, markersize=4,
                       color=colors[idx], label=dataset_name, alpha=0.8)
        except Exception as e:
            print(f"    ⚠ Skipping {dataset_name}: {str(e)}")

    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_xlabel('Percentile Error ($\\epsilon$)', fontweight='bold')
    ax.set_ylabel(f'{metric.upper()}', fontweight='bold')
    ax.set_title(f'Percentile Error Sensitivity Across Datasets\n'
                f'(Fixed Window Size = {FIXED_WINDOW_SIZE})',
                fontweight='bold', pad=15)
    ax.legend(loc='best', ncol=2, framealpha=0.9, fontsize=8)

    plt.tight_layout()
    output_path = PERCENTILE_OUTPUT_DIR / f"combined_{metric}_percentile_error.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"    ✓ Saved to: {output_path}")


def main():
    """Generate all line plots."""
    print("=" * 70)
    print("Generating LGTD Parameter Sensitivity Line Plots")
    print("=" * 70)

    # Find all datasets
    csv_files = list(SENSITIVITY_DIR.glob("*_sensitivity.csv"))
    dataset_names = sorted([f.stem.replace('_sensitivity', '') for f in csv_files])

    print(f"Datasets: {', '.join(dataset_names)}")
    print(f"Fixed percentile error: {FIXED_PERCENTILE}")
    print(f"Fixed window size: {FIXED_WINDOW_SIZE}\n")

    # Generate all plots
    for metric in ['mae', 'mse']:
        print(f"\n{metric.upper()} Plots:")
        print("-" * 70)

        # Individual plots for window size
        print("\nWindow Size Sensitivity:")
        for dataset_name in dataset_names:
            try:
                plot_window_size_sensitivity(dataset_name, metric)
            except Exception as e:
                print(f"    ✗ Error: {dataset_name}: {str(e)}")

        # Individual plots for percentile error
        print("\nPercentile Error Sensitivity:")
        for dataset_name in dataset_names:
            try:
                plot_percentile_sensitivity(dataset_name, metric)
            except Exception as e:
                print(f"    ✗ Error: {dataset_name}: {str(e)}")

        # Combined plots
        if len(dataset_names) > 1:
            try:
                plot_combined_window_size(dataset_names, metric)
                plot_combined_percentile(dataset_names, metric)
            except Exception as e:
                print(f"    ✗ Error creating combined plots: {str(e)}")

    print("\n" + "=" * 70)
    print("ALL LINE PLOTS COMPLETED")
    print("=" * 70)
    print(f"Window size plots saved to: {WINDOW_OUTPUT_DIR}")
    print(f"Percentile error plots saved to: {PERCENTILE_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
