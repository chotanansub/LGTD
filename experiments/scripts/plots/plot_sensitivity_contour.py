#!/usr/bin/env python3
"""
Generate contour plots for LGTD parameter sensitivity analysis.

This script creates contour plots showing how MAE/MSE varies with window_size
and percentile_error parameters for each synthetic dataset.

Usage:
    python plot_sensitivity_contour.py
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Configuration
SENSITIVITY_DIR = project_root / "experiments" / "results" / "sensitivity"
OUTPUT_DIR = project_root / "experiments" / "results" / "sensitivity" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data(dataset_name):
    """Load sensitivity test results for a dataset."""
    csv_path = SENSITIVITY_DIR / f"{dataset_name}_sensitivity.csv"
    return pd.read_csv(csv_path)


def create_contour_plot(dataset_name, metric='mae'):
    """Create a contour plot for a single dataset."""
    print(f"  Creating {dataset_name} ({metric.upper()})...")

    df = load_data(dataset_name)
    df_valid = df[df['valid'] == 1].copy()

    if len(df_valid) == 0:
        print(f"    ✗ No valid results, skipping...")
        return

    # Prepare data
    window_sizes = sorted(df_valid['window_size'].unique())
    percentile_errors = sorted(df_valid['percentile_error'].unique())
    X, Y = np.meshgrid(window_sizes, percentile_errors)
    Z = np.full((len(percentile_errors), len(window_sizes)), np.nan)

    for i, pe in enumerate(percentile_errors):
        for j, ws in enumerate(window_sizes):
            row = df_valid[(df_valid['window_size'] == ws) &
                          (df_valid['percentile_error'] == pe)]
            if len(row) > 0:
                Z[i, j] = row[metric].values[0]

    # Find best parameters
    best_idx = df_valid[metric].idxmin()
    best_ws = df_valid.loc[best_idx, 'window_size']
    best_pe = df_valid.loc[best_idx, 'percentile_error']
    best_val = df_valid.loc[best_idx, metric]

    # Create plot
    _, ax = plt.subplots(figsize=(10, 8))
    contour_filled = ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.9)
    contour_lines = ax.contour(X, Y, Z, levels=20, colors='black',
                               linewidths=0.5, alpha=0.3)
    ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.2f')

    # Mark data points and best
    ax.scatter(df_valid['window_size'], df_valid['percentile_error'],
              c='red', s=20, alpha=0.5, marker='o', edgecolors='darkred',
              linewidths=0.5, label='Data points')
    ax.scatter([best_ws], [best_pe], c='lime', s=200, marker='*',
              edgecolors='darkgreen', linewidths=2,
              label=f'Best: ws={best_ws:.0f}, pe={best_pe:.0f}', zorder=5)

    # Formatting
    cbar = plt.colorbar(contour_filled, ax=ax, label=metric.upper())
    cbar.ax.tick_params(labelsize=10)
    ax.set_xlabel('Window Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentile Error', fontsize=12, fontweight='bold')
    ax.set_title(f'{dataset_name.upper()}: {metric.upper()} vs Parameters\n'
                f'Best: ws={best_ws:.0f}, pe={best_pe:.0f}, '
                f'{metric.upper()}={best_val:.4f}',
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.set_xticks(window_sizes)
    ax.set_yticks(percentile_errors)

    plt.tight_layout()
    output_path = OUTPUT_DIR / f"{dataset_name}_{metric}_contour.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"    ✓ Saved to: {output_path}")


def create_combined_subplot(dataset_names, metric='mae'):
    """Create a subplot grid showing all datasets."""
    print(f"\n  Creating combined subplot ({metric.upper()})...")

    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    axes = axes.flatten()

    # Find global min/max for consistent color scale
    all_data = {}
    global_min, global_max = float('inf'), float('-inf')

    for dataset_name in dataset_names:
        try:
            df = load_data(dataset_name)
            df_valid = df[df['valid'] == 1]
            if len(df_valid) > 0:
                all_data[dataset_name] = df_valid
                global_min = min(global_min, df_valid[metric].min())
                global_max = max(global_max, df_valid[metric].max())
        except FileNotFoundError:
            print(f"    ⚠ Skipping {dataset_name}: file not found")

    plot_levels = np.linspace(global_min, global_max, 15)

    # Create plots
    for idx, dataset_name in enumerate(dataset_names):
        ax = axes[idx]

        if dataset_name not in all_data:
            ax.axis('off')
            continue

        df_valid = all_data[dataset_name]
        window_sizes = sorted(df_valid['window_size'].unique())
        percentile_errors = sorted(df_valid['percentile_error'].unique())
        X, Y = np.meshgrid(window_sizes, percentile_errors)
        Z = np.full((len(percentile_errors), len(window_sizes)), np.nan)

        for i, pe in enumerate(percentile_errors):
            for j, ws in enumerate(window_sizes):
                row = df_valid[(df_valid['window_size'] == ws) &
                              (df_valid['percentile_error'] == pe)]
                if len(row) > 0:
                    Z[i, j] = row[metric].values[0]

        # Create contour
        contour_filled = ax.contourf(X, Y, Z, levels=plot_levels,
                                    cmap='viridis', alpha=0.9,
                                    vmin=global_min, vmax=global_max)
        ax.contour(X, Y, Z, levels=plot_levels, colors='black',
                  linewidths=0.3, alpha=0.2)

        # Find and mark best
        best_idx = df_valid[metric].idxmin()
        best_ws = df_valid.loc[best_idx, 'window_size']
        best_pe = df_valid.loc[best_idx, 'percentile_error']
        best_val = df_valid.loc[best_idx, metric]

        ax.scatter([best_ws], [best_pe], c='lime', s=150, marker='*',
                  edgecolors='darkgreen', linewidths=1.5, zorder=5)

        # Formatting
        ax.set_title(f'{dataset_name}\nBest: ws={best_ws:.0f}, pe={best_pe:.0f}, '
                    f'{metric.upper()}={best_val:.3f}',
                    fontsize=10, fontweight='bold')
        if idx % 3 == 0:
            ax.set_ylabel('Percentile Error', fontsize=9)
        if idx >= 6:
            ax.set_xlabel('Window Size', fontsize=9)
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
        ax.set_xticks(window_sizes[::2])
        ax.set_yticks(percentile_errors[::2])
        ax.tick_params(labelsize=8)

    # Remove unused subplots
    for idx in range(len(dataset_names), len(axes)):
        axes[idx].axis('off')

    # Add shared colorbar
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(contour_filled, cax=cbar_ax, label=metric.upper())
    cbar.ax.tick_params(labelsize=10)

    fig.suptitle(f'LGTD Parameter Sensitivity Analysis: {metric.upper()} Across Datasets',
                fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 0.92, 0.96])
    output_path = OUTPUT_DIR / f"all_datasets_{metric}_contour.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"    ✓ Saved combined plot to: {output_path}")


def main():
    """Generate all contour plots."""
    print("=" * 70)
    print("Generating LGTD Parameter Sensitivity Contour Plots")
    print("=" * 70)

    # Find all datasets
    csv_files = list(SENSITIVITY_DIR.glob("*_sensitivity.csv"))
    dataset_names = sorted([f.stem.replace('_sensitivity', '') for f in csv_files])

    print(f"Datasets: {', '.join(dataset_names)}")
    print(f"Output directory: {OUTPUT_DIR}\n")

    # Generate all plots
    for metric in ['mae', 'mse']:
        print(f"\n{metric.upper()} Plots:")
        print("-" * 70)

        # Individual plots
        for dataset_name in dataset_names:
            try:
                create_contour_plot(dataset_name, metric)
            except Exception as e:
                print(f"    ✗ Error: {dataset_name}: {str(e)}")

        # Combined plot
        if len(dataset_names) > 1:
            try:
                create_combined_subplot(dataset_names, metric)
            except Exception as e:
                print(f"    ✗ Error creating combined plot: {str(e)}")

    print("\n" + "=" * 70)
    print("ALL PLOTS COMPLETED")
    print("=" * 70)
    print(f"Plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
