#!/usr/bin/env python
"""
Analyze experiment results.

Usage:
    python experiments/scripts/analyze_results.py experiments/results/synthetic/experiment_results_*.csv
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def analyze_results(results_file: str):
    """Analyze and visualize experiment results."""

    # Load results
    df = pd.read_csv(results_file)

    print("="*80)
    print(f"EXPERIMENT RESULTS ANALYSIS")
    print(f"File: {results_file}")
    print("="*80)

    # Basic stats
    print(f"\nTotal experiments: {len(df)}")
    print(f"Datasets: {df['dataset'].nunique()} ({', '.join(df['dataset'].unique())})")
    print(f"Models: {df['model'].nunique()} ({', '.join(df['model'].unique())})")

    # MSE by model
    print("\n" + "="*80)
    print("MSE (Trend) by Model")
    print("="*80)
    mse_by_model = df.groupby('model')['mse_trend'].agg(['mean', 'std', 'min', 'max'])
    print(mse_by_model.to_string())

    # MSE by dataset
    print("\n" + "="*80)
    print("MSE (Trend) by Dataset")
    print("="*80)
    mse_by_dataset = df.groupby('dataset')['mse_trend'].agg(['mean', 'std', 'min', 'max'])
    print(mse_by_dataset.to_string())

    # LGTD auto-selection analysis
    if 'selected_method' in df.columns:
        lgtd_results = df[df['model'] == 'LGTD'].copy()
        if not lgtd_results.empty:
            print("\n" + "="*80)
            print("LGTD Auto-Selection Analysis")
            print("="*80)
            selection_counts = lgtd_results['selected_method'].value_counts()
            print(f"\nMethod Selection:")
            print(selection_counts.to_string())

            print(f"\nMSE by Selected Method:")
            mse_by_method = lgtd_results.groupby('selected_method')['mse_trend'].agg(['mean', 'std'])
            print(mse_by_method.to_string())

    # Execution time analysis
    print("\n" + "="*80)
    print("Execution Time (seconds)")
    print("="*80)
    time_by_model = df.groupby('model')['time'].agg(['mean', 'std', 'min', 'max'])
    print(time_by_model.to_string())

    # Comparison matrix
    if len(df['model'].unique()) > 1 and len(df['dataset'].unique()) > 1:
        print("\n" + "="*80)
        print("MSE (Trend) Comparison Matrix")
        print("="*80)
        comparison = df.pivot_table(
            index='dataset',
            columns='model',
            values='mse_trend',
            aggfunc='mean'
        )
        print(comparison.to_string())

        # Find best model for each dataset
        print("\n" + "="*80)
        print("Best Model per Dataset")
        print("="*80)
        for dataset in comparison.index:
            best_model = comparison.loc[dataset].idxmin()
            best_mse = comparison.loc[dataset].min()
            print(f"{dataset}: {best_model} (MSE={best_mse:.4f})")

    # Visualizations
    print("\n" + "="*80)
    print("Generating Visualizations...")
    print("="*80)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: MSE by Model
    df.groupby('model')['mse_trend'].mean().sort_values().plot(
        kind='barh', ax=axes[0, 0], color='steelblue'
    )
    axes[0, 0].set_title('Average MSE (Trend) by Model')
    axes[0, 0].set_xlabel('MSE')

    # Plot 2: MSE by Dataset
    df.groupby('dataset')['mse_trend'].mean().sort_values().plot(
        kind='barh', ax=axes[0, 1], color='coral'
    )
    axes[0, 1].set_title('Average MSE (Trend) by Dataset')
    axes[0, 1].set_xlabel('MSE')

    # Plot 3: Execution Time by Model
    df.groupby('model')['time'].mean().sort_values().plot(
        kind='barh', ax=axes[1, 0], color='green'
    )
    axes[1, 0].set_title('Average Execution Time by Model')
    axes[1, 0].set_xlabel('Time (seconds)')

    # Plot 4: LGTD Selection (if available)
    if 'selected_method' in df.columns and 'LGTD' in df['model'].values:
        lgtd_results = df[df['model'] == 'LGTD']
        selection_counts = lgtd_results['selected_method'].value_counts()
        selection_counts.plot(kind='pie', ax=axes[1, 1], autopct='%1.1f%%')
        axes[1, 1].set_title('LGTD Auto-Selection Distribution')
        axes[1, 1].set_ylabel('')
    else:
        axes[1, 1].text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=20)
        axes[1, 1].set_title('LGTD Auto-Selection')
        axes[1, 1].axis('off')

    plt.tight_layout()

    # Save plot
    output_path = Path(results_file).parent / f"analysis_{Path(results_file).stem}.png"
    plt.savefig(output_path, dpi=150)
    print(f"Visualization saved to: {output_path}")

    plt.show()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <results_file.csv>")
        print("\nExample:")
        print("  python experiments/scripts/analyze_results.py experiments/results/synthetic/experiment_results_*.csv")
        sys.exit(1)

    results_file = sys.argv[1]
    analyze_results(results_file)
