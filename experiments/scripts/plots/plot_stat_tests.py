#!/usr/bin/env python3
"""
Compact Diagnostic Figure for Statistical Test Results

Generates publication-quality plot showing:
- Seasonality magnitude (Kruskal-Wallis η²)
- Residual autocorrelation (Ljung-Box Q-statistic)

Output: polished_diagnostics.png
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Peer-Review Style Configuration
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'] + plt.rcParams['font.serif'],
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'axes.linewidth': 0.6,
    'lines.linewidth': 0.5,
    'figure.dpi': 300
})

MODEL_COLORS = {
    'LGTD': '#e41a1c',
    'STL': '#377eb8',
    'OnlineSTL': '#4daf4a',
    'ASTD': '#984ea3',
    'ASTD_Online': '#ff7f00',
    'STR': '#a65628',
    'RobustSTL': '#f781bf',
    'FastRobustSTL': '#999999',
    'OneShotSTL': '#66c2a5',
    'LGTD_LOWESS': '#fc8d62',
    'LGTD_Linear': '#8da0cb'
}

# Default model selection for standard comparison
DEFAULT_MODELS = ['LGTD', 'STL', 'OnlineSTL', 'ASTD_Online', 'FastRobustSTL', 'OneShotSTL', 'STR']

def generate_polished_compact_plot(lb_path, kw_path, output_dir, models=None, suffix=''):
    """
    Generate compact diagnostic plot for statistical test results.

    Args:
        lb_path: Path to Ljung-Box test results CSV
        kw_path: Path to Kruskal-Wallis test results CSV
        output_dir: Directory to save output plot
        models: List of models to include (None = use DEFAULT_MODELS)
        suffix: Suffix to add to output filename (e.g., '_full' for all models)
    """
    # Load and filter
    df_lb = pd.read_csv(lb_path)
    df_kw = pd.read_csv(kw_path)

    # Determine which models to include
    if models is None:
        models = DEFAULT_MODELS

    # Filter to specified models that exist in the data
    available_models = set(df_lb['model'].unique()) & set(df_kw['model'].unique())
    valid_models = [m for m in models if m in available_models]

    if not valid_models:
        print(f"Warning: No valid models found. Available: {sorted(available_models)}")
        return

    df_lb = df_lb[df_lb['model'].isin(valid_models)]
    df_kw = df_kw[df_kw['model'].isin(valid_models)]
    
    datasets = sorted(df_lb['dataset'].unique())
    num_ds = len(datasets)

    # Preserve order of valid_models for plotting
    model_order = [m for m in valid_models if m in df_lb['model'].values]
    plot_order = model_order[::-1]  # bottom-up for barh
    y_pos = np.arange(len(plot_order))
    colors = [MODEL_COLORS.get(m, '#000000') for m in plot_order]

    print(f"Generating plot with {len(plot_order)} models: {', '.join(plot_order[::-1])}")
    print(f"Datasets: {', '.join(datasets)}")

    # Figure layout
    fig, axes = plt.subplots(
        num_ds, 2,
        figsize=(3.3, 1.1 * num_ds),
        squeeze=False
    )

    for i, ds in enumerate(datasets):
        ax_kw, ax_lb = axes[i, 0], axes[i, 1]
        
        ds_lb = df_lb[df_lb['dataset'] == ds].set_index('model').reindex(plot_order)
        ds_kw = df_kw[df_kw['dataset'] == ds].set_index('model').reindex(plot_order)

        # Seasonality Magnitude (left)
        ax_kw.barh(
            y_pos,
            ds_kw['eta_squared'],
            color=colors,
            edgecolor='black',
            linewidth=0.5,
            height=0.7
        )
        ax_kw.set_xlim(0, 1.0)
        ax_kw.set_xticks([0, 0.5, 1.0])
        ax_kw.set_yticks(y_pos)
        ax_kw.set_yticklabels(plot_order, fontsize=6)
        ax_kw.set_ylabel(ds, fontweight='bold', rotation=0, labelpad=30, va='center')

        # Residual Autocorrelation (right)
        ax_lb.barh(
            y_pos,
            ds_lb['lag_10_statistic'],
            color=colors,
            edgecolor='black',
            linewidth=0.5,
            height=0.7
        )
        ax_lb.set_xscale('log')
        ax_lb.set_yticks(y_pos)
        ax_lb.set_yticklabels([])

        # Style cleanup
        for ax in (ax_kw, ax_lb):
            ax.grid(False)
            ax.tick_params(axis='both', which='both', length=2, pad=1, width=0.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        # Bottom-row x-labels only
        if i == num_ds - 1:
            ax_kw.set_xlabel(r'Kruskal-Wallis $\eta^2$', labelpad=2)
            ax_lb.set_xlabel(r'Ljung-Box $Q_{10}$', labelpad=2)

    # Figure-level column titles (stable with bbox_inches="tight")
    fig.text(
        0.33, 0.92,
        r'Seasonality Magnitude ($\eta^2$)',
        ha='center', va='bottom', fontsize=8
    )
    fig.text(
        0.77, 0.92,
        r'Residual Autocorrelation ($Q$)',
        ha='center', va='bottom', fontsize=8
    )

    # Spacing
    plt.subplots_adjust(
        wspace=0.15,
        hspace=0.5,
        left=0.3,
        right=0.96,
        top=0.88,
        bottom=0.12
    )
    
    out_file = Path(output_dir) / f"polished_diagnostics{suffix}.png"
    plt.savefig(out_file, bbox_inches='tight', dpi=300, facecolor='white')
    print(f"✓ PNG saved to: {out_file}")

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Generate statistical test diagnostic plots'
    )

    parser.add_argument(
        '--models',
        nargs='+',
        default=None,
        help='List of models to include in plot (default: standard selection)'
    )

    parser.add_argument(
        '--input-dir',
        type=str,
        default='experiments/results/stat_tests/real_world',
        help='Directory containing statistical test results'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save output plots (default: same as input-dir)'
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir

    lb_path = input_dir / 'residual_ljung_box.csv'
    kw_path = input_dir / 'seasonality_kruskal_wallis.csv'

    # Check if files exist
    if not lb_path.exists():
        print(f"Error: {lb_path} not found")
        sys.exit(1)
    if not kw_path.exists():
        print(f"Error: {kw_path} not found")
        sys.exit(1)

    print("=" * 70)
    print("Generating Statistical Test Diagnostic Plot")
    print("=" * 70)

    # Determine which models to use (always use default, only generate one plot)
    if args.models:
        models = args.models
        print(f"Mode: Custom selection ({len(models)} models)")
    else:
        models = DEFAULT_MODELS
        print(f"Mode: Default selection ({len(models)} models)")

    print(f"Models: {', '.join(models)}")
    print()

    # Always use empty suffix to generate polished_diagnostics.png
    generate_polished_compact_plot(
        lb_path,
        kw_path,
        output_dir,
        models=models,
        suffix=''
    )

    print()
    print("=" * 70)
    print("Plot Generation Complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
