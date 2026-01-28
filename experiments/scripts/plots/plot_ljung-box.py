#!/usr/bin/env python3
"""
Diagnostic Figure for Ljung-Box Test Results (Multi-Lag)

Generates publication-quality plot showing:
- Residual autocorrelation (Ljung-Box Q-statistic for lags 10, 20, 30)

Output: ljung_box.png
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from matplotlib.patches import Patch

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

}

DEFAULT_MODELS = ['LGTD', 'STL', 'OnlineSTL', 'ASTD_Online', 'FastRobustSTL', 'OneShotSTL', 'STR']

def plot_combined_horizontal_barplot(lb_path, output_dir, models=None, suffix=''):
    """
    Generate horizontal barplot for Ljung-Box statistical test results.
    Refined for minimal whitespace and high density.
    """
    df_lb = pd.read_csv(lb_path)

    if models is None:
        models = DEFAULT_MODELS

    available_models = set(df_lb['model'].unique())
    valid_models = [m for m in models if m in available_models]

    if not valid_models:
        return

    df_lb = df_lb[df_lb['model'].isin(valid_models)]
    datasets = sorted(df_lb['dataset'].unique())
    num_ds = len(datasets)

    model_order = [m for m in valid_models if m in df_lb['model'].values]
    plot_order = model_order[::-1]
    y_pos = np.arange(len(plot_order))

    lags = ['lag_10_statistic', 'lag_20_statistic', 'lag_30_statistic']
    lag_labels = [r'$Q_{10}$', r'$Q_{20}$', r'$Q_{30}$']
    
    # SLIGHTLY increased bar_height to reduce gap between model groups
    bar_height = 0.26  
    offsets = [bar_height, 0, -bar_height]
    alphas = [1.0, 0.65, 0.35]

    fig, axes = plt.subplots(
        num_ds, 1,
        figsize=(3.0, 1.1 * num_ds), # Reduced width and height per subplot
        squeeze=False
    )

    for i, ds in enumerate(datasets):
        ax_lb = axes[i, 0]
        ds_lb = df_lb[df_lb['dataset'] == ds].set_index('model').reindex(plot_order)

        for j, (lag_col, offset, alpha) in enumerate(zip(lags, offsets, alphas)):
            colors = [MODEL_COLORS.get(m, '#000000') for m in plot_order]
            ax_lb.barh(
                y_pos + offset,
                ds_lb[lag_col],
                height=bar_height,
                color=colors,
                alpha=alpha,
                edgecolor='black',
                linewidth=0.3 # Thinner lines for a cleaner look
            )
        
        ax_lb.set_xscale('log')
        ax_lb.set_yticks(y_pos)
        ax_lb.set_yticklabels(plot_order, fontsize=6)
        
        # MINIMAL labelpad to bring Dataset names (ETTh1) right next to model names
        ax_lb.set_ylabel(ds, fontweight='bold', rotation=0, labelpad=22, va='center')

        ax_lb.grid(False)
        
        # AGGRESSIVE pad reduction (0.2) for the model labels
        ax_lb.tick_params(axis='both', which='both', length=1.5, pad=0.2, width=0.5)
        
        ax_lb.spines['top'].set_visible(False)
        ax_lb.spines['right'].set_visible(False)

        if i == num_ds - 1:
            ax_lb.set_xlabel(r'Ljung-Box $Q$ Statistic (Log Scale)', labelpad=1)

    legend_elements = [
        Patch(facecolor='gray', alpha=alphas[0], edgecolor='black', label=lag_labels[0]),
        Patch(facecolor='gray', alpha=alphas[1], edgecolor='black', label=lag_labels[1]),
        Patch(facecolor='gray', alpha=alphas[2], edgecolor='black', label=lag_labels[2])
    ]
    
    # Adjusting legend position to sit tighter to the top plot
    axes[0, 0].legend(
        handles=legend_elements, 
        loc='lower left', 
        bbox_to_anchor=(0, 1.02),
        ncol=3, 
        fontsize=6, 
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.0
    )

    # Tightened hspace and left margin
    plt.subplots_adjust(
        hspace=0.45, 
        left=0.18,
        right=0.98,
        top=0.92,
        bottom=0.10
    )
    
    out_file = Path(output_dir) / f"ljung_box{suffix}.png"
    # Using bbox_inches='tight' with a small pad to clip remaining empty space
    plt.savefig(out_file, bbox_inches='tight', pad_inches=0.02, dpi=300, facecolor='white')
    print(f"✓ PNG saved to: {out_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate Ljung-Box diagnostic plots')
    parser.add_argument('--models', nargs='+', default=None)
    parser.add_argument('--input-dir', type=str, default='experiments/results/stat_tests/real_world')
    parser.add_argument('--output-dir', type=str, default=None)

    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    lb_path = input_dir / 'residual_ljung_box.csv'

    if not lb_path.exists():
        print(f"Error: {lb_path} not found")
        sys.exit(1)

    models = args.models if args.models else DEFAULT_MODELS
    
    plot_combined_horizontal_barplot(
        lb_path,
        output_dir,
        models=models,
        suffix=''
    )

if __name__ == "__main__":
    main()