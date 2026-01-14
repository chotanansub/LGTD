#!/usr/bin/env python3
"""
Compact Diagnostic Figure (Stable Titles, No Collapse)
"""

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
    'ASTD_Online': '#ff7f00'
}

def generate_polished_compact_plot(lb_path, kw_path, output_dir):
    # Load and filter
    df_lb = pd.read_csv(lb_path)
    df_kw = pd.read_csv(kw_path)
    valid_models = list(MODEL_COLORS.keys())
    df_lb = df_lb[df_lb['model'].isin(valid_models)]
    df_kw = df_kw[df_kw['model'].isin(valid_models)]
    
    datasets = sorted(df_lb['dataset'].unique())
    num_ds = len(datasets)
    
    # Academic sorting: proposed model first
    model_order = [m for m in MODEL_COLORS.keys() if m in df_lb['model'].values]
    plot_order = model_order[::-1]  # bottom-up for barh
    y_pos = np.arange(len(plot_order))
    colors = [MODEL_COLORS[m] for m in plot_order]

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
    
    out_file = Path(output_dir) / "polished_diagnostics.png"
    plt.savefig(out_file, bbox_inches='tight', dpi=300, facecolor='white')
    print(f"Final PNG saved to: {out_file}")

if __name__ == "__main__":
    out_dir = 'experiments/results/stat_tests/real_world'
    generate_polished_compact_plot(
        Path(f'{out_dir}/residual_ljung_box.csv'),
        Path(f'{out_dir}/seasonality_kruskal_wallis.csv'),
        out_dir
    )
