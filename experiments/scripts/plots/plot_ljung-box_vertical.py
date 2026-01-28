#!/usr/bin/env python3
"""
Diagnostic Figure for Ljung-Box Test Results (Multi-Lag)
Generates both 1x3 and 3x1 (Compact) publication-quality plots with rotated x-labels.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from matplotlib.patches import Patch

# --- CONFIGURATION ---
INPUT_CSV = Path('experiments/results/stat_tests/real_world/residual_ljung_box.csv')
OUTPUT_DIR = Path('experiments/results/stat_tests/real_world')
DEFAULT_MODELS = ['LGTD', 'STL', 'OnlineSTL', 'ASTD_Online', 'FastRobustSTL', 'OneShotSTL', 'STR']

# Peer-Review Style Configuration
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'] + plt.rcParams['font.serif'],
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'xtick.labelsize': 6.5,
    'ytick.labelsize': 6.5,
    'axes.linewidth': 0.6,
    'lines.linewidth': 0.5,
    'figure.dpi': 300
})

MODEL_COLORS = {
    'LGTD': '#e41a1c',
    'STL': "#3A5C57",
    'STR': '#3A5C57',  
    'OnlineSTL': '#3A5C57',
    'ASTD_Online': '#3A5C57',
    'FastRobustSTL': '#3A5C57'
}

def generate_plots():
    if not INPUT_CSV.exists():
        print(f"Error: File not found at {INPUT_CSV}")
        return

    df_lb = pd.read_csv(INPUT_CSV)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    available_models = set(df_lb['model'].unique())
    valid_models = [m for m in DEFAULT_MODELS if m in available_models]
    df_lb = df_lb[df_lb['model'].isin(valid_models)]
    datasets = sorted(df_lb['dataset'].unique())
    
    x_pos = np.arange(len(valid_models))
    lags = ['lag_10_statistic', 'lag_20_statistic', 'lag_30_statistic']
    lag_labels = [r'$Q_{10}$', r'$Q_{20}$', r'$Q_{30}$']
    bar_width, offsets, alphas = 0.25, [-0.25, 0, 0.25], [1.0, 0.65, 0.35]

    for layout in ['1x3', '3x1']:
        if layout == '1x3':
            fig, axes = plt.subplots(1, len(datasets), figsize=(8.5, 2.8), squeeze=False)
            leg_y = 1.12
        else:
            # Increased height slightly to 6.0 to accommodate rotated labels
            fig, axes = plt.subplots(len(datasets), 1, figsize=(4.0, 3.9), squeeze=False, sharex=True)
            leg_y = 1.05

        for i, ds in enumerate(datasets):
            ax = axes[0, i] if layout == '1x3' else axes[i, 0]
            ds_lb = df_lb[df_lb['dataset'] == ds].set_index('model').reindex(valid_models)

            for j, (lag_col, offset, alpha) in enumerate(zip(lags, offsets, alphas)):
                colors = [MODEL_COLORS.get(m, '#3A5C57') for m in valid_models]
                ax.bar(x_pos + offset, ds_lb[lag_col], width=bar_width, color=colors,
                       alpha=alpha, edgecolor='black', linewidth=0.2)
            
            ax.set_yscale('log')
            ax.set_xticks(x_pos)
            
            # --- ROTATION LOGIC ---
            if layout == '1x3':
                ax.set_xticklabels(valid_models, rotation=45, ha='right')
            elif i == len(datasets) - 1:
                # Apply rotation to the bottom subplot of the 3x1 stack
                ax.set_xticklabels(valid_models, rotation=45, ha='right')
            
            ax.set_title(ds, fontweight='bold', pad=6)
            ax.spines[['top', 'right']].set_visible(False)
            
            if (layout == '1x3' and i == 0) or layout == '3x1':
                ax.set_ylabel(r'LB $Q$ (Log)', fontsize=7)

        # Legend
        legend_elements = [Patch(facecolor='gray', alpha=alphas[k], edgecolor='black', label=lag_labels[k]) for k in range(3)]
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, leg_y),
                   ncol=3, fontsize=7, frameon=False, handletextpad=0.3)

        # Use bbox_inches='tight' to ensure rotated labels aren't cropped
        plt.tight_layout()
        save_path = OUTPUT_DIR / f"ljung_box_{layout}.png"
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.05)
        plt.close()
        print(f"✓ Created: {save_path}")

if __name__ == "__main__":
    generate_plots()