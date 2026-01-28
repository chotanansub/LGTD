#!/usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from matplotlib.ticker import LogLocator, NullFormatter
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# --- ACADEMIC CONFIG ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 8,
    'axes.titlesize': 10,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'axes.linewidth': 0.6,
    'figure.dpi': 300
})

MODEL_COLORS = {
    'LGTD': '#e41a1c', 'STL': '#377eb8', 'OnlineSTL': '#4daf4a',
    'OneShotSTL': '#66c2a5', 'ASTD': '#984ea3', 'ASTD_Online': '#ff7f00',
    'RobustSTL': '#a65628', 'FastRobustSTL': '#f781bf', 'STR': '#999999'
}

LAG_MARKERS = ['o', 's', '^', 'D', 'p']
LAG_VALS = [10, 20, 30, 40, 50]
MAX_LENGTH = 400

# --- PATHS ---
PROJECT_ROOT = Path(__file__).parent.parent.parent
DECOMP_DIR = PROJECT_ROOT / 'results' / 'decompositions' / 'real_world'
STAT_DIR = PROJECT_ROOT / 'results' / 'stat_tests' / 'real_world'
OUTPUT_DIR = STAT_DIR

def load_residual(decomp_dir: Path, dataset: str, model: str):
    decomp_path = decomp_dir / dataset / f"{model}.json"
    if not decomp_path.exists(): return None
    try:
        with open(decomp_path, 'r') as f:
            data = json.load(f)
        res = np.array(data['residual'])
        if len(res) > MAX_LENGTH:
            res = res[np.linspace(0, len(res)-1, MAX_LENGTH, dtype=int)]
        return res
    except: return None

def plot_combined_horizontal_barplot(ax, model_name, dataset_name, df_lb, ds_limits, plot_type='line'):
    """ REQUIRED FUNCTION NAME - Preserved. Bars use global scale; Lines use local zoom. """
    lags = [f'lag_{i}_statistic' for i in LAG_VALS]
    
    row = df_lb[(df_lb['dataset'] == dataset_name) & (df_lb['model'] == model_name)]
    color = MODEL_COLORS.get(model_name, '#333333')
    
    values = [max(row[lag].values[0], 10**1.0) if (not row.empty and lag in row.columns) else 10**1.0 for lag in lags]
    
    if plot_type == 'line':
        local_min = min(values) * 0.5
        local_max = max(values) * 2.0
        x_pos = np.linspace(0.7, 1.3, 5)
        ax.plot(x_pos, values, color=color, alpha=0.3, lw=2, zorder=2)
        for i, (x, val) in enumerate(zip(x_pos, values)):
            ax.scatter(x, val, marker=LAG_MARKERS[i], color=color, s=60, alpha=0.8, edgecolors='none', zorder=3)
        ax.set_xlim(0.6, 1.4)
        ax.set_xticks([])
        ax.set_yscale('log')
        ax.set_ylim(local_min, local_max)
    else:
        lower_limit, upper_limit = ds_limits
        y_pos = np.linspace(-0.3, 0.3, 5)
        for i, (y, val) in enumerate(zip(y_pos, values)):
            ax.barh(y, val, height=0.1, color=color, alpha=0.6, edgecolor='none')
        
        # ADDED: Explicit Y-axis labels for lags
        ax.set_yticks(y_pos)
        ax.set_yticklabels(LAG_VALS)
        
        ax.set_ylim(-0.5, 0.5)
        ax.set_xscale('log')
        ax.set_xlim(lower_limit, upper_limit)

    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    if plot_type == 'line': ax.spines['bottom'].set_visible(False)

def main():
    lb_path = STAT_DIR / 'residual_ljung_box.csv'
    if not lb_path.exists(): 
        print(f"Error: CSV not found at {lb_path}")
        return
    df_lb = pd.read_csv(lb_path)
    datasets = sorted(df_lb['dataset'].unique())[:3]
    exclude_models = {'LGTD_LOWESS', 'LGTD_Linear'}
    valid_models = [m for m in MODEL_COLORS.keys() if m in df_lb['model'].values and m not in exclude_models]
    
    fig = plt.figure(figsize=(18, 11))
    outer_grid = gridspec.GridSpec(len(datasets), 1, figure=fig, hspace=0.08)

    for d_idx, ds_name in enumerate(datasets):
        ds_df = df_lb[df_lb['dataset'] == ds_name]
        lags_cols = [f'lag_{i}_statistic' for i in LAG_VALS]
        relevant_vals = ds_df[ds_df['model'].isin(valid_models)][lags_cols].values
        global_min = max(10**1.0, np.nanmin(relevant_vals) * 0.5)
        global_max = np.nanmax(relevant_vals) * 2.0
        ds_limits = (global_min, global_max)

        inner_grid = gridspec.GridSpecFromSubplotSpec(3, len(valid_models), 
                                                      subplot_spec=outer_grid[d_idx], 
                                                      height_ratios=[1.2, 0.7, 0.6],
                                                      wspace=0.08, hspace=0.0)
        
        fig.text(0.012, 1 - (d_idx * 0.31) - 0.15, ds_name.upper().replace("_", " "), 
                 va='center', rotation=90, fontweight='bold', fontsize=14)

        for m_idx, m_name in enumerate(valid_models):
            ax_top = fig.add_subplot(inner_grid[0, m_idx])
            ax_res = fig.add_subplot(inner_grid[1, m_idx])
            ax_bot = fig.add_subplot(inner_grid[2, m_idx])
            
            # 1. Top Line Plot
            plot_combined_horizontal_barplot(ax_top, m_name, ds_name, df_lb, ds_limits, plot_type='line')
            if d_idx == 0: 
                ax_top.set_title(m_name, fontsize=14, fontweight='bold', pad=8)
            
            # 2. Middle Residual Plot
            res = load_residual(DECOMP_DIR, ds_name, m_name)
            if res is not None:
                res_norm = (res - np.mean(res)) / (np.std(res) + 1e-8)
                ax_res.plot(res_norm, color=MODEL_COLORS[m_name], lw=0.6, alpha=0.7)
                ax_res.axhline(0, color='black', lw=0.4, ls='-', alpha=0.2)
                padding = (res_norm.max() - res_norm.min()) * 0.1
                ax_res.set_ylim(res_norm.min() - padding, res_norm.max() + padding)
            
            if m_idx == 0:
                ax_res.set_ylabel("Residual", fontsize=7, fontweight='semibold', labelpad=2)
            
            ax_res.set_xticks([]); ax_res.set_yticks([])
            for s in ax_res.spines.values(): s.set_visible(False)

            # 3. Bottom Bar Plot
            plot_combined_horizontal_barplot(ax_bot, m_name, ds_name, df_lb, ds_limits, plot_type='bar')
            
            # Labeling strategy: Only show y-axis ticks/labels on the first column
            if m_idx != 0:
                ax_top.set_yticklabels([])
                ax_bot.set_yticklabels([])
            else:
                ax_top.set_ylabel(r"$Q$ (Zoom)", fontsize=9)
                ax_bot.set_ylabel(r"Lag", fontsize=9)

            if d_idx == len(datasets)-1:
                ax_bot.set_xlabel(r"Ljung--Box ($Q_m$)", fontsize=10)
            else:
                ax_bot.set_xticklabels([])

    # Legend
    legend_elements = [Line2D([0], [0], marker=m, color='w', markerfacecolor='#555555', markersize=7, label=f"Lag {l}") 
                       for m, l in zip(LAG_MARKERS, LAG_VALS)]
    fig.legend(handles=legend_elements, 
               loc='lower center', 
               bbox_to_anchor=(0.5, 0.01), 
               ncol=5, 
               frameon=False,
               fontsize=12,        
               markerscale=1.2,    
               handletextpad=0.5) 
    plt.subplots_adjust(left=0.07, right=0.98, bottom=0.1, top=0.94)
    plt.savefig(OUTPUT_DIR / "realworld_residual.png", dpi=300)
    print(f"✓ Saved plot with Lag labels (10-50) to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()