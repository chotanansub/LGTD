#!/usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from matplotlib.ticker import LogLocator, NullFormatter
from matplotlib.lines import Line2D

# --- ACADEMIC CONFIG: LARGE TEXT ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'axes.linewidth': 1.0,
    'grid.linewidth': 0.5,
    'lines.linewidth': 0.8,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05
})

MODEL_COLORS = {
    'LGTD': '#e41a1c',
    'STL': '#377eb8',
    'OnlineSTL': '#4daf4a',
    'OneShotSTL': '#66c2a5',
    'ASTD': '#984ea3',
    'ASTD_Online': '#ff7f00',
    'RobustSTL': '#a65628',
    'FastRobustSTL': '#f781bf',
    'STR': '#999999'
}

# Reverted to 3 alpha levels for 3 lags
LAG_ALPHAS = [1.0, 0.7, 0.4]
MAX_LENGTH = 500

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

def plot_combined_horizontal_barplot(ax, model_name, dataset_name, df_lb):
    """ REQUIRED FUNCTION NAME - Strictly preserved """
    # Reverted to lags 10, 20, 30
    lags = ['lag_10_statistic', 'lag_20_statistic', 'lag_30_statistic']
    lower_limit = 10**2.0
    upper_limit = 10**5.0
    
    # Adjusted bar height and offsets for 3 bars to look balanced
    bar_h = 0.08  
    offsets = [0.1, 0, -0.1] 
    
    row = df_lb[(df_lb['dataset'] == dataset_name) & (df_lb['model'] == model_name)]
    color = MODEL_COLORS.get(model_name, '#333333')
    
    ax.grid(False)
    
    for lag, alpha, offset in zip(lags, LAG_ALPHAS, offsets):
        val = row[lag].values[0] if not row.empty and lag in row.columns else lower_limit
        val = max(val, lower_limit) 
        
        ax.barh(offset, val, height=bar_h, color=color,
                alpha=alpha, edgecolor='none', zorder=3)
    
    ax.set_xscale('log')
    ax.set_xlim(lower_limit, upper_limit) 
    
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=4))
    ax.xaxis.set_minor_formatter(NullFormatter())
    
    ax.set_ylim(-0.4, 0.4)
    ax.set_yticks([])
    
    for s in ['left', 'top', 'right']:
        ax.spines[s].set_visible(False)

def main():
    lb_path = STAT_DIR / 'residual_ljung_box.csv'
    if not lb_path.exists(): 
        print(f"File not found: {lb_path}")
        return

    df_lb = pd.read_csv(lb_path)
    datasets = sorted(df_lb['dataset'].unique())[:3]
    exclude_models = {'LGTD_LOWESS', 'LGTD_Linear'}

    fig = plt.figure(figsize=(18.0, 9.5))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.15, top=0.90)
    subfigs = fig.subfigures(1, 3, wspace=0.02) 
    
    for d_idx, (ds_name, subfig) in enumerate(zip(datasets, subfigs)):
        ds_df = df_lb[df_lb['dataset'] == ds_name]
        valid_models = [m for m in MODEL_COLORS.keys() 
                        if m in ds_df['model'].values and m not in exclude_models]
        
        subfig.suptitle(ds_name.replace("_", " ").upper(), fontweight='bold', fontsize=20, y=0.96)
        
        axes = subfig.subplots(len(valid_models), 2, 
                                gridspec_kw={'width_ratios': [1, 1.2], 'hspace': 0.0, 'wspace': 0.05})
        
        for m_idx, m_name in enumerate(valid_models):
            ax_res = axes[m_idx, 0]
            ax_bar = axes[m_idx, 1]
            
            res = load_residual(DECOMP_DIR, ds_name, m_name)
            if res is not None:
                res_norm = (res - np.mean(res)) / (np.std(res) + 1e-8)
                ax_res.plot(res_norm, color=MODEL_COLORS[m_name], lw=0.7)
                ax_res.axhline(0, color='black', lw=0.5, ls='--', alpha=0.3)
            
            ax_res.set_xticks([]); ax_res.set_yticks([])
            for spine in ax_res.spines.values():
                spine.set_visible(False)
            
            if d_idx == 0:
                ax_res.annotate(m_name, xy=(-0.12, 0.5), xycoords='axes fraction',
                               ha='right', va='center', fontweight='bold', fontsize=16)
            
            plot_combined_horizontal_barplot(ax_bar, m_name, ds_name, df_lb)
            
            if m_idx < len(valid_models) - 1:
                ax_bar.set_xticklabels([])
            else:
                ax_bar.set_xlabel(r"Stat $Q_m$", fontsize=12)

            if m_idx == 0:
                ax_res.set_title(r"Resid. $\epsilon_t$", fontsize=13)
                ax_bar.set_title("Ljung–Box", fontsize=13)

    # Reverted Legend for 3 lags
    legend_elements = [
        Line2D([0], [0], color='gray', alpha=LAG_ALPHAS[0], lw=2, label=r'$Q_{10}$'),
        Line2D([0], [0], color='gray', alpha=LAG_ALPHAS[1], lw=2, label=r'$Q_{20}$'),
        Line2D([0], [0], color='gray', alpha=LAG_ALPHAS[2], lw=2, label=r'$Q_{30}$'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.03),
                ncol=3, frameon=False, fontsize=13)

    out_file_png = OUTPUT_DIR / "realworld_residual.png"
    plt.savefig(out_file_png, format='png', dpi=300)
    print(f"✓ Saved: {out_file_png}")

if __name__ == "__main__":
    main()