#!/usr/bin/env python3
"""
Generate contour plots for LGTD parameter sensitivity analysis for MAE and MSE.
Includes a 3x3 grid with Row (Trend) and Column (Periodicity) headers.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from pathlib import Path

# --- Academic Style Configuration ---
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 16,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 12,
    "figure.dpi": 300,
    "mathtext.fontset": "cm"
})

project_root = Path(__file__).parent.parent.parent.parent
SENSITIVITY_DIR = project_root / "experiments" / "results" / "sensitivity"
BASE_OUTPUT = project_root / "experiments" / "results" / "sensitivity" / "contour_plot"

DEFAULT_PARAM_MARKER_COLOR = '#ff00f2'
DEF_W, DEF_P = 5, 50

def setup_dirs(metric):
    """Creates specific directories for the given metric."""
    metric_dir = BASE_OUTPUT / metric
    log_dir = metric_dir / "logscale"
    clip_dir = metric_dir / "clip"
    for d in [log_dir, clip_dir]:
        d.mkdir(parents=True, exist_ok=True)
    return log_dir, clip_dir

def load_data(dataset_name):
    csv_path = SENSITIVITY_DIR / f"{dataset_name}_sensitivity.csv"
    return pd.read_csv(csv_path)

def _plot_markers_and_line(ax, best_x, best_y, best_val, def_val, mode='individual'):
    """Renders markers and a connecting line with percentage difference."""
    base_s = 500 if mode == 'individual' else 700
    
    ax.plot([DEF_W, best_x], [DEF_P, best_y], color='black', 
            linewidth=5, alpha=0.5, zorder=14)
    ax.plot([DEF_W, best_x], [DEF_P, best_y], color='white', linestyle='--',
            linewidth=3, alpha=1, zorder=15)
    
    # if def_val is not None and def_val != 0:
    #     diff_pct = ((def_val - best_val) / def_val) * 100
    #     mid_x = (DEF_W + best_x) / 2
    #     mid_y = (DEF_P + best_y) / 2
    #     ax.text(mid_x, mid_y, f"{diff_pct:+.1f}%", color='black', fontweight='bold',
    #             fontsize=8 if mode == 'combined' else 10, ha='center', va='center',
    #             zorder=30, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    ax.scatter([best_x], [best_y], c='white', s=base_s * 2.5, marker='*', alpha=0.8, zorder=20)
    ax.scatter([best_x], [best_y], c='black', s=base_s * 1.5, marker='*', alpha=1.0, zorder=21)
    best_m = ax.scatter([best_x], [best_y], c='lime', s=base_s, marker='*', edgecolors='none', zorder=22)

    ax.scatter([DEF_W], [DEF_P], c='white', s=base_s * 0.4, marker='o', alpha=0.8, zorder=20)
    def_m = ax.scatter([DEF_W], [DEF_P], c=DEFAULT_PARAM_MARKER_COLOR, s=base_s * 0.2, marker='o', 
                        edgecolors='black', linewidths=1, zorder=21)
    
    return [best_m, def_m]

def get_robust_vmax(Z):
    q1, q3 = np.percentile(Z, [25, 75])
    iqr = q3 - q1
    vmax = q3 + 3 * iqr
    if vmax <= np.min(Z) or np.isclose(vmax, np.min(Z)):
        vmax = np.percentile(Z, 95)
    return min(vmax, np.max(Z))

def create_contour_plot(dataset_name, metric='mae', mode='log', dirs=None):
    df = load_data(dataset_name)
    df_v = df[df['valid'] == 1]
    if df_v.empty: return

    pivot = df_v.pivot(index='percentile_error', columns='window_size', values=metric)
    X, Y = np.meshgrid(pivot.columns, pivot.index)
    Z = pivot.values
    
    fig, ax = plt.subplots(figsize=(7, 7))
    log_dir, clip_dir = dirs
    
    if mode == 'log':
        vmin_val = max(Z.min(), 1e-3)
        norm = LogNorm(vmin=vmin_val, vmax=Z.max())
        lev_exp = np.linspace(np.log10(vmin_val), np.log10(Z.max()), 30)
        levels = np.power(10, lev_exp)
        cp = ax.contourf(X, Y, Z, levels=levels, norm=norm, cmap='RdYlBu_r', extend='both')
        out_path = log_dir / f"{dataset_name}_{metric}_log.png"
    else:
        vmax = get_robust_vmax(Z)
        cp = ax.contourf(X, Y, Z, levels=30, vmax=vmax, cmap='RdYlBu_r', extend='max')
        out_path = clip_dir / f"{dataset_name}_{metric}_clip.png"

    ax.contour(X, Y, Z, levels=15, colors='black', linewidths=0.3, alpha=0.2)
    
    best = df_v.loc[df_v[metric].idxmin()]
    def_row = df_v[(df_v['window_size'] == DEF_W) & (df_v['percentile_error'] == DEF_P)]
    def_val = def_row[metric].values[0] if not def_row.empty else None

    _plot_markers_and_line(ax, best['window_size'], best['percentile_error'], best[metric], def_val, mode='individual')

    ax.set_box_aspect(1) 
    cbar = plt.colorbar(cp, fraction=0.046, pad=0.1)
    cbar.ax.set_title(metric.upper(), fontsize=10, pad=10, fontweight='normal')
    
    ax.set_xlabel('Window Size ($W$)')
    ax.set_ylabel('Percentile Error ($p$)', labelpad=10)
    ax.set_title(f"{dataset_name.upper()} ({metric.upper()})", pad=15)
    
    legend_handles = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor='lime', markersize=14, 
               label=f'Optimal: {best[metric]:.4f}', markeredgecolor='black'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=DEFAULT_PARAM_MARKER_COLOR, markersize=9, 
               label=r'Default ($W=5$, $p=50^{th}$)', markeredgecolor='black'),
    ]
    
    ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
              ncol=3, frameon=True, fontsize=9.5, columnspacing=0.8)
    
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

def create_combined_subplot(dataset_names, metric='mae', mode='log', dirs=None):
    cols = 3
    rows = (len(dataset_names) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, 6 * rows)) 
    axes = axes.flatten()
    log_dir, clip_dir = dirs

    col_headers = ['Fixed-Period', 'Transitive-Period', 'Variable-Period']
    row_headers = ['Linear', 'Inverted-V', 'Piecewise']

    for i, dname in enumerate(dataset_names):
        ax = axes[i]
        df = load_data(dname)[lambda x: x['valid'] == 1]
        pivot = df.pivot(index='percentile_error', columns='window_size', values=metric)
        X, Y = np.meshgrid(pivot.columns, pivot.index)
        Z = pivot.values

        if mode == 'log':
            vmin_val = max(Z.min(), 1e-3)
            norm = LogNorm(vmin=vmin_val, vmax=Z.max())
            lev_exp = np.linspace(np.log10(vmin_val), np.log10(Z.max()), 25)
            levels = np.power(10, lev_exp)
            cp = ax.contourf(X, Y, Z, levels=levels, norm=norm, cmap='RdYlBu_r', extend='both')
        else:
            vmax = get_robust_vmax(Z)
            cp = ax.contourf(X, Y, Z, levels=25, vmax=vmax, cmap='RdYlBu_r', extend='max')

        ax.set_box_aspect(1) 
        cbar = plt.colorbar(cp, ax=ax, fraction=0.046, pad=0.1)
        cbar.ax.set_title(metric.upper(), fontsize=10, pad=10, fontweight='normal')
        cbar.ax.tick_params(labelsize=9)

        best = df.loc[df[metric].idxmin()]
        def_row = df[(df['window_size'] == DEF_W) & (df['percentile_error'] == DEF_P)]
        def_val = def_row[metric].values[0] if not def_row.empty else None

        _plot_markers_and_line(ax, best['window_size'], best['percentile_error'], best[metric], def_val, mode='combined')
        
        ax.set_title(f"{dname.upper()}", pad=10, fontsize=12, alpha=0.6)
        
        if i < cols:
            ax.annotate(col_headers[i], xy=(0.5, 1.25), xycoords='axes fraction',
                        ha='center', va='baseline', fontsize=20, fontweight='bold')

        if i % cols == 0:
            row_idx = i // cols
            ax.annotate(row_headers[row_idx], xy=(-0.35, 0.5), xycoords='axes fraction',
                        ha='right', va='center', fontsize=20, fontweight='bold', rotation=90)
            ax.set_ylabel("Percentile Error ($p$)", labelpad=12)
        
        if i >= (rows-1)*cols: 
            ax.set_xlabel("Window Size ($W$)", labelpad=10)

    for j in range(i + 1, len(axes)): 
        fig.delaxes(axes[j])
        
    global_legend_handles = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor='lime', markersize=24, 
               label='Optimal Configuration', markeredgecolor='black'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=DEFAULT_PARAM_MARKER_COLOR, markersize=16, 
               label=r'Default ($W=5$, $p=50^{th}$)', markeredgecolor='black'),
    ]
    
    fig.legend(handles=global_legend_handles, loc='lower center', 
               bbox_to_anchor=(0.5, 0.02), ncol=3, fontsize=18, frameon=True)
    
    plt.tight_layout(rect=[0.05, 0.08, 0.98, 0.92]) 
    
    folder = log_dir if mode == 'log' else clip_dir
    plt.savefig(folder / f"combined_{metric}_{mode}.png", bbox_inches='tight')
    plt.close()

def main():
    csv_files = list(SENSITIVITY_DIR.glob("*_sensitivity.csv"))
    datasets = sorted([f.stem.replace('_sensitivity', '') for f in csv_files])

    if not datasets:
        print(f"No CSV files found in {SENSITIVITY_DIR}")
        return

    for metric in ['mae', 'mse']:
        dirs = setup_dirs(metric)
        for mode in ['log', 'clip']:
            print(f"Generating {metric.upper()} {mode.upper()} visualization...")
            for d in datasets:
                create_contour_plot(d, metric=metric, mode=mode, dirs=dirs)
            if len(datasets) > 1:
                create_combined_subplot(datasets, metric=metric, mode=mode, dirs=dirs)

if __name__ == "__main__":
    main()