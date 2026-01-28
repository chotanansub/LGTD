#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

# --- Academic Style Configuration ---
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "figure.dpi": 300,
    "mathtext.fontset": "cm"
})

project_root = Path(__file__).parent.parent.parent.parent
SENSITIVITY_DIR = project_root / "experiments" / "results" / "sensitivity"
OUTPUT_DIR = project_root / "experiments" / "results" / "sensitivity" / "color_themes"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Define Custom Academic Palettes ---
def get_custom_cmaps():
    customs = {}
    
    # 1. Deep Sea (Professional Dark Blue to Cyan)
    customs['deep_sea'] = LinearSegmentedColormap.from_list('deep_sea', ['#001219', '#005f73', '#0a9396', '#94d2bd'])
    
    # 2. Sunset Gold (High contrast for identifying minima)
    customs['sunset_gold'] = LinearSegmentedColormap.from_list('sunset', ['#000814', '#003566', '#ffc300', '#ffd60a'])
    
    # 3. Earth & Sky (Soft aesthetic, easy on the eyes)
    customs['earth_sky'] = LinearSegmentedColormap.from_list('earth', ['#264653', '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51'])
    
    # 4. Modern Slate (Greyscale friendly for print)
    customs['modern_slate'] = LinearSegmentedColormap.from_list('slate', ['#212529', '#495057', '#adb5bd', '#dee2e6'])
    
    return customs

# Standard Matplotlib themes to include
STANDARD_THEMES = ['viridis', 'plasma', 'cividis', 'Spectral_r', 'coolwarm', 'YlGnBu_r']

def load_data(dataset_name):
    csv_path = SENSITIVITY_DIR / f"{dataset_name}_sensitivity.csv"
    return pd.read_csv(csv_path)

def _plot_best_marker(ax, x, y, val, size_mult=1.0):
    ax.axvline(x=x, color='white', linestyle='--', linewidth=1.5, alpha=0.5, zorder=18)
    ax.axhline(y=y, color='white', linestyle='--', linewidth=1.5, alpha=0.5, zorder=18)
    ax.scatter([x], [y], c='white', s=800*size_mult, marker='*', alpha=0.7, zorder=20)
    ax.scatter([x], [y], c='black', s=450*size_mult, marker='*', alpha=1.0, zorder=21)
    return ax.scatter([x], [y], c='#00ff00', s=250*size_mult, marker='*', edgecolors='none', zorder=22)

def create_combined_mae_global(dataset_names, cmap, theme_name):
    print(f"  Rendering theme: {theme_name}")
    metric = 'mae'
    cols = 3
    rows = (len(dataset_names) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 5.5 * rows))
    axes = axes.flatten()

    g_min, g_max = float('inf'), float('-inf')
    data_cache = {}
    for dname in dataset_names:
        df = load_data(dname)[lambda x: x['valid'] == 1]
        data_cache[dname] = df
        g_min, g_max = min(g_min, df[metric].min()), max(g_max, df[metric].max())

    for i, dname in enumerate(dataset_names):
        ax = axes[i]
        df = data_cache[dname]
        pivot = df.pivot(index='percentile_error', columns='window_size', values=metric)
        X, Y = np.meshgrid(pivot.columns, pivot.index)

        cp = ax.contourf(X, Y, pivot.values, levels=np.linspace(g_min, g_max, 30), 
                         cmap=cmap, extend='both')
        ax.set_box_aspect(1) 
        best = df.loc[df[metric].idxmin()]
        _plot_best_marker(ax, best['window_size'], best['percentile_error'], best[metric], size_mult=0.6)
        ax.set_title(f"{dname}\nMin: {best[metric]:.4f}", pad=8)
        
        if i >= (rows-1)*cols: ax.set_xlabel("Window Size ($W$)")
        if i % cols == 0: ax.set_ylabel("Percentile Error ($\epsilon$)")

    for j in range(len(dataset_names), len(axes)): fig.delaxes(axes[j])

    fig.subplots_adjust(right=0.88, top=0.95, hspace=0.3, wspace=0.3)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.015, 0.7])
    fig.colorbar(cp, cax=cbar_ax, label=f"Global MAE ({theme_name})")

    plt.savefig(OUTPUT_DIR / f"global_mae_{theme_name}.png", bbox_inches='tight')
    plt.close()

def main():
    csv_files = list(SENSITIVITY_DIR.glob("*_sensitivity.csv"))
    datasets = sorted([f.stem.replace('_sensitivity', '') for f in csv_files])
    
    custom_themes = get_custom_cmaps()

    # Process Standard Themes
    for theme in STANDARD_THEMES:
        create_combined_mae_global(datasets, theme, theme)
        
    # Process Custom Themes
    for name, cmap in custom_themes.items():
        create_combined_mae_global(datasets, cmap, name)

    print(f"\nSuccess! Check {OUTPUT_DIR} for all theme variations.")

if __name__ == "__main__":
    main()