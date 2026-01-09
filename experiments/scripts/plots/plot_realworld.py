#!/usr/bin/env python3
"""
Generate publication-quality plots for real-world dataset decomposition results.

Strategy:
- Column 1: Always Full Time Series (Original in gray, Trend in model color).
- Column 2 & 3: Full OR Zoomed (Middle 500 points) based on file version.
- Residuals share Y-scale within the visible window.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import matplotlib.gridspec as gridspec

# Publication-quality plot settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['figure.titlesize'] = 13

MODEL_COLORS = {
    'LGTD': '#e41a1c',
    'STL': '#377eb8',
    'OnlineSTL': '#4daf4a',
    'ASTD': '#984ea3',
    'ASTD_Online': '#ff7f00'
}

def load_result(dataset_name, model_name):
    results_dir = Path('experiments/results/real_world/decompositions')
    result_file = results_dir / f"{dataset_name}_{model_name}.json"
    if not result_file.exists():
        return None
    with open(result_file, 'r') as f:
        data = json.load(f)
    return {k: np.array(data[k]) for k in ['y', 'trend', 'seasonal', 'residual']}

def get_zoom_indices(n_points, zoom_enabled, window_size=500):
    if zoom_enabled and n_points > window_size:
        mid = n_points // 2
        return mid - (window_size // 2), mid + (window_size // 2)
    return 0, n_points

def plot_dataset_comparison(dataset_name, models, zoom_enabled):
    results = {}
    for model in models:
        result = load_result(dataset_name, model)
        if result is not None:
            results[model] = result
    if not results: return

    y_ref = list(results.values())[0]['y']
    n_points = len(y_ref)
    
    # Indices for zoomed columns (2 and 3)
    start, end = get_zoom_indices(n_points, zoom_enabled)
    time_zoom = np.arange(start, end)
    
    # Indices for full column (1)
    time_full = np.arange(n_points)

    # Calculate shared residual Y-scale for the visible window
    res_min, res_max = np.inf, -np.inf
    for r in results.values():
        res_chunk = r['residual'][start:end]
        res_min, res_max = min(res_min, np.min(res_chunk)), max(res_max, np.max(res_chunk))
    
    res_range = max(res_max - res_min, 1e-6)
    res_ylim = (res_min - 0.05 * res_range, res_max + 0.05 * res_range)

    n_models = len(results)
    fig = plt.figure(figsize=(12, 2.5 * n_models))
    gs = gridspec.GridSpec(n_models, 3, hspace=0.3, wspace=0.3)
    
    # FIGURE HEADLINE REMOVED
    # fig.suptitle(f'{dataset_name}{title_suffix} - Decomposition Comparison', fontsize=14, fontweight='bold')

    for idx, (model_name, result) in enumerate(results.items()):
        color = MODEL_COLORS.get(model_name, '#000000')
        
        # Column 1: Original + Trend (ALWAYS FULL)
        ax1 = fig.add_subplot(gs[idx, 0])
        ax1.plot(time_full, result['y'], color='gray', linewidth=0.8, alpha=0.5)
        ax1.plot(time_full, result['trend'], color=color, linewidth=1.2)
        ax1.set_ylabel(model_name, fontweight='bold')
        if idx == 0: ax1.set_title('Original + Trend (Full)')
        if idx == n_models - 1: ax1.set_xlabel('Time')

        # Column 2: Seasonal (Full or Zoomed)
        ax2 = fig.add_subplot(gs[idx, 1])
        ax2.plot(time_zoom, result['seasonal'][start:end], color=color, linewidth=1.0)
        if idx == 0: ax2.set_title('Seasonal' + (' (Zoomed)' if zoom_enabled else ''))
        if idx == n_models - 1: ax2.set_xlabel('Time')

        # Column 3: Residual (Full or Zoomed)
        ax3 = fig.add_subplot(gs[idx, 2])
        ax3.plot(time_zoom, result['residual'][start:end], color=color, linewidth=0.8, alpha=0.7)
        ax3.axhline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        ax3.set_ylim(res_ylim)
        if idx == 0: ax3.set_title('Residual' + (' (Zoomed)' if zoom_enabled else ''))
        if idx == n_models - 1: ax3.set_xlabel('Time')

    out_dir = Path('experiments/results/real_world/figures')
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_zoom" if zoom_enabled else ""
    plt.savefig(out_dir / f"{dataset_name}_comparison{suffix}.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_single_model_all_datasets(model_name, datasets, zoom_enabled):
    results = {d: load_result(d, model_name) for d in datasets if load_result(d, model_name) is not None}
    if not results: return

    n_datasets = len(results)
    fig = plt.figure(figsize=(12, 2.5 * n_datasets))
    gs = gridspec.GridSpec(n_datasets, 3, hspace=0.3, wspace=0.3)
    
    # FIGURE HEADLINE REMOVED
    # fig.suptitle(f'{model_name}{title_suffix} - Multi-Dataset', fontsize=14, fontweight='bold')

    for idx, (dataset_name, result) in enumerate(results.items()):
        n_points = len(result['y'])
        start, end = get_zoom_indices(n_points, zoom_enabled)
        
        time_full = np.arange(n_points)
        time_zoom = np.arange(start, end)
        color = MODEL_COLORS.get(model_name, '#000000')

        # Col 1: Always Full
        ax1 = fig.add_subplot(gs[idx, 0])
        ax1.plot(time_full, result['y'], color='gray', linewidth=0.8, alpha=0.5)
        ax1.plot(time_full, result['trend'], color=color, linewidth=1.2)
        ax1.set_ylabel(dataset_name, fontweight='bold')
        if idx == 0: ax1.set_title('Original + Trend (Full)')

        # Col 2: Seasonal (Zoomed)
        ax2 = fig.add_subplot(gs[idx, 1])
        ax2.plot(time_zoom, result['seasonal'][start:end], color=color, linewidth=1.0)
        if idx == 0: ax2.set_title('Seasonal')

        # Col 3: Residual (Zoomed)
        ax3 = fig.add_subplot(gs[idx, 2])
        ax3.plot(time_zoom, result['residual'][start:end], color=color, linewidth=0.8, alpha=0.7)
        ax3.axhline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        if idx == 0: ax3.set_title('Residual')

    out_dir = Path('experiments/results/real_world/figures')
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_zoom" if zoom_enabled else ""
    plt.savefig(out_dir / f"{model_name}_multidataset{suffix}.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['by-dataset', 'by-model', 'both'], default='both')
    parser.add_argument('--datasets', nargs='+', default=['sunspot', 'ETTh1', 'ETTh2'])
    parser.add_argument('--models', nargs='+', default=['LGTD', 'STL', 'OnlineSTL', 'ASTD_Online'])
    args = parser.parse_args()

    for is_zoom in [False, True]:
        status = "Zoomed (Cols 2-3)" if is_zoom else "Full"
        print(f"Generating {status} plots...")
        
        if args.mode in ['by-dataset', 'both']:
            for d in args.datasets:
                plot_dataset_comparison(d, args.models, is_zoom)

        if args.mode in ['by-model', 'both']:
            for m in args.models:
                plot_single_model_all_datasets(m, args.datasets, is_zoom)

    print(f"Done! Files saved in results/real_world/figures/")

if __name__ == '__main__':
    main()