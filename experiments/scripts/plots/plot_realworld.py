#!/usr/bin/env python3
"""
Generate publication-quality plots for real-world dataset decomposition results.

This script generates both selective (default models) and full (all models) comparison plots.

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
    'ASTD_Online': '#ff7f00',
    'STR': '#a65628',
    'RobustSTL': '#f781bf',
    'FastRobustSTL': '#999999',
    'OneShotSTL': '#66c2a5',
    'LGTD_LOWESS': '#fc8d62',
    'LGTD_Linear': '#8da0cb'
}

def load_result(dataset_name, model_name):
    """Load decomposition result from JSON file."""
    results_dir = Path('experiments/results/decompositions/real_world')
    # Try subdirectory structure first (current format)
    result_file = results_dir / dataset_name / f"{model_name}.json"
    if not result_file.exists():
        # Fallback to flat structure (legacy format)
        result_file = results_dir / f"{dataset_name}_{model_name}.json"
        if not result_file.exists():
            return None
    with open(result_file, 'r') as f:
        data = json.load(f)
    return {k: np.array(data[k]) for k in ['y', 'trend', 'seasonal', 'residual']}

def get_zoom_indices(n_points, zoom_enabled, window_size=500):
    """Get start and end indices for zoom window."""
    if zoom_enabled and n_points > window_size:
        mid = n_points // 2
        return mid - (window_size // 2), mid + (window_size // 2)
    return 0, n_points

def plot_dataset_comparison(dataset_name, models, zoom_enabled, suffix=''):
    """Plot comparison for one dataset with specified models."""
    results = {}
    for model in models:
        result = load_result(dataset_name, model)
        if result is not None:
            results[model] = result
    if not results:
        print(f"  ⚠ No results found for {dataset_name}")
        return

    y_ref = list(results.values())[0]['y']
    n_points = len(y_ref)
    
    # Indices for zoomed columns (2 and 3)
    start, end = get_zoom_indices(n_points, zoom_enabled)
    time_zoom = np.arange(start, end)
    
    # Indices for full column (1)
    time_full = np.arange(n_points)

    # Calculate shared seasonal Y-scale for the visible window
    seas_min, seas_max = np.inf, -np.inf
    for r in results.values():
        seas_chunk = r['seasonal'][start:end]
        seas_min, seas_max = min(seas_min, np.min(seas_chunk)), max(seas_max, np.max(seas_chunk))

    seas_range = max(seas_max - seas_min, 1e-6)
    seas_ylim = (seas_min - 0.05 * seas_range, seas_max + 0.05 * seas_range)

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
        ax2.set_ylim(seas_ylim)
        if idx == 0: ax2.set_title('Seasonal' + (' (Zoomed)' if zoom_enabled else ''))
        if idx == n_models - 1: ax2.set_xlabel('Time')

        # Column 3: Residual (Full or Zoomed)
        ax3 = fig.add_subplot(gs[idx, 2])
        ax3.plot(time_zoom, result['residual'][start:end], color=color, linewidth=0.8, alpha=0.7)
        ax3.axhline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        ax3.set_ylim(res_ylim)
        if idx == 0: ax3.set_title('Residual' + (' (Zoomed)' if zoom_enabled else ''))
        if idx == n_models - 1: ax3.set_xlabel('Time')

    out_dir = Path('experiments/results/figures/real_world')
    out_dir.mkdir(parents=True, exist_ok=True)
    zoom_suffix = "_zoom" if zoom_enabled else ""
    output_path = out_dir / f"{dataset_name}_comparison{suffix}{zoom_suffix}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved {output_path}")

def plot_single_model_all_datasets(model_name, datasets, zoom_enabled, suffix=''):
    """Plot single model across all datasets."""
    results = {d: load_result(d, model_name) for d in datasets if load_result(d, model_name) is not None}
    if not results:
        print(f"  ⚠ No results found for {model_name}")
        return

    n_datasets = len(results)
    fig = plt.figure(figsize=(12, 2.5 * n_datasets))
    gs = gridspec.GridSpec(n_datasets, 3, hspace=0.3, wspace=0.3)

    # FIGURE HEADLINE REMOVED
    # fig.suptitle(f'{model_name}{title_suffix} - Multi-Dataset', fontsize=14, fontweight='bold')

    # Calculate shared seasonal Y-scale across all datasets for this model
    seas_min_global, seas_max_global = np.inf, -np.inf
    res_min_global, res_max_global = np.inf, -np.inf

    zoom_ranges = {}
    for dataset_name, result in results.items():
        n_points = len(result['y'])
        start, end = get_zoom_indices(n_points, zoom_enabled)
        zoom_ranges[dataset_name] = (start, end)

        seas_chunk = result['seasonal'][start:end]
        seas_min_global = min(seas_min_global, np.min(seas_chunk))
        seas_max_global = max(seas_max_global, np.max(seas_chunk))

        res_chunk = result['residual'][start:end]
        res_min_global = min(res_min_global, np.min(res_chunk))
        res_max_global = max(res_max_global, np.max(res_chunk))

    seas_range = max(seas_max_global - seas_min_global, 1e-6)
    seas_ylim = (seas_min_global - 0.05 * seas_range, seas_max_global + 0.05 * seas_range)

    res_range = max(res_max_global - res_min_global, 1e-6)
    res_ylim = (res_min_global - 0.05 * res_range, res_max_global + 0.05 * res_range)

    for idx, (dataset_name, result) in enumerate(results.items()):
        start, end = zoom_ranges[dataset_name]
        n_points = len(result['y'])

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
        ax2.set_ylim(seas_ylim)
        if idx == 0: ax2.set_title('Seasonal')

        # Col 3: Residual (Zoomed)
        ax3 = fig.add_subplot(gs[idx, 2])
        ax3.plot(time_zoom, result['residual'][start:end], color=color, linewidth=0.8, alpha=0.7)
        ax3.axhline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        ax3.set_ylim(res_ylim)
        if idx == 0: ax3.set_title('Residual')

    out_dir = Path('experiments/results/figures/real_world')
    out_dir.mkdir(parents=True, exist_ok=True)
    zoom_suffix = "_zoom" if zoom_enabled else ""
    output_path = out_dir / f"{model_name}_multidataset{suffix}{zoom_suffix}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved {output_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Generate real-world decomposition comparison plots'
    )
    parser.add_argument('--mode', choices=['by-dataset', 'by-model', 'both'], default='both',
                        help='Plot organization mode')
    parser.add_argument('--datasets', nargs='+', default=['Sunspot', 'ETTh1', 'ETTh2'],
                        help='Datasets to plot')
    parser.add_argument('--default-models', nargs='+',
                        default=['LGTD', 'STL', 'OnlineSTL', 'ASTD_Online'],
                        help='Default models for selective comparison')
    parser.add_argument('--full-models', nargs='+',
                        default=['LGTD', 'STL', 'OnlineSTL', 'ASTD', 'ASTD_Online',
                                'STR', 'RobustSTL', 'FastRobustSTL'],
                        help='All models for full comparison')
    parser.add_argument('--plot-type', choices=['default', 'full', 'both'], default='both',
                        help='Generate default plots, full plots, or both')
    args = parser.parse_args()

    print("=" * 70)
    print("Generating Real-World Decomposition Plots")
    print("=" * 70)
    print(f"Datasets: {', '.join(args.datasets)}")
    print()

    # Generate plots based on plot_type
    plot_configs = []
    if args.plot_type in ['default', 'both']:
        plot_configs.append(('Default', args.default_models, ''))
    if args.plot_type in ['full', 'both']:
        plot_configs.append(('Full', args.full_models, '_full'))

    for config_name, models, suffix in plot_configs:
        print(f"\n{config_name} Model Set ({len(models)} models):")
        print(f"  Models: {', '.join(models)}")
        print()

        for is_zoom in [False, True]:
            status = "Zoomed" if is_zoom else "Full"
            print(f"  Generating {status} plots...")

            if args.mode in ['by-dataset', 'both']:
                for d in args.datasets:
                    print(f"    Processing {d}...")
                    plot_dataset_comparison(d, models, is_zoom, suffix)

            if args.mode in ['by-model', 'both']:
                for m in models:
                    print(f"    Processing {m}...")
                    plot_single_model_all_datasets(m, args.datasets, is_zoom, suffix)

    print()
    print("=" * 70)
    print("Plot Generation Complete!")
    print("=" * 70)
    print("Output directory: experiments/results/figures/real_world/")
    print()

if __name__ == '__main__':
    main()