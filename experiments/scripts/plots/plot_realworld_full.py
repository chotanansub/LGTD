#!/usr/bin/env python3
"""
Generate full comparison plots with all available models.

This script generates plots with all models and saves them with "_full" suffix
to distinguish from the standard 4-model comparison.
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
    'FastRobustSTL': '#999999'
}


def load_result(dataset_name, model_name):
    """Load decomposition result from JSON file."""
    results_dir = Path('experiments/results/real_world/decompositions')
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


def plot_dataset_comparison_full(dataset_name, models, zoom_enabled):
    """Plot comparison for one dataset with all models."""
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
    suffix = "_zoom_full" if zoom_enabled else "_full"
    output_path = out_dir / f"{dataset_name}_comparison{suffix}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved {output_path}")


def main():
    """Generate full model comparison plots."""

    # All available models
    models = ['LGTD', 'STL', 'OnlineSTL', 'ASTD', 'ASTD_Online', 'STR', 'RobustSTL', 'FastRobustSTL']
    datasets = ['sunspot', 'ETTh1', 'ETTh2']

    print("="*70)
    print("Generating Full Real-World Decomposition Plots")
    print("="*70)
    print(f"Models: {', '.join(models)}")
    print(f"Datasets: {', '.join(datasets)}")
    print()

    for is_zoom in [False, True]:
        status = "Zoomed" if is_zoom else "Full"
        print(f"\nGenerating {status} plots...")

        for dataset in datasets:
            print(f"  Processing {dataset}...")
            plot_dataset_comparison_full(dataset, models, is_zoom)

    print()
    print("="*70)
    print("Full Comparison Plots Generated!")
    print("="*70)
    print("Output directory: results/real_world/figures/")
    print(f"Files with '_full' suffix include all {len(models)} models")
    print()


if __name__ == '__main__':
    main()
