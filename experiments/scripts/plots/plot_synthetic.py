#!/usr/bin/env python3
"""
Generate publication-quality plots for synthetic dataset decomposition results.
Enhanced with 'colorful' mode for LGTD rainbow gradients.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection

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
    'LGTD_Linear': '#e41a1c',
    'LGTD_LOWESS': '#e41a1c',
    'STL': '#377eb8',
    'OnlineSTL': '#4daf4a',
    'OneShotSTL': '#66c2a5',
    'ASTD': '#984ea3',
    'ASTD_Online': '#ff7f00',
    'RobustSTL': '#a65628',
    'FastRobustSTL': '#f781bf',
    'STR': '#999999'
}

def load_result_json(dataset_name, model_name, results_dir):
    path = Path(results_dir) / dataset_name / f"{model_name}.json"
    if not path.exists(): return None
    with open(path) as f:
        d = json.load(f)
    return {k: np.array(d[k]) for k in ['y', 'trend', 'seasonal', 'residual']}

def load_result_npz(dataset_name, model_name, results_dir):
    path = Path(results_dir) / dataset_name / f"{model_name}.npz"
    if not path.exists(): return None
    d = np.load(path)
    dataset_file = Path('data/synthetic/datasets') / f"{dataset_name}_data.json"
    if dataset_file.exists():
        with open(dataset_file) as f:
            dataset = json.load(f)
            y = np.array(dataset['data']['y'])
    else:
        y = d['trend'] + d['seasonal'] + d['residual']
    return {'y': y, 'trend': d['trend'], 'seasonal': d['seasonal'], 'residual': d['residual']}

def load_result(dataset_name, model_name, results_dir):
    r = load_result_json(dataset_name, model_name, results_dir)
    if r is None: r = load_result_npz(dataset_name, model_name, results_dir)
    return r

def load_ground_truth(dataset_name):
    dataset_file = Path('data/synthetic/datasets') / f"{dataset_name}_data.json"
    if not dataset_file.exists(): return None
    with open(dataset_file) as f:
        dataset = json.load(f)
    return {'seasonal': np.array(dataset['data']['seasonal']), 'residual': np.array(dataset['data']['residual'])}

import matplotlib.colors as mcolors
def plot_rainbow_line(ax, x, y, lw=1.5, alpha=1.0):
    """
    Helper to plot a line with a darker, dull rainbow gradient.
    Uses a gamma-adjusted 'Spectral' map to keep colors muted but make them darker.
    """
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Get the Spectral colormap
    base_cmap = plt.get_cmap('Spectral')
    
    # Create a darkened version by adjusting the lookup table
    # Higher values in the power function make the colormap 'darker'
    colors = base_cmap(np.linspace(0, 1, 256))
    darkened_colors = colors ** 1.5  # Adjust 1.5 to 2.0 for even darker
    dark_cmap = mcolors.ListedColormap(darkened_colors)

    lc = LineCollection(segments, cmap=dark_cmap, linewidth=lw, alpha=alpha)
    
    lc.set_array(np.linspace(0, 1, len(x)))
    ax.add_collection(lc)
    ax.autoscale_view()

def plot_dataset_comparison(dataset_name, models, output_file, results_dir, colorful=False):
    results = {m: load_result(dataset_name, m, results_dir) for m in models if load_result(dataset_name, m, results_dir)}
    if not results: return

    y = next(iter(results.values()))['y']
    time = np.arange(len(y))
    gt = load_ground_truth(dataset_name)

    res_vals = [np.min(r['residual']) for r in results.values()] + ([np.min(gt['residual'])] if gt else [])
    res_maxs = [np.max(r['residual']) for r in results.values()] + ([np.max(gt['residual'])] if gt else [])
    res_min, res_max = min(res_vals), max(res_maxs)
    res_range = max(res_max - res_min, 1.0)
    res_ylim = (res_min - 0.05 * res_range, res_max + 0.05 * res_range)

    n_models = len(results)
    fig = plt.figure(figsize=(12, 2.5 * n_models))
    gs = gridspec.GridSpec(n_models, 3, hspace=0.3, wspace=0.3)

    for idx, (model_name, result) in enumerate(results.items()):
        color = MODEL_COLORS.get(model_name, '#000000')
        is_lgtd = "LGTD" in model_name and colorful

        # Trend
        ax1 = fig.add_subplot(gs[idx, 0])
        ax1.plot(time, y, color='gray', linewidth=1.0, alpha=0.7)
        if is_lgtd: plot_rainbow_line(ax1, time, result['trend'], lw=2.0)
        else: ax1.plot(time, result['trend'], color=color, linewidth=1.5)
        ax1.set_ylabel(model_name, fontweight='bold')
        if idx == 0: ax1.set_title('Original + Trend')

        # Seasonal
        ax2 = fig.add_subplot(gs[idx, 1])
        if gt is not None: ax2.plot(time, gt['seasonal'], color='gray', linewidth=1.2, alpha=0.6)
        if is_lgtd: plot_rainbow_line(ax2, time, result['seasonal'], lw=1.5)
        else: ax2.plot(time, result['seasonal'], color=color, linewidth=1.0)
        if idx == 0: ax2.set_title('Seasonal')

        # Residual
        ax3 = fig.add_subplot(gs[idx, 2])
        if gt is not None: ax3.plot(time, gt['residual'], color='gray', linewidth=1.2, alpha=0.6)
        if is_lgtd: plot_rainbow_line(ax3, time, result['residual'], lw=1.2, alpha=0.8)
        else: ax3.plot(time, result['residual'], color=color, linewidth=0.8, alpha=0.7)
        ax3.axhline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        ax3.set_ylim(res_ylim)
        if idx == 0: ax3.set_title('Residual')

    suffix = "_colorful.png" if colorful else ".png"
    out_path = Path('experiments/results/synthetic/figures') / f"{Path(output_file).stem}{suffix}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_single_model_all_datasets(model_name, datasets, output_file, results_dir, colorful=False):
    results = {d: load_result(d, model_name, results_dir) for d in datasets if load_result(d, model_name, results_dir)}
    if not results: return

    n_datasets = len(results)
    fig = plt.figure(figsize=(12, 2.5 * n_datasets))
    gs = gridspec.GridSpec(n_datasets, 3, hspace=0.3, wspace=0.3)
    color = MODEL_COLORS.get(model_name, '#000000')
    is_lgtd = "LGTD" in model_name and colorful

    for idx, (dataset_name, result) in enumerate(results.items()):
        time = np.arange(len(result['y']))
        gt = load_ground_truth(dataset_name)

        ax1 = fig.add_subplot(gs[idx, 0])
        ax1.plot(time, result['y'], color='gray', linewidth=1.0, alpha=0.7)
        if is_lgtd: plot_rainbow_line(ax1, time, result['trend'], lw=2.0)
        else: ax1.plot(time, result['trend'], color=color, linewidth=1.5)
        ax1.set_ylabel(dataset_name, fontweight='bold')

        ax2 = fig.add_subplot(gs[idx, 1])
        if gt is not None: ax2.plot(time, gt['seasonal'], color='gray', linewidth=1.2, alpha=0.6)
        if is_lgtd: plot_rainbow_line(ax2, time, result['seasonal'], lw=1.5)
        else: ax2.plot(time, result['seasonal'], color=color, linewidth=1.0)

        ax3 = fig.add_subplot(gs[idx, 2])
        if gt is not None: ax3.plot(time, gt['residual'], color='gray', linewidth=1.2, alpha=0.6)
        if is_lgtd: plot_rainbow_line(ax3, time, result['residual'], lw=1.2, alpha=0.8)
        else: ax3.plot(time, result['residual'], color=color, linewidth=0.8, alpha=0.7)
        ax3.axhline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)

    suffix = "_colorful.png" if colorful else ".png"
    out_path = Path('experiments/results/synthetic/figures') / f"{Path(output_file).stem}{suffix}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Plot synthetic decomposition results')
    parser.add_argument('--mode', choices=['by-dataset', 'by-model', 'colorful', 'all'], default='all')
    parser.add_argument('--datasets', nargs='+', default=['synth1', 'synth2', 'synth3', 'synth4', 'synth5', 'synth6', 'synth7', 'synth8', 'synth9'])
    parser.add_argument('--models', nargs='+', default=['LGTD', 'STL', 'OnlineSTL', 'ASTD_Online', 'FastRobustSTL'])
    parser.add_argument('--results-dir', default='experiments/results/synthetic/decompositions')

    args = parser.parse_args()
    
    modes_to_run = []
    if args.mode == 'all': modes_to_run = ['by-dataset', 'by-model', 'colorful']
    else: modes_to_run = [args.mode]

    for m in modes_to_run:
        is_cf = (m == 'colorful')
        if m in ['by-dataset', 'colorful']:
            for ds in args.datasets:
                plot_dataset_comparison(ds, args.models, f'{ds}_comparison', args.results_dir, colorful=is_cf)
        if m in ['by-model', 'colorful']:
            for mdl in args.models:
                plot_single_model_all_datasets(mdl, args.datasets, f'{mdl}_multidataset', args.results_dir, colorful=is_cf)

if __name__ == '__main__':
    main()