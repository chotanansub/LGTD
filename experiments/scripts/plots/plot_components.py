#!/usr/bin/env python3
"""
Shared plotting components for consistent visualization across synthetic and real-world datasets.

This module provides:
- Common color schemes
- File name mapping utilities
- Shared plotting functions
- Rainbow gradient effects for LGTD models
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from pathlib import Path

# Publication-quality plot settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['figure.titlesize'] = 13

# Consistent color scheme across all plots
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

# Mapping from display names to file names
MODEL_FILE_NAMES = {
    'LGTD': 'LGTD',
    'LGTD_Linear': 'LGTD_LINEAR',
    'LGTD_LOWESS': 'LGTD_LOWESS',
    'STL': 'STL',
    'RobustSTL': 'ROBUST_STL',
    'FastRobustSTL': 'FAST_ROBUST_STL',
    'STR': 'STR',
    'OnlineSTL': 'ONLINE_STL',
    'OneShotSTL': 'ONESHOT_STL',
    'ASTD': 'ASTD',
    'ASTD_Online': 'ASTD_ONLINE'
}


def plot_rainbow_line(ax, x, y, lw=1.5, alpha=1.0):
    """
    Plot a line with a darker, dull rainbow gradient.

    Uses a gamma-adjusted 'Spectral' colormap to keep colors muted but darker.

    Args:
        ax: Matplotlib axis
        x: X coordinates
        y: Y coordinates
        lw: Line width
        alpha: Line transparency
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


def create_output_path(base_dir, output_file, plot_type='default', zoom=False, colorful=False):
    """
    Create organized output path with subfolder structure.

    Args:
        base_dir: Base output directory (e.g., 'experiments/results/figures/synthetic')
        output_file: Base filename without extension
        plot_type: Type of plot ('default', 'full', 'by_model', 'by_dataset')
        zoom: Whether this is a zoomed plot
        colorful: Whether this is a colorful plot

    Returns:
        Path object for the output file
    """
    base_path = Path(base_dir)

    # Determine subfolder based on plot characteristics
    if colorful:
        subfolder = 'colorful'
    elif zoom:
        subfolder = f'{plot_type}_zoom' if plot_type != 'default' else 'zoom'
    else:
        subfolder = plot_type if plot_type != 'default' else 'standard'

    # Create full path
    output_dir = base_path / subfolder
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build filename
    suffix = "_colorful" if colorful else ""
    suffix += "_zoom" if zoom else ""
    filename = f"{output_file}{suffix}.png"

    return output_dir / filename


def get_model_color(model_name):
    """
    Get the color for a given model name.

    Args:
        model_name: Name of the model

    Returns:
        Color hex string
    """
    return MODEL_COLORS.get(model_name, '#000000')


def get_model_file_name(model_name):
    """
    Convert display model name to file name.

    Args:
        model_name: Display name of the model

    Returns:
        File name for the model
    """
    return MODEL_FILE_NAMES.get(model_name, model_name)


def is_lgtd_variant(model_name):
    """
    Check if a model is an LGTD variant.

    Args:
        model_name: Name of the model

    Returns:
        True if model is an LGTD variant
    """
    return "LGTD" in model_name


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

import json
import matplotlib.gridspec as gridspec


def load_synthetic_result_json(dataset_name, model_name, results_dir):
    """Load synthetic dataset result from JSON file."""
    file_name = get_model_file_name(model_name)
    path = Path(results_dir) / dataset_name / f"{file_name}.json"
    if not path.exists():
        return None
    with open(path) as f:
        d = json.load(f)
    return {k: np.array(d[k]) for k in ['y', 'trend', 'seasonal', 'residual']}


def load_synthetic_result_npz(dataset_name, model_name, results_dir):
    """Load synthetic dataset result from NPZ file."""
    file_name = get_model_file_name(model_name)
    path = Path(results_dir) / dataset_name / f"{file_name}.npz"
    if not path.exists():
        return None
    d = np.load(path)
    dataset_file = Path('data/synthetic/datasets') / f"{dataset_name}_data.json"
    if dataset_file.exists():
        with open(dataset_file) as f:
            dataset = json.load(f)
            y = np.array(dataset['data']['y'])
    else:
        y = d['trend'] + d['seasonal'] + d['residual']
    return {'y': y, 'trend': d['trend'], 'seasonal': d['seasonal'], 'residual': d['residual']}


def load_synthetic_result(dataset_name, model_name, results_dir):
    """Load synthetic dataset result (tries JSON first, then NPZ)."""
    r = load_synthetic_result_json(dataset_name, model_name, results_dir)
    if r is None:
        r = load_synthetic_result_npz(dataset_name, model_name, results_dir)
    return r


def load_ground_truth(dataset_name):
    """Load ground truth for synthetic dataset."""
    dataset_file = Path('data/synthetic/datasets') / f"{dataset_name}_data.json"
    if not dataset_file.exists():
        return None
    with open(dataset_file) as f:
        dataset = json.load(f)
    return {
        'seasonal': np.array(dataset['data']['seasonal']),
        'residual': np.array(dataset['data']['residual'])
    }


def load_realworld_result(dataset_name, model_name, results_dir):
    """Load real-world dataset result from JSON file."""
    file_name = get_model_file_name(model_name)

    # Try subdirectory structure with converted file name
    result_file = Path(results_dir) / dataset_name / f"{file_name}.json"

    # If not found, try with original model name (real-world uses mixed case)
    if not result_file.exists():
        result_file = Path(results_dir) / dataset_name / f"{model_name}.json"

    # Fallback to flat structure (legacy format)
    if not result_file.exists():
        result_file = Path(results_dir) / f"{dataset_name}_{file_name}.json"

    if not result_file.exists():
        result_file = Path(results_dir) / f"{dataset_name}_{model_name}.json"

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


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_dataset_comparison(dataset_name, models, results_dir, output_dir,
                           load_func, gt_func=None, zoom=False, colorful=False, plot_type='by_dataset'):
    """
    Plot comparison of multiple models for one dataset.

    Args:
        dataset_name: Name of the dataset
        models: List of model names to compare
        results_dir: Directory containing results
        output_dir: Base directory for output figures
        load_func: Function to load results (load_synthetic_result or load_realworld_result)
        gt_func: Function to load ground truth (optional, for synthetic only)
        zoom: Whether to zoom in on seasonal/residual
        colorful: Whether to use rainbow gradients for LGTD
        plot_type: Type of plot ('by_dataset' or 'by_dataset_full')
    """
    # Load results
    results = {}
    for model in models:
        result = load_func(dataset_name, model, results_dir)
        if result is not None:
            results[model] = result

    if not results:
        print(f"  ⚠ No results found for {dataset_name}")
        return

    # Get reference data
    y_ref = list(results.values())[0]['y']
    n_points = len(y_ref)

    # Determine zoom indices
    start, end = get_zoom_indices(n_points, zoom)
    time_zoom = np.arange(start, end)
    time_full = np.arange(n_points)

    # Load ground truth if available
    gt = gt_func(dataset_name) if gt_func else None

    # Calculate shared Y-scales for seasonal
    seas_min, seas_max = np.inf, -np.inf
    for r in results.values():
        seas_chunk = r['seasonal'][start:end]
        seas_min = min(seas_min, np.min(seas_chunk))
        seas_max = max(seas_max, np.max(seas_chunk))

    seas_range = max(seas_max - seas_min, 1e-6)
    seas_ylim = (seas_min - 0.05 * seas_range, seas_max + 0.05 * seas_range)

    # Calculate shared Y-scales for residual
    res_vals = [np.min(r['residual'][start:end]) for r in results.values()]
    res_maxs = [np.max(r['residual'][start:end]) for r in results.values()]
    if gt:
        res_vals.append(np.min(gt['residual'][start:end]))
        res_maxs.append(np.max(gt['residual'][start:end]))

    res_min, res_max = min(res_vals), max(res_maxs)
    res_range = max(res_max - res_min, 1e-6)
    res_ylim = (res_min - 0.05 * res_range, res_max + 0.05 * res_range)

    # Create figure
    n_models = len(results)
    fig = plt.figure(figsize=(12, 2.5 * n_models))
    gs = gridspec.GridSpec(n_models, 3, hspace=0.3, wspace=0.3)

    for idx, (model_name, result) in enumerate(results.items()):
        color = get_model_color(model_name)
        is_lgtd = is_lgtd_variant(model_name) and colorful

        # Column 1: Original + Trend (ALWAYS FULL)
        ax1 = fig.add_subplot(gs[idx, 0])
        ax1.plot(time_full, y_ref, color='gray', linewidth=0.8, alpha=0.5)
        if is_lgtd:
            plot_rainbow_line(ax1, time_full, result['trend'], lw=1.5)
        else:
            ax1.plot(time_full, result['trend'], color=color, linewidth=1.2)
        ax1.set_ylabel(model_name, fontweight='bold')
        if idx == 0:
            ax1.set_title('Original + Trend (Full)')
        if idx == n_models - 1:
            ax1.set_xlabel('Time')

        # Column 2: Seasonal
        ax2 = fig.add_subplot(gs[idx, 1])
        if gt is not None:
            ax2.plot(time_zoom, gt['seasonal'][start:end],
                    color='gray', linewidth=1.2, alpha=0.6, label='Ground Truth')
        if is_lgtd:
            plot_rainbow_line(ax2, time_zoom, result['seasonal'][start:end], lw=1.0)
        else:
            ax2.plot(time_zoom, result['seasonal'][start:end], color=color, linewidth=1.0)
        ax2.set_ylim(seas_ylim)
        if idx == 0:
            title = 'Seasonal' + (' (Zoomed)' if zoom else '')
            ax2.set_title(title)
        if idx == n_models - 1:
            ax2.set_xlabel('Time')

        # Column 3: Residual
        ax3 = fig.add_subplot(gs[idx, 2])
        if gt is not None:
            ax3.plot(time_zoom, gt['residual'][start:end],
                    color='gray', linewidth=1.2, alpha=0.6, label='Ground Truth')
        if is_lgtd:
            plot_rainbow_line(ax3, time_zoom, result['residual'][start:end], lw=0.8, alpha=0.7)
        else:
            ax3.plot(time_zoom, result['residual'][start:end],
                    color=color, linewidth=0.8, alpha=0.7)
        ax3.axhline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        ax3.set_ylim(res_ylim)
        if idx == 0:
            title = 'Residual' + (' (Zoomed)' if zoom else '')
            ax3.set_title(title)
        if idx == n_models - 1:
            ax3.set_xlabel('Time')

    # Save figure
    output_path = create_output_path(
        output_dir,
        f"{dataset_name}_comparison",
        plot_type=plot_type,
        zoom=zoom,
        colorful=colorful
    )
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved {output_path}")


def plot_single_model_all_datasets(model_name, datasets, results_dir, output_dir,
                                   load_func, gt_func=None, zoom=False, colorful=False):
    """
    Plot single model across multiple datasets.

    Args:
        model_name: Name of the model
        datasets: List of dataset names
        results_dir: Directory containing results
        output_dir: Base directory for output figures
        load_func: Function to load results
        gt_func: Function to load ground truth (optional, for synthetic only)
        zoom: Whether to zoom in on seasonal/residual
        colorful: Whether to use rainbow gradients for LGTD
    """
    # Load results
    results = {}
    for dataset in datasets:
        result = load_func(dataset, model_name, results_dir)
        if result is not None:
            results[dataset] = result

    if not results:
        print(f"  ⚠ No results found for {model_name}")
        return

    # Calculate zoom ranges and global Y-scales
    zoom_ranges = {}
    seas_min_global, seas_max_global = np.inf, -np.inf
    res_min_global, res_max_global = np.inf, -np.inf

    for dataset_name, result in results.items():
        n_points = len(result['y'])
        start, end = get_zoom_indices(n_points, zoom)
        zoom_ranges[dataset_name] = (start, end)

        seas_chunk = result['seasonal'][start:end]
        seas_min_global = min(seas_min_global, np.min(seas_chunk))
        seas_max_global = max(seas_max_global, np.max(seas_chunk))

        res_chunk = result['residual'][start:end]
        res_min_global = min(res_min_global, np.min(res_chunk))
        res_max_global = max(res_max_global, np.max(res_chunk))

        # Include ground truth if available
        if gt_func:
            gt = gt_func(dataset_name)
            if gt:
                seas_min_global = min(seas_min_global, np.min(gt['seasonal'][start:end]))
                seas_max_global = max(seas_max_global, np.max(gt['seasonal'][start:end]))
                res_min_global = min(res_min_global, np.min(gt['residual'][start:end]))
                res_max_global = max(res_max_global, np.max(gt['residual'][start:end]))

    seas_range = max(seas_max_global - seas_min_global, 1e-6)
    seas_ylim = (seas_min_global - 0.05 * seas_range, seas_max_global + 0.05 * seas_range)

    res_range = max(res_max_global - res_min_global, 1e-6)
    res_ylim = (res_min_global - 0.05 * res_range, res_max_global + 0.05 * res_range)

    # Create figure
    n_datasets = len(results)
    fig = plt.figure(figsize=(12, 2.5 * n_datasets))
    gs = gridspec.GridSpec(n_datasets, 3, hspace=0.3, wspace=0.3)

    color = get_model_color(model_name)
    is_lgtd = is_lgtd_variant(model_name) and colorful

    for idx, (dataset_name, result) in enumerate(results.items()):
        start, end = zoom_ranges[dataset_name]
        n_points = len(result['y'])

        time_full = np.arange(n_points)
        time_zoom = np.arange(start, end)

        gt = gt_func(dataset_name) if gt_func else None

        # Col 1: Original + Trend (Always Full)
        ax1 = fig.add_subplot(gs[idx, 0])
        ax1.plot(time_full, result['y'], color='gray', linewidth=0.8, alpha=0.5)
        if is_lgtd:
            plot_rainbow_line(ax1, time_full, result['trend'], lw=1.5)
        else:
            ax1.plot(time_full, result['trend'], color=color, linewidth=1.2)
        ax1.set_ylabel(dataset_name, fontweight='bold')
        if idx == 0:
            ax1.set_title('Original + Trend (Full)')
        if idx == n_datasets - 1:
            ax1.set_xlabel('Time')

        # Col 2: Seasonal
        ax2 = fig.add_subplot(gs[idx, 1])
        if gt is not None:
            ax2.plot(time_zoom, gt['seasonal'][start:end],
                    color='gray', linewidth=1.2, alpha=0.6)
        if is_lgtd:
            plot_rainbow_line(ax2, time_zoom, result['seasonal'][start:end], lw=1.0)
        else:
            ax2.plot(time_zoom, result['seasonal'][start:end], color=color, linewidth=1.0)
        ax2.set_ylim(seas_ylim)
        if idx == 0:
            title = 'Seasonal' + (' (Zoomed)' if zoom else '')
            ax2.set_title(title)
        if idx == n_datasets - 1:
            ax2.set_xlabel('Time')

        # Col 3: Residual
        ax3 = fig.add_subplot(gs[idx, 2])
        if gt is not None:
            ax3.plot(time_zoom, gt['residual'][start:end],
                    color='gray', linewidth=1.2, alpha=0.6)
        if is_lgtd:
            plot_rainbow_line(ax3, time_zoom, result['residual'][start:end], lw=0.8, alpha=0.7)
        else:
            ax3.plot(time_zoom, result['residual'][start:end],
                    color=color, linewidth=0.8, alpha=0.7)
        ax3.axhline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        ax3.set_ylim(res_ylim)
        if idx == 0:
            title = 'Residual' + (' (Zoomed)' if zoom else '')
            ax3.set_title(title)
        if idx == n_datasets - 1:
            ax3.set_xlabel('Time')

    # Save figure
    output_path = create_output_path(
        output_dir,
        f"{model_name}_multidataset",
        plot_type='by_model',
        zoom=zoom,
        colorful=colorful
    )
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved {output_path}")
