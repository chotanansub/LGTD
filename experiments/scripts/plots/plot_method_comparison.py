#!/usr/bin/env python3
"""
Generate method comparison plots for each dataset.

Creates one figure per dataset with subplots showing decomposition results
from all methods (LGTD, STL, ASTD, OnlineSTL, OneShotSTL, etc.).
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiments.runners.experiment_runner import ExperimentRunner


def load_dataset(dataset_path: Path) -> Dict[str, np.ndarray]:
    """Load dataset from JSON file."""
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    # Data is nested under 'data' key
    data = dataset['data']

    return {
        'y': np.array(data['y']),
        'trend': np.array(data['trend']),
        'seasonal': np.array(data['seasonal']),
        'residual': np.array(data['residual'])
    }


def plot_method_comparison(
    dataset_name: str,
    data: Dict[str, np.ndarray],
    results: Dict[str, Dict[str, np.ndarray]],
    init_point: int = 0,
    save_path: Optional[Path] = None,
    show: bool = False
):
    """
    Create comparison plot showing all methods for one dataset.

    Args:
        dataset_name: Name of the dataset
        data: Ground truth data dictionary
        results: Dictionary of {model_name: decomposition_result}
        init_point: Index where initialization ends
        save_path: Path to save the figure
        show: Whether to display the plot
    """
    # Filter out models that failed
    valid_results = {name: res for name, res in results.items() if res is not None}

    if not valid_results:
        print(f"⚠️  No valid results for {dataset_name}")
        return

    n_methods = len(valid_results)
    n_components = 4  # Original, Trend, Seasonal, Residual

    # Create figure with subplots: rows = methods, cols = components
    fig, axes = plt.subplots(n_methods, n_components, figsize=(20, 4*n_methods))
    fig.suptitle(f'{dataset_name} - Method Comparison', fontsize=18, fontweight='bold', y=0.995)

    # Ensure axes is 2D even with single method
    if n_methods == 1:
        axes = axes.reshape(1, -1)

    time = np.arange(len(data['y']))

    # Color scheme
    colors = {
        'gt': '#2563eb',      # Blue for ground truth
        'est': '#dc2626',     # Red for estimated
        'original': '#000000' # Black for original
    }

    # Plot each method
    for i, (method_name, result) in enumerate(sorted(valid_results.items())):
        # Column 0: Original Signal
        ax = axes[i, 0]
        if init_point > 0:
            ax.axvspan(0, init_point, alpha=0.15, color='gray', zorder=0)
        ax.plot(time, data['y'], color=colors['original'], linewidth=1.5, label='Original', alpha=0.8)
        ax.set_ylabel(method_name, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
        if i == 0:
            ax.set_title('Original Signal', fontsize=12, fontweight='bold')
        if i == n_methods - 1:
            ax.set_xlabel('Time', fontsize=10)

        # Column 1: Trend
        ax = axes[i, 1]
        if init_point > 0:
            ax.axvspan(0, init_point, alpha=0.15, color='gray', zorder=0)
        ax.plot(time, data['trend'], color=colors['gt'], linewidth=1.5,
                label='Ground Truth', alpha=0.7, linestyle='--')
        ax.plot(time, result['trend'], color=colors['est'], linewidth=1.5,
                label=f'{method_name}', alpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
        if i == 0:
            ax.set_title('Trend', fontsize=12, fontweight='bold')
        if i == n_methods - 1:
            ax.set_xlabel('Time', fontsize=10)

        # Column 2: Seasonal
        ax = axes[i, 2]
        if init_point > 0:
            ax.axvspan(0, init_point, alpha=0.15, color='gray', zorder=0)
        ax.plot(time, data['seasonal'], color=colors['gt'], linewidth=1.5,
                label='Ground Truth', alpha=0.7, linestyle='--')
        ax.plot(time, result['seasonal'], color=colors['est'], linewidth=1.5,
                label=f'{method_name}', alpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
        if i == 0:
            ax.set_title('Seasonal', fontsize=12, fontweight='bold')
        if i == n_methods - 1:
            ax.set_xlabel('Time', fontsize=10)

        # Column 3: Residual
        ax = axes[i, 3]
        if init_point > 0:
            ax.axvspan(0, init_point, alpha=0.15, color='gray', zorder=0)
        ax.plot(time, data['residual'], color=colors['gt'], linewidth=1.5,
                label='Ground Truth', alpha=0.7, linestyle='--')
        ax.plot(time, result['residual'], color=colors['est'], linewidth=1.5,
                label=f'{method_name}', alpha=0.9)
        ax.axhline(y=0, color='black', linestyle=':', alpha=0.5, linewidth=1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
        if i == 0:
            ax.set_title('Residual', fontsize=12, fontweight='bold')
        if i == n_methods - 1:
            ax.set_xlabel('Time', fontsize=10)

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def load_cached_decomposition(
    dataset_name: str,
    model_name: str,
    cache_dir: str = "results/synthetic/decompositions"
) -> Optional[Dict[str, np.ndarray]]:
    """Load cached decomposition from disk."""
    cache_path = Path(cache_dir) / dataset_name / f"{model_name}.npz"
    if cache_path.exists():
        data = np.load(cache_path)
        return {
            'trend': data['trend'],
            'seasonal': data['seasonal'],
            'residual': data['residual']
        }
    return None


def save_decomposition(
    dataset_name: str,
    model_name: str,
    result: Dict[str, np.ndarray],
    cache_dir: str = "results/synthetic/decompositions"
):
    """Save decomposition to disk for faster loading later."""
    cache_path = Path(cache_dir) / dataset_name
    cache_path.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        cache_path / f"{model_name}.npz",
        trend=result['trend'],
        seasonal=result['seasonal'],
        residual=result['residual']
    )


def generate_all_comparison_plots(
    datasets: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    output_dir: str = "results/synthetic/comparison_plots",
    init_ratio: float = 0.3,
    use_cached: bool = True,
    cache_dir: str = "results/synthetic/decompositions"
):
    """
    Generate method comparison plots for all datasets.

    Args:
        datasets: List of dataset names (None = all)
        models: List of model names (None = all enabled)
        output_dir: Directory to save comparison plots
        init_ratio: Initialization ratio for online models (default: 0.3)
        use_cached: Use cached decompositions from disk (default: True, much faster!)
        cache_dir: Directory containing cached decomposition .npz files
    """
    print("="*70)
    print("Generating Method Comparison Plots")
    print("="*70)

    runner = ExperimentRunner()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get list of datasets to process
    if datasets is None:
        dataset_configs = sorted(runner.config_dir.glob('synth*_params.json'))
        datasets = [cfg.stem.replace('_params', '') for cfg in dataset_configs]

    print(f"\n📊 Processing {len(datasets)} dataset(s)")
    if models:
        print(f"🔧 Models: {', '.join(models)}")
    print(f"💾 Output: {output_path}")
    if use_cached:
        print(f"⚡ Using cached decompositions from {cache_dir} (faster)")
    print()

    for dataset_name in datasets:
        print(f"\n{'='*70}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*70}")

        # Load dataset configuration
        config_path = runner.config_dir / f"{dataset_name}_params.json"
        if not config_path.exists():
            print(f"❌ Config not found: {config_path}")
            continue

        with open(config_path, 'r') as f:
            config = json.load(f)

        # Load ground truth data
        data_path = Path(config['dataset']['path'])
        if not data_path.exists():
            print(f"❌ Data not found: {data_path}")
            continue

        data = load_dataset(data_path)
        init_point = int(len(data['y']) * init_ratio)

        # Determine which models to run
        enabled_models = [
            name for name, cfg in config['models'].items()
            if cfg.get('enabled', False)
        ]
        if models:
            models_to_run = [m for m in models if m in enabled_models]
        else:
            models_to_run = enabled_models

        print(f"\n🔧 Running {len(models_to_run)} model(s):")
        for model in models_to_run:
            print(f"   • {model}")

        # Run all models and collect results
        results = {}
        for model_name in models_to_run:
            try:
                # Try to load cached decomposition first
                if use_cached:
                    cached_result = load_cached_decomposition(dataset_name, model_name, cache_dir)
                    if cached_result is not None:
                        print(f"\n   ⚡ Loaded cached {model_name}...", end=' ')
                        results[model_name] = cached_result
                        print("✅")
                        continue

                print(f"\n   Running {model_name}...", end=' ')

                # Run decomposition
                model_params = config['models'][model_name]['params']
                result = runner._run_model(
                    model_name=model_name,
                    data=data['y'],
                    params=model_params
                )

                if result is not None:
                    results[model_name] = result
                    print("✅")

                    # Save to cache for future use
                    save_decomposition(dataset_name, model_name, result, cache_dir)
                else:
                    print("❌ Failed")

            except Exception as e:
                print(f"❌ Error: {str(e)}")

        # Generate comparison plot
        if results:
            save_path = output_path / f"{dataset_name}_comparison.png"
            plot_method_comparison(
                dataset_name=dataset_name,
                data=data,
                results=results,
                init_point=init_point,
                save_path=save_path,
                show=False
            )
            print(f"\n✅ Generated comparison plot: {save_path}")
        else:
            print(f"\n⚠️  No valid results to plot for {dataset_name}")

    print(f"\n{'='*70}")
    print(f"✅ Comparison plots saved to: {output_path}")
    print(f"{'='*70}\n")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate method comparison plots for each dataset'
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        help='Dataset names to process (default: all)'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        help='Model names to compare (default: all enabled)'
    )
    parser.add_argument(
        '--output-dir',
        default='experiments/results/synthetic/comparison_plots',
        help='Output directory for plots (default: experiments/results/synthetic/comparison_plots)'
    )
    parser.add_argument(
        '--init-ratio',
        type=float,
        default=0.3,
        help='Initialization ratio for online models (default: 0.3)'
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Do not use cached decompositions (slower)'
    )
    parser.add_argument(
        '--cache-dir',
        default='experiments/results/synthetic/decompositions',
        help='Directory for cached decomposition files (default: experiments/results/synthetic/decompositions)'
    )

    args = parser.parse_args()

    generate_all_comparison_plots(
        datasets=args.datasets,
        models=args.models,
        output_dir=args.output_dir,
        init_ratio=args.init_ratio,
        use_cached=not args.no_cache,
        cache_dir=args.cache_dir
    )


if __name__ == '__main__':
    main()
