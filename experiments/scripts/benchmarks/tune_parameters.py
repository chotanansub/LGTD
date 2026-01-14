#!/usr/bin/env python3
"""
Parameter tuning for all decomposition methods using cross-validation.

This script performs fair parameter optimization across all methods:
1. Uses k-fold cross-validation across datasets
2. Tunes each method with the same approach
3. Reports optimized parameters for all methods
4. Updates config files with tuned parameters

Usage:
    python scripts/tune_all_methods.py --models LGTD STL FastRobustSTL
    python scripts/tune_all_methods.py --all  # Tune all methods
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple
from itertools import product

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiments.runners.experiment_runner import ExperimentRunner
from lgtd.evaluation.metrics import compute_mae


# Parameter search spaces for each method
PARAM_GRIDS = {
    'LGTD': {
        'window_size': [3, 5, 7],
        'error_percentile': [40, 50, 60, 70],
        'lowess_frac': [0.05, 0.1, 0.15, 0.2],
        'threshold_r2': [0.85, 0.90, 0.92, 0.95]
    },
    'STL': {
        'seasonal': [7, 11, 13, 15, 21],
        'robust': [False, True]
    },
    'FastRobustSTL': {
        'reg1': [0.5, 1.0, 2.0, 5.0],
        'reg2': [5.0, 10.0, 20.0],
        'K': [2, 3, 4],
        'H': [3, 5, 7]
    },
    'STR': {
        'trend_lambda': [50.0, 100.0, 200.0, 500.0],
        'seasonal_lambda': [0.5, 1.0, 2.0, 5.0],
        'robust': [False, True]
    },
    'ASTD': {
        'seasonality_smoothing': [0.5, 0.6, 0.7, 0.8, 0.9]
    },
    'ASTD_Online': {
        'seasonality_smoothing': [0.5, 0.6, 0.7, 0.8, 0.9],
        'init_ratio': [0.2, 0.3, 0.4]
    },
    'OnlineSTL': {
        'lam': [0.1, 0.3, 0.5, 0.7],
        'init_window_ratio': [0.2, 0.3, 0.4]
    },
    'OneShotSTL': {
        'init_ratio': [0.2, 0.3, 0.4, 0.5],
        'shift_window': [0, 1, 2]
    }
}


def generate_param_combinations(param_grid: Dict[str, List]) -> List[Dict[str, Any]]:
    """Generate all combinations of parameters from grid."""
    if not param_grid:
        return [{}]

    keys = list(param_grid.keys())
    values = list(param_grid.values())

    combinations = []
    for combo in product(*values):
        combinations.append(dict(zip(keys, combo)))

    return combinations


def evaluate_params(
    runner: ExperimentRunner,
    model_name: str,
    params: Dict[str, Any],
    train_datasets: List[str],
    config_template: Dict
) -> float:
    """
    Evaluate a parameter configuration on training datasets.

    Returns average MAE across all components.
    """
    total_mae = 0.0
    num_datasets = 0

    for dataset_name in train_datasets:
        try:
            # Load dataset
            config = runner.configs[dataset_name]
            data_dict = runner._load_dataset(config)

            # Merge template params with test params
            model_params = {**config_template, **params}

            # Run model
            result = runner._run_model(
                model_name=model_name,
                data=data_dict['y'],
                params=model_params
            )

            if result is None:
                continue

            # Calculate MAE
            init_point = result.get('init_window_size', int(len(data_dict['y']) * 0.3))

            gt = {
                'trend': data_dict['trend'][init_point:],
                'seasonal': data_dict['seasonal'][init_point:],
                'residual': data_dict['residual'][init_point:]
            }

            res = {
                'trend': result['trend'][init_point:],
                'seasonal': result['seasonal'][init_point:],
                'residual': result['residual'][init_point:]
            }

            mae = compute_mae(gt, res)
            avg_mae = np.mean([mae['trend'], mae['seasonal'], mae['residual']])

            total_mae += avg_mae
            num_datasets += 1

        except Exception as e:
            print(f"      Error on {dataset_name}: {e}")
            continue

    if num_datasets == 0:
        return float('inf')

    return total_mae / num_datasets


def tune_method_cv(
    runner: ExperimentRunner,
    model_name: str,
    param_grid: Dict[str, List],
    all_datasets: List[str],
    n_folds: int = 3,
    max_combinations: int = 100
) -> Tuple[Dict[str, Any], float]:
    """
    Tune method parameters using k-fold cross-validation.

    Args:
        runner: ExperimentRunner instance
        model_name: Name of model to tune
        param_grid: Parameter search space
        all_datasets: List of all dataset names
        n_folds: Number of CV folds
        max_combinations: Maximum parameter combinations to try

    Returns:
        (best_params, best_score)
    """
    print(f"\n{'='*70}")
    print(f"Tuning {model_name}")
    print(f"{'='*70}")

    # Get base config template for this model
    sample_config = runner.configs[all_datasets[0]]
    config_template = sample_config['models'][model_name]['params'].copy()

    # Generate parameter combinations
    all_combinations = generate_param_combinations(param_grid)

    # Limit combinations if too many
    if len(all_combinations) > max_combinations:
        print(f"  Limiting search to {max_combinations} random combinations (from {len(all_combinations)} total)")
        import random
        random.seed(42)
        combinations_to_try = random.sample(all_combinations, max_combinations)
    else:
        combinations_to_try = all_combinations

    print(f"  Testing {len(combinations_to_try)} parameter combinations")
    print(f"  Using {n_folds}-fold cross-validation on {len(all_datasets)} datasets")

    # Prepare CV folds
    np.random.seed(42)
    fold_size = len(all_datasets) // n_folds
    dataset_indices = np.random.permutation(len(all_datasets))

    best_params = None
    best_avg_score = float('inf')

    for idx, params in enumerate(combinations_to_try, 1):
        print(f"\n  [{idx}/{len(combinations_to_try)}] Testing: {params}")

        fold_scores = []

        # K-fold CV
        for fold in range(n_folds):
            # Split into train/val
            val_indices = dataset_indices[fold * fold_size:(fold + 1) * fold_size]
            train_indices = np.concatenate([
                dataset_indices[:fold * fold_size],
                dataset_indices[(fold + 1) * fold_size:]
            ])

            train_datasets = [all_datasets[i] for i in train_indices]

            # Evaluate on training fold
            score = evaluate_params(
                runner, model_name, params, train_datasets, config_template
            )

            fold_scores.append(score)
            print(f"    Fold {fold + 1}: MAE = {score:.4f}")

        # Average across folds
        avg_score = np.mean(fold_scores)
        print(f"    Average MAE: {avg_score:.4f}")

        if avg_score < best_avg_score:
            best_avg_score = avg_score
            best_params = params
            print(f"    ✨ New best!")

    print(f"\n{'='*70}")
    if best_params is None:
        print(f"⚠️  No valid parameters found for {model_name}")
        print(f"All parameter combinations failed or returned inf")
        print(f"{'='*70}")
        return None, float('inf')

    print(f"Best parameters for {model_name}:")
    print(f"{'='*70}")
    for key, val in best_params.items():
        print(f"  {key}: {val}")
    print(f"\nCV MAE: {best_avg_score:.4f}")

    return best_params, best_avg_score


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Tune parameters for all methods')
    parser.add_argument(
        '--models', '-m',
        nargs='+',
        help='Models to tune (default: all)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Tune all methods'
    )
    parser.add_argument(
        '--n-folds',
        type=int,
        default=3,
        help='Number of CV folds (default: 3)'
    )
    parser.add_argument(
        '--max-combinations',
        type=int,
        default=100,
        help='Max parameter combinations to try (default: 100)'
    )
    parser.add_argument(
        '--update-configs',
        action='store_true',
        help='Update config files with tuned parameters'
    )

    args = parser.parse_args()

    print("="*70)
    print("Parameter Tuning for All Methods")
    print("="*70)
    print("\nThis script performs fair cross-validation tuning for all methods.")
    print()

    # Initialize runner
    runner = ExperimentRunner(save_decompositions=False)

    # Get all datasets
    all_datasets = sorted([name for name in runner.configs.keys() if name.startswith('synth')])
    print(f"Using {len(all_datasets)} datasets: {', '.join(all_datasets)}")

    # Determine which models to tune
    if args.all:
        models_to_tune = list(PARAM_GRIDS.keys())
    elif args.models:
        models_to_tune = [m for m in args.models if m in PARAM_GRIDS]
    else:
        models_to_tune = ['LGTD']  # Default to just LGTD

    print(f"\nTuning {len(models_to_tune)} method(s): {', '.join(models_to_tune)}")
    print()

    # Tune each method
    results_file = Path('experiments/results/tuned_parameters.json')
    results_file.parent.mkdir(parents=True, exist_ok=True)

    # Load existing results if available
    if results_file.exists():
        with open(results_file, 'r') as f:
            tuned_params = json.load(f)
        print(f"\nLoaded existing results from {results_file}")
        print(f"Already tuned: {', '.join(tuned_params.keys())}\n")
    else:
        tuned_params = {}

    for model_name in models_to_tune:
        # Skip if already tuned
        if model_name in tuned_params:
            print(f"\n⏭️  Skipping {model_name} (already tuned)")
            continue

        param_grid = PARAM_GRIDS[model_name]

        best_params, best_score = tune_method_cv(
            runner=runner,
            model_name=model_name,
            param_grid=param_grid,
            all_datasets=all_datasets,
            n_folds=args.n_folds,
            max_combinations=args.max_combinations
        )

        tuned_params[model_name] = {
            'params': best_params,
            'cv_mae': best_score
        }

        # Save after each method
        with open(results_file, 'w') as f:
            json.dump(tuned_params, f, indent=2)
        print(f"\n💾 Saved intermediate results to {results_file}")

    # Save final results
    results_file.parent.mkdir(parents=True, exist_ok=True)

    with open(results_file, 'w') as f:
        json.dump(tuned_params, f, indent=2)

    print(f"\n{'='*70}")
    print("Tuning Complete!")
    print(f"{'='*70}")
    print(f"\nResults saved to: {results_file}")

    # Update configs if requested
    if args.update_configs:
        print("\nUpdating config files with tuned parameters...")

        for dataset_name in all_datasets:
            config_file = runner.config_dir / f"{dataset_name}_params.json"

            with open(config_file, 'r') as f:
                config = json.load(f)

            # Update parameters for tuned models
            for model_name, tuned_data in tuned_params.items():
                if model_name in config['models'] and tuned_data['params'] is not None:
                    config['models'][model_name]['params'].update(tuned_data['params'])

            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)

            print(f"  Updated: {config_file}")

        print("\n✅ Config files updated with tuned parameters!")
        print("   Run experiments again to see improved results.")
    else:
        print("\n💡 To update config files, run with --update-configs flag")

    print()


if __name__ == '__main__':
    main()
