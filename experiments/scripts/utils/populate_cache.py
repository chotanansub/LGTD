#!/usr/bin/env python3
"""
Populate decomposition cache by running experiments.

This script runs experiments and saves the decomposition results to disk,
which makes generating comparison plots much faster (especially for slow methods
like FastRobustSTL).

Usage:
    # Populate cache for all datasets and models
    python scripts/populate_decomposition_cache.py

    # Populate cache for specific datasets
    python scripts/populate_decomposition_cache.py --datasets synth1 synth2

    # Populate cache for specific models
    python scripts/populate_decomposition_cache.py --models LGTD STL FastRobustSTL
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiments.runners.experiment_runner import ExperimentRunner


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Populate decomposition cache for faster plot generation'
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        help='Dataset names to process (default: all)'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        help='Model names to process (default: all enabled)'
    )
    parser.add_argument(
        '--config-dir',
        default='experiments/configs/dataset_params',
        help='Directory containing parameter JSON files'
    )
    parser.add_argument(
        '--results-dir',
        default='experiments/results/synthetic',
        help='Directory to save results and decompositions'
    )

    args = parser.parse_args()

    print("="*70)
    print("Populating Decomposition Cache")
    print("="*70)
    print(f"\nThis will run experiments and save decomposition arrays to:")
    print(f"  {args.results_dir}/decompositions/")
    print(f"\nThis makes generating comparison plots much faster!")
    print()

    # Create runner with decomposition saving enabled
    runner = ExperimentRunner(
        config_dir=args.config_dir,
        results_dir=args.results_dir,
        save_decompositions=True  # Enable decomposition caching
    )

    # Run experiments (this will automatically save decompositions)
    runner.run_experiment(
        datasets=args.datasets,
        models=args.models,
        save_results=True,
        verbose=True
    )

    print(f"\n{'='*70}")
    print("Cache Population Complete!")
    print(f"{'='*70}")
    print(f"\nDecompositions saved to: {runner.decomp_dir}")
    print(f"\nNow you can run generate_method_comparison_plots.py much faster:")
    print(f"  python scripts/generate_method_comparison_plots.py")
    print()


if __name__ == '__main__':
    main()
