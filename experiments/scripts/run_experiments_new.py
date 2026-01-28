#!/usr/bin/env python
"""
Run experiments using the new model-based configuration structure.

Usage:
    # Run all experiments
    python run_experiments_new.py

    # Run specific datasets
    python run_experiments_new.py --datasets synth1 synth2 synth7

    # Run specific models
    python run_experiments_new.py --models lgtd lgtd_linear lgtd_lowess

    # Run specific datasets and models
    python run_experiments_new.py --datasets synth7 synth8 synth9 --models lgtd lgtd_linear
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.runners.model_based_runner import ModelBasedExperimentRunner


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Run experiments using model-based configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--datasets',
        nargs='+',
        help='Specific datasets to run (e.g., synth1 synth2). If not specified, runs all.'
    )

    parser.add_argument(
        '--models',
        nargs='+',
        help='Specific models to run (e.g., lgtd stl). If not specified, runs all enabled models.'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress output'
    )

    args = parser.parse_args()

    print("="*70)
    print("LGTD EXPERIMENTS - Model-Based Configuration")
    print("="*70)
    print()

    # Initialize runner
    runner = ModelBasedExperimentRunner()

    # Run experiments
    try:
        results = runner.run_all_experiments(
            dataset_filter=args.datasets,
            model_filter=args.models,
            verbose=not args.quiet
        )

        if not args.quiet:
            print("\n✓ Experiments completed successfully!")

        # Print summary
        print("\nResults summary:")
        for dataset_name, dataset_results in results.items():
            print(f"\n{dataset_name}:")
            for model_name, result in dataset_results.items():
                if 'error' in result:
                    print(f"  {model_name}: ERROR - {result['error']}")
                else:
                    print(f"  {model_name}: {result['time']:.3f}s")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
