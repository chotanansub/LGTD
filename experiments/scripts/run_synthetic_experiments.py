#!/usr/bin/env python
"""
Script to run synthetic experiments for LGTD evaluation (Tables 1-3, Figures 2-4).

Usage:
    # Run all experiments
    python run_synthetic_experiments.py

    # Run specific datasets
    python run_synthetic_experiments.py --datasets synth1 synth2

    # Run specific models
    python run_synthetic_experiments.py --models LGTD ASTD_Online

    # Run specific datasets and models
    python run_synthetic_experiments.py --datasets synth1 synth2 --models LGTD ASTD
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.runners.synthetic_runner import SyntheticExperimentRunner
from experiments.runners.base_experiment import BaseExperiment


def main():
    """Main execution function."""
    print("="*70)
    print("LGTD SYNTHETIC EXPERIMENTS (Tables 1-3, Figures 2-4)")
    print("="*70)

    # Parse command-line arguments
    parser = BaseExperiment.create_argument_parser(
        description="Run synthetic experiments for LGTD evaluation"
    )
    args = parser.parse_args()

    # Configuration path
    if args.config:
        config_path = args.config
    else:
        config_path = str(project_root / "experiments" / "configs" / "synthetic_experiments.yaml")

    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = str(project_root / "experiments" / "results" / "synthetic")

    # Initialize runner
    runner = SyntheticExperimentRunner(
        config_path=config_path,
        output_dir=output_dir
    )

    # Run experiments with filtering
    # Results will be saved to:
    # - experiments/results/decompositions/synthetic/ (JSON files)
    # - experiments/results/figures/synthetic/ (plots)
    # - experiments/results/accuracy/synthetic/ (CSV files)
    try:
        runner.run_all_experiments(
            dataset_filter=args.datasets,
            method_filter=args.models,
            save_results=not args.no_save
        )

        if not args.quiet:
            print("\n✓ Experiments completed successfully!")
            print(f"✓ Results saved to: {runner.output_dir}")
    except ValueError as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
