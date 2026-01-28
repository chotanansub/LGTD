#!/usr/bin/env python
"""
Script to run real-world experiments for LGTD evaluation (Table 4, Figure 5).

Runs experiments on:
- ETTh1: Electricity Transformer Temperature (hourly)
- ETTh2: Electricity Transformer Temperature (hourly)
- Sunspot: Monthly sunspot numbers

Usage:
    # Run all real-world experiments
    python run_realworld_experiments.py

    # Run specific datasets
    python run_realworld_experiments.py --datasets ETTh1 ETTh2

    # Run specific models
    python run_realworld_experiments.py --models LGTD ASTD_Online

    # Run specific datasets and models
    python run_realworld_experiments.py --datasets ETTh1 --models LGTD ASTD
"""

import sys
from pathlib import Path

# Add parent directory to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.runners.real_world_runner import RealWorldExperimentRunner
from experiments.runners.base_experiment import BaseExperiment


def main():
    """Main execution function."""
    print("="*70)
    print("LGTD REAL-WORLD EXPERIMENTS (Table 4, Figure 5)")
    print("="*70)

    # Parse command-line arguments
    parser = BaseExperiment.create_argument_parser(
        description="Run real-world experiments for LGTD evaluation"
    )
    args = parser.parse_args()

    # Configuration path (optional - runner uses model configs by default)
    config_path = args.config if args.config else None

    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = str(project_root / "experiments" / "results")

    # Initialize runner
    # Model parameters are loaded from experiments/configs/models/*.yaml
    runner = RealWorldExperimentRunner(
        config_path=config_path,
        output_dir=output_dir
    )

    # Run experiments with filtering
    # Results will be saved to:
    # - experiments/results/decompositions/real_world/{dataset}/ (JSON files)
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


if __name__ == '__main__':
    main()
