"""
Script to run synthetic experiments for LGTD evaluation.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.runners.synthetic_runner import SyntheticExperimentRunner


def main():
    """Main execution function."""
    print("="*70)
    print("LGTD SYNTHETIC EXPERIMENTS")
    print("="*70)

    # Configuration path
    config_path = project_root / "experiments" / "configs" / "synthetic_experiments.yaml"

    # Initialize and run experiments
    runner = SyntheticExperimentRunner(
        config_path=str(config_path),
        output_dir=str(project_root / "results" / "synthetic")
    )

    runner.run_all_experiments()

    print("\n✓ Experiments completed successfully!")
    print(f"✓ Results saved to: {runner.output_dir}")


if __name__ == "__main__":
    main()
