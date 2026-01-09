"""
Script to run benchmark comparisons between decomposition methods.
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.runners.synthetic_runner import SyntheticExperimentRunner


def generate_comparison_table(results_dir: Path) -> pd.DataFrame:
    """
    Generate comprehensive comparison table from results.

    Args:
        results_dir: Directory containing result CSV files

    Returns:
        Combined DataFrame with all results
    """
    all_results = []

    for result_file in results_dir.glob("*_results.csv"):
        df = pd.read_csv(result_file)
        dataset_name = result_file.stem.replace('_results', '')
        df['dataset'] = dataset_name
        all_results.append(df)

    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        return combined_df
    else:
        return pd.DataFrame()


def print_comparison_table(df: pd.DataFrame) -> None:
    """
    Print formatted comparison table.

    Args:
        df: Combined results DataFrame
    """
    print("\n" + "="*100)
    print("COMPREHENSIVE BENCHMARK COMPARISON")
    print("="*100)

    for metric in ['MSE', 'MAE']:
        print(f"\n{metric}:")
        print("-"*100)

        metric_df = df[df['metric'] == metric]

        if not metric_df.empty:
            pivot = metric_df.pivot_table(
                index=['dataset', 'model'],
                values=['trend', 'seasonal', 'residual'],
                aggfunc='first'
            )
            print(pivot.to_string())

    print("="*100)


def main():
    """Main execution function."""
    print("="*70)
    print("LGTD BENCHMARK COMPARISON")
    print("="*70)

    # Configuration path
    config_path = project_root / "experiments" / "configs" / "synthetic_experiments.yaml"
    results_dir = project_root / "results" / "synthetic"

    # Run experiments
    runner = SyntheticExperimentRunner(
        config_path=str(config_path),
        output_dir=str(results_dir)
    )

    runner.run_all_experiments()

    # Generate comparison table
    print("\n" + "="*70)
    print("GENERATING COMPARISON TABLE")
    print("="*70)

    combined_df = generate_comparison_table(results_dir)

    if not combined_df.empty:
        # Print comparison
        print_comparison_table(combined_df)

        # Save combined results
        output_file = results_dir / "benchmark_comparison.csv"
        combined_df.to_csv(output_file, index=False)
        print(f"\n✓ Comprehensive results saved to: {output_file}")
    else:
        print("⚠ No results found to compare")

    print("\n✓ Benchmark comparison completed!")


if __name__ == "__main__":
    main()
