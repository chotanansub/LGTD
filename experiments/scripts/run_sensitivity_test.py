#!/usr/bin/env python
"""
Parameter sensitivity test for LGTD.

This script evaluates the sensitivity of LGTD to the window_size and error_percentile
parameters on synthetic datasets.

Usage:
    python run_sensitivity_test.py
    python run_sensitivity_test.py --datasets synth1 synth2
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from data.synthetic.generators import generate_synthetic_data
from lgtd import lgtd
from lgtd.evaluation.metrics import compute_mse, compute_mae


class SensitivityTest:
    """Run parameter sensitivity tests for LGTD."""

    def __init__(self, output_dir: str = None):
        """
        Initialize sensitivity test runner.

        Args:
            output_dir: Directory to save results (default: experiments/results/sensitivity)
        """
        if output_dir is None:
            output_dir = str(project_root / "experiments" / "results" / "sensitivity")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Parameter ranges
        self.window_sizes = list(range(2, 30))  
        self.percentile_errors = list(range(1, 100))

    def load_dataset_config(self, dataset_name: str) -> Dict[str, Any]:
        """
        Load dataset configuration from datasets.yaml.

        Args:
            dataset_name: Name of the dataset (e.g., 'synth1')

        Returns:
            Dataset configuration dictionary
        """
        config_path = project_root / "experiments" / "configs" / "datasets.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Find the dataset
        for dataset in config['datasets']:
            if dataset['name'] == dataset_name:
                return dataset

        raise ValueError(f"Dataset {dataset_name} not found in configuration")

    def generate_dataset(self, dataset_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a synthetic dataset from configuration.

        Args:
            dataset_config: Dataset configuration dictionary

        Returns:
            Dataset with ground truth components
        """
        data = generate_synthetic_data(
            n=dataset_config.get('n_samples', 2000),
            trend_type=dataset_config['trend_type'],
            seasonality_type=dataset_config['seasonality_type'],
            seasonal_params=dataset_config.get('seasonal_params', {}),
            residual_std=dataset_config.get('noise_std', 1.0),
            seed=dataset_config.get('seed', 69)
        )

        return data

    def run_lgtd_with_params(
        self,
        data: np.ndarray,
        window_size: int,
        error_percentile: int
    ) -> Dict[str, np.ndarray]:
        """
        Run LGTD with specific parameters.

        Args:
            data: Time series data
            window_size: Window size parameter
            error_percentile: Error percentile parameter

        Returns:
            Decomposition result dictionary

        Raises:
            Exception: If decomposition fails
        """
        model = lgtd(
            window_size=window_size,
            error_percentile=error_percentile,
            verbose=False
        )
        result = model.fit_transform(data)

        return {
            'trend': result.trend,
            'seasonal': result.seasonal,
            'residual': result.residual
        }

    def test_dataset(self, dataset_name: str) -> pd.DataFrame:
        """
        Run sensitivity test on a single dataset.

        Args:
            dataset_name: Name of the synthetic dataset

        Returns:
            DataFrame with results for all parameter combinations
        """
        print(f"\n{'='*70}")
        print(f"Sensitivity Test: {dataset_name}")
        print(f"{'='*70}")

        # Load and generate dataset
        dataset_config = self.load_dataset_config(dataset_name)
        data = self.generate_dataset(dataset_config)

        print(f"Dataset: {dataset_name}")
        print(f"Trend Type: {dataset_config['trend_type']}")
        print(f"Seasonality Type: {dataset_config['seasonality_type']}")
        print(f"Samples: {len(data['y'])}")
        print(f"\nTesting {len(self.window_sizes)} window sizes × "
              f"{len(self.percentile_errors)} percentile values = "
              f"{len(self.window_sizes) * len(self.percentile_errors)} combinations")

        # Store results
        results = []
        total_tests = len(self.window_sizes) * len(self.percentile_errors)
        current_test = 0

        # Test all parameter combinations
        for window_size in self.window_sizes:
            for percentile_error in self.percentile_errors:
                current_test += 1

                if current_test % 10 == 0 or current_test == total_tests:
                    print(f"  Progress: {current_test}/{total_tests} "
                          f"({100*current_test/total_tests:.1f}%)")

                result_row = {
                    'dataset': dataset_name,
                    'window_size': window_size,
                    'percentile_error': percentile_error,
                    'valid': 0,
                    'mae': np.nan,
                    'mse': np.nan
                }

                try:
                    # Run LGTD
                    result = self.run_lgtd_with_params(
                        data['y'],
                        window_size,
                        percentile_error
                    )

                    # Compute metrics
                    mae_dict = compute_mae(data, result)
                    mse_dict = compute_mse(data, result)

                    # Compute overall metrics as mean of components
                    mae_overall = np.mean([mae_dict[k] for k in mae_dict.keys()])
                    mse_overall = np.mean([mse_dict[k] for k in mse_dict.keys()])

                    # Update result
                    result_row['valid'] = 1
                    result_row['mae'] = mae_overall
                    result_row['mse'] = mse_overall

                except Exception as e:
                    # Keep valid=0 and mae/mse as NaN
                    if current_test % 10 == 0:
                        print(f"    ✗ Error with window_size={window_size}, "
                              f"percentile_error={percentile_error}: {str(e)[:50]}")

                results.append(result_row)

        # Create DataFrame
        df = pd.DataFrame(results)

        # Print summary
        valid_count = df['valid'].sum()
        print(f"\n{'='*70}")
        print(f"Results Summary for {dataset_name}")
        print(f"{'='*70}")
        print(f"Total tests: {len(df)}")
        print(f"Valid results: {valid_count} ({100*valid_count/len(df):.1f}%)")
        print(f"Failed results: {len(df) - valid_count}")

        if valid_count > 0:
            valid_df = df[df['valid'] == 1]
            print(f"\nMAE Statistics:")
            print(f"  Min: {valid_df['mae'].min():.6f}")
            print(f"  Max: {valid_df['mae'].max():.6f}")
            print(f"  Mean: {valid_df['mae'].mean():.6f}")
            print(f"  Median: {valid_df['mae'].median():.6f}")
            print(f"\nMSE Statistics:")
            print(f"  Min: {valid_df['mse'].min():.6f}")
            print(f"  Max: {valid_df['mse'].max():.6f}")
            print(f"  Mean: {valid_df['mse'].mean():.6f}")
            print(f"  Median: {valid_df['mse'].median():.6f}")

            # Best parameters
            best_mae_idx = valid_df['mae'].idxmin()
            best_mse_idx = valid_df['mse'].idxmin()

            print(f"\nBest parameters (by MAE):")
            print(f"  window_size={df.loc[best_mae_idx, 'window_size']:.0f}, "
                  f"percentile_error={df.loc[best_mae_idx, 'percentile_error']:.0f}")
            print(f"  MAE={df.loc[best_mae_idx, 'mae']:.6f}, "
                  f"MSE={df.loc[best_mae_idx, 'mse']:.6f}")

            print(f"\nBest parameters (by MSE):")
            print(f"  window_size={df.loc[best_mse_idx, 'window_size']:.0f}, "
                  f"percentile_error={df.loc[best_mse_idx, 'percentile_error']:.0f}")
            print(f"  MAE={df.loc[best_mse_idx, 'mae']:.6f}, "
                  f"MSE={df.loc[best_mse_idx, 'mse']:.6f}")

        return df

    def save_results(self, dataset_name: str, df: pd.DataFrame) -> None:
        """
        Save results to CSV file.

        Args:
            dataset_name: Name of the dataset
            df: Results DataFrame
        """
        output_path = self.output_dir / f"{dataset_name}_sensitivity.csv"
        df.to_csv(output_path, index=False)
        print(f"\n✓ Results saved to: {output_path}")

    def run_all_tests(self, dataset_names: List[str] = None) -> None:
        """
        Run sensitivity tests on all or specified synthetic datasets.

        Args:
            dataset_names: List of dataset names to test (None = all synthetic datasets)
        """
        print("\n" + "="*70)
        print("LGTD PARAMETER SENSITIVITY TEST")
        print("="*70)
        print(f"Window sizes: {self.window_sizes}")
        print(f"Percentile errors: {self.percentile_errors}")

        # Load all synthetic datasets if not specified
        if dataset_names is None:
            config_path = project_root / "experiments" / "configs" / "datasets.yaml"
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            dataset_names = [d['name'] for d in config['datasets']
                           if d['name'].startswith('synth')]

        print(f"Datasets to test: {dataset_names}")

        # Run tests
        all_results = {}
        for dataset_name in dataset_names:
            try:
                df = self.test_dataset(dataset_name)
                self.save_results(dataset_name, df)
                all_results[dataset_name] = df
            except Exception as e:
                print(f"\n✗ Failed to process {dataset_name}: {str(e)}")
                continue

        print("\n" + "="*70)
        print("ALL SENSITIVITY TESTS COMPLETED")
        print("="*70)
        print(f"Results saved to: {self.output_dir}")


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run LGTD parameter sensitivity tests on synthetic datasets"
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        help='Specific datasets to test (e.g., synth1 synth2). Default: all synthetic datasets'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for results (default: experiments/results/sensitivity)'
    )

    args = parser.parse_args()

    # Create and run sensitivity test
    tester = SensitivityTest(output_dir=args.output_dir)
    tester.run_all_tests(dataset_names=args.datasets)


if __name__ == "__main__":
    main()
