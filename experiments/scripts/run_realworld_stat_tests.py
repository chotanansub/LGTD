#!/usr/bin/env python
"""
Statistical testing script for real-world time-series decomposition results.

This script performs:
1. Seasonality validation using Kruskal-Wallis H-test with adaptive phase binning
2. Residual validation using Ljung-Box test at configurable lags

Results are saved to CSV files in experiments/results/stat_tests/real_world/

Usage:
    # Run all statistical tests on all datasets and models
    python run_realworld_stat_tests.py

    # Run tests on specific datasets
    python run_realworld_stat_tests.py --datasets ETTh1 ETTh2

    # Run tests on specific models
    python run_realworld_stat_tests.py --models LGTD ASTD

    # Configure test parameters
    python run_realworld_stat_tests.py --num_bins 12 --ljung_box_lags 10,20,30
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.stats import kruskal
from statsmodels.stats.diagnostic import acorr_ljungbox


# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class RealWorldStatisticalTester:
    """
    Statistical tester for real-world time-series decomposition results.

    Performs seasonality validation (Kruskal-Wallis) and residual validation (Ljung-Box).
    """

    def __init__(
        self,
        decomposition_dir: str,
        output_dir: str,
        num_bins: int = 12,
        ljung_box_lags: List[int] = None
    ):
        """
        Initialize the statistical tester.

        Args:
            decomposition_dir: Directory containing decomposition results
            output_dir: Directory to save statistical test results
            num_bins: Number of adaptive phase bins for seasonality test
            ljung_box_lags: List of lags for Ljung-Box test (default: [10, 20, 30])
        """
        self.decomposition_dir = Path(decomposition_dir)
        self.output_dir = Path(output_dir)
        self.num_bins = num_bins
        self.ljung_box_lags = ljung_box_lags or [10, 20, 30]

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Results storage
        self.seasonality_results = []
        self.residual_results = []

    def detect_adaptive_bins(self, seasonal: np.ndarray, num_bins: int) -> np.ndarray:
        """
        Create adaptive phase bins for seasonal component without assuming fixed period.

        Groups seasonal values into bins based on their position in the emergent cycle.

        Args:
            seasonal: Seasonal component array
            num_bins: Number of bins to create

        Returns:
            Array of bin indices for each point
        """
        n = len(seasonal)

        # Method 1: Use position-based binning (treats as emergent cyclic structure)
        # Divide the time series into equal-sized bins
        bin_indices = np.floor(np.linspace(0, num_bins, n, endpoint=False)).astype(int)

        # Ensure bin indices are within valid range [0, num_bins-1]
        bin_indices = np.clip(bin_indices, 0, num_bins - 1)

        return bin_indices

    def compute_effect_size_eta_squared(
        self,
        groups: List[np.ndarray],
        h_statistic: float,
        n: int
    ) -> float:
        """
        Compute eta-squared (η²) effect size for Kruskal-Wallis test.

        η² = (H - k + 1) / (n - k)
        where H is the test statistic, k is number of groups, n is total sample size

        Args:
            groups: List of arrays, one per group
            h_statistic: Kruskal-Wallis H statistic
            n: Total sample size

        Returns:
            Eta-squared effect size
        """
        k = len(groups)
        if n <= k:
            return 0.0

        eta_squared = (h_statistic - k + 1) / (n - k)
        # Clip to [0, 1] range
        eta_squared = max(0.0, min(1.0, eta_squared))

        return eta_squared

    def compute_effect_size_epsilon_squared(
        self,
        groups: List[np.ndarray],
        h_statistic: float,
        n: int
    ) -> float:
        """
        Compute epsilon-squared (ε²) effect size for Kruskal-Wallis test.

        ε² = H / ((n² - 1) / (n + 1))

        Args:
            groups: List of arrays, one per group
            h_statistic: Kruskal-Wallis H statistic
            n: Total sample size

        Returns:
            Epsilon-squared effect size
        """
        if n <= 1:
            return 0.0

        epsilon_squared = h_statistic / ((n**2 - 1) / (n + 1))
        # Clip to [0, 1] range
        epsilon_squared = max(0.0, min(1.0, epsilon_squared))

        return epsilon_squared

    def test_seasonality_kruskal_wallis(
        self,
        seasonal: np.ndarray,
        num_bins: int
    ) -> Dict:
        """
        Test seasonality using Kruskal-Wallis H-test with adaptive phase binning.

        Treats seasonal component as emergent cyclic structure without assuming
        a fixed period.

        Args:
            seasonal: Seasonal component array
            num_bins: Number of adaptive phase bins

        Returns:
            Dictionary with test results
        """
        # Check for all-zero or all-identical seasonal component
        if np.all(seasonal == seasonal[0]) or np.std(seasonal) < 1e-10:
            return {
                'h_statistic': 0.0,
                'p_value': 1.0,
                'num_bins': 0,
                'eta_squared': 0.0,
                'epsilon_squared': 0.0
            }

        # Create adaptive phase bins
        bin_indices = self.detect_adaptive_bins(seasonal, num_bins)

        # Group seasonal values by bin
        groups = []
        for bin_idx in range(num_bins):
            mask = bin_indices == bin_idx
            if np.sum(mask) > 0:  # Only include non-empty bins
                groups.append(seasonal[mask])

        # Perform Kruskal-Wallis H-test
        if len(groups) < 2:
            # Not enough groups for test
            return {
                'h_statistic': np.nan,
                'p_value': np.nan,
                'num_bins': len(groups),
                'eta_squared': np.nan,
                'epsilon_squared': np.nan
            }

        try:
            h_statistic, p_value = kruskal(*groups)
        except ValueError as e:
            # Handle case where all values are identical
            if "identical" in str(e).lower():
                return {
                    'h_statistic': 0.0,
                    'p_value': 1.0,
                    'num_bins': len(groups),
                    'eta_squared': 0.0,
                    'epsilon_squared': 0.0
                }
            else:
                raise

        # Compute effect sizes
        n = len(seasonal)
        eta_squared = self.compute_effect_size_eta_squared(groups, h_statistic, n)
        epsilon_squared = self.compute_effect_size_epsilon_squared(groups, h_statistic, n)

        return {
            'h_statistic': h_statistic,
            'p_value': p_value,
            'num_bins': len(groups),
            'eta_squared': eta_squared,
            'epsilon_squared': epsilon_squared
        }

    def test_residual_ljung_box(
        self,
        residual: np.ndarray,
        lags: List[int]
    ) -> Dict:
        """
        Test residual autocorrelation using Ljung-Box test.

        Args:
            residual: Residual component array
            lags: List of lags to test

        Returns:
            Dictionary with test results for each lag
        """
        results = {}

        for lag in lags:
            if lag >= len(residual):
                # Skip if lag is too large
                results[lag] = {
                    'test_statistic': np.nan,
                    'p_value': np.nan
                }
                continue

            try:
                # Perform Ljung-Box test
                lb_result = acorr_ljungbox(residual, lags=[lag], return_df=False)
                test_statistic = lb_result[0][0]  # Q-statistic
                p_value = lb_result[1][0]  # p-value

                results[lag] = {
                    'test_statistic': test_statistic,
                    'p_value': p_value
                }
            except Exception as e:
                print(f"Warning: Ljung-Box test failed for lag {lag}: {e}")
                results[lag] = {
                    'test_statistic': np.nan,
                    'p_value': np.nan
                }

        return results

    def load_decomposition(self, dataset_name: str, model_name: str) -> Optional[Dict]:
        """
        Load decomposition results for a specific dataset and model.

        Args:
            dataset_name: Name of the dataset (e.g., ETTh1, ETTh2, Sunspot)
            model_name: Name of the model (e.g., LGTD, ASTD, STL)

        Returns:
            Dictionary with decomposition components or None if not found
        """
        decomp_path = self.decomposition_dir / dataset_name / f"{model_name}.json"

        if not decomp_path.exists():
            return None

        try:
            with open(decomp_path, 'r') as f:
                data = json.load(f)

            return {
                'y': np.array(data['y']),
                'trend': np.array(data['trend']),
                'seasonal': np.array(data['seasonal']),
                'residual': np.array(data['residual'])
            }
        except Exception as e:
            print(f"Error loading {decomp_path}: {e}")
            return None

    def run_tests_for_model(
        self,
        dataset_name: str,
        model_name: str
    ) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        Run all statistical tests for a specific model on a dataset.

        Args:
            dataset_name: Name of the dataset
            model_name: Name of the model

        Returns:
            Tuple of (seasonality_result, residual_result)
        """
        # Load decomposition
        decomp = self.load_decomposition(dataset_name, model_name)

        if decomp is None:
            return None, None

        # Test seasonality
        seasonality_result = self.test_seasonality_kruskal_wallis(
            decomp['seasonal'],
            self.num_bins
        )
        seasonality_result['dataset'] = dataset_name
        seasonality_result['model'] = model_name

        # Test residual
        residual_result_dict = self.test_residual_ljung_box(
            decomp['residual'],
            self.ljung_box_lags
        )

        # Format residual results for CSV
        residual_result = {
            'dataset': dataset_name,
            'model': model_name
        }
        for lag, lag_result in residual_result_dict.items():
            residual_result[f'lag_{lag}_statistic'] = lag_result['test_statistic']
            residual_result[f'lag_{lag}_p_value'] = lag_result['p_value']

        return seasonality_result, residual_result

    def run_all_tests(
        self,
        datasets: Optional[List[str]] = None,
        models: Optional[List[str]] = None
    ):
        """
        Run statistical tests on all datasets and models.

        Args:
            datasets: List of dataset names to test (None = all)
            models: List of model names to test (None = all)
        """
        print("="*70)
        print("STATISTICAL TESTING FOR REAL-WORLD DECOMPOSITIONS")
        print("="*70)
        print(f"Decomposition directory: {self.decomposition_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Number of bins for seasonality test: {self.num_bins}")
        print(f"Ljung-Box test lags: {self.ljung_box_lags}")
        print("="*70)

        # Get all available datasets
        available_datasets = [d.name for d in self.decomposition_dir.iterdir() if d.is_dir()]
        datasets_to_test = datasets if datasets else available_datasets

        # Validate datasets
        for dataset in datasets_to_test:
            if dataset not in available_datasets:
                print(f"Warning: Dataset '{dataset}' not found. Available: {available_datasets}")

        datasets_to_test = [d for d in datasets_to_test if d in available_datasets]

        if not datasets_to_test:
            print("Error: No valid datasets to test")
            return

        # Process each dataset
        for dataset_name in datasets_to_test:
            print(f"\n{'='*70}")
            print(f"Testing dataset: {dataset_name}")
            print(f"{'='*70}")

            dataset_dir = self.decomposition_dir / dataset_name

            # Get all available models for this dataset
            available_models = [
                f.stem for f in dataset_dir.glob("*.json")
            ]
            models_to_test = models if models else available_models

            # Filter to only available models
            models_to_test = [m for m in models_to_test if m in available_models]

            if not models_to_test:
                print(f"  No models found for {dataset_name}")
                continue

            print(f"  Testing models: {models_to_test}")

            # Run tests for each model
            for model_name in models_to_test:
                print(f"    - {model_name}...", end=" ")

                seasonality_result, residual_result = self.run_tests_for_model(
                    dataset_name,
                    model_name
                )

                if seasonality_result is not None:
                    self.seasonality_results.append(seasonality_result)
                if residual_result is not None:
                    self.residual_results.append(residual_result)

                print("✓")

        print(f"\n{'='*70}")
        print("TESTS COMPLETED")
        print(f"{'='*70}")
        print(f"Total seasonality tests: {len(self.seasonality_results)}")
        print(f"Total residual tests: {len(self.residual_results)}")

    def save_results(self):
        """
        Save statistical test results to CSV files.
        """
        print(f"\n{'='*70}")
        print("SAVING RESULTS")
        print(f"{'='*70}")

        # Save seasonality test results
        if self.seasonality_results:
            seasonality_df = pd.DataFrame(self.seasonality_results)

            # Reorder columns for clarity
            column_order = [
                'dataset', 'model', 'h_statistic', 'p_value',
                'num_bins', 'eta_squared', 'epsilon_squared'
            ]
            seasonality_df = seasonality_df[column_order]

            seasonality_path = self.output_dir / "seasonality_kruskal_wallis.csv"
            seasonality_df.to_csv(seasonality_path, index=False, float_format='%.6f')
            print(f"✓ Seasonality results saved to: {seasonality_path}")
            print(f"  Columns: {list(seasonality_df.columns)}")
        else:
            print("  No seasonality results to save")

        # Save residual test results
        if self.residual_results:
            residual_df = pd.DataFrame(self.residual_results)

            # Reorder columns for clarity: group by lag (statistic, p_value)
            column_order = ['dataset', 'model']
            # Extract lag numbers and sort them
            lag_numbers = sorted(set(
                int(col.split('_')[1])
                for col in residual_df.columns
                if col.startswith('lag_') and col.endswith('_statistic')
            ))
            # Add columns in order: lag_X_statistic, lag_X_p_value for each lag
            for lag in lag_numbers:
                column_order.append(f'lag_{lag}_statistic')
                column_order.append(f'lag_{lag}_p_value')
            residual_df = residual_df[column_order]

            residual_path = self.output_dir / "residual_ljung_box.csv"
            residual_df.to_csv(residual_path, index=False, float_format='%.6f')
            print(f"✓ Residual results saved to: {residual_path}")
            print(f"  Columns: {list(residual_df.columns)}")
        else:
            print("  No residual results to save")

        print(f"\n{'='*70}")
        print("ALL RESULTS SAVED SUCCESSFULLY")
        print(f"{'='*70}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Run statistical tests on real-world decomposition results"
    )

    parser.add_argument(
        '--datasets',
        nargs='+',
        default=None,
        help='List of datasets to test (default: all)'
    )

    parser.add_argument(
        '--models',
        nargs='+',
        default=None,
        help='List of models to test (default: all)'
    )

    parser.add_argument(
        '--num_bins',
        type=int,
        default=12,
        help='Number of adaptive phase bins for seasonality test (default: 12)'
    )

    parser.add_argument(
        '--ljung_box_lags',
        type=str,
        default='10,20,30',
        help='Comma-separated list of lags for Ljung-Box test (default: 10,20,30)'
    )

    parser.add_argument(
        '--decomposition_dir',
        type=str,
        default=None,
        help='Directory containing decomposition results (default: experiments/results/decompositions/real_world)'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directory to save statistical test results (default: experiments/results/stat_tests/real_world)'
    )

    args = parser.parse_args()

    # Parse lags
    ljung_box_lags = [int(lag.strip()) for lag in args.ljung_box_lags.split(',')]

    # Set default paths
    decomposition_dir = args.decomposition_dir or str(
        project_root / "experiments" / "results" / "decompositions" / "real_world"
    )
    output_dir = args.output_dir or str(
        project_root / "experiments" / "results" / "stat_tests" / "real_world"
    )

    # Initialize tester
    tester = RealWorldStatisticalTester(
        decomposition_dir=decomposition_dir,
        output_dir=output_dir,
        num_bins=args.num_bins,
        ljung_box_lags=ljung_box_lags
    )

    # Run tests
    tester.run_all_tests(
        datasets=args.datasets,
        models=args.models
    )

    # Save results
    tester.save_results()


if __name__ == '__main__':
    main()
