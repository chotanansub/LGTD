"""
Base experiment runner for decomposition methods.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml

from lgtd.evaluation.metrics import compute_mse, compute_mae


class BaseExperiment:
    """
    Base class for running decomposition experiments.

    This class provides common functionality for:
    - Loading datasets
    - Running decomposition methods
    - Computing evaluation metrics
    - Saving results
    - Command-line argument parsing
    """

    def __init__(self, output_dir: str = 'results', config_path: Optional[str] = None):
        """
        Initialize base experiment.

        Args:
            output_dir: Directory to save results
            config_path: Path to configuration file (optional)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        self.config = None
        if config_path:
            self.config = self.load_config(config_path)

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load experiment configuration from YAML file.

        Args:
            config_path: Path to configuration file

        Returns:
            Configuration dictionary
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def run_method(
        self,
        method_name: str,
        method_func: callable,
        dataset: Dict[str, np.ndarray],
        params: Dict[str, Any]
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Run a single decomposition method.

        Args:
            method_name: Name of the method
            method_func: Decomposition function
            dataset: Dataset dictionary
            params: Method parameters

        Returns:
            Result dictionary or None if failed
        """
        print(f"  → Running {method_name}...")

        try:
            result = method_func(dataset, **params)
            print(f"    ✓ {method_name} completed")
            return result
        except Exception as e:
            print(f"    ✗ {method_name} failed: {str(e)}")
            return None

    def compute_metrics(
        self,
        ground_truth: Dict[str, np.ndarray],
        result: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute evaluation metrics.

        Args:
            ground_truth: Ground truth components
            result: Predicted components

        Returns:
            Dictionary of metrics
        """
        mse = compute_mse(ground_truth, result)
        mae = compute_mae(ground_truth, result)

        return {
            'mse': mse,
            'mae': mae
        }

    def create_evaluation_dataframe(
        self,
        ground_truth: Dict[str, np.ndarray],
        results_dict: Dict[str, Dict[str, np.ndarray]]
    ) -> pd.DataFrame:
        """
        Create evaluation DataFrame with metrics for all models.

        Args:
            ground_truth: Ground truth dataset
            results_dict: Dictionary with model names as keys and results as values

        Returns:
            Pandas DataFrame with evaluation metrics
        """
        records = []

        for model_name, result in results_dict.items():
            if result is None:
                continue

            metrics = self.compute_metrics(ground_truth, result)

            for metric_name, metric_values in metrics.items():
                record = {
                    **metric_values,
                    'metric': metric_name.upper(),
                    'model': model_name
                }
                records.append(record)

        df = pd.DataFrame(records)
        return df

    def save_results(
        self,
        dataset_name: str,
        results: Dict[str, Any],
        format: str = 'csv'
    ) -> None:
        """
        Save experiment results to file.

        Args:
            dataset_name: Name of the dataset
            results: Results dictionary
            format: Output format ('csv' or 'json')
        """
        # Create subdirectories for organized storage
        # Save accuracy metrics separately from decomposition components
        # Determine dataset type from output_dir or default to synthetic
        dataset_type = "synthetic"  # Default for backward compatibility
        accuracy_dir = self.output_dir / "accuracy" / dataset_type
        accuracy_dir.mkdir(parents=True, exist_ok=True)

        if format == 'csv' and 'evaluation' in results:
            # Save evaluation metrics as CSV
            output_path = accuracy_dir / f"{dataset_name}_metrics.{format}"
            results['evaluation'].to_csv(output_path, index=False)
            print(f"✓ Metrics saved to: {output_path}")
        elif format == 'json':
            import json
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {}
            for key, value in results.items():
                if isinstance(value, pd.DataFrame):
                    serializable_results[key] = value.to_dict()
                elif isinstance(value, dict):
                    serializable_results[key] = {
                        k: v.tolist() if isinstance(v, np.ndarray) else v
                        for k, v in value.items()
                    }
                else:
                    serializable_results[key] = value

            output_path = accuracy_dir / f"{dataset_name}_metrics.{format}"
            with open(output_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            print(f"✓ Metrics saved to: {output_path}")

    def print_summary(
        self,
        dataset_name: str,
        evaluation_df: pd.DataFrame
    ) -> None:
        """
        Print evaluation summary.

        Args:
            dataset_name: Name of the dataset
            evaluation_df: Evaluation DataFrame
        """
        print(f"\nEvaluation Summary for {dataset_name}:")
        print("-" * 70)

        for model in evaluation_df['model'].unique():
            model_eval = evaluation_df[evaluation_df['model'] == model]
            mse_row = model_eval[model_eval['metric'] == 'MSE']

            if not mse_row.empty:
                mse_row = mse_row.iloc[0]
                print(f"  {model:<15} MSE → "
                      f"Trend: {mse_row.get('trend', 0):.6f}, "
                      f"Seasonal: {mse_row.get('seasonal', 0):.6f}, "
                      f"Residual: {mse_row.get('residual', 0):.6f}")

    @staticmethod
    def create_argument_parser(description: str = "Run decomposition experiments") -> argparse.ArgumentParser:
        """
        Create command-line argument parser with common options.

        Args:
            description: Description for the argument parser

        Returns:
            Configured ArgumentParser instance
        """
        parser = argparse.ArgumentParser(description=description)

        parser.add_argument(
            '--datasets', '-d',
            nargs='+',
            help='Dataset names to run (e.g., synth1 synth2). Default: all configured datasets'
        )

        parser.add_argument(
            '--models', '-m',
            nargs='+',
            help='Model names to run (e.g., LGTD ASTD_Online). Default: all enabled models'
        )

        parser.add_argument(
            '--config',
            type=str,
            help='Path to configuration file (overrides default)'
        )

        parser.add_argument(
            '--output-dir', '-o',
            type=str,
            help='Directory to save results (overrides default)'
        )

        parser.add_argument(
            '--no-save',
            action='store_true',
            help='Do not save results to file'
        )

        parser.add_argument(
            '--quiet', '-q',
            action='store_true',
            help='Suppress progress output'
        )

        return parser

    def filter_datasets(self, dataset_filter: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Filter datasets based on user specification.

        Args:
            dataset_filter: List of dataset names to include (None = all)

        Returns:
            List of filtered dataset configurations
        """
        if self.config is None or 'datasets' not in self.config:
            return []

        datasets = self.config['datasets']

        if dataset_filter is None:
            return datasets

        # Filter datasets by name
        filtered = [d for d in datasets if d.get('name') in dataset_filter]

        if not filtered:
            available = [d.get('name') for d in datasets]
            raise ValueError(
                f"No matching datasets found. "
                f"Requested: {dataset_filter}, Available: {available}"
            )

        return filtered

    def filter_methods(self, method_filter: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Filter methods based on user specification.

        Args:
            method_filter: List of method names to include (None = all enabled)

        Returns:
            Dictionary of filtered method configurations
        """
        if self.config is None or 'methods' not in self.config:
            return {}

        methods = self.config['methods']

        if method_filter is None:
            # Return all enabled methods
            return {k: v for k, v in methods.items() if v.get('enabled', True)}

        # Normalize method names to lowercase for comparison
        method_filter_lower = [m.lower() for m in method_filter]

        # Filter methods by name
        filtered = {}
        for method_name, method_config in methods.items():
            if method_name.lower() in method_filter_lower:
                filtered[method_name] = method_config

        if not filtered:
            available = list(methods.keys())
            raise ValueError(
                f"No matching methods found. "
                f"Requested: {method_filter}, Available: {available}"
            )

        return filtered
