"""
Base experiment runner for decomposition methods.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml

from LGTD.evaluation.metrics import compute_mse, compute_mae


class BaseExperiment:
    """
    Base class for running decomposition experiments.

    This class provides common functionality for:
    - Loading datasets
    - Running decomposition methods
    - Computing evaluation metrics
    - Saving results
    """

    def __init__(self, output_dir: str = 'results'):
        """
        Initialize base experiment.

        Args:
            output_dir: Directory to save results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}

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
        # Save metrics separately from decomposition components
        metrics_dir = self.output_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)

        if format == 'csv' and 'evaluation' in results:
            # Save evaluation metrics as CSV
            output_path = metrics_dir / f"{dataset_name}_metrics.{format}"
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

            output_path = metrics_dir / f"{dataset_name}_metrics.{format}"
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
