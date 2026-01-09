"""
Flexible experiment runner for LGTD experiments.

Supports running:
- All models on all datasets
- Specific model(s) on all datasets
- All models on specific dataset(s)
- Specific model(s) on specific dataset(s)
"""

import json
import os
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd

from LGTD import LGTD
from LGTD.evaluation.metrics import compute_mse, compute_mae, compute_rmse, compute_correlation, compute_psnr
from LGTD.evaluation.visualization import plot_decomposition


class ExperimentRunner:
    """
    Flexible experiment runner that executes decomposition experiments
    based on JSON parameter files.
    """

    def __init__(
        self,
        config_dir: str = "experiments/configs/dataset_params",
        results_dir: str = "experiments/results/synthetic"
    ):
        """
        Initialize experiment runner.

        Args:
            config_dir: Directory containing dataset parameter JSON files
            results_dir: Directory to save results
        """
        self.config_dir = Path(config_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Available models
        self.available_models = [
            'LGTD', 'LGTD_Linear', 'LGTD_LOWESS',
            'STL', 'RobustSTL', 'ASTD'
        ]

        # Load all dataset configs
        self.configs = self._load_configs()

    def _load_configs(self) -> Dict[str, Dict]:
        """Load all dataset configuration files."""
        configs = {}
        for config_file in self.config_dir.glob("*_params.json"):
            dataset_name = config_file.stem.replace('_params', '')
            with open(config_file, 'r') as f:
                configs[dataset_name] = json.load(f)
        return configs

    def _load_dataset(self, config: Dict) -> Dict[str, np.ndarray]:
        """Load dataset from JSON file."""
        dataset_path = config['dataset']['path']
        with open(dataset_path, 'r') as f:
            data = json.load(f)

        return {
            'y': np.array(data['data']['y']),
            'trend': np.array(data['data']['trend']),
            'seasonal': np.array(data['data']['seasonal']),
            'residual': np.array(data['data']['residual']),
            'time': np.array(data['data']['time'])
        }

    def _run_lgtd_variant(
        self,
        data: np.ndarray,
        model_name: str,
        params: Dict
    ) -> Dict[str, Any]:
        """Run LGTD or its variants."""
        start_time = time.time()

        model = LGTD(**params)
        result = model.fit_transform(data)

        elapsed_time = time.time() - start_time

        return {
            'trend': result.trend,
            'seasonal': result.seasonal,
            'residual': result.residual,
            'time': elapsed_time,
            'trend_info': result.trend_info,
            'detected_periods': result.detected_periods
        }

    def _run_stl(self, data: np.ndarray, params: Dict) -> Dict[str, Any]:
        """Run STL decomposition."""
        from statsmodels.tsa.seasonal import STL
        import pandas as pd

        start_time = time.time()

        ts = pd.Series(data)
        stl = STL(
            ts,
            period=params['period'],
            seasonal=params.get('seasonal', 13),
            trend=params.get('trend'),
            robust=params.get('robust', False)
        )
        result = stl.fit()

        elapsed_time = time.time() - start_time

        return {
            'trend': result.trend.values,
            'seasonal': result.seasonal.values,
            'residual': result.resid.values,
            'time': elapsed_time
        }

    def _run_model(
        self,
        model_name: str,
        data: np.ndarray,
        params: Dict
    ) -> Dict[str, Any]:
        """Run a specific model."""
        if model_name in ['LGTD', 'LGTD_Linear', 'LGTD_LOWESS']:
            return self._run_lgtd_variant(data, model_name, params)
        elif model_name == 'STL':
            return self._run_stl(data, params)
        elif model_name == 'RobustSTL':
            # TODO: Implement if needed
            raise NotImplementedError(f"{model_name} not yet implemented")
        elif model_name == 'ASTD':
            # TODO: Implement if needed
            raise NotImplementedError(f"{model_name} not yet implemented")
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def _compute_metrics(
        self,
        ground_truth: Dict[str, np.ndarray],
        result: Dict[str, np.ndarray],
        metrics: List[str]
    ) -> Dict[str, float]:
        """Compute evaluation metrics."""
        gt = {
            'trend': ground_truth['trend'],
            'seasonal': ground_truth['seasonal'],
            'residual': ground_truth['residual']
        }

        res = {
            'trend': result['trend'],
            'seasonal': result['seasonal'],
            'residual': result['residual']
        }

        computed_metrics = {}

        if 'mse' in metrics:
            mse = compute_mse(gt, res)
            for comp, val in mse.items():
                computed_metrics[f'mse_{comp}'] = val

        if 'mae' in metrics:
            mae = compute_mae(gt, res)
            for comp, val in mae.items():
                computed_metrics[f'mae_{comp}'] = val

        if 'rmse' in metrics:
            rmse = compute_rmse(gt, res)
            for comp, val in rmse.items():
                computed_metrics[f'rmse_{comp}'] = val

        if 'correlation' in metrics:
            corr = compute_correlation(gt, res)
            for comp, val in corr.items():
                computed_metrics[f'corr_{comp}'] = val

        if 'psnr' in metrics:
            psnr = compute_psnr(gt, res)
            for comp, val in psnr.items():
                computed_metrics[f'psnr_{comp}'] = val

        return computed_metrics

    def run_experiment(
        self,
        datasets: Optional[List[str]] = None,
        models: Optional[List[str]] = None,
        save_results: bool = True,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Run experiments on specified datasets and models.

        Args:
            datasets: List of dataset names to run (None = all)
            models: List of model names to run (None = all enabled)
            save_results: Whether to save results to CSV
            verbose: Whether to print progress

        Returns:
            DataFrame with all experiment results
        """
        # Determine which datasets to run
        if datasets is None:
            datasets_to_run = list(self.configs.keys())
        else:
            datasets_to_run = [d for d in datasets if d in self.configs]
            if not datasets_to_run:
                raise ValueError(f"No valid datasets found. Available: {list(self.configs.keys())}")

        # Determine which models to run
        if models is None:
            models_to_run = self.available_models
        else:
            models_to_run = [m for m in models if m in self.available_models]
            if not models_to_run:
                raise ValueError(f"No valid models found. Available: {self.available_models}")

        results = []

        for dataset_name in datasets_to_run:
            config = self.configs[dataset_name]

            if verbose:
                print(f"\n{'='*80}")
                print(f"Running experiments on {dataset_name}")
                print(f"{'='*80}")

            # Load dataset
            data = self._load_dataset(config)

            for model_name in models_to_run:
                # Check if model is enabled in config
                if model_name not in config['models']:
                    continue

                model_config = config['models'][model_name]
                if not model_config.get('enabled', False):
                    if verbose:
                        print(f"  Skipping {model_name} (disabled)")
                    continue

                if verbose:
                    print(f"  Running {model_name}...", end=' ')

                try:
                    # Run model
                    result = self._run_model(
                        model_name,
                        data['y'],
                        model_config['params']
                    )

                    # Compute metrics
                    metrics = self._compute_metrics(
                        data,
                        result,
                        config['evaluation']['metrics']
                    )

                    # Compile result
                    result_entry = {
                        'dataset': dataset_name,
                        'trend_type': config['dataset']['trend_type'],
                        'period_type': config['dataset']['period_type'],
                        'model': model_name,
                        'time': result['time'],
                        **metrics
                    }

                    # Add model-specific info
                    if 'trend_info' in result:
                        result_entry['selected_method'] = result['trend_info'].get('method', 'N/A')

                    results.append(result_entry)

                    if verbose:
                        print(f"✓ (MSE trend: {metrics.get('mse_trend', 0):.2f}, time: {result['time']:.3f}s)")

                    # Save plot if enabled
                    if config['evaluation'].get('save_plots', False):
                        plot_dir = Path(config['evaluation']['plot_dir'])
                        plot_dir.mkdir(parents=True, exist_ok=True)

                        from LGTD.decomposition.lgtd import LGTDResult

                        # Create LGTDResult object for plotting
                        plot_result = LGTDResult(
                            trend=result['trend'],
                            seasonal=result['seasonal'],
                            residual=result['residual'],
                            y=data['y'],
                            detected_periods=result.get('detected_periods', []),
                            trend_info=result.get('trend_info', {})
                        )

                        plot_path = plot_dir / f"{model_name}_decomposition.png"
                        plot_decomposition(
                            plot_result,
                            ground_truth={
                                'trend': data['trend'],
                                'seasonal': data['seasonal'],
                                'residual': data['residual']
                            },
                            title=f"{dataset_name} - {model_name}",
                            save_path=str(plot_path),
                            show=False
                        )

                except Exception as e:
                    if verbose:
                        print(f"✗ Error: {e}")
                    results.append({
                        'dataset': dataset_name,
                        'trend_type': config['dataset']['trend_type'],
                        'period_type': config['dataset']['period_type'],
                        'model': model_name,
                        'error': str(e)
                    })

        # Convert to DataFrame
        results_df = pd.DataFrame(results)

        # Save results
        if save_results and not results_df.empty:
            # Create benchmarks directory
            benchmarks_dir = Path('experiments/results/benchmarks')
            benchmarks_dir.mkdir(parents=True, exist_ok=True)

            # Save canonical version with descriptive name
            canonical_file = benchmarks_dir / "synthetic_benchmarks.csv"
            results_df.to_csv(canonical_file, index=False)

            # Also save timestamped version for history
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            timestamped_file = benchmarks_dir / f"synthetic_benchmarks_{timestamp}.csv"
            results_df.to_csv(timestamped_file, index=False)

            if verbose:
                print(f"\nResults saved to:")
                print(f"  {canonical_file}")
                print(f"  {timestamped_file}")

        return results_df


def main():
    """Main entry point for running experiments from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Run LGTD experiments")
    parser.add_argument(
        '--datasets', '-d',
        nargs='+',
        help='Dataset names to run (e.g., synth1 synth2). Default: all'
    )
    parser.add_argument(
        '--models', '-m',
        nargs='+',
        help='Model names to run (e.g., LGTD STL). Default: all enabled'
    )
    parser.add_argument(
        '--config-dir',
        default='experiments/configs/dataset_params',
        help='Directory containing parameter JSON files'
    )
    parser.add_argument(
        '--results-dir',
        default='experiments/results/synthetic',
        help='Directory to save results'
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

    args = parser.parse_args()

    # Create runner
    runner = ExperimentRunner(
        config_dir=args.config_dir,
        results_dir=args.results_dir
    )

    # Run experiments
    results = runner.run_experiment(
        datasets=args.datasets,
        models=args.models,
        save_results=not args.no_save,
        verbose=not args.quiet
    )

    # Print summary
    if not args.quiet and not results.empty:
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"Total experiments: {len(results)}")
        print(f"Datasets tested: {results['dataset'].nunique()}")
        print(f"Models tested: {results['model'].nunique()}")

        if 'mse_trend' in results.columns:
            print(f"\nAverage MSE (trend): {results['mse_trend'].mean():.4f}")
            print("\nBest MSE by model:")
            print(results.groupby('model')['mse_trend'].mean().sort_values().to_string())


if __name__ == '__main__':
    main()
