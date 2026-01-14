"""
Synthetic experiment runner for LGTD evaluation.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List

from experiments.runners.base_experiment import BaseExperiment
from data.synthetic.generators import generate_synthetic_data
from lgtd import LGTD
from experiments.baselines.stl import STLDecomposer
from experiments.baselines.robust_stl import RobustSTLDecomposer
from experiments.baselines.fast_robust_stl import FastRobustSTLDecomposer
from experiments.baselines.str_decomposer import STRDecomposer
from experiments.baselines.online_stl import OnlineSTLDecomposer
from experiments.baselines.oneshot_stl import OneShotSTLDecomposer
from experiments.baselines.astd import ASTDDecomposer


class SyntheticExperimentRunner(BaseExperiment):
    """
    Runner for synthetic data experiments.

    This class handles:
    - Generating synthetic datasets
    - Running multiple decomposition methods
    - Computing evaluation metrics
    - Saving results
    """

    def __init__(self, config_path: str = None, output_dir: str = 'results'):
        """
        Initialize synthetic experiment runner.

        Args:
            config_path: Path to configuration YAML file
            output_dir: Directory to save results
        """
        super().__init__(output_dir=output_dir, config_path=config_path)

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

        return {
            'name': dataset_config['name'],
            'data': data,
            'config': dataset_config
        }

    def run_lgtd(self, dataset: Dict[str, Any], **params) -> Dict[str, np.ndarray]:
        """Run LGTD decomposition."""
        model = LGTD(**params)
        result = model.fit_transform(dataset['data']['y'])

        return {
            'time': dataset['data']['time'],
            'y': dataset['data']['y'],
            'trend': result.trend,
            'seasonal': result.seasonal,
            'residual': result.residual
        }

    def run_stl(self, dataset: Dict[str, Any], **params) -> Dict[str, np.ndarray]:
        """Run STL decomposition."""
        model = STLDecomposer(**params)
        return model.fit_transform(dataset['data']['y'])

    def run_robust_stl(self, dataset: Dict[str, Any], **params) -> Dict[str, np.ndarray]:
        """Run RobustSTL decomposition."""
        model = RobustSTLDecomposer(**params)
        return model.fit_transform(dataset['data']['y'])

    def run_astd(self, dataset: Dict[str, Any], **params) -> Dict[str, np.ndarray]:
        """Run ASTD decomposition."""
        model = ASTDDecomposer(**params)
        return model.fit_transform(dataset['data']['y'])

    def run_astd_online(self, dataset: Dict[str, Any], **params) -> Dict[str, np.ndarray]:
        """Run ASTD Online decomposition."""
        model = ASTDDecomposer(**params)
        return model.fit_transform(dataset['data']['y'])

    def run_fast_robust_stl(self, dataset: Dict[str, Any], **params) -> Dict[str, np.ndarray]:
        """Run FastRobustSTL decomposition."""
        model = FastRobustSTLDecomposer(**params)
        return model.fit_transform(dataset['data']['y'])

    def run_str(self, dataset: Dict[str, Any], **params) -> Dict[str, np.ndarray]:
        """Run STR decomposition."""
        model = STRDecomposer(**params)
        return model.fit_transform(dataset['data']['y'])

    def run_online_stl(self, dataset: Dict[str, Any], **params) -> Dict[str, np.ndarray]:
        """Run OnlineSTL decomposition."""
        model = OnlineSTLDecomposer(**params)
        return model.fit_transform(dataset['data']['y'])

    def run_oneshot_stl(self, dataset: Dict[str, Any], **params) -> Dict[str, np.ndarray]:
        """Run OneShotSTL decomposition."""
        model = OneShotSTLDecomposer(**params)
        return model.fit_transform(dataset['data']['y'])

    def run_lgtd_linear(self, dataset: Dict[str, Any], **params) -> Dict[str, np.ndarray]:
        """Run LGTD with linear trend."""
        params['trend_selection'] = 'linear'
        model = LGTD(**params)
        result = model.fit_transform(dataset['data']['y'])
        return {
            'time': dataset['data']['time'],
            'y': dataset['data']['y'],
            'trend': result.trend,
            'seasonal': result.seasonal,
            'residual': result.residual
        }

    def run_lgtd_lowess(self, dataset: Dict[str, Any], **params) -> Dict[str, np.ndarray]:
        """Run LGTD with LOWESS trend."""
        params['trend_selection'] = 'lowess'
        model = LGTD(**params)
        result = model.fit_transform(dataset['data']['y'])
        return {
            'time': dataset['data']['time'],
            'y': dataset['data']['y'],
            'trend': result.trend,
            'seasonal': result.seasonal,
            'residual': result.residual
        }

    def run_single_experiment(
        self,
        dataset_config: Dict[str, Any],
        methods_to_run: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run experiment on a single dataset.

        Args:
            dataset_config: Dataset configuration
            methods_to_run: Specific methods to run (None = all enabled from config)

        Returns:
            Dictionary with results and evaluation
        """
        print(f"\n{'='*70}")
        print(f"Running Experiment: {dataset_config['name'].upper()}")
        print(f"{'='*70}")
        print(f"Trend Type: {dataset_config['trend_type']}")
        print(f"Seasonality Type: {dataset_config['seasonality_type']}")
        print(f"Samples: {dataset_config.get('n_samples', 2000)}")
        print(f"Noise: {dataset_config.get('noise_std', 1.0)}")

        # Generate dataset
        dataset = self.generate_dataset(dataset_config)
        ground_truth = dataset['data']

        # Run methods
        results = {}
        methods_config = methods_to_run if methods_to_run is not None else self.config['methods']

        method_runners = {
            'lgtd': self.run_lgtd,
            'lgtd_linear': self.run_lgtd_linear,
            'lgtd_lowess': self.run_lgtd_lowess,
            'stl': self.run_stl,
            'robust_stl': self.run_robust_stl,
            'fast_robust_stl': self.run_fast_robust_stl,
            'str': self.run_str,
            'online_stl': self.run_online_stl,
            'oneshot_stl': self.run_oneshot_stl,
            'astd': self.run_astd,
            'astd_online': self.run_astd_online
        }

        for method_name, method_config in methods_config.items():
            if not method_config.get('enabled', True):
                continue

            if method_name in method_runners:
                result = self.run_method(
                    method_name.upper(),
                    method_runners[method_name],
                    dataset,
                    method_config.get('params', {})
                )
                if result is not None:
                    results[method_name.upper()] = result

        # Compute evaluation
        evaluation_df = self.create_evaluation_dataframe(ground_truth, results)

        # Print summary
        self.print_summary(dataset_config['name'], evaluation_df)

        return {
            'dataset': dataset,
            'results': results,
            'evaluation': evaluation_df
        }

    def save_decomposition_components(
        self,
        dataset_name: str,
        experiment_result: Dict[str, Any]
    ) -> None:
        """
        Save decomposition components for plotting.

        Saves each model's decomposition as JSON in the format expected by plot_synthetic.py:
        experiments/results/synthetic/decompositions/{dataset_name}/{model_name}.json

        Args:
            dataset_name: Name of the dataset (e.g., "synth1_linear_fixed")
            experiment_result: Results dictionary containing 'dataset' and 'results'
        """
        import json
        import re

        # Extract base dataset name (synth1, synth2, etc.) from full name
        # e.g., "synth1_linear_fixed" -> "synth1"
        base_name = re.match(r'(synth\d+)', dataset_name)
        if base_name:
            short_name = base_name.group(1)
        else:
            short_name = dataset_name

        # Create dataset subdirectory using short name
        dataset_dir = self.output_dir / "decompositions" / "synthetic" / short_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        dataset = experiment_result['dataset']
        ground_truth = dataset['data']

        # Save each model's decomposition
        for model_name, result in experiment_result['results'].items():
            if result is None:
                continue

            # Prepare data in the format expected by plot_synthetic.py
            decomposition_data = {
                'y': ground_truth['y'].tolist(),
                'trend': result['trend'].tolist(),
                'seasonal': result['seasonal'].tolist(),
                'residual': result['residual'].tolist()
            }

            # Save as JSON
            output_path = dataset_dir / f"{model_name}.json"
            with open(output_path, 'w') as f:
                json.dump(decomposition_data, f, indent=2)

        print(f"✓ Decomposition components saved to: {dataset_dir}")

    def run_all_experiments(
        self,
        dataset_filter: Optional[List[str]] = None,
        method_filter: Optional[List[str]] = None,
        save_results: bool = True
    ) -> None:
        """
        Run all experiments defined in configuration.

        Args:
            dataset_filter: List of dataset names to run (None = all)
            method_filter: List of method names to run (None = all enabled)
            save_results: Whether to save results to disk
        """
        print("\n" + "="*70)
        print("STARTING SYNTHETIC EXPERIMENTS")
        print("="*70)

        # Filter datasets and methods
        datasets_to_run = self.filter_datasets(dataset_filter)
        methods_to_run = self.filter_methods(method_filter)

        if dataset_filter:
            print(f"Running on datasets: {[d['name'] for d in datasets_to_run]}")
        if method_filter:
            print(f"Running methods: {list(methods_to_run.keys())}")

        all_results = {}

        for dataset_config in datasets_to_run:
            experiment_result = self.run_single_experiment(
                dataset_config,
                methods_to_run=methods_to_run
            )
            all_results[dataset_config['name']] = experiment_result

            # Save evaluation metrics
            if save_results and self.config['output'].get('save_metrics', True):
                self.save_results(
                    dataset_config['name'],
                    experiment_result,
                    format=self.config['output'].get('results_format', 'csv')
                )

            # Save decomposition components for plotting
            if save_results and self.config['output'].get('save_decompositions', True):
                self.save_decomposition_components(
                    dataset_config['name'],
                    experiment_result
                )

        self.results = all_results

        print("\n" + "="*70)
        print("ALL EXPERIMENTS COMPLETED")
        print("="*70)
        print(f"Results saved to: {self.output_dir}")
