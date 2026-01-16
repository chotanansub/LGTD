"""
Real-world experiment runner for LGTD evaluation.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List

from experiments.runners.base_experiment import BaseExperiment
from data.real_world.loaders import load_ett_dataset, load_sunspot_dataset
from lgtd import lgtd
from experiments.baselines.stl import STLDecomposer
from experiments.baselines.robust_stl import RobustSTLDecomposer
from experiments.baselines.fast_robust_stl import FastRobustSTLDecomposer
from experiments.baselines.str_decomposer import STRDecomposer
from experiments.baselines.online_stl import OnlineSTLDecomposer
from experiments.baselines.oneshot_stl import OneShotSTLDecomposer
from experiments.baselines.astd import ASTDDecomposer


class RealWorldExperimentRunner(BaseExperiment):
    """
    Runner for real-world data experiments.

    This class handles:
    - Loading real-world datasets (ETTh1, ETTh2, Sunspot)
    - Running multiple decomposition methods
    - Limiting to first 2500 points for consistency
    - Saving results
    """

    # Real-world datasets don't have ground truth
    MAX_POINTS = 2500

    def __init__(self, config_path: str = None, output_dir: str = 'results/real_world'):
        """
        Initialize real-world experiment runner.

        Args:
            config_path: Path to configuration YAML file
            output_dir: Directory to save results
        """
        super().__init__(output_dir=output_dir, config_path=config_path)

    def load_dataset(self, dataset_name: str, **loader_params) -> Dict[str, np.ndarray]:
        """
        Load a real-world dataset.

        Args:
            dataset_name: Name of the dataset (ETTh1, ETTh2, Sunspot)
            **loader_params: Additional parameters for the loader

        Returns:
            Dataset dictionary with time series
        """
        # Load based on dataset name
        if dataset_name in ['ETTh1', 'ETTh2']:
            data = load_ett_dataset(dataset_name, **loader_params)
        elif dataset_name == 'Sunspot':
            data = load_sunspot_dataset(**loader_params)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        # Limit to first MAX_POINTS for consistency
        y_data = data['y'][:self.MAX_POINTS]

        return {
            'y': y_data,
            'time': np.arange(len(y_data)),
            # Real-world datasets don't have ground truth
            'trend': None,
            'seasonal': None,
            'residual': None
        }

    def run_lgtd(self, dataset: Dict[str, Any], **params) -> Dict[str, np.ndarray]:
        """Run LGTD decomposition."""
        model = lgtd(**params)
        result = model.fit_transform(dataset['y'])

        return {
            'time': dataset['time'],
            'y': dataset['y'],
            'trend': result.trend,
            'seasonal': result.seasonal,
            'residual': result.residual
        }

    def run_stl(self, dataset: Dict[str, Any], **params) -> Dict[str, np.ndarray]:
        """Run STL decomposition."""
        model = STLDecomposer(**params)
        return model.fit_transform(dataset['y'])

    def run_robust_stl(self, dataset: Dict[str, Any], **params) -> Dict[str, np.ndarray]:
        """Run RobustSTL decomposition."""
        model = RobustSTLDecomposer(**params)
        return model.fit_transform(dataset['y'])

    def run_astd(self, dataset: Dict[str, Any], **params) -> Dict[str, np.ndarray]:
        """Run ASTD decomposition."""
        model = ASTDDecomposer(**params)
        return model.fit_transform(dataset['y'])

    def run_astd_online(self, dataset: Dict[str, Any], **params) -> Dict[str, np.ndarray]:
        """Run ASTD Online decomposition."""
        model = ASTDDecomposer(**params)
        return model.fit_transform(dataset['y'])

    def run_fast_robust_stl(self, dataset: Dict[str, Any], **params) -> Dict[str, np.ndarray]:
        """Run FastRobustSTL decomposition."""
        model = FastRobustSTLDecomposer(**params)
        return model.fit_transform(dataset['y'])

    def run_str(self, dataset: Dict[str, Any], **params) -> Dict[str, np.ndarray]:
        """Run STR decomposition."""
        model = STRDecomposer(**params)
        return model.fit_transform(dataset['y'])

    def run_online_stl(self, dataset: Dict[str, Any], **params) -> Dict[str, np.ndarray]:
        """Run OnlineSTL decomposition."""
        model = OnlineSTLDecomposer(**params)
        return model.fit_transform(dataset['y'])

    def run_oneshot_stl(self, dataset: Dict[str, Any], **params) -> Dict[str, np.ndarray]:
        """Run OneShotSTL decomposition."""
        model = OneShotSTLDecomposer(**params)
        return model.fit_transform(dataset['y'])

    def run_lgtd_linear(self, dataset: Dict[str, Any], **params) -> Dict[str, np.ndarray]:
        """Run LGTD with linear trend."""
        params['trend_selection'] = 'linear'
        model = lgtd(**params)
        result = model.fit_transform(dataset['y'])
        return {
            'time': dataset['time'],
            'y': dataset['y'],
            'trend': result.trend,
            'seasonal': result.seasonal,
            'residual': result.residual
        }

    def run_lgtd_lowess(self, dataset: Dict[str, Any], **params) -> Dict[str, np.ndarray]:
        """Run LGTD with LOWESS trend."""
        params['trend_selection'] = 'lowess'
        model = lgtd(**params)
        result = model.fit_transform(dataset['y'])
        return {
            'time': dataset['time'],
            'y': dataset['y'],
            'trend': result.trend,
            'seasonal': result.seasonal,
            'residual': result.residual
        }

    def run_single_experiment(
        self,
        dataset_name: str,
        loader_params: Dict[str, Any],
        methods_to_run: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run experiment on a single real-world dataset.

        Args:
            dataset_name: Name of the dataset
            loader_params: Parameters for loading the dataset
            methods_to_run: Specific methods to run (None = all enabled from config)

        Returns:
            Dictionary with results
        """
        print(f"\n{'='*70}")
        print(f"Running Experiment: {dataset_name.upper()}")
        print(f"{'='*70}")
        print(f"Dataset: {dataset_name}")
        print(f"Max points: {self.MAX_POINTS}")

        # Load dataset
        dataset = self.load_dataset(dataset_name, **loader_params)

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

        return {
            'dataset': {'name': dataset_name, 'y': dataset['y']},
            'results': results
        }

    def save_decomposition_components(
        self,
        dataset_name: str,
        experiment_result: Dict[str, Any]
    ) -> None:
        """
        Save decomposition components for plotting.

        Saves each model's decomposition as JSON in the format:
        experiments/results/decompositions/real_world/{dataset_name}/{model_name}.json

        Args:
            dataset_name: Name of the dataset
            experiment_result: Results dictionary containing 'dataset' and 'results'
        """
        import json

        # Create dataset subdirectory with real_world prefix
        dataset_dir = self.output_dir / "decompositions" / "real_world" / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        y_data = experiment_result['dataset']['y']

        # Save each model's decomposition
        for model_name, result in experiment_result['results'].items():
            if result is None:
                continue

            # Prepare data
            decomposition_data = {
                'y': y_data.tolist(),
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
        Run all real-world experiments defined in configuration.

        Args:
            dataset_filter: List of dataset names to run (None = all)
            method_filter: List of method names to run (None = all enabled)
            save_results: Whether to save results to disk
        """
        print("\n" + "="*70)
        print("STARTING REAL-WORLD EXPERIMENTS")
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
            dataset_name = dataset_config['name']
            loader_params = dataset_config.get('loader_params', {})

            experiment_result = self.run_single_experiment(
                dataset_name,
                loader_params,
                methods_to_run=methods_to_run
            )
            all_results[dataset_name] = experiment_result

            # Save decomposition components for plotting
            if save_results and self.config['output'].get('save_decompositions', True):
                self.save_decomposition_components(
                    dataset_name,
                    experiment_result
                )

        self.results = all_results

        print("\n" + "="*70)
        print("ALL EXPERIMENTS COMPLETED")
        print("="*70)
        print(f"Results saved to: {self.output_dir}")
