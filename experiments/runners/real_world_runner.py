"""
Real-world experiment runner for LGTD evaluation.
"""

import numpy as np
import yaml
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
    - Loading model parameters from experiments/configs/models/*.yaml
    """

    # Real-world datasets don't have ground truth
    MAX_POINTS = 2500

    def __init__(self, config_path: str = None, output_dir: str = 'results'):
        """
        Initialize real-world experiment runner.

        Args:
            config_path: Path to configuration YAML file (optional, for datasets config)
            output_dir: Directory to save results (base directory)
        """
        super().__init__(output_dir=output_dir, config_path=config_path)

        # Load model configurations from experiments/configs/models/
        self.models_config = self._load_model_configs()

    def _load_model_configs(self) -> Dict[str, Dict]:
        """Load all model configurations from experiments/configs/models/."""
        project_root = Path(__file__).parent.parent.parent
        models_dir = project_root / "experiments" / "configs" / "models"

        models = {}
        for model_file in models_dir.glob("*.yaml"):
            with open(model_file, 'r') as f:
                model_config = yaml.safe_load(f)
                model_name = model_config['model_name']
                models[model_name] = model_config

        return models

    def get_model_params(self, model_name: str, dataset_name: str) -> Dict[str, Any]:
        """
        Get model parameters for a specific dataset from model config files.

        Args:
            model_name: Name of the model
            dataset_name: Name of the dataset

        Returns:
            Dictionary of model parameters
        """
        if model_name not in self.models_config:
            raise ValueError(f"Model {model_name} not found in configurations")

        model_config = self.models_config[model_name]

        # Get dataset-specific parameters
        dataset_params = model_config.get('dataset_params', {})
        if dataset_name not in dataset_params:
            raise ValueError(f"No parameters found for {model_name} on dataset {dataset_name}")

        return dataset_params[dataset_name].copy()

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
        methods_to_run: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run experiment on a single real-world dataset.

        Args:
            dataset_name: Name of the dataset
            loader_params: Parameters for loading the dataset
            methods_to_run: Specific methods to run (None = all enabled from model configs)

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

        # Determine which models to run
        if methods_to_run is not None:
            models_to_run = methods_to_run
        else:
            # Run all enabled models from model configs
            models_to_run = [name for name, config in self.models_config.items()
                           if config.get('enabled', True)]

        for method_name in models_to_run:
            if method_name not in method_runners:
                print(f"  ✗ Skipping unknown method: {method_name}")
                continue

            try:
                # Get model parameters from model config file
                params = self.get_model_params(method_name, dataset_name)

                result = self.run_method(
                    method_name.upper(),
                    method_runners[method_name],
                    dataset,
                    params
                )
                if result is not None:
                    results[method_name] = result
            except ValueError as e:
                print(f"  ✗ Skipping {method_name}: {e}")
                continue

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

        # Create dataset subdirectory under decompositions/real_world/
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

            # Save as JSON with lowercase filename
            output_path = dataset_dir / f"{model_name.lower()}.json"
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

        # Filter datasets
        datasets_to_run = self.filter_datasets(dataset_filter) if self.config else []

        # If no config file provided, use datasets from model configs
        if not datasets_to_run:
            # Get datasets from any model config (all should have the same datasets)
            if self.models_config:
                first_model = next(iter(self.models_config.values()))
                dataset_names = list(first_model.get('dataset_params', {}).keys())
                # Filter to real-world datasets only
                real_world_datasets = ['ETTh1', 'ETTh2', 'Sunspot']
                dataset_names = [d for d in dataset_names if d in real_world_datasets]

                if dataset_filter:
                    dataset_names = [d for d in dataset_names if d in dataset_filter]

                datasets_to_run = [{'name': name, 'loader_params': {}} for name in dataset_names]

        # Filter methods
        if method_filter:
            methods_to_run = [m.lower() for m in method_filter]
        else:
            # Get all enabled models from model configs
            methods_to_run = [name for name, config in self.models_config.items()
                            if config.get('enabled', True)]

        if dataset_filter:
            print(f"Running on datasets: {[d['name'] for d in datasets_to_run]}")
        if method_filter:
            print(f"Running methods: {methods_to_run}")

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
            if save_results:
                self.save_decomposition_components(
                    dataset_name,
                    experiment_result
                )

        self.results = all_results

        print("\n" + "="*70)
        print("ALL EXPERIMENTS COMPLETED")
        print("="*70)
        print(f"Results saved to: {self.output_dir}")
