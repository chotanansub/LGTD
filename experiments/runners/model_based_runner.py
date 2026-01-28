"""
Model-based experiment runner for LGTD evaluation.

This runner uses the new model-based configuration structure:
- experiments/configs/datasets.yaml - Dataset definitions
- experiments/configs/models/*.yaml - Model-specific configurations
- experiments/configs/experiment_settings.yaml - General settings
"""

import json
import time
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List

from lgtd import lgtd
from experiments.baselines.stl import STLDecomposer
from experiments.baselines.robust_stl import RobustSTLDecomposer
from experiments.baselines.fast_robust_stl import FastRobustSTLDecomposer
from experiments.baselines.str_decomposer import STRDecomposer
from experiments.baselines.online_stl import OnlineSTLDecomposer
from experiments.baselines.oneshot_stl import OneShotSTLDecomposer
from experiments.baselines.astd import ASTDDecomposer


class ModelBasedExperimentRunner:
    """
    Experiment runner using model-based configuration structure.

    Configuration files:
    - datasets.yaml: Dataset definitions
    - models/*.yaml: Per-model configurations with dataset overrides
    - experiment_settings.yaml: General experiment settings
    """

    def __init__(
        self,
        datasets_config: str = "experiments/configs/datasets.yaml",
        models_dir: str = "experiments/configs/models",
        settings_config: str = "experiments/configs/experiment_settings.yaml",
        results_dir: str = "experiments/results/synthetic"
    ):
        """
        Initialize model-based experiment runner.

        Args:
            datasets_config: Path to datasets YAML file
            models_dir: Directory containing model YAML files
            settings_config: Path to experiment settings YAML file
            results_dir: Directory to save results
        """
        self.datasets_config_path = Path(datasets_config)
        self.models_dir = Path(models_dir)
        self.settings_config_path = Path(settings_config)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Load configurations
        self.datasets = self._load_datasets()
        self.models_config = self._load_models()
        self.settings = self._load_settings()

    def _load_datasets(self) -> Dict[str, Dict]:
        """Load dataset configurations."""
        with open(self.datasets_config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Convert list to dict keyed by dataset name
        datasets = {}
        for dataset in config['datasets']:
            datasets[dataset['name']] = dataset

        return datasets

    def _load_models(self) -> Dict[str, Dict]:
        """Load all model configurations."""
        models = {}
        for model_file in self.models_dir.glob("*.yaml"):
            with open(model_file, 'r') as f:
                model_config = yaml.safe_load(f)
                model_name = model_config['model_name']
                models[model_name] = model_config

        return models

    def _load_settings(self) -> Dict[str, Any]:
        """Load experiment settings."""
        with open(self.settings_config_path, 'r') as f:
            return yaml.safe_load(f)

    def get_model_params(self, model_name: str, dataset_name: str) -> Dict[str, Any]:
        """
        Get model parameters for a specific dataset.

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

    def _load_dataset(self, dataset_name: str) -> Dict[str, np.ndarray]:
        """Load dataset from configuration."""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not found")

        dataset_config = self.datasets[dataset_name]
        dataset_type = dataset_config.get('type', 'synthetic')

        if dataset_type == 'real_world':
            # Load real-world dataset
            from data.real_world.loaders import load_ett_dataset, load_sunspot_dataset

            loader_name = dataset_config.get('loader')
            loader_params = dataset_config.get('loader_params', {})

            if 'ETT' in dataset_name:
                data = load_ett_dataset(dataset_name, **loader_params)
            elif dataset_name == 'Sunspot':
                data = load_sunspot_dataset(**loader_params)
            else:
                raise ValueError(f"Unknown real-world dataset: {dataset_name}")

            # Limit to first 2500 points
            max_points = 2500
            y_data = data['y'][:max_points]

            return {
                'y': y_data,
                'time': np.arange(len(y_data)),
                'trend': None,
                'seasonal': None,
                'residual': None
            }
        else:
            # Load synthetic dataset from JSON file
            dataset_path = dataset_config['path']
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

        model = lgtd(**params)
        result = model.fit_transform(data)

        elapsed_time = time.time() - start_time

        return {
            'trend': result.trend,
            'seasonal': result.seasonal,
            'residual': result.residual,
            'time': elapsed_time,
        }

    def _run_stl(self, data: np.ndarray, params: Dict) -> Dict[str, Any]:
        """Run STL decomposition."""
        start_time = time.time()

        model = STLDecomposer(**params)
        result = model.fit_transform(data)

        elapsed_time = time.time() - start_time

        return {
            'trend': result['trend'],
            'seasonal': result['seasonal'],
            'residual': result['residual'],
            'time': elapsed_time,
        }

    def _run_fast_robust_stl(self, data: np.ndarray, params: Dict) -> Dict[str, Any]:
        """Run FastRobustSTL decomposition."""
        start_time = time.time()

        model = FastRobustSTLDecomposer(**params)
        result = model.fit_transform(data)

        elapsed_time = time.time() - start_time

        return {
            'trend': result['trend'],
            'seasonal': result['seasonal'],
            'residual': result['residual'],
            'time': elapsed_time,
        }

    def _run_str(self, data: np.ndarray, params: Dict) -> Dict[str, Any]:
        """Run STR decomposition."""
        start_time = time.time()

        model = STRDecomposer(**params)
        result = model.fit_transform(data)

        elapsed_time = time.time() - start_time

        return {
            'trend': result['trend'],
            'seasonal': result['seasonal'],
            'residual': result['residual'],
            'time': elapsed_time,
        }

    def _run_online_stl(self, data: np.ndarray, params: Dict) -> Dict[str, Any]:
        """Run OnlineSTL decomposition."""
        start_time = time.time()

        model = OnlineSTLDecomposer(**params)
        result = model.fit_transform(data)

        elapsed_time = time.time() - start_time

        return {
            'trend': result['trend'],
            'seasonal': result['seasonal'],
            'residual': result['residual'],
            'time': elapsed_time,
        }

    def _run_oneshot_stl(self, data: np.ndarray, params: Dict) -> Dict[str, Any]:
        """Run OneShotSTL decomposition."""
        start_time = time.time()

        model = OneShotSTLDecomposer(**params)
        result = model.fit_transform(data)

        elapsed_time = time.time() - start_time

        return {
            'trend': result['trend'],
            'seasonal': result['seasonal'],
            'residual': result['residual'],
            'time': elapsed_time,
        }

    def _run_astd(self, data: np.ndarray, params: Dict) -> Dict[str, Any]:
        """Run ASTD decomposition."""
        start_time = time.time()

        model = ASTDDecomposer(**params)
        result = model.fit_transform(data)

        elapsed_time = time.time() - start_time

        return {
            'trend': result['trend'],
            'seasonal': result['seasonal'],
            'residual': result['residual'],
            'time': elapsed_time,
        }

    def run_model_on_dataset(
        self,
        model_name: str,
        dataset_name: str,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run a specific model on a specific dataset.

        Args:
            model_name: Name of the model
            dataset_name: Name of the dataset
            verbose: Whether to print progress

        Returns:
            Dictionary containing decomposition results and metrics
        """
        if verbose:
            print(f"Running {model_name} on {dataset_name}...")

        # Load dataset
        dataset = self._load_dataset(dataset_name)

        # Get model parameters for this dataset
        params = self.get_model_params(model_name, dataset_name)

        # Run the appropriate model
        if model_name in ['lgtd', 'lgtd_linear', 'lgtd_lowess']:
            result = self._run_lgtd_variant(dataset['y'], model_name, params)
        elif model_name == 'stl':
            result = self._run_stl(dataset['y'], params)
        elif model_name == 'fast_robust_stl':
            result = self._run_fast_robust_stl(dataset['y'], params)
        elif model_name == 'str':
            result = self._run_str(dataset['y'], params)
        elif model_name == 'online_stl':
            result = self._run_online_stl(dataset['y'], params)
        elif model_name == 'oneshot_stl':
            result = self._run_oneshot_stl(dataset['y'], params)
        elif model_name in ['astd', 'astd_online']:
            result = self._run_astd(dataset['y'], params)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Add dataset info to result
        result['dataset'] = dataset
        result['model_name'] = model_name
        result['dataset_name'] = dataset_name
        result['params'] = params

        if verbose:
            print(f"  Completed in {result['time']:.2f}s")

        return result

    def run_all_experiments(
        self,
        dataset_filter: Optional[List[str]] = None,
        model_filter: Optional[List[str]] = None,
        verbose: bool = True
    ):
        """
        Run experiments on multiple datasets and models.

        Args:
            dataset_filter: List of dataset names to run (None = all)
            model_filter: List of model names to run (None = all enabled)
            verbose: Whether to print progress
        """
        # Filter datasets
        datasets_to_run = list(self.datasets.keys())
        if dataset_filter:
            datasets_to_run = [d for d in datasets_to_run if d in dataset_filter]

        # Filter models (only enabled ones)
        models_to_run = [m for m, config in self.models_config.items() if config.get('enabled', True)]
        if model_filter:
            models_to_run = [m for m in models_to_run if m in model_filter]

        if verbose:
            print(f"Running {len(models_to_run)} models on {len(datasets_to_run)} datasets")
            print(f"Models: {', '.join(models_to_run)}")
            print(f"Datasets: {', '.join(datasets_to_run)}")
            print()

        results = {}
        for dataset_name in datasets_to_run:
            results[dataset_name] = {}
            for model_name in models_to_run:
                try:
                    result = self.run_model_on_dataset(model_name, dataset_name, verbose=verbose)
                    results[dataset_name][model_name] = result
                except Exception as e:
                    if verbose:
                        print(f"  ERROR: {e}")
                    results[dataset_name][model_name] = {'error': str(e)}

        return results


def main():
    """Example usage."""
    runner = ModelBasedExperimentRunner()

    # Example: Run LGTD on synth1
    result = runner.run_model_on_dataset('lgtd', 'synth1')
    print(f"LGTD parameters used: {result['params']}")
    print(f"Execution time: {result['time']:.3f}s")


if __name__ == '__main__':
    main()
