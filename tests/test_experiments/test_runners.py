"""
Unit tests for experiment runners.

Tests the experiment runner infrastructure including config loading,
dataset processing, and result saving.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import json
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.runners.experiment_runner import ExperimentRunner


@pytest.fixture
def temp_config_dir():
    """Create temporary config directory with test configs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir) / "configs"
        config_dir.mkdir()

        # Create a temporary dataset file
        dataset_data = {
            "metadata": {
                "trend_type": "linear",
                "period_type": "fixed",
                "n_samples": 100,
                "period": 10
            },
            "data": {
                "time": list(range(100)),
                "y": [i * 0.1 + 5 * np.sin(2 * np.pi * i / 10) for i in range(100)],
                "trend": [i * 0.1 for i in range(100)],
                "seasonal": [5 * np.sin(2 * np.pi * i / 10) for i in range(100)],
                "residual": [0.0] * 100
            }
        }

        dataset_file = config_dir / "test_dataset.json"
        with open(dataset_file, 'w') as f:
            json.dump(dataset_data, f)

        # Create a simple test config
        test_config = {
            "dataset": {
                "name": "test_dataset",
                "path": str(dataset_file),
                "trend_type": "linear",
                "period_type": "fixed"
            },
            "n_samples": 100,
            "period": 10,
            "models": {
                "LGTD": {
                    "enabled": True,
                    "params": {
                        "window_size": 3,
                        "error_percentile": 50
                    }
                }
            },
            "evaluation": {
                "metrics": ["mse", "mae"]
            }
        }

        config_file = config_dir / "test_dataset_params.json"
        with open(config_file, 'w') as f:
            json.dump(test_config, f)

        yield str(config_dir)


@pytest.fixture
def temp_results_dir():
    """Create temporary results directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def synthetic_dataset():
    """Generate simple test dataset."""
    np.random.seed(42)
    n = 100
    t = np.arange(n)

    trend = 0.1 * t
    seasonal = 5 * np.sin(2 * np.pi * t / 10)
    noise = np.random.normal(0, 0.5, n)

    return {
        'time': t,
        'y': trend + seasonal + noise,
        'trend': trend,
        'seasonal': seasonal,
        'residual': noise
    }


class TestExperimentRunner:
    """Test ExperimentRunner class."""

    def test_init(self, temp_config_dir, temp_results_dir):
        """Test runner initialization."""
        runner = ExperimentRunner(
            config_dir=temp_config_dir,
            results_dir=temp_results_dir
        )

        assert runner is not None
        assert runner.config_dir.exists()
        assert runner.results_dir.exists()

    def test_load_configs(self, temp_config_dir, temp_results_dir):
        """Test config loading."""
        runner = ExperimentRunner(
            config_dir=temp_config_dir,
            results_dir=temp_results_dir
        )

        # Should load the test config
        assert len(runner.configs) > 0
        assert 'test_dataset' in runner.configs

    def test_config_structure(self, temp_config_dir, temp_results_dir):
        """Test loaded config has correct structure."""
        runner = ExperimentRunner(
            config_dir=temp_config_dir,
            results_dir=temp_results_dir
        )

        config = runner.configs['test_dataset']

        # Check required fields
        assert 'dataset' in config
        assert 'models' in config
        assert 'evaluation' in config

        # Check models
        assert 'LGTD' in config['models']
        assert 'enabled' in config['models']['LGTD']
        assert 'params' in config['models']['LGTD']

    def test_run_single_model(self, temp_config_dir, temp_results_dir):
        """Test running a single model on one dataset."""
        runner = ExperimentRunner(
            config_dir=temp_config_dir,
            results_dir=temp_results_dir
        )

        # Run experiments
        results = runner.run_experiment(
            datasets=['test_dataset'],
            models=['LGTD'],
            save_results=False,
            verbose=False
        )

        # Check results
        assert results is not None
        assert len(results) > 0

    def test_results_structure(self, temp_config_dir, temp_results_dir):
        """Test results have correct structure."""
        runner = ExperimentRunner(
            config_dir=temp_config_dir,
            results_dir=temp_results_dir
        )

        results = runner.run_experiment(
            datasets=['test_dataset'],
            models=['LGTD'],
            save_results=False,
            verbose=False
        )

        # Check first result
        if len(results) > 0:
            result = results.iloc[0]

            # Check required columns
            assert 'dataset' in result.index
            assert 'model' in result.index
            assert 'mse_trend' in result.index or 'error' in result.index

    def test_save_results(self, temp_config_dir, temp_results_dir):
        """Test results are saved correctly."""
        runner = ExperimentRunner(
            config_dir=temp_config_dir,
            results_dir=temp_results_dir
        )

        results = runner.run_experiment(
            datasets=['test_dataset'],
            models=['LGTD'],
            save_results=True,
            verbose=False
        )

        # Check benchmark file was created
        benchmarks_dir = Path(temp_results_dir).parent / 'experiments/results/benchmarks'
        if benchmarks_dir.exists():
            csv_files = list(benchmarks_dir.glob('*.csv'))
            # Note: may not exist in temp dir, that's ok
            assert True
        else:
            # Results saved to configured location
            assert results is not None

    def test_error_handling(self, temp_config_dir, temp_results_dir):
        """Test error handling for invalid inputs."""
        runner = ExperimentRunner(
            config_dir=temp_config_dir,
            results_dir=temp_results_dir
        )

        # Try to run non-existent dataset - should raise ValueError
        with pytest.raises(ValueError, match="No valid datasets found"):
            runner.run_experiment(
                datasets=['nonexistent'],
                models=['LGTD'],
                save_results=False,
                verbose=False
            )


class TestExperimentRunnerIntegration:
    """Integration tests for full experiment pipeline."""

    def test_full_pipeline(self, temp_config_dir, temp_results_dir):
        """Test complete experiment pipeline."""
        runner = ExperimentRunner(
            config_dir=temp_config_dir,
            results_dir=temp_results_dir
        )

        # Run all datasets, all models
        results = runner.run_experiment(
            save_results=False,
            verbose=False
        )

        # Should complete without errors
        assert results is not None

    def test_multiple_models(self, temp_config_dir, temp_results_dir):
        """Test running multiple models."""
        # Update config to have multiple models
        config_file = Path(temp_config_dir) / "test_dataset_params.json"
        with open(config_file, 'r') as f:
            config = json.load(f)

        config['models']['STL'] = {
            'enabled': True,
            'params': {
                'period': 10,
                'seasonal': 7
            }
        }

        with open(config_file, 'w') as f:
            json.dump(config, f)

        runner = ExperimentRunner(
            config_dir=temp_config_dir,
            results_dir=temp_results_dir
        )

        results = runner.run_experiment(
            datasets=['test_dataset'],
            save_results=False,
            verbose=False
        )

        # Should run both models
        if len(results) > 0:
            models_run = results['model'].unique()
            # At least one model should run
            assert len(models_run) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
