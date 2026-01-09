#!/usr/bin/env python
"""
Synchronize dataset configurations between generated datasets and experiment configs.

This script:
1. Scans all dataset JSON files in data/synthetic/datasets/
2. Extracts metadata (period, amplitude, n_samples, trend_type, etc.)
3. Updates synthetic_experiments.yaml with dataset information
4. Creates/updates corresponding JSON parameter files in dataset_params/

Usage:
    # Sync all datasets
    python experiments/scripts/sync_dataset_configs.py

    # Sync specific datasets
    python experiments/scripts/sync_dataset_configs.py --datasets synth1 synth2

    # Dry run (show what would change without modifying files)
    python experiments/scripts/sync_dataset_configs.py --dry-run
"""

import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional


class DatasetConfigSynchronizer:
    """Synchronizes dataset metadata across configuration files."""

    def __init__(
        self,
        datasets_dir: str = "data/synthetic/datasets",
        yaml_config_path: str = "experiments/configs/synthetic_experiments.yaml",
        params_dir: str = "experiments/configs/dataset_params"
    ):
        self.datasets_dir = Path(datasets_dir)
        self.yaml_config_path = Path(yaml_config_path)
        self.params_dir = Path(params_dir)

        # Ensure directories exist
        self.params_dir.mkdir(parents=True, exist_ok=True)

    def load_dataset_metadata(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """Load metadata from a dataset JSON file."""
        dataset_path = self.datasets_dir / f"{dataset_name}_data.json"

        if not dataset_path.exists():
            print(f"⚠️  Dataset file not found: {dataset_path}")
            return None

        try:
            with open(dataset_path, 'r') as f:
                data = json.load(f)

            if 'meta' not in data:
                print(f"⚠️  No metadata found in {dataset_name}")
                return None

            return data['meta']

        except Exception as e:
            print(f"❌ Error loading {dataset_name}: {e}")
            return None

    def convert_meta_to_yaml_format(
        self,
        dataset_name: str,
        meta: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert dataset metadata to YAML config format."""

        # Map period_type to seasonality_type
        period_type = meta.get('period_type', 'fixed')
        seasonality_type = period_type  # same terminology

        # Build seasonal_params based on period_type
        seasonal_params = {
            'amplitude': meta.get('seasonal_amplitude', 50.0)
        }

        if period_type == 'fixed':
            seasonal_params['period'] = meta.get('period', 120)

        elif period_type == 'transitive':
            # For transitive, we need main_period and transition_period
            # These should be in meta, but fallback to defaults
            seasonal_params['main_period'] = meta.get('main_period', meta.get('period', 120))
            seasonal_params['transition_period'] = meta.get('transition_period', 60)

        elif period_type == 'variable':
            # For variable, we need a list of periods
            periods = meta.get('periods', [100, 300, 150, 400, 120, 350, 180, 450, 200, 250])
            seasonal_params['periods'] = periods

        # Build the YAML entry
        yaml_entry = {
            'name': f"{dataset_name}_{meta.get('trend_type', 'unknown')}_{period_type}",
            'trend_type': meta.get('trend_type', 'linear'),
            'seasonality_type': seasonality_type,
            'seasonal_params': seasonal_params,
            'n_samples': meta.get('n', 2000),
            'noise_std': meta.get('noise_std', 1.0)
        }

        return yaml_entry

    def create_param_json(
        self,
        dataset_name: str,
        meta: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create parameter JSON structure for a dataset."""

        period_type = meta.get('period_type', 'fixed')
        period_value = meta.get('period', 120)

        # For variable period, use string "variable"
        if period_type == 'variable':
            period_value = "variable"

        param_json = {
            "dataset": {
                "name": dataset_name,
                "path": f"data/synthetic/datasets/{dataset_name}_data.json",
                "trend_type": meta.get('trend_type', 'linear'),
                "period_type": period_type,
                "period": period_value
            },
            "models": {
                "LGTD": {
                    "enabled": True,
                    "params": {
                        "window_size": 3,
                        "error_percentile": 50,
                        "trend_selection": "auto",
                        "lowess_frac": 0.1,
                        "threshold_r2": 0.92,
                        "verbose": False
                    }
                },
                "LGTD_Linear": {
                    "enabled": True,
                    "params": {
                        "window_size": 3,
                        "error_percentile": 50,
                        "trend_selection": "linear",
                        "verbose": False
                    }
                },
                "LGTD_LOWESS": {
                    "enabled": True,
                    "params": {
                        "window_size": 3,
                        "error_percentile": 50,
                        "trend_selection": "lowess",
                        "lowess_frac": 0.1,
                        "verbose": False
                    }
                },
                "STL": {
                    "enabled": True,
                    "params": {
                        "period": meta.get('period', 120) if period_type != 'variable' else 120,
                        "seasonal": 13,
                        "trend": None,
                        "robust": False
                    }
                },
                "RobustSTL": {
                    "enabled": False,
                    "params": {
                        "period": meta.get('period', 120) if period_type != 'variable' else 120,
                        "reg1": 10.0,
                        "reg2": 0.5,
                        "K": 2,
                        "H": 5,
                        "dn1": 1.0,
                        "dn2": 1.0,
                        "ds1": 50.0,
                        "ds2": 1.0
                    }
                },
                "ASTD": {
                    "enabled": False,
                    "params": {
                        "period": meta.get('period', 120) if period_type != 'variable' else 120,
                        "alpha": 0.1,
                        "beta": 0.1
                    }
                }
            },
            "evaluation": {
                "metrics": [
                    "mse",
                    "mae",
                    "rmse",
                    "correlation",
                    "psnr"
                ],
                "save_plots": True,
                "plot_dir": f"experiments/results/synthetic/plots/{dataset_name}"
            }
        }

        return param_json

    def sync_datasets(
        self,
        dataset_names: Optional[List[str]] = None,
        dry_run: bool = False
    ):
        """
        Synchronize dataset configurations.

        Args:
            dataset_names: List of dataset names to sync (e.g., ['synth1', 'synth2'])
                          If None, scans all datasets in datasets_dir
            dry_run: If True, show what would change without modifying files
        """

        # Find all datasets if not specified
        if dataset_names is None:
            dataset_files = sorted(self.datasets_dir.glob("synth*_data.json"))
            dataset_names = [f.stem.replace('_data', '') for f in dataset_files]

        print(f"{'=' * 80}")
        print(f"DATASET CONFIGURATION SYNCHRONIZATION")
        print(f"Mode: {'DRY RUN' if dry_run else 'LIVE UPDATE'}")
        print(f"{'=' * 80}\n")

        if not dataset_names:
            print("No datasets found to synchronize.")
            return

        print(f"Found {len(dataset_names)} datasets to sync: {', '.join(dataset_names)}\n")

        # Load existing YAML config
        yaml_datasets = []
        if self.yaml_config_path.exists():
            with open(self.yaml_config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)

            # Keep existing methods, metrics, output settings
            methods = yaml_config.get('methods', {})
            metrics = yaml_config.get('metrics', [])
            output = yaml_config.get('output', {})
            experiment = yaml_config.get('experiment', {})
        else:
            # Default settings
            methods = {
                'lgtd': {'enabled': True, 'params': {'window_size': 3, 'error_percentile': 50, 'trend_selection': 'auto'}},
                'stl': {'enabled': True, 'params': {'period': 120, 'seasonal': 7}}
            }
            metrics = ['mse', 'mae', 'rmse', 'correlation', 'psnr']
            output = {'save_decompositions': True, 'save_visualizations': True, 'save_metrics': True, 'results_format': 'csv'}
            experiment = {'name': 'lgtd_synthetic_evaluation', 'description': 'Synthetic data experiments for LGTD method validation', 'output_dir': 'experiments/results/synthetic'}

        # Process each dataset
        for dataset_name in dataset_names:
            print(f"Processing {dataset_name}...")

            # Load metadata
            meta = self.load_dataset_metadata(dataset_name)
            if meta is None:
                continue

            # Create YAML entry
            yaml_entry = self.convert_meta_to_yaml_format(dataset_name, meta)
            yaml_datasets.append(yaml_entry)

            # Create/update param JSON
            param_json = self.create_param_json(dataset_name, meta)
            param_path = self.params_dir / f"{dataset_name}_params.json"

            if dry_run:
                print(f"  Would update: {param_path}")
                print(f"    Trend: {meta.get('trend_type')}, Period type: {meta.get('period_type')}, Period: {meta.get('period')}")
            else:
                with open(param_path, 'w') as f:
                    json.dump(param_json, f, indent=2)
                print(f"  ✓ Updated: {param_path}")

        # Update YAML config
        new_yaml_config = {
            'experiment': experiment,
            'datasets': yaml_datasets,
            'methods': methods,
            'metrics': metrics,
            'output': output
        }

        if dry_run:
            print(f"\nWould update: {self.yaml_config_path}")
            print(f"  Total datasets: {len(yaml_datasets)}")
        else:
            with open(self.yaml_config_path, 'w') as f:
                yaml.dump(new_yaml_config, f, default_flow_style=False, sort_keys=False, indent=2)
            print(f"\n✓ Updated: {self.yaml_config_path}")
            print(f"  Total datasets: {len(yaml_datasets)}")

        print(f"\n{'=' * 80}")
        print(f"SYNCHRONIZATION {'PREVIEW' if dry_run else 'COMPLETE'}")
        print(f"{'=' * 80}")


def main():
    parser = argparse.ArgumentParser(
        description='Synchronize dataset configurations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sync all datasets
  python experiments/scripts/sync_dataset_configs.py

  # Sync specific datasets
  python experiments/scripts/sync_dataset_configs.py --datasets synth1 synth2 synth10

  # Preview changes without modifying files
  python experiments/scripts/sync_dataset_configs.py --dry-run
        """
    )

    parser.add_argument(
        '--datasets',
        nargs='+',
        help='Specific datasets to sync (e.g., synth1 synth2). If not specified, syncs all.'
    )

    parser.add_argument(
        '--datasets-dir',
        default='data/synthetic/datasets',
        help='Directory containing dataset JSON files (default: data/synthetic/datasets)'
    )

    parser.add_argument(
        '--yaml-config',
        default='experiments/configs/synthetic_experiments.yaml',
        help='Path to YAML config file (default: experiments/configs/synthetic_experiments.yaml)'
    )

    parser.add_argument(
        '--params-dir',
        default='experiments/configs/dataset_params',
        help='Directory for parameter JSON files (default: experiments/configs/dataset_params)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without modifying files'
    )

    args = parser.parse_args()

    # Create synchronizer
    syncer = DatasetConfigSynchronizer(
        datasets_dir=args.datasets_dir,
        yaml_config_path=args.yaml_config,
        params_dir=args.params_dir
    )

    # Sync datasets
    syncer.sync_datasets(
        dataset_names=args.datasets,
        dry_run=args.dry_run
    )


if __name__ == '__main__':
    main()
