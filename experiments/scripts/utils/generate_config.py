#!/usr/bin/env python3
"""
Generate experiment configuration YAML from dataset JSON metadata.

This ensures dataset properties are only defined once (in JSON files)
and automatically synced to the experiment configuration.

Usage:
    python scripts/generate_experiment_config.py
"""

import json
import yaml
from pathlib import Path


def load_dataset_configs_from_json():
    """Generate dataset configurations from JSON metadata files."""
    datasets_dir = Path('data/synthetic/datasets')
    configs = []
    
    for i in range(1, 10):
        json_file = datasets_dir / f'synth{i}_data.json'
        if json_file.exists():
            with open(json_file, 'r') as f:
                data = json.load(f)
                meta = data.get('meta', {})
                
                config = {
                    'name': f'synth{i}_{meta.get("trend_type", "unknown")}_{meta.get("period_type", "unknown")}',
                    'trend_type': meta.get('trend_type', 'linear'),
                    'seasonality_type': meta.get('period_type', 'fixed'),
                    'seasonal_params': {
                        'amplitude': meta.get('seasonal_amplitude', 50.0),
                    },
                    'n_samples': meta.get('n', 2000),
                    'noise_std': meta.get('noise_std', 1.0)
                }
                
                # Add period-specific parameters
                if meta.get('period_type') == 'fixed':
                    config['seasonal_params']['period'] = meta.get('period')
                elif meta.get('period_type') == 'transitive':
                    config['seasonal_params']['main_period'] = meta.get('main_period')
                    config['seasonal_params']['transition_period'] = meta.get('transition_period')
                    config['seasonal_params']['transition_start_ratio'] = meta.get('transition_start_ratio', 0.4)
                    config['seasonal_params']['transition_end_ratio'] = meta.get('transition_end_ratio', 0.6)
                elif meta.get('period_type') == 'variable':
                    config['seasonal_params']['periods'] = meta.get('periods', [])
                
                configs.append(config)
    
    return configs


def generate_experiment_config(output_file='experiments/configs/synthetic_experiments.yaml'):
    """Generate full experiment configuration YAML."""
    
    # Base experiment configuration
    experiment_config = {
        'experiment': {
            'name': 'lgtd_synthetic_evaluation',
            'description': 'Synthetic data experiments for LGTD method validation',
            'output_dir': 'experiments/results/synthetic'
        },
        'datasets': load_dataset_configs_from_json(),
        'methods': {
            'lgtd': {
                'enabled': True,
                'params': {
                    'window_size': 3,
                    'error_percentile': 50,
                    'trend_selection': 'auto'
                }
            },
            'stl': {
                'enabled': True,
                'params': {
                    'period': 120,
                    'seasonal': 7
                }
            },
            'robust_stl': {
                'enabled': True,
                'params': {
                    'period': 120,
                    'reg1': 5.0,
                    'reg2': 0.1,
                    'K': 1,
                    'H': 2,
                    'dn1': 2.0,
                    'dn2': 2.0,
                    'ds1': 20.0,
                    'ds2': 2.0
                }
            },
            'astd': {
                'enabled': True,
                'params': {
                    'seasonality_smoothing': 0.7
                }
            }
        },
        'metrics': ['mse', 'mae', 'rmse', 'correlation', 'psnr'],
        'output': {
            'save_decompositions': True,
            'save_visualizations': True,
            'save_metrics': True,
            'results_format': 'csv'
        }
    }
    
    # Write to YAML file
    output_path = Path(output_file)
    with open(output_path, 'w') as f:
        yaml.dump(experiment_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✅ Generated experiment config: {output_path}")
    print(f"   Datasets: {len(experiment_config['datasets'])}")
    print(f"   Methods: {len(experiment_config['methods'])}")
    print(f"\nDatasets included:")
    for ds in experiment_config['datasets']:
        print(f"   - {ds['name']}")


if __name__ == '__main__':
    generate_experiment_config()
