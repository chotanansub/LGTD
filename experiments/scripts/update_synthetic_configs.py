#!/usr/bin/env python3
"""
Update all synthetic experiment configs to include all baseline models.
"""

import json
from pathlib import Path

# Template for all models with their default parameters
ALL_MODELS = {
    "lgtd": {
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
    "lgtd_linear": {
        "enabled": True,
        "params": {
            "window_size": 3,
            "error_percentile": 50,
            "lowess_frac": 0.1,
            "verbose": False
        }
    },
    "lgtd_lowess": {
        "enabled": True,
        "params": {
            "window_size": 3,
            "error_percentile": 50,
            "lowess_frac": 0.1,
            "verbose": False
        }
    },
    "stl": {
        "enabled": True,
        "params": {}  # Will be filled with period-specific values
    },
    "robust_stl": {
        "enabled": False,
        "params": {}  # Will be filled with period-specific values
    },
    "fast_robust_stl": {
        "enabled": True,
        "params": {}  # Will be filled with period-specific values
    },
    "str": {
        "enabled": True,
        "params": {}  # Will be filled with period-specific values
    },
    "online_stl": {
        "enabled": True,
        "params": {}  # Will be filled with period-specific values
    },
    "oneshot_stl": {
        "enabled": True,
        "params": {}  # Will be filled with period-specific values
    },
    "astd": {
        "enabled": True,
        "params": {
            "seasonality_smoothing": 0.7,
            "mode": "batch"
        }
    }
}

def get_period_specific_params(period):
    """Get period-specific parameters for baseline models."""
    return {
        "stl": {
            "period": period,
            "seasonal": 13,
            "trend": None,
            "robust": False
        },
        "robust_stl": {
            "period": period,
            "reg1": 10.0,
            "reg2": 0.5,
            "K": 2,
            "H": 5,
            "dn1": 1.0,
            "dn2": 1.0,
            "ds1": 50.0,
            "ds2": 1.0
        },
        "fast_robust_stl": {
            "period": period,
            "reg1": 10.0,
            "reg2": 0.5
        },
        "str": {
            "seasonal_periods": [period],
            "trend_lambda": 100.0,
            "seasonal_lambda": 1.0,
            "robust": False,
            "auto_params": False,
            "n_trials": 20
        },
        "online_stl": {
            "periods": [period],
            "lam": 0.3,
            "init_window_ratio": 0.3
        },
        "oneshot_stl": {
            "period": period,
            "shift_window": 0,
            "init_ratio": 0.3
        }
    }

def update_config_file(config_path):
    """Update a single config file to include all models."""
    print(f"Updating {config_path}...")

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Get period from dataset config
    period = config['dataset'].get('period', 120)

    # Get period-specific parameters
    period_params = get_period_specific_params(period)

    # Build new models config
    new_models = {}
    for model_name, model_config in ALL_MODELS.items():
        new_models[model_name] = model_config.copy()

        # Add period-specific params if they exist
        if model_name in period_params:
            new_models[model_name]['params'] = period_params[model_name]

    # Update config
    config['models'] = new_models

    # Write back
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"  ✓ Updated with {len(new_models)} models")

def main():
    config_dir = Path('experiments/configs/dataset_params')

    # Find all synthetic config files
    synth_configs = sorted(config_dir.glob('synth*_params.json'))

    print(f"Found {len(synth_configs)} synthetic config files\n")

    for config_path in synth_configs:
        update_config_file(config_path)

    print(f"\n✓ All synthetic configs updated!")

if __name__ == '__main__':
    main()
