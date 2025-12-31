#!/usr/bin/env python3
"""
Add OneShotSTL configuration to all dataset parameter files.
"""

import json
from pathlib import Path


def add_oneshot_stl_to_config(config_path: Path):
    """Add OneShotSTL configuration to a dataset config."""

    with open(config_path, 'r') as f:
        config = json.load(f)

    models = config.get('models', {})
    dataset = config.get('dataset', {})
    period = dataset.get('period', 120)

    # For variable period datasets, use a sensible default
    if isinstance(period, str) or period is None:
        period = 120  # Default period for variable datasets

    # Check if OneShotSTL already exists
    if 'OneShotSTL' in models:
        print(f"✓  {config_path.name}: OneShotSTL already exists")
        return False

    # Add OneShotSTL configuration
    models['OneShotSTL'] = {
        "enabled": True,
        "model_type": "online",
        "params": {
            "period": period,
            "shift_window": 0,
            "init_ratio": 0.3
        }
    }

    # Write back
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"✅ {config_path.name}: OneShotSTL added")
    return True


def main():
    """Add OneShotSTL to all dataset configurations."""

    configs_dir = Path('experiments/configs/dataset_params')

    print("📋 Adding OneShotSTL to all dataset configurations\n")

    added_count = 0
    for config_path in sorted(configs_dir.glob('synth*_params.json')):
        if add_oneshot_stl_to_config(config_path):
            added_count += 1

    print(f"\n✅ Added OneShotSTL to {added_count} dataset(s)")


if __name__ == '__main__':
    main()
