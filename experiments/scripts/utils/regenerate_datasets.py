#!/usr/bin/env python
"""
Regenerate variable period datasets (synth7, synth8, synth9) with more than 20 periods.
"""

import sys
import json
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.synthetic.generators import (
    generate_linear_trend,
    generate_inverted_v_trend,
    generate_piecewise_trend,
    generate_variable_period_seasonality,
    DEFAULT_N,
    DEFAULT_NOISE_STD,
    GLOBAL_SEED
)


def generate_variable_period_dataset(
    name: str,
    trend_type: str,
    n_periods: int = 25,
    n_samples: int = DEFAULT_N,
    noise_std: float = DEFAULT_NOISE_STD,
    seed: int = GLOBAL_SEED
) -> dict:
    """
    Generate dataset with variable period seasonality.

    Args:
        name: Dataset name (e.g., 'synth7')
        trend_type: 'linear', 'inverted_v', or 'piecewise'
        n_periods: Number of different periods (should be > 20)
        n_samples: Number of time points
        noise_std: Noise standard deviation
        seed: Random seed

    Returns:
        Dataset dictionary with data and metadata
    """
    np.random.seed(seed)

    # Generate time array
    time = np.arange(n_samples)

    # Generate trend component
    if trend_type == 'linear':
        trend = generate_linear_trend(time, slope=0.02)
    elif trend_type == 'inverted_v':
        trend = generate_inverted_v_trend(time)
    elif trend_type == 'piecewise':
        trend = generate_piecewise_trend(time)
    else:
        raise ValueError(f"Unknown trend type: {trend_type}")

    # Generate variable periods
    # Create diverse periods ranging from 50 to 500
    min_period = 50
    max_period = 500
    periods = []

    # Generate n_periods with some randomness but ensure diversity
    np.random.seed(seed)
    for i in range(n_periods):
        # Mix of regular intervals and some randomness
        if i % 3 == 0:
            # Short periods
            period = np.random.randint(min_period, 150)
        elif i % 3 == 1:
            # Medium periods
            period = np.random.randint(150, 350)
        else:
            # Long periods
            period = np.random.randint(350, max_period + 1)
        periods.append(period)

    # Sort periods for better distribution
    periods = sorted(periods)

    # Generate amplitude (different based on trend type)
    if trend_type == 'piecewise':
        amplitude = 80.0
    else:
        amplitude = 50.0

    # Generate variable seasonality
    seasonal = generate_variable_period_seasonality(
        time=time,
        periods=periods,
        amplitude=amplitude
    )

    # Generate noise
    np.random.seed(seed + 1)
    residual = np.random.normal(0, noise_std, n_samples)

    # Combine components
    y = trend + seasonal + residual

    # Create metadata
    meta = {
        "n": n_samples,
        "noise_std": noise_std,
        "seed": seed,
        "trend_type": trend_type,
        "period_type": "variable",
        "seasonal_amplitude": amplitude,
        "periods": periods,
        "n_periods": len(periods),
        "mean_period": float(np.mean(periods)),
        "min_period": int(np.min(periods)),
        "max_period": int(np.max(periods))
    }

    # Create dataset dictionary
    dataset = {
        "name": name,
        "data": {
            "time": time.tolist(),
            "y": y.tolist(),
            "trend": trend.tolist(),
            "seasonal": seasonal.tolist(),
            "residual": residual.tolist()
        },
        "meta": meta
    }

    return dataset


def main():
    """Regenerate synth7, synth8, synth9 with more than 20 periods."""

    datasets_to_generate = [
        {
            "name": "synth7",
            "trend_type": "linear",
            "n_periods": 25,
            "description": "Linear trend, Variable period (25 periods)"
        },
        {
            "name": "synth8",
            "trend_type": "inverted_v",
            "n_periods": 25,
            "description": "Inverted-V trend, Variable period (25 periods)"
        },
        {
            "name": "synth9",
            "trend_type": "piecewise",
            "n_periods": 25,
            "description": "Piecewise trend, Variable period (25 periods)"
        }
    ]

    print("=" * 80)
    print("REGENERATING VARIABLE PERIOD DATASETS (synth7, synth8, synth9)")
    print("=" * 80)
    print()

    output_dir = Path("data/synthetic/datasets")
    output_dir.mkdir(parents=True, exist_ok=True)

    for config in datasets_to_generate:
        print(f"Generating {config['name']}: {config['description']}...")

        # Generate dataset
        dataset = generate_variable_period_dataset(
            name=config['name'],
            trend_type=config['trend_type'],
            n_periods=config['n_periods'],
            n_samples=DEFAULT_N,
            noise_std=DEFAULT_NOISE_STD,
            seed=GLOBAL_SEED
        )

        # Save to file
        output_path = output_dir / f"{config['name']}_data.json"
        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)

        # Print summary
        meta = dataset['meta']
        print(f"  ✓ Saved to: {output_path}")
        print(f"    Samples: {meta['n']}")
        print(f"    Periods: {meta['n_periods']} (range: {meta['min_period']}-{meta['max_period']}, mean: {meta['mean_period']:.1f})")
        print(f"    Amplitude: {meta['seasonal_amplitude']}")
        print(f"    Noise: {meta['noise_std']}")
        print()

    print("=" * 80)
    print("REGENERATION COMPLETE")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. Sync configurations:")
    print("   python experiments/scripts/sync_dataset_configs.py")
    print()
    print("2. Run experiments:")
    print("   python experiments/scripts/run_experiments.py --datasets synth7 synth8 synth9")


if __name__ == '__main__':
    main()
