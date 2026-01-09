#!/usr/bin/env python3
"""
Run decomposition experiments on real-world datasets.

Processes real-world time series data (ETTh1, ETTh2, sunspot) using various
decomposition methods and saves results as JSON for plotting.
"""

import numpy as np
import pandas as pd
import json
import sys
from pathlib import Path

# Add LGTD to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from LGTD import LGTD


def load_dataset(dataset_name: str, n_points: int = 2500):
    """
    Load real-world dataset.

    Args:
        dataset_name: Name of dataset (ETTh1, ETTh2, sunspot)
        n_points: Number of data points to use

    Returns:
        numpy array of time series data
    """
    raw_dir = Path('data/real_world/raw')

    if dataset_name == 'sunspot':
        df = pd.read_csv(raw_dir / 'sunspot.csv')
        data = df['sunspot_mean'].values[:n_points]
    elif dataset_name == 'ETTh1':
        df = pd.read_csv(raw_dir / 'ETTh1.csv')
        data = df['OT'].values[:n_points]  # Oil Temperature
    elif dataset_name == 'ETTh2':
        df = pd.read_csv(raw_dir / 'ETTh2.csv')
        data = df['OT'].values[:n_points]  # Oil Temperature
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return data


def run_lgtd(data, params):
    """Run LGTD decomposition."""
    lgtd = LGTD(
        window_size=params.get('window_size', 3),
        error_percentile=params.get('error_percentile', 40),
        trend_selection=params.get('trend_selection', 'auto'),
        lowess_frac=params.get('lowess_frac', 0.15),
        threshold_r2=params.get('threshold_r2', 0.85),
        verbose=False
    )

    result = lgtd.fit_transform(data)

    return {
        'trend': result.trend.tolist(),
        'seasonal': result.seasonal.tolist(),
        'residual': result.residual.tolist(),
        'y': data.tolist()
    }


def run_stl(data, params):
    """Run STL decomposition."""
    from statsmodels.tsa.seasonal import STL

    period = params.get('period', 12)
    seasonal = params.get('seasonal', 13)
    robust = params.get('robust', False)

    stl = STL(data, period=period, seasonal=seasonal, robust=robust)
    result = stl.fit()

    return {
        'trend': result.trend.tolist(),
        'seasonal': result.seasonal.tolist(),
        'residual': result.resid.tolist(),
        'y': data.tolist()
    }


def run_online_stl(data, params):
    """Run OnlineSTL decomposition."""
    from experiments.baselines.online_stl import OnlineSTLDecomposer

    period = params.get('period', 12)
    periods = params.get('periods', [period])
    lam = params.get('lam', 0.3)
    init_window_ratio = params.get('init_window_ratio', 0.3)

    decomposer = OnlineSTLDecomposer(
        periods=periods,
        lam=lam,
        init_window_ratio=init_window_ratio
    )

    result = decomposer.decompose(data, period=period)

    return {
        'trend': result['trend'].tolist(),
        'seasonal': result['seasonal'].tolist(),
        'residual': result['residual'].tolist(),
        'y': data.tolist()
    }


def run_astd(data, params):
    """Run ASTD decomposition."""
    from experiments.baselines.astd import ASTDDecomposer

    seasonality_smoothing = params.get('seasonality_smoothing', 0.7)

    decomposer = ASTDDecomposer(seasonality_smoothing=seasonality_smoothing)
    result = decomposer.decompose(data)

    return {
        'trend': result['trend'].tolist(),
        'seasonal': result['seasonal'].tolist(),
        'residual': result['residual'].tolist(),
        'y': data.tolist()
    }


def run_astd_online(data, params):
    """Run ASTD Online decomposition."""
    from experiments.baselines.astd_online import ASTDOnlineDecomposer

    seasonality_smoothing = params.get('seasonality_smoothing', 0.7)
    init_window_size = params.get('init_window_size', 300)
    init_ratio = params.get('init_ratio', 0.3)

    decomposer = ASTDOnlineDecomposer(
        seasonality_smoothing=seasonality_smoothing,
        init_window_size=init_window_size,
        init_ratio=init_ratio
    )

    result = decomposer.decompose(data)

    return {
        'trend': result['trend'].tolist(),
        'seasonal': result['seasonal'].tolist(),
        'residual': result['residual'].tolist(),
        'y': data.tolist()
    }


def run_str(data, params):
    """Run STR decomposition."""
    from experiments.baselines.str_decomposer import STRDecomposer

    seasonal_periods = params.get('seasonal_periods', [12])
    trend_lambda = params.get('trend_lambda', 1000.0)
    seasonal_lambda = params.get('seasonal_lambda', 100.0)
    auto_params = params.get('auto_params', False)

    decomposer = STRDecomposer(
        seasonal_periods=seasonal_periods,
        trend_lambda=trend_lambda,
        seasonal_lambda=seasonal_lambda,
        auto_params=auto_params
    )

    result = decomposer.decompose(data)

    return {
        'trend': result['trend'].tolist(),
        'seasonal': result['seasonal'].tolist(),
        'residual': result['residual'].tolist(),
        'y': data.tolist()
    }


def run_robust_stl(data, params):
    """Run Robust STL decomposition."""
    from experiments.baselines.robust_stl import RobustSTLDecomposer

    period = params.get('period', 12)
    reg1 = params.get('reg1', 10.0)
    reg2 = params.get('reg2', 0.5)
    K = params.get('K', 2)
    H = params.get('H', 5)

    decomposer = RobustSTLDecomposer(
        period=period,
        reg1=reg1,
        reg2=reg2,
        K=K,
        H=H
    )

    result = decomposer.decompose(data)

    return {
        'trend': result['trend'].tolist(),
        'seasonal': result['seasonal'].tolist(),
        'residual': result['residual'].tolist(),
        'y': data.tolist()
    }


def run_fast_robust_stl(data, params):
    """Run Fast Robust STL decomposition."""
    from experiments.baselines.fast_robust_stl import FastRobustSTLDecomposer

    period = params.get('period', 12)
    reg1 = params.get('reg1', 1.0)
    reg2 = params.get('reg2', 10.0)
    K = params.get('K', 2)
    H = params.get('H', 5)

    decomposer = FastRobustSTLDecomposer(
        period=period,
        reg1=reg1,
        reg2=reg2,
        K=K,
        H=H
    )

    result = decomposer.decompose(data)

    return {
        'trend': result['trend'].tolist(),
        'seasonal': result['seasonal'].tolist(),
        'residual': result['residual'].tolist(),
        'y': data.tolist()
    }


def run_oneshot_stl(data, params):
    """Run OneShotSTL decomposition."""
    from experiments.baselines.oneshot_stl import OneShotSTLDecomposer

    period = params.get('period', 12)
    init_ratio = params.get('init_ratio', 0.3)
    shift_window = params.get('shift_window', 0)

    decomposer = OneShotSTLDecomposer(
        period=period,
        init_ratio=init_ratio,
        shift_window=shift_window
    )

    result = decomposer.decompose(data, period=period)

    return {
        'trend': result['trend'].tolist(),
        'seasonal': result['seasonal'].tolist(),
        'residual': result['residual'].tolist(),
        'y': data.tolist()
    }


# Model runners
MODEL_RUNNERS = {
    'LGTD': run_lgtd,
    'STL': run_stl,
    'OnlineSTL': run_online_stl,
    'ASTD': run_astd,
    'ASTD_Online': run_astd_online,
    'STR': run_str,
    'RobustSTL': run_robust_stl,
    'FastRobustSTL': run_fast_robust_stl,
    'OneShotSTL': run_oneshot_stl,
}


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Run real-world experiments')
    parser.add_argument(
        '--datasets', '-d',
        nargs='+',
        default=['sunspot', 'ETTh1', 'ETTh2'],
        help='Datasets to process'
    )
    parser.add_argument(
        '--models', '-m',
        nargs='+',
        default=list(MODEL_RUNNERS.keys()),
        help='Models to run'
    )
    parser.add_argument(
        '--n-points', '-n',
        type=int,
        default=2500,
        help='Number of data points to use (default: 2500)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='experiments/results/real_world/decompositions',
        help='Output directory for results'
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_dir = Path('experiments/configs/realworld_params')

    print("="*70)
    print("Real-World Dataset Experiments")
    print("="*70)
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Data points: {args.n_points}")
    print()

    for dataset_name in args.datasets:
        print(f"\n{'='*70}")
        print(f"Processing: {dataset_name}")
        print(f"{'='*70}")

        # Load data
        try:
            data = load_dataset(dataset_name, args.n_points)
            print(f"✓ Loaded {len(data)} data points")
        except Exception as e:
            print(f"✗ Failed to load dataset: {e}")
            continue

        # Load config if exists
        config_file = config_dir / f"{dataset_name}_params.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
            print(f"✓ Loaded config from {config_file}")
        else:
            print(f"⚠ No config found, using defaults")
            config = {'models': {}}

        # Run each model
        for model_name in args.models:
            if model_name not in MODEL_RUNNERS:
                print(f"  ⚠ Unknown model: {model_name}")
                continue

            print(f"  Running {model_name}... ", end='', flush=True)

            # Get model parameters from config or use defaults
            model_params = config.get('models', {}).get(model_name, {}).get('params', {})

            try:
                result = MODEL_RUNNERS[model_name](data, model_params)

                # Save result
                output_file = output_dir / f"{dataset_name}_{model_name}.json"
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)

                print(f"✓ Saved to {output_file}")

            except Exception as e:
                print(f"✗ Failed: {e}")
                continue

    print(f"\n{'='*70}")
    print("Experiments Complete!")
    print(f"{'='*70}")
    print(f"Results saved to: {output_dir}/")
    print()


if __name__ == '__main__':
    main()
