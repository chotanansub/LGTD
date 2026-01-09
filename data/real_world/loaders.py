"""
Data loaders for real-world datasets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any


def load_ett_dataset(dataset_name: str, column: str = 'OT', subset: str = 'all') -> Dict[str, np.ndarray]:
    """
    Load ETT (Electricity Transformer Temperature) dataset.

    Args:
        dataset_name: 'ETTh1' or 'ETTh2'
        column: Which column to use ('OT' for oil temperature, or other columns)
        subset: 'all', 'train', 'val', or 'test'

    Returns:
        Dictionary with 'y' (time series) and 'time' (indices)
    """
    data_path = Path('data/real_world/raw') / f'{dataset_name}.csv'

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)

    # Use specified column (default: OT - Oil Temperature)
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found. Available: {df.columns.tolist()}")

    y = df[column].values

    # Apply subset if specified
    if subset != 'all':
        n = len(y)
        if subset == 'train':
            y = y[:int(0.7 * n)]
        elif subset == 'val':
            y = y[int(0.7 * n):int(0.85 * n)]
        elif subset == 'test':
            y = y[int(0.85 * n):]

    return {
        'y': y,
        'time': np.arange(len(y))
    }


def load_sunspot_dataset(subset: str = 'all') -> Dict[str, np.ndarray]:
    """
    Load Sunspot dataset (monthly sunspot numbers).

    Args:
        subset: 'all', 'train', 'val', or 'test'

    Returns:
        Dictionary with 'y' (time series) and 'time' (indices)
    """
    data_path = Path('data/real_world/raw/sunspot.csv')

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)

    # Use sunspot_mean column
    y = df['sunspot_mean'].values

    # Apply subset if specified
    if subset != 'all':
        n = len(y)
        if subset == 'train':
            y = y[:int(0.7 * n)]
        elif subset == 'val':
            y = y[int(0.7 * n):int(0.85 * n)]
        elif subset == 'test':
            y = y[int(0.85 * n):]

    return {
        'y': y,
        'time': np.arange(len(y))
    }


def load_real_world_dataset(dataset_name: str, **kwargs) -> Dict[str, np.ndarray]:
    """
    Load any real-world dataset by name.

    Args:
        dataset_name: 'ETTh1', 'ETTh2', or 'Sunspot'
        **kwargs: Additional arguments passed to specific loaders

    Returns:
        Dictionary with 'y' (time series) and 'time' (indices)
    """
    if dataset_name in ['ETTh1', 'ETTh2']:
        return load_ett_dataset(dataset_name, **kwargs)
    elif dataset_name == 'Sunspot':
        return load_sunspot_dataset(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: ETTh1, ETTh2, Sunspot")
