"""
Synthetic Time Series Data Generator for LGTD Research Experiments

This module provides functions to generate synthetic time series data with
different trend and seasonality patterns for evaluating decomposition methods.
"""

import numpy as np
from typing import Dict, List, Optional, Union


# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

DEFAULT_N = 2000
DEFAULT_NOISE_STD = 1.0
GLOBAL_SEED = 69


def set_all_seeds(seed: int = GLOBAL_SEED) -> None:
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)


# =============================================================================
# TREND GENERATORS
# =============================================================================

def generate_linear_trend(time: np.ndarray, slope: float = 0.02) -> np.ndarray:
    """
    Generate linear trend component.

    Args:
        time: Time array
        slope: Slope of the linear trend

    Returns:
        Linear trend array
    """
    return slope * time


def generate_inverted_v_trend(
    time: np.ndarray,
    peak_position: float = 0.6,
    max_height: float = 100.0,
    curve_sharpness: float = 3.0
) -> np.ndarray:
    """
    Generate inverted-V shaped trend component.

    Args:
        time: Time array
        peak_position: Position of peak (0-1)
        max_height: Maximum height of the peak
        curve_sharpness: Sharpness of the curve

    Returns:
        Inverted-V trend array
    """
    n = len(time)
    normalized_time = time / (n - 1)
    peak_idx = peak_position

    trend = np.where(
        normalized_time <= peak_idx,
        max_height * (normalized_time / peak_idx) ** curve_sharpness,
        max_height * ((1 - normalized_time) / (1 - peak_idx)) ** curve_sharpness
    )
    return trend


def generate_piecewise_trend(
    time: np.ndarray,
    n_segments: int = 4,
    slopes: Optional[List[float]] = None
) -> np.ndarray:
    """
    Generate piecewise linear trend component.

    Args:
        time: Time array
        n_segments: Number of segments
        slopes: List of slopes for each segment

    Returns:
        Piecewise linear trend array
    """
    n = len(time)
    if slopes is None:
        slopes = [0.15, -0.08, 0.20, -0.05]

    breakpoints = [int(n * i / n_segments) for i in range(n_segments + 1)]
    trend = np.zeros(n)

    for i in range(len(slopes)):
        start = breakpoints[i]
        end = min(breakpoints[i + 1], n)
        segment_length = end - start
        segment_trend = np.linspace(0, slopes[i] * segment_length, segment_length)

        if i > 0:
            offset = trend[start - 1]
            segment_trend += offset

        trend[start:end] = segment_trend

    return trend


# =============================================================================
# SEASONALITY GENERATORS
# =============================================================================

def generate_fixed_period_seasonality(
    time: np.ndarray,
    period: int,
    amplitude: float = 50.0
) -> np.ndarray:
    """
    Generate fixed-period seasonal component.

    Args:
        time: Time array
        period: Seasonal period
        amplitude: Seasonal amplitude

    Returns:
        Fixed-period seasonal array
    """
    return amplitude * np.sin(2 * np.pi * time / period)


def generate_transitive_period_seasonality(
    time: np.ndarray,
    main_period: int,
    transition_period: int,
    transition_start_ratio: float = 0.4,
    transition_end_ratio: float = 0.6,
    amplitude: float = 50.0
) -> np.ndarray:
    """
    Generate seasonal component with transitive period.

    Period transitions from main_period to transition_period and back.
    Example: [120, 120, 120, ..., 60, 60, 60, ..., 120, 120, 120]

    Args:
        time: Time array
        main_period: Main seasonal period
        transition_period: Period during transition phase
        transition_start_ratio: When transition starts (0-1)
        transition_end_ratio: When transition ends (0-1)
        amplitude: Seasonal amplitude

    Returns:
        Transitive-period seasonal array
    """
    n = len(time)
    transition_start = int(n * transition_start_ratio)
    transition_end = int(n * transition_end_ratio)

    seasonal = np.zeros(n)
    current_phase = 0

    for i in range(n):
        # Determine which period to use
        if transition_start <= i < transition_end:
            period = transition_period
        else:
            period = main_period

        # Generate sine wave with accumulated phase
        seasonal[i] = amplitude * np.sin(current_phase)
        current_phase += 2 * np.pi / period

    return seasonal


def generate_variable_period_seasonality(
    time: np.ndarray,
    periods: List[int],
    amplitude: float = 50.0
) -> np.ndarray:
    """
    Generate seasonal component with variable periods.

    Args:
        time: Time array
        periods: List of period lengths
        amplitude: Seasonal amplitude

    Returns:
        Variable-period seasonal array
    """
    n = len(time)
    seasonal = np.zeros(n)
    current_idx = 0
    cycle_count = 0

    while current_idx < n:
        period = periods[cycle_count % len(periods)]
        segment_length = min(period, n - current_idx)

        t_segment = np.linspace(0, 2 * np.pi * (segment_length / period), segment_length)
        seasonal[current_idx:current_idx + segment_length] = amplitude * np.sin(t_segment)

        current_idx += segment_length
        cycle_count += 1

    return seasonal


# =============================================================================
# UNIFIED SYNTHETIC DATA GENERATOR
# =============================================================================

def generate_synthetic_data(
    n: int = DEFAULT_N,
    trend_type: str = 'linear',
    seasonality_type: str = 'fixed',
    seasonal_params: Optional[Dict] = None,
    residual_std: float = DEFAULT_NOISE_STD,
    seed: Optional[int] = None
) -> Dict[str, Union[np.ndarray, Dict]]:
    """
    Unified synthetic data generator.

    Args:
        n: Number of time points
        trend_type: 'linear', 'inverted_v', or 'piecewise'
        seasonality_type: 'fixed', 'transitive', or 'variable'
        seasonal_params: Dictionary with seasonality parameters
        residual_std: Standard deviation of noise
        seed: Random seed

    Returns:
        Dictionary with time, y, trend, seasonal, residual, and config
    """
    if seed is not None:
        np.random.seed(seed)

    time = np.arange(n)

    # Generate trend
    if trend_type == 'linear':
        trend = generate_linear_trend(time, slope=0.02)
    elif trend_type == 'inverted_v':
        trend = generate_inverted_v_trend(time)
    elif trend_type == 'piecewise':
        trend = generate_piecewise_trend(time)
    else:
        raise ValueError(f"Unknown trend_type: {trend_type}")

    # Generate seasonality
    if seasonal_params is None:
        seasonal_params = {}

    if seasonality_type == 'fixed':
        period = seasonal_params.get('period', 120)
        amplitude = seasonal_params.get('amplitude', 50.0)
        seasonal = generate_fixed_period_seasonality(time, period, amplitude)

    elif seasonality_type == 'transitive':
        main_period = seasonal_params.get('main_period', 120)
        transition_period = seasonal_params.get('transition_period', 60)
        amplitude = seasonal_params.get('amplitude', 50.0)
        seasonal = generate_transitive_period_seasonality(
            time, main_period, transition_period, amplitude=amplitude
        )

    elif seasonality_type == 'variable':
        periods = seasonal_params.get('periods', [100, 300, 150, 400, 120, 350, 180, 450, 200, 250])
        amplitude = seasonal_params.get('amplitude', 50.0)
        seasonal = generate_variable_period_seasonality(time, periods, amplitude)

    else:
        raise ValueError(f"Unknown seasonality_type: {seasonality_type}")

    # Generate noise
    residual = np.random.normal(0, residual_std, n)

    # Combine components
    y = trend + seasonal + residual

    return {
        "time": time,
        "y": y,
        "trend": trend,
        "seasonal": seasonal,
        "residual": residual,
        "config": {
            "trend_type": trend_type,
            "seasonality_type": seasonality_type,
            "seasonal_params": seasonal_params
        }
    }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def generate_trend_series(
    n: int = DEFAULT_N,
    trend_type: str = 'linear',
    noise_std: float = DEFAULT_NOISE_STD,
    seed: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Generate time series with only trend component.

    Args:
        n: Number of time points
        trend_type: Type of trend ('linear', 'inverted_v', 'piecewise')
        noise_std: Standard deviation of noise
        seed: Random seed

    Returns:
        Dictionary with time, y, and trend
    """
    if seed is not None:
        np.random.seed(seed)

    time = np.arange(n)

    if trend_type == 'linear':
        trend = generate_linear_trend(time)
    elif trend_type == 'inverted_v':
        trend = generate_inverted_v_trend(time)
    elif trend_type == 'piecewise':
        trend = generate_piecewise_trend(time)
    else:
        raise ValueError(f"Unknown trend_type: {trend_type}")

    noise = np.random.normal(0, noise_std, n)
    y = trend + noise

    return {
        "time": time,
        "y": y,
        "trend": trend
    }


def generate_seasonal_series(
    n: int = DEFAULT_N,
    period: int = 120,
    amplitude: float = 50.0,
    noise_std: float = DEFAULT_NOISE_STD,
    seed: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Generate time series with only seasonal component.

    Args:
        n: Number of time points
        period: Seasonal period
        amplitude: Seasonal amplitude
        noise_std: Standard deviation of noise
        seed: Random seed

    Returns:
        Dictionary with time, y, and seasonal
    """
    if seed is not None:
        np.random.seed(seed)

    time = np.arange(n)
    seasonal = generate_fixed_period_seasonality(time, period, amplitude)
    noise = np.random.normal(0, noise_std, n)
    y = seasonal + noise

    return {
        "time": time,
        "y": y,
        "seasonal": seasonal
    }


def generate_trend_seasonal_series(
    n: int = DEFAULT_N,
    trend_type: str = 'linear',
    period: int = 120,
    amplitude: float = 50.0,
    noise_std: float = DEFAULT_NOISE_STD,
    seed: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Generate time series with trend and seasonal components.

    Args:
        n: Number of time points
        trend_type: Type of trend ('linear', 'inverted_v', 'piecewise')
        period: Seasonal period
        amplitude: Seasonal amplitude
        noise_std: Standard deviation of noise
        seed: Random seed

    Returns:
        Dictionary with time, y, trend, and seasonal
    """
    result = generate_synthetic_data(
        n=n,
        trend_type=trend_type,
        seasonality_type='fixed',
        seasonal_params={'period': period, 'amplitude': amplitude},
        residual_std=noise_std,
        seed=seed
    )

    return {
        "time": result["time"],
        "y": result["y"],
        "trend": result["trend"],
        "seasonal": result["seasonal"]
    }
