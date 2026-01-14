"""
Utility functions for LGTD decomposition.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
from statsmodels.nonparametric.smoothers_lowess import lowess as lowess_smooth
from typing import Tuple, Literal


def estimate_period(seq: np.ndarray, max_lag: int = None) -> int:
    """
    Estimate dominant period using autocorrelation.

    Args:
        seq: Input time series
        max_lag: Maximum lag to consider (default: len(seq)//2)

    Returns:
        Estimated period (default to len(seq)//4 if no clear period found)
    """
    n = len(seq)
    if max_lag is None:
        max_lag = min(n // 2, 500)  # Limit to 500 for efficiency

    # Compute autocorrelation
    mean = np.mean(seq)
    c0 = np.sum((seq - mean) ** 2) / n

    if c0 < 1e-10:  # Constant signal
        return n // 4

    acf = np.array([np.sum((seq[:n-k] - mean) * (seq[k:] - mean)) / n / c0
                    for k in range(1, max_lag)])

    # Find peaks in autocorrelation
    peaks, properties = find_peaks(acf, prominence=0.1, distance=5)

    if len(peaks) > 0:
        # Return the first significant peak as the period
        return int(peaks[0]) + 1  # +1 because we started from lag=1
    else:
        # No clear period found, use default
        return max(n // 4, 20)


def select_trend_method(
    seq: np.ndarray,
    threshold_r2: float = 0.92
) -> Literal['linear', 'lowess']:
    """
    Auto-select between linear and lowess based on data characteristics.

    This function uses a three-step approach:
    1. Estimate period using autocorrelation
    2. Rough deseasonalization using window size based on estimated period
    3. Test linearity on the smoothed trend using dual-threshold system

    Criteria for linear trend:
    - Primary threshold: R² >= threshold_r2 (default: 0.92)
    - Fallback threshold: R² >= 0.75 for harder cases (variable periods)
    - Monotonicity check via sign changes

    Args:
        seq: Input time series
        threshold_r2: Primary R² threshold for selecting linear trend (default: 0.92)

    Returns:
        Selected trend method ('linear' or 'lowess')
    """
    n = len(seq)

    # Step 1: Estimate period
    estimated_period = estimate_period(seq)

    # Step 2: Rough deseasonalization using period-based window
    # Use ~2 periods for effective deseasonalization
    window_size = min(2 * estimated_period + 1, n // 2)
    if window_size < 5:
        window_size = min(51, n // 4)

    # Use uniform (simple moving average) filter - works better for preserving linear trends
    # than Savitzky-Golay which can introduce curvature
    try:
        rough_trend = uniform_filter1d(seq, window_size, mode='nearest')
    except Exception:
        # Fallback to np.convolve
        if window_size % 2 == 0:
            window_size += 1
        rough_trend = np.convolve(seq, np.ones(window_size) / window_size, mode='same')

    # Step 3: Test linearity of the rough trend
    x = np.arange(n).reshape(-1, 1)
    model = LinearRegression()
    model.fit(x, rough_trend)
    y_pred = model.predict(x)

    # Metric 1: R² score
    r2 = r2_score(rough_trend, y_pred)

    # Metric 2: Monotonicity check (sign changes in first derivative)
    diff1 = np.diff(y_pred)
    sign_changes = np.sum(np.diff(np.sign(diff1)) != 0)

    # Metric 3: Relative residual standard deviation
    # This helps distinguish piecewise trends from true linear trends
    residuals = rough_trend - y_pred
    trend_range = np.ptp(rough_trend)
    if trend_range > 1e-10:
        relative_residual_std = np.std(residuals) / trend_range
    else:
        relative_residual_std = 0.0

    # Decision logic with enhanced dual-threshold system:
    # Primary threshold: Strict R² for clean linear cases
    # Fallback threshold: Relaxed R² with residual check for harder cases
    # Always require monotonicity (sign_changes <= 1)

    # Primary decision: strict threshold for clean linear trends
    if r2 >= threshold_r2 and sign_changes <= 1:
        return 'linear'

    # Fallback decision: relaxed threshold for harder cases (e.g., variable periods)
    # Additional check: relative_residual_std must be low to avoid piecewise trends
    # Piecewise trends have high R² but also high systematic residuals
    fallback_threshold_r2 = min(threshold_r2 * 0.82, 0.75)  # ~0.75 when threshold_r2=0.92
    fallback_threshold_std = 0.135  # Relative std threshold to detect piecewise patterns

    if (r2 >= fallback_threshold_r2 and
        sign_changes <= 1 and
        relative_residual_std <= fallback_threshold_std):
        return 'linear'

    # If neither threshold is met, use lowess
    return 'lowess'


def extract_linear_trend(seq: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Extract linear trend using linear regression.

    Args:
        seq: Input time series

    Returns:
        Tuple of (trend array, R² score)
    """
    x = np.arange(len(seq)).reshape(-1, 1)
    model = LinearRegression()
    model.fit(x, seq)
    trend = model.predict(x)
    r2 = r2_score(seq, trend)

    return trend, r2


def extract_lowess_trend(
    seq: np.ndarray,
    frac: float = 0.1
) -> Tuple[np.ndarray, float]:
    """
    Extract non-linear trend using LOWESS.

    Args:
        seq: Input time series
        frac: Fraction of data to use for smoothing

    Returns:
        Tuple of (trend array, R² score)
    """
    smoothed = lowess_smooth(
        seq,
        np.arange(len(seq)),
        frac=frac,
        return_sorted=False
    )
    r2 = r2_score(seq, smoothed)

    return smoothed, r2


def detrend(seq: np.ndarray, trend: np.ndarray) -> np.ndarray:
    """
    Remove trend from time series.

    Args:
        seq: Input time series
        trend: Trend component

    Returns:
        Detrended series
    """
    return seq - trend
