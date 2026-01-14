"""Tests for evaluation metrics."""

import pytest
import numpy as np
from lgtd.evaluation.metrics import (
    mean_squared_error,
    mean_absolute_error,
    root_mean_squared_error,
    correlation_coefficient,
    peak_signal_noise_ratio,
    compute_mse,
    compute_mae,
    compute_decomposition_metrics
)


def test_mean_squared_error():
    """Test MSE calculation."""
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 3, 4, 5])

    mse = mean_squared_error(y_true, y_pred)
    assert mse == 0.0

    y_pred = np.array([2, 3, 4, 5, 6])
    mse = mean_squared_error(y_true, y_pred)
    assert mse == 1.0


def test_mean_absolute_error():
    """Test MAE calculation."""
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 3, 4, 5])

    mae = mean_absolute_error(y_true, y_pred)
    assert mae == 0.0

    y_pred = np.array([2, 3, 4, 5, 6])
    mae = mean_absolute_error(y_true, y_pred)
    assert mae == 1.0


def test_root_mean_squared_error():
    """Test RMSE calculation."""
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 3, 4, 5])

    rmse = root_mean_squared_error(y_true, y_pred)
    assert rmse == 0.0

    y_pred = np.array([2, 3, 4, 5, 6])
    rmse = root_mean_squared_error(y_true, y_pred)
    assert rmse == 1.0


def test_correlation_coefficient():
    """Test correlation coefficient calculation."""
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 3, 4, 5])

    corr = correlation_coefficient(y_true, y_pred)
    assert np.isclose(corr, 1.0)

    y_pred = np.array([5, 4, 3, 2, 1])
    corr = correlation_coefficient(y_true, y_pred)
    assert np.isclose(corr, -1.0)


def test_peak_signal_noise_ratio():
    """Test PSNR calculation."""
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 3, 4, 5])

    psnr = peak_signal_noise_ratio(y_true, y_pred)
    assert psnr == float('inf')  # Perfect match

    y_pred = np.array([2, 3, 4, 5, 6])
    psnr = peak_signal_noise_ratio(y_true, y_pred)
    assert psnr > 0  # Positive PSNR for non-perfect match


def test_compute_mse():
    """Test MSE computation for decomposition components."""
    ground_truth = {
        'trend': np.array([1, 2, 3, 4, 5]),
        'seasonal': np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
        'residual': np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    }

    result = {
        'trend': np.array([1, 2, 3, 4, 5]),
        'seasonal': np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
        'residual': np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    }

    mse = compute_mse(ground_truth, result)

    assert 'trend' in mse
    assert 'seasonal' in mse
    assert 'residual' in mse
    assert mse['trend'] == 0.0
    assert mse['seasonal'] == 0.0
    assert mse['residual'] == 0.0


def test_compute_mae():
    """Test MAE computation for decomposition components."""
    ground_truth = {
        'trend': np.array([1, 2, 3, 4, 5]),
        'seasonal': np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
        'residual': np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    }

    result = {
        'trend': np.array([2, 3, 4, 5, 6]),
        'seasonal': np.array([0.2, 0.3, 0.4, 0.5, 0.6]),
        'residual': np.array([0.02, 0.03, 0.04, 0.05, 0.06])
    }

    mae = compute_mae(ground_truth, result)

    assert 'trend' in mae
    assert 'seasonal' in mae
    assert 'residual' in mae
    assert mae['trend'] == 1.0
    assert mae['seasonal'] == 0.1
    assert np.isclose(mae['residual'], 0.01)


def test_compute_decomposition_metrics():
    """Test comprehensive decomposition metrics computation."""
    ground_truth = {
        'trend': np.array([1, 2, 3, 4, 5]),
        'seasonal': np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
        'residual': np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    }

    result = {
        'trend': np.array([1, 2, 3, 4, 5]),
        'seasonal': np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
        'residual': np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    }

    metrics = compute_decomposition_metrics(ground_truth, result)

    assert 'trend' in metrics
    assert 'seasonal' in metrics
    assert 'residual' in metrics

    # Check that all expected metrics are present
    for component in ['trend', 'seasonal', 'residual']:
        assert 'mse' in metrics[component]
        assert 'mae' in metrics[component]
        assert 'rmse' in metrics[component]
        assert 'correlation' in metrics[component]
        assert 'psnr' in metrics[component]
