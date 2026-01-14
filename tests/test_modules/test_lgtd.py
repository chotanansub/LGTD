"""Tests for LGTD decomposition."""

import pytest
import numpy as np
from lgtd import LGTD
from lgtd.decomposition.lgtd import LGTDResult


def test_lgtd_initialization():
    """Test LGTD can be initialized."""
    model = LGTD()
    assert model is not None
    assert model.window_size == 3
    assert model.error_percentile == 50
    assert model.trend_selection == 'auto'


def test_lgtd_initialization_with_params():
    """Test LGTD initialization with custom parameters."""
    model = LGTD(
        window_size=5,
        error_percentile=75,
        trend_selection='linear',
        lowess_frac=0.2
    )
    assert model.window_size == 5
    assert model.error_percentile == 75
    assert model.trend_selection == 'linear'
    assert model.lowess_frac == 0.2


def test_lgtd_fit_transform():
    """Test LGTD fit_transform on simple data."""
    # Generate simple test data
    t = np.arange(100)
    data = 0.5 * t + 10 * np.sin(2 * np.pi * t / 12)

    model = LGTD()
    result = model.fit_transform(data)

    assert isinstance(result, LGTDResult)
    assert hasattr(result, 'trend')
    assert hasattr(result, 'seasonal')
    assert hasattr(result, 'residual')
    assert len(result.trend) == len(data)
    assert len(result.seasonal) == len(data)
    assert len(result.residual) == len(data)


def test_lgtd_decomposition_components_sum():
    """Test that decomposition components sum to original series."""
    t = np.arange(100)
    data = 0.5 * t + 10 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 0.1, 100)

    model = LGTD()
    result = model.fit_transform(data)

    # Check that trend + seasonal + residual ≈ original
    reconstructed = result.trend + result.seasonal + result.residual
    np.testing.assert_allclose(reconstructed, data, rtol=1e-10)


def test_lgtd_with_noise():
    """Test LGTD with noisy data."""
    np.random.seed(42)
    t = np.arange(200)
    trend = 0.3 * t
    seasonal = 15 * np.sin(2 * np.pi * t / 24)
    noise = np.random.normal(0, 2, 200)
    data = trend + seasonal + noise

    model = LGTD(window_size=3, verbose=False)
    result = model.fit_transform(data)

    # Basic sanity checks
    assert len(result.trend) == 200
    assert len(result.seasonal) == 200
    assert len(result.residual) == 200
    assert not np.any(np.isnan(result.trend))
    assert not np.any(np.isnan(result.seasonal))


def test_lgtd_properties():
    """Test LGTD properties after fitting."""
    t = np.arange(50)
    data = 0.5 * t + 5 * np.sin(2 * np.pi * t / 12)

    model = LGTD()
    model.fit_transform(data)

    assert model.trend is not None
    assert model.seasonal is not None
    assert model.residual is not None
    assert model.detected_periods is not None


def test_lgtd_with_linear_trend():
    """Test LGTD with explicit linear trend selection."""
    t = np.arange(100)
    data = 2 * t + np.random.normal(0, 1, 100)

    model = LGTD(trend_selection='linear')
    result = model.fit_transform(data)

    assert result.trend_info['method'] == 'linear'
    # Linear trend should fit well
    assert result.trend_info['r2'] > 0.9


def test_lgtd_with_lowess_trend():
    """Test LGTD with LOWESS trend selection."""
    t = np.arange(100)
    # Non-linear trend
    data = 0.01 * t**2 + np.random.normal(0, 1, 100)

    model = LGTD(trend_selection='lowess', lowess_frac=0.3)
    result = model.fit_transform(data)

    assert result.trend_info['method'] == 'lowess'


def test_lgtd_invalid_input():
    """Test LGTD with invalid input."""
    model = LGTD()

    # 2D array should raise error
    with pytest.raises(ValueError):
        model.fit_transform(np.array([[1, 2], [3, 4]]))


def test_lgtd_empty_input():
    """Test LGTD with empty input."""
    model = LGTD()
    data = np.array([])

    # Should handle gracefully or raise appropriate error
    try:
        result = model.fit_transform(data)
        assert len(result.trend) == 0
    except (ValueError, IndexError):
        # Expected behavior for empty input
        pass
