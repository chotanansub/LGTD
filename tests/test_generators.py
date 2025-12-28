"""Tests for synthetic data generators."""

import pytest
import numpy as np
from data.synthetic.generators import (
    generate_linear_trend,
    generate_inverted_v_trend,
    generate_piecewise_trend,
    generate_fixed_period_seasonality,
    generate_transitive_period_seasonality,
    generate_variable_period_seasonality,
    generate_synthetic_data,
    generate_trend_series,
    generate_seasonal_series,
    generate_trend_seasonal_series
)


def test_generate_linear_trend():
    """Test linear trend generation."""
    time = np.arange(100)
    trend = generate_linear_trend(time, slope=0.5)

    assert len(trend) == 100
    assert trend[0] == 0
    assert np.isclose(trend[-1], 0.5 * 99)


def test_generate_inverted_v_trend():
    """Test inverted-V trend generation."""
    time = np.arange(100)
    trend = generate_inverted_v_trend(time, peak_position=0.5)

    assert len(trend) == 100
    # Peak should be near the middle
    peak_idx = np.argmax(trend)
    assert 40 <= peak_idx <= 60


def test_generate_piecewise_trend():
    """Test piecewise linear trend generation."""
    time = np.arange(100)
    trend = generate_piecewise_trend(time, n_segments=4)

    assert len(trend) == 100
    # Should be continuous
    assert not np.any(np.isnan(trend))


def test_generate_fixed_period_seasonality():
    """Test fixed period seasonality generation."""
    time = np.arange(120)
    seasonal = generate_fixed_period_seasonality(time, period=12, amplitude=10.0)

    assert len(seasonal) == 120
    # Should be periodic with period 12
    assert np.isclose(seasonal[0], seasonal[12], atol=0.1)
    # Amplitude check
    assert np.max(np.abs(seasonal)) <= 10.0 + 0.1


def test_generate_transitive_period_seasonality():
    """Test transitive period seasonality generation."""
    time = np.arange(100)
    seasonal = generate_transitive_period_seasonality(
        time,
        main_period=20,
        transition_period=10,
        amplitude=5.0
    )

    assert len(seasonal) == 100
    assert not np.any(np.isnan(seasonal))


def test_generate_variable_period_seasonality():
    """Test variable period seasonality generation."""
    time = np.arange(200)
    periods = [20, 30, 25]
    seasonal = generate_variable_period_seasonality(time, periods, amplitude=5.0)

    assert len(seasonal) == 200
    assert not np.any(np.isnan(seasonal))


def test_generate_synthetic_data_linear_fixed():
    """Test synthetic data generation with linear trend and fixed seasonality."""
    data = generate_synthetic_data(
        n=100,
        trend_type='linear',
        seasonality_type='fixed',
        seasonal_params={'period': 12, 'amplitude': 5.0},
        residual_std=0.1,
        seed=42
    )

    assert 'time' in data
    assert 'y' in data
    assert 'trend' in data
    assert 'seasonal' in data
    assert 'residual' in data
    assert 'config' in data

    assert len(data['time']) == 100
    assert len(data['y']) == 100
    assert len(data['trend']) == 100
    assert len(data['seasonal']) == 100
    assert len(data['residual']) == 100


def test_generate_synthetic_data_inverted_v():
    """Test synthetic data with inverted-V trend."""
    data = generate_synthetic_data(
        n=100,
        trend_type='inverted_v',
        seasonality_type='fixed',
        seed=42
    )

    assert data['config']['trend_type'] == 'inverted_v'
    # Check that trend has a peak
    peak_idx = np.argmax(data['trend'])
    assert peak_idx > 0 and peak_idx < len(data['trend']) - 1


def test_generate_synthetic_data_piecewise():
    """Test synthetic data with piecewise trend."""
    data = generate_synthetic_data(
        n=100,
        trend_type='piecewise',
        seasonality_type='fixed',
        seed=42
    )

    assert data['config']['trend_type'] == 'piecewise'


def test_generate_synthetic_data_transitive():
    """Test synthetic data with transitive seasonality."""
    data = generate_synthetic_data(
        n=200,
        trend_type='linear',
        seasonality_type='transitive',
        seasonal_params={'main_period': 30, 'transition_period': 15},
        seed=42
    )

    assert data['config']['seasonality_type'] == 'transitive'


def test_generate_synthetic_data_variable():
    """Test synthetic data with variable seasonality."""
    data = generate_synthetic_data(
        n=200,
        trend_type='linear',
        seasonality_type='variable',
        seasonal_params={'periods': [20, 30, 25]},
        seed=42
    )

    assert data['config']['seasonality_type'] == 'variable'


def test_generate_trend_series():
    """Test trend-only series generation."""
    data = generate_trend_series(n=100, trend_type='linear', seed=42)

    assert 'time' in data
    assert 'y' in data
    assert 'trend' in data
    assert len(data['y']) == 100


def test_generate_seasonal_series():
    """Test seasonal-only series generation."""
    data = generate_seasonal_series(n=120, period=12, amplitude=5.0, seed=42)

    assert 'time' in data
    assert 'y' in data
    assert 'seasonal' in data
    assert len(data['y']) == 120


def test_generate_trend_seasonal_series():
    """Test trend + seasonal series generation."""
    data = generate_trend_seasonal_series(
        n=100,
        trend_type='linear',
        period=12,
        amplitude=5.0,
        seed=42
    )

    assert 'time' in data
    assert 'y' in data
    assert 'trend' in data
    assert 'seasonal' in data
    assert len(data['y']) == 100


def test_generate_synthetic_data_reproducibility():
    """Test that same seed produces same results."""
    data1 = generate_synthetic_data(n=100, seed=42)
    data2 = generate_synthetic_data(n=100, seed=42)

    np.testing.assert_array_equal(data1['y'], data2['y'])
    np.testing.assert_array_equal(data1['trend'], data2['trend'])
    np.testing.assert_array_equal(data1['seasonal'], data2['seasonal'])


def test_generate_synthetic_data_invalid_trend():
    """Test error handling for invalid trend type."""
    with pytest.raises(ValueError):
        generate_synthetic_data(n=100, trend_type='invalid')


def test_generate_synthetic_data_invalid_seasonality():
    """Test error handling for invalid seasonality type."""
    with pytest.raises(ValueError):
        generate_synthetic_data(n=100, seasonality_type='invalid')
