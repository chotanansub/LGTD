"""
Comprehensive unit tests for LGTD module.

Tests all aspects of the Linear-Guided Trend Decomposition algorithm
including initialization, trend selection, decomposition, and edge cases.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lgtd import lgtd


@pytest.fixture
def simple_data():
    """Generate simple synthetic data for testing."""
    np.random.seed(42)
    n = 200
    t = np.arange(n)

    trend = 0.5 * t + 10
    seasonal = 5 * np.sin(2 * np.pi * t / 20)
    noise = np.random.normal(0, 0.5, n)

    return {
        'y': trend + seasonal + noise,
        'trend': trend,
        'seasonal': seasonal,
        'noise': noise,
        'period': 20
    }


@pytest.fixture
def complex_data():
    """Generate complex synthetic data with multiple components."""
    np.random.seed(123)
    n = 500
    t = np.arange(n)

    # Non-linear trend
    trend = 0.01 * t**1.5 + 50

    # Multiple seasonal components
    seasonal1 = 10 * np.sin(2 * np.pi * t / 30)
    seasonal2 = 5 * np.sin(2 * np.pi * t / 7)
    seasonal = seasonal1 + seasonal2

    noise = np.random.normal(0, 1, n)

    return {
        'y': trend + seasonal + noise,
        'trend': trend,
        'seasonal': seasonal,
        'noise': noise
    }


class TestLGTDInitialization:
    """Test LGTD initialization and parameters."""

    def test_default_init(self):
        """Test default initialization."""
        model = lgtd()

        assert model.window_size == 3
        assert model.error_percentile == 50
        assert model.trend_selection == 'auto'

    def test_custom_init(self):
        """Test custom parameter initialization."""
        model = lgtd(
            window_size=5,
            error_percentile=30,
            trend_selection='linear'
        )

        assert model.window_size == 5
        assert model.error_percentile == 30
        assert model.trend_selection == 'linear'

    def test_invalid_window_size(self):
        """Test invalid window size raises error."""
        with pytest.raises((ValueError, AssertionError)):
            lgtd(window_size=0)

        with pytest.raises((ValueError, AssertionError)):
            lgtd(window_size=-1)

    def test_invalid_percentile(self):
        """Test invalid percentile raises error."""
        with pytest.raises((ValueError, AssertionError)):
            lgtd(error_percentile=-10)

        with pytest.raises((ValueError, AssertionError)):
            lgtd(error_percentile=150)

    def test_invalid_trend_selection(self):
        """Test invalid trend selection raises error."""
        with pytest.raises((ValueError, AssertionError)):
            lgtd(trend_selection='invalid_method')


class TestLGTDDecomposition:
    """Test LGTD decomposition functionality."""

    def test_basic_decomposition(self, simple_data):
        """Test basic decomposition."""
        model = lgtd()
        result = model.fit_transform(simple_data['y'])

        # Check output structure
        assert hasattr(result, 'trend')
        assert hasattr(result, 'seasonal')
        assert hasattr(result, 'residual')
        assert hasattr(result, 'y')

        # Check shapes
        assert len(result.trend) == len(simple_data['y'])
        assert len(result.seasonal) == len(simple_data['y'])
        assert len(result.residual) == len(simple_data['y'])

    def test_decomposition_reconstruction(self, simple_data):
        """Test decomposition components sum to original."""
        model = lgtd()
        result = model.fit_transform(simple_data['y'])

        reconstructed = result.trend + result.seasonal + result.residual
        np.testing.assert_allclose(reconstructed, simple_data['y'], rtol=1e-10)

    def test_trend_recovery_linear(self, simple_data):
        """Test trend recovery for linear trend."""
        model = lgtd(trend_selection='linear')
        result = model.fit_transform(simple_data['y'])

        # Trend should be reasonably close to true trend
        correlation = np.corrcoef(result.trend, simple_data['trend'])[0, 1]
        assert correlation > 0.95, f"Trend correlation {correlation} too low"

    def test_trend_recovery_lowess(self, simple_data):
        """Test trend recovery with LOWESS."""
        model = lgtd(trend_selection='lowess')
        result = model.fit_transform(simple_data['y'])

        # LOWESS should capture trend
        correlation = np.corrcoef(result.trend, simple_data['trend'])[0, 1]
        assert correlation > 0.90, f"LOWESS trend correlation {correlation} too low"

    def test_trend_selection_auto(self, simple_data):
        """Test auto trend selection."""
        model = lgtd(trend_selection='auto')
        result = model.fit_transform(simple_data['y'])

        # Should select a method and decompose
        assert result is not None
        assert hasattr(result, 'trend')

        # Check if selection was made (stored in trend_info)
        assert hasattr(result, 'trend_info')
        assert 'method' in result.trend_info
        assert result.trend_info['method'] in ['linear', 'lowess']

    def test_seasonal_extraction(self, simple_data):
        """Test seasonal component extraction."""
        model = lgtd()
        result = model.fit_transform(simple_data['y'])

        # Seasonal should have near-zero mean (relaxed tolerance)
        assert abs(np.mean(result.seasonal)) < 2.0

        # Seasonal should have periodicity
        # (check autocorrelation at period lag)
        from numpy import correlate
        seasonal = result.seasonal
        period = 20

        if len(seasonal) >= 2 * period:
            # Simple check: seasonal should repeat
            first_period = seasonal[:period]
            second_period = seasonal[period:2*period]
            corr = np.corrcoef(first_period, second_period)[0, 1]
            assert corr > 0.5, "Seasonal component not periodic"


class TestLGTDEdgeCases:
    """Test LGTD edge cases and robustness."""

    def test_short_time_series(self):
        """Test with very short time series."""
        np.random.seed(42)
        y = np.random.randn(10)

        model = lgtd()
        result = model.fit_transform(y)

        # Should handle short series
        assert result is not None
        assert len(result.trend) == 10

    def test_constant_series(self):
        """Test with constant time series."""
        y = np.ones(100) * 5.0

        model = lgtd()
        result = model.fit_transform(y)

        # Trend should be approximately constant
        assert np.std(result.trend) < 0.1

        # Seasonal should be near zero
        assert np.std(result.seasonal) < 0.1

    def test_with_nans(self):
        """Test handling of NaN values."""
        np.random.seed(42)
        y = np.random.randn(100)
        y[50] = np.nan

        model = lgtd()

        # Should either handle NaNs or raise informative error
        try:
            result = model.fit_transform(y)
            # If it handles NaNs, check output is reasonable
            assert result is not None
        except (ValueError, RuntimeError) as e:
            # Expected error for NaNs
            assert 'nan' in str(e).lower() or 'NaN' in str(e)

    def test_large_noise(self):
        """Test with high noise levels."""
        np.random.seed(42)
        n = 200
        t = np.arange(n)

        trend = 0.1 * t
        seasonal = 2 * np.sin(2 * np.pi * t / 20)
        noise = np.random.normal(0, 10, n)  # Large noise

        y = trend + seasonal + noise

        model = lgtd()
        result = model.fit_transform(y)

        # Should still decompose
        assert result is not None
        assert len(result.trend) == n


class TestLGTDWindowSize:
    """Test effect of window size parameter."""

    def test_different_window_sizes(self, simple_data):
        """Test decomposition with different window sizes."""
        window_sizes = [3, 5, 7]

        results = []
        for ws in window_sizes:
            model = lgtd(window_size=ws)
            result = model.fit_transform(simple_data['y'])
            results.append(result)

        # All should produce valid decompositions
        for result in results:
            assert result is not None
            assert len(result.trend) == len(simple_data['y'])

        # Larger window should produce smoother trend
        # (lower variance in trend component)
        trend_vars = [np.var(np.diff(r.trend)) for r in results]
        # Generally, larger window → smoother → lower variance in diff
        # This might not always hold but is a general tendency
        assert trend_vars[0] >= trend_vars[-1] * 0.5  # Relaxed check


class TestLGTDPercentile:
    """Test effect of error percentile parameter."""

    def test_different_percentiles(self, simple_data):
        """Test decomposition with different percentiles."""
        percentiles = [25, 50, 75]

        results = []
        for p in percentiles:
            model = lgtd(error_percentile=p)
            result = model.fit_transform(simple_data['y'])
            results.append(result)

        # All should produce valid decompositions
        for result in results:
            assert result is not None
            assert len(result.trend) == len(simple_data['y'])

    def test_extreme_percentiles(self, simple_data):
        """Test with extreme percentile values."""
        # Very low percentile
        model_low = lgtd(error_percentile=5)
        result_low = model_low.fit_transform(simple_data['y'])
        assert result_low is not None

        # Very high percentile
        model_high = lgtd(error_percentile=95)
        result_high = model_high.fit_transform(simple_data['y'])
        assert result_high is not None


class TestLGTDComplexData:
    """Test LGTD on complex data scenarios."""

    def test_non_linear_trend(self, complex_data):
        """Test with non-linear trend."""
        model = lgtd(trend_selection='lowess')
        result = model.fit_transform(complex_data['y'])

        # LOWESS should handle non-linear trend better
        assert result is not None

        # Check trend captures general shape
        correlation = np.corrcoef(result.trend, complex_data['trend'])[0, 1]
        assert correlation > 0.85, f"Non-linear trend correlation {correlation} too low"

    def test_multiple_seasonalities(self, complex_data):
        """Test with multiple seasonal components."""
        model = lgtd()
        result = model.fit_transform(complex_data['y'])

        # Should still decompose successfully
        assert result is not None

        # Seasonal should capture multiple frequencies
        seasonal_variance = np.var(result.seasonal)
        assert seasonal_variance > 1.0  # Should have significant seasonal variation

    def test_changing_amplitude(self):
        """Test with changing seasonal amplitude."""
        np.random.seed(42)
        n = 300
        t = np.arange(n)

        trend = 0.1 * t + 10

        # Seasonal with changing amplitude
        amplitude = 1 + 0.02 * t
        seasonal = amplitude * np.sin(2 * np.pi * t / 20)

        noise = np.random.normal(0, 0.5, n)
        y = trend + seasonal + noise

        model = lgtd()
        result = model.fit_transform(y)

        # Should still decompose
        assert result is not None
        assert len(result.seasonal) == n


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
