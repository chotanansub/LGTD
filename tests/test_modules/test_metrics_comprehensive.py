"""
Comprehensive unit tests for evaluation metrics.

Tests all metrics used to evaluate decomposition quality including
MSE, MAE, RMSE, correlation, and PSNR.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from LGTD.evaluation.metrics import (
    mean_squared_error,
    mean_absolute_error,
    root_mean_squared_error,
    correlation_coefficient,
    peak_signal_noise_ratio
)


class TestMSE:
    """Test Mean Squared Error metric."""

    def test_mse_identical(self):
        """Test MSE of identical arrays is zero."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])

        mse = mean_squared_error(y_true, y_pred)
        assert mse == 0.0

    def test_mse_simple(self):
        """Test MSE with simple case."""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([2, 3, 4])  # Off by 1

        mse = mean_squared_error(y_true, y_pred)
        assert mse == 1.0  # (1^2 + 1^2 + 1^2) / 3 = 1

    def test_mse_symmetry(self):
        """Test MSE is symmetric."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([2, 3, 4, 5, 6])

        mse1 = mean_squared_error(y_true, y_pred)
        mse2 = mean_squared_error(y_pred, y_true)

        assert mse1 == mse2

    def test_mse_positive(self):
        """Test MSE is always non-negative."""
        np.random.seed(42)
        y_true = np.random.randn(100)
        y_pred = np.random.randn(100)

        mse = mean_squared_error(y_true, y_pred)
        assert mse >= 0.0

    def test_mse_larger_error(self):
        """Test MSE increases with larger errors."""
        y_true = np.array([1, 2, 3, 4, 5])

        y_pred1 = y_true + 1  # Small error
        y_pred2 = y_true + 5  # Large error

        mse1 = mean_squared_error(y_true, y_pred1)
        mse2 = mean_squared_error(y_true, y_pred2)

        assert mse2 > mse1


class TestMAE:
    """Test Mean Absolute Error metric."""

    def test_mae_identical(self):
        """Test MAE of identical arrays is zero."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])

        mae = mean_absolute_error(y_true, y_pred)
        assert mae == 0.0

    def test_mae_simple(self):
        """Test MAE with simple case."""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([2, 3, 4])  # Off by 1

        mae = mean_absolute_error(y_true, y_pred)
        assert mae == 1.0  # (|1| + |1| + |1|) / 3 = 1

    def test_mae_vs_mse(self):
        """Test MAE is less sensitive to outliers than MSE."""
        y_true = np.array([1, 2, 3, 4, 5])

        # Prediction with one large outlier
        y_pred = np.array([1, 2, 3, 4, 15])  # Last value off by 10

        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)

        # MAE = (0+0+0+0+10)/5 = 2
        # MSE = (0+0+0+0+100)/5 = 20
        assert mse > mae  # MSE penalizes outlier more

    def test_mae_positive(self):
        """Test MAE is always non-negative."""
        np.random.seed(42)
        y_true = np.random.randn(100)
        y_pred = np.random.randn(100)

        mae = mean_absolute_error(y_true, y_pred)
        assert mae >= 0.0


class TestRMSE:
    """Test Root Mean Squared Error metric."""

    def test_rmse_identical(self):
        """Test RMSE of identical arrays is zero."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])

        rmse = root_mean_squared_error(y_true, y_pred)
        assert rmse == 0.0

    def test_rmse_vs_mse(self):
        """Test RMSE is square root of MSE."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([2, 3, 4, 5, 6])

        mse = mean_squared_error(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)

        assert np.isclose(rmse, np.sqrt(mse))

    def test_rmse_same_units(self):
        """Test RMSE has same units as input."""
        y_true = np.array([10, 20, 30])
        y_pred = np.array([12, 22, 32])  # Off by 2

        rmse = root_mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)

        # Both should be in same units
        assert abs(rmse - mae) < 10  # Reasonable difference


class TestCorrelation:
    """Test Correlation coefficient metric."""

    def test_correlation_identical(self):
        """Test correlation of identical arrays is 1."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])

        corr = correlation_coefficient(y_true, y_pred)
        assert np.isclose(corr, 1.0)

    def test_correlation_perfect_linear(self):
        """Test correlation of perfectly linear relationship is 1."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = 2 * y_true + 3  # Perfect linear relationship

        corr = correlation_coefficient(y_true, y_pred)
        assert np.isclose(corr, 1.0)

    def test_correlation_negative(self):
        """Test correlation can be negative."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = -y_true  # Perfect negative correlation

        corr = correlation_coefficient(y_true, y_pred)
        assert np.isclose(corr, -1.0)

    def test_correlation_range(self):
        """Test correlation is in [-1, 1]."""
        np.random.seed(42)
        y_true = np.random.randn(100)
        y_pred = np.random.randn(100)

        corr = correlation_coefficient(y_true, y_pred)
        assert -1.0 <= corr <= 1.0

    def test_correlation_uncorrelated(self):
        """Test correlation of uncorrelated data is near 0."""
        np.random.seed(42)
        y_true = np.random.randn(1000)
        y_pred = np.random.randn(1000)  # Independent random data

        corr = correlation_coefficient(y_true, y_pred)
        assert abs(corr) < 0.1  # Should be close to 0

    def test_correlation_constant(self):
        """Test correlation with constant array."""
        y_true = np.ones(10)
        y_pred = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        # Correlation with constant should be NaN or 0
        corr = correlation_coefficient(y_true, y_pred)
        assert np.isnan(corr) or corr == 0.0


class TestPSNR:
    """Test Peak Signal-to-Noise Ratio metric."""

    def test_psnr_identical(self):
        """Test PSNR of identical arrays is infinity."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        psnr = peak_signal_noise_ratio(y_true, y_pred)
        assert np.isinf(psnr) or psnr > 100  # Very high PSNR

    def test_psnr_higher_is_better(self):
        """Test higher PSNR means better quality."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        y_pred1 = y_true + 0.1  # Small error
        y_pred2 = y_true + 1.0  # Large error

        psnr1 = peak_signal_noise_ratio(y_true, y_pred1)
        psnr2 = peak_signal_noise_ratio(y_true, y_pred2)

        # Higher PSNR is better (smaller error)
        assert psnr1 > psnr2

    def test_psnr_positive(self):
        """Test PSNR is typically positive."""
        np.random.seed(42)
        y_true = np.random.rand(100) * 10
        y_pred = y_true + np.random.randn(100) * 0.5

        psnr = peak_signal_noise_ratio(y_true, y_pred)
        # PSNR can be negative for very poor quality, but should be positive for reasonable predictions
        assert psnr > 0.0 or np.isinf(psnr)


class TestMetricsEdgeCases:
    """Test edge cases for all metrics."""

    def test_empty_arrays(self):
        """Test metrics with empty arrays."""
        y_true = np.array([])
        y_pred = np.array([])

        # Should either handle gracefully or raise informative error
        try:
            mse = mean_squared_error(y_true, y_pred)
            # If it handles empty, result should be NaN or 0
            assert np.isnan(mse) or mse == 0.0
        except (ValueError, ZeroDivisionError):
            pass  # Expected error

    def test_single_value(self):
        """Test metrics with single value."""
        y_true = np.array([5.0])
        y_pred = np.array([7.0])

        mse = mean_squared_error(y_true, y_pred)
        assert mse == 4.0  # (7-5)^2 = 4

        mae = mean_absolute_error(y_true, y_pred)
        assert mae == 2.0  # |7-5| = 2

    def test_mismatched_lengths(self):
        """Test metrics with mismatched array lengths."""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2])

        # Should raise error
        with pytest.raises((ValueError, IndexError)):
            mean_squared_error(y_true, y_pred)

    def test_very_large_values(self):
        """Test metrics with very large values."""
        y_true = np.array([1e10, 2e10, 3e10])
        y_pred = np.array([1e10 + 1, 2e10 + 1, 3e10 + 1])

        # Should handle large values without overflow
        mse = mean_squared_error(y_true, y_pred)
        assert np.isfinite(mse)

    def test_very_small_values(self):
        """Test metrics with very small values."""
        y_true = np.array([1e-10, 2e-10, 3e-10])
        y_pred = np.array([1.1e-10, 2.1e-10, 3.1e-10])

        # Should handle small values without underflow
        mae = mean_absolute_error(y_true, y_pred)
        assert np.isfinite(mae)
        assert mae > 0


class TestMetricsIntegration:
    """Integration tests for metrics on realistic data."""

    def test_all_metrics_on_decomposition(self):
        """Test all metrics on decomposition results."""
        np.random.seed(42)
        n = 200
        t = np.arange(n)

        # True components
        true_trend = 0.1 * t + 10
        true_seasonal = 5 * np.sin(2 * np.pi * t / 20)

        # Predicted components (with some error)
        pred_trend = true_trend + np.random.normal(0, 0.5, n)
        pred_seasonal = true_seasonal + np.random.normal(0, 0.3, n)

        # Compute all metrics
        mse = mean_squared_error(true_trend, pred_trend)
        mae = mean_absolute_error(true_trend, pred_trend)
        rmse = root_mean_squared_error(true_trend, pred_trend)
        corr = correlation_coefficient(true_trend, pred_trend)
        psnr = peak_signal_noise_ratio(true_trend, pred_trend)

        # All should be computed successfully
        assert np.isfinite(mse)
        assert np.isfinite(mae)
        assert np.isfinite(rmse)
        assert np.isfinite(corr) or np.isnan(corr)
        assert np.isfinite(psnr) or np.isinf(psnr)

        # Relationships between metrics
        assert np.isclose(rmse, np.sqrt(mse))
        assert mae <= rmse  # MAE <= RMSE (AM-GM inequality)
        assert 0 <= corr <= 1  # Good correlation for small errors

    def test_metrics_sensitivity(self):
        """Test relative sensitivity of different metrics."""
        y_true = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        # Small errors
        y_pred_small = y_true + np.array([0.1, 0.1, 0.1, 0.1, 0.1,
                                          0.1, 0.1, 0.1, 0.1, 0.1])

        # Large errors
        y_pred_large = y_true + np.array([0.5, 0.5, 0.5, 0.5, 0.5,
                                          0.5, 0.5, 0.5, 0.5, 0.5])

        # All metrics should increase with larger errors
        assert mean_squared_error(y_true, y_pred_large) > mean_squared_error(y_true, y_pred_small)
        assert mean_absolute_error(y_true, y_pred_large) > mean_absolute_error(y_true, y_pred_small)
        assert root_mean_squared_error(y_true, y_pred_large) > root_mean_squared_error(y_true, y_pred_small)

        # Correlation should decrease with larger errors
        corr_small = correlation_coefficient(y_true, y_pred_small)
        corr_large = correlation_coefficient(y_true, y_pred_large)
        assert corr_small >= corr_large

        # PSNR should decrease with larger errors
        psnr_small = peak_signal_noise_ratio(y_true, y_pred_small)
        psnr_large = peak_signal_noise_ratio(y_true, y_pred_large)
        if np.isfinite(psnr_small) and np.isfinite(psnr_large):
            assert psnr_small > psnr_large


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
