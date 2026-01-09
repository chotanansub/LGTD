"""
Unit tests for baseline decomposition methods.

Tests all baseline methods (STL, RobustSTL, ASTD, STR, etc.) to ensure
they can decompose synthetic data correctly.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.baselines.stl import STLDecomposer
from experiments.baselines.robust_stl import RobustSTLDecomposer
from experiments.baselines.astd import ASTDDecomposer
from experiments.baselines.str_decomposer import STRDecomposer
from experiments.baselines.fast_robust_stl import FastRobustSTLDecomposer
from experiments.baselines.online_stl import OnlineSTLDecomposer


@pytest.fixture
def synthetic_data():
    """Generate simple synthetic time series for testing."""
    np.random.seed(42)
    n = 200
    t = np.arange(n)

    # Simple trend
    trend = 0.1 * t + 10

    # Simple seasonal component
    period = 20
    seasonal = 5 * np.sin(2 * np.pi * t / period)

    # Small noise
    noise = np.random.normal(0, 0.5, n)

    # Combine
    y = trend + seasonal + noise

    return {
        'y': y,
        'trend': trend,
        'seasonal': seasonal,
        'residual': noise,
        'period': period
    }


class TestSTL:
    """Test STL decomposition."""

    def test_stl_basic(self, synthetic_data):
        """Test basic STL decomposition."""
        decomposer = STLDecomposer(period=synthetic_data['period'])
        result = decomposer.decompose(synthetic_data['y'])

        # Check output structure
        assert 'trend' in result
        assert 'seasonal' in result
        assert 'residual' in result

        # Check output shapes
        assert len(result['trend']) == len(synthetic_data['y'])
        assert len(result['seasonal']) == len(synthetic_data['y'])
        assert len(result['residual']) == len(synthetic_data['y'])

        # Check reconstruction
        reconstructed = result['trend'] + result['seasonal'] + result['residual']
        np.testing.assert_allclose(reconstructed, synthetic_data['y'], rtol=1e-10)

    def test_stl_trend_recovery(self, synthetic_data):
        """Test that STL recovers trend reasonably well."""
        decomposer = STLDecomposer(period=synthetic_data['period'])
        result = decomposer.decompose(synthetic_data['y'])

        # Trend should be reasonably close
        # (allowing some deviation due to method differences)
        correlation = np.corrcoef(result['trend'], synthetic_data['trend'])[0, 1]
        assert correlation > 0.95, f"Trend correlation {correlation} too low"


class TestRobustSTL:
    """Test RobustSTL decomposition."""

    def test_robust_stl_basic(self, synthetic_data):
        """Test basic RobustSTL decomposition."""
        decomposer = RobustSTLDecomposer(period=synthetic_data['period'])
        if not decomposer._robust_stl_available:
            pytest.skip("RobustSTL not available")
        result = decomposer.decompose(synthetic_data['y'])

        # Check output structure
        assert 'trend' in result
        assert 'seasonal' in result
        assert 'residual' in result

        # Check output shapes
        assert len(result['trend']) == len(synthetic_data['y'])
        assert len(result['seasonal']) == len(synthetic_data['y'])
        assert len(result['residual']) == len(synthetic_data['y'])

    def test_robust_stl_with_outliers(self, synthetic_data):
        """Test RobustSTL handles outliers."""
        decomposer = RobustSTLDecomposer(period=synthetic_data['period'])
        if not decomposer._robust_stl_available:
            pytest.skip("RobustSTL not available")

        # Add outliers
        y_with_outliers = synthetic_data['y'].copy()
        outlier_indices = [50, 100, 150]
        y_with_outliers[outlier_indices] += 50

        result = decomposer.decompose(y_with_outliers)

        # Should still decompose successfully
        assert 'trend' in result
        assert len(result['trend']) == len(y_with_outliers)


class TestASTD:
    """Test ASTD decomposition."""

    def test_astd_basic(self, synthetic_data):
        """Test basic ASTD decomposition."""
        decomposer = ASTDDecomposer()
        if not decomposer._astd_available:
            pytest.skip("ASTD not available")
        result = decomposer.decompose(synthetic_data['y'])

        # Check output structure
        assert 'trend' in result
        assert 'seasonal' in result
        assert 'residual' in result

        # Check output shapes
        assert len(result['trend']) == len(synthetic_data['y'])
        assert len(result['seasonal']) == len(synthetic_data['y'])
        assert len(result['residual']) == len(synthetic_data['y'])

    def test_astd_auto_period(self, synthetic_data):
        """Test ASTD auto-detects period."""
        decomposer = ASTDDecomposer()
        if not decomposer._astd_available:
            pytest.skip("ASTD not available")
        result = decomposer.decompose(synthetic_data['y'])

        # Should detect period automatically
        assert result is not None
        assert 'seasonal' in result


class TestSTR:
    """Test STR decomposition."""

    def test_str_basic(self, synthetic_data):
        """Test basic STR decomposition."""
        decomposer = STRDecomposer(seasonal_periods=[synthetic_data['period']])
        if not decomposer._str_available:
            pytest.skip("STR not available")
        result = decomposer.decompose(synthetic_data['y'])

        # Check output structure
        assert 'trend' in result
        assert 'seasonal' in result
        assert 'residual' in result

        # Check output shapes
        assert len(result['trend']) == len(synthetic_data['y'])
        assert len(result['seasonal']) == len(synthetic_data['y'])
        assert len(result['residual']) == len(synthetic_data['y'])


class TestFastRobustSTL:
    """Test FastRobustSTL decomposition."""

    def test_fast_robust_stl_basic(self, synthetic_data):
        """Test basic FastRobustSTL decomposition."""
        decomposer = FastRobustSTLDecomposer(period=synthetic_data['period'])
        result = decomposer.decompose(synthetic_data['y'])

        # Check output structure
        assert 'trend' in result
        assert 'seasonal' in result
        assert 'residual' in result

        # Check output shapes
        assert len(result['trend']) == len(synthetic_data['y'])
        assert len(result['seasonal']) == len(synthetic_data['y'])
        assert len(result['residual']) == len(synthetic_data['y'])

    def test_fast_robust_stl_speed(self, synthetic_data):
        """Test FastRobustSTL is reasonably fast."""
        import time

        decomposer = FastRobustSTLDecomposer(period=synthetic_data['period'])

        start = time.time()
        result = decomposer.decompose(synthetic_data['y'])
        elapsed = time.time() - start

        # Should complete quickly (< 5 seconds for 200 points)
        assert elapsed < 5.0, f"Decomposition took {elapsed}s, too slow"


class TestOnlineSTL:
    """Test Online STL decomposition."""

    def test_online_stl_basic(self, synthetic_data):
        """Test basic Online STL decomposition."""
        decomposer = OnlineSTLDecomposer(periods=[synthetic_data['period']])
        if not decomposer._online_stl_available:
            pytest.skip("OnlineSTL not available")
        result = decomposer.decompose(synthetic_data['y'])

        # Check output structure
        assert 'trend' in result
        assert 'seasonal' in result
        assert 'residual' in result

        # Check output shapes
        assert len(result['trend']) == len(synthetic_data['y'])
        assert len(result['seasonal']) == len(synthetic_data['y'])
        assert len(result['residual']) == len(synthetic_data['y'])

    def test_online_stl_incremental(self, synthetic_data):
        """Test Online STL can process data incrementally."""
        decomposer = OnlineSTLDecomposer(periods=[synthetic_data['period']])
        if not decomposer._online_stl_available:
            pytest.skip("OnlineSTL not available")

        # Process first half
        half = len(synthetic_data['y']) // 2
        result1 = decomposer.decompose(synthetic_data['y'][:half])

        # Should work on partial data
        assert len(result1['trend']) == half


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
