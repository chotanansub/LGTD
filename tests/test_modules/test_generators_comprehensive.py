"""
Comprehensive unit tests for synthetic data generators.

Tests the main generate_synthetic_data function with different trend types,
seasonality patterns, and configurations.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.synthetic.generators import generate_synthetic_data


class TestSyntheticDataGeneration:
    """Test main synthetic data generation function."""

    def test_generate_simple_data(self):
        """Test generating simple synthetic dataset."""
        data = generate_synthetic_data(
            n=200,
            trend_type='linear',
            seasonality_type='fixed',
            seasonal_params={'period': 20, 'amplitude': 5},
            residual_std=0.5
        )

        assert 'time' in data
        assert 'y' in data
        assert 'trend' in data
        assert 'seasonal' in data
        assert 'residual' in data

        # Check lengths
        assert len(data['time']) == 200
        assert len(data['y']) == 200
        assert len(data['trend']) == 200
        assert len(data['seasonal']) == 200
        assert len(data['residual']) == 200

    def test_data_reconstruction(self):
        """Test that y = trend + seasonal + residual."""
        data = generate_synthetic_data(
            n=100,
            trend_type='linear',
            seasonality_type='fixed',
            seasonal_params={'period': 10, 'amplitude': 3},
            residual_std=0.1
        )

        reconstructed = data['trend'] + data['seasonal'] + data['residual']
        np.testing.assert_allclose(reconstructed, data['y'], rtol=1e-10)

    def test_linear_trend(self):
        """Test linear trend generation."""
        data = generate_synthetic_data(
            n=150,
            trend_type='linear',
            seasonality_type='fixed',
            seasonal_params={'period': 15, 'amplitude': 3}
        )

        assert data is not None
        assert 'trend' in data
        assert len(data['trend']) == 150

        # Linear trend should have constant first derivative
        first_diff = np.diff(data['trend'])
        assert np.std(first_diff) < 0.01  # Nearly constant slope

    def test_inverted_v_trend(self):
        """Test inverted-V trend generation."""
        data = generate_synthetic_data(
            n=200,
            trend_type='inverted_v',
            seasonality_type='fixed',
            seasonal_params={'period': 20, 'amplitude': 3}
        )

        assert data is not None
        assert 'trend' in data

        # Should have peak in middle
        mid = len(data['trend']) // 2
        quarter = len(data['trend']) // 4

        # Peak should be higher than endpoints
        assert data['trend'][mid] > data['trend'][quarter]
        assert data['trend'][mid] > data['trend'][3 * quarter]

    def test_piecewise_trend(self):
        """Test piecewise linear trend."""
        data = generate_synthetic_data(
            n=300,
            trend_type='piecewise',
            seasonality_type='fixed',
            seasonal_params={'period': 30, 'amplitude': 5}
        )

        assert data is not None
        assert 'trend' in data
        assert len(data['trend']) == 300

        # Trend should exist and vary
        assert np.std(data['trend']) > 0

    def test_fixed_seasonality(self):
        """Test fixed period seasonality."""
        data = generate_synthetic_data(
            n=200,
            trend_type='linear',
            seasonality_type='fixed',
            seasonal_params={'period': 20, 'amplitude': 5}
        )

        assert 'seasonal' in data

        # Check periodicity - first two periods should be similar
        period = 20
        if len(data['seasonal']) >= 2 * period:
            first_period = data['seasonal'][:period]
            second_period = data['seasonal'][period:2*period]
            correlation = np.corrcoef(first_period, second_period)[0, 1]
            assert correlation > 0.95, "Fixed seasonality not periodic enough"

    def test_transitive_seasonality(self):
        """Test transitive seasonality (changing period)."""
        data = generate_synthetic_data(
            n=400,
            trend_type='linear',
            seasonality_type='transitive',
            seasonal_params={'main_period': 40, 'transition_period': 20, 'amplitude': 5}
        )

        assert data is not None
        assert 'seasonal' in data
        assert len(data['seasonal']) == 400

        # Should have seasonal variation
        assert np.std(data['seasonal']) > 1.0

    def test_variable_seasonality(self):
        """Test variable period seasonality."""
        data = generate_synthetic_data(
            n=500,
            trend_type='linear',
            seasonality_type='variable',
            seasonal_params={'periods': [20, 40, 30, 50], 'amplitude': 5}
        )

        assert data is not None
        assert 'seasonal' in data
        assert len(data['seasonal']) == 500

        # Should have variation
        assert np.std(data['seasonal']) > 0

    def test_no_noise_data(self):
        """Test generation with no noise."""
        data = generate_synthetic_data(
            n=100,
            trend_type='linear',
            seasonality_type='fixed',
            seasonal_params={'period': 10, 'amplitude': 5},
            residual_std=0.0
        )

        # Residual should be all zeros (or very close)
        assert np.allclose(data['residual'], 0, atol=1e-10)

        # y should equal trend + seasonal exactly
        expected = data['trend'] + data['seasonal']
        np.testing.assert_allclose(data['y'], expected, rtol=1e-10)

    def test_with_noise(self):
        """Test generation with noise."""
        np.random.seed(42)
        data = generate_synthetic_data(
            n=200,
            trend_type='linear',
            seasonality_type='fixed',
            seasonal_params={'period': 20, 'amplitude': 5},
            residual_std=1.0
        )

        # Noise should have std close to specified
        assert 0.5 < np.std(data['residual']) < 1.5

    def test_reproducibility(self):
        """Test data generation is reproducible with same seed."""
        data1 = generate_synthetic_data(
            n=100,
            trend_type='linear',
            seasonality_type='fixed',
            seasonal_params={'period': 10, 'amplitude': 3},
            residual_std=1.0,
            seed=42
        )

        data2 = generate_synthetic_data(
            n=100,
            trend_type='linear',
            seasonality_type='fixed',
            seasonal_params={'period': 10, 'amplitude': 3},
            residual_std=1.0,
            seed=42
        )

        np.testing.assert_allclose(data1['y'], data2['y'])
        np.testing.assert_allclose(data1['residual'], data2['residual'])

    def test_different_lengths(self):
        """Test generation with different data lengths."""
        lengths = [50, 100, 500, 1000]

        for n in lengths:
            data = generate_synthetic_data(
                n=n,
                trend_type='linear',
                seasonality_type='fixed',
                seasonal_params={'period': min(20, n//5), 'amplitude': 3}
            )

            assert len(data['y']) == n
            assert len(data['trend']) == n
            assert len(data['seasonal']) == n

    def test_different_amplitudes(self):
        """Test seasonal amplitude parameter."""
        data_amp5 = generate_synthetic_data(
            n=200,
            trend_type='linear',
            seasonality_type='fixed',
            seasonal_params={'period': 20, 'amplitude': 5}
        )

        data_amp10 = generate_synthetic_data(
            n=200,
            trend_type='linear',
            seasonality_type='fixed',
            seasonal_params={'period': 20, 'amplitude': 10}
        )

        # Larger amplitude should have larger seasonal range
        assert np.ptp(data_amp10['seasonal']) > np.ptp(data_amp5['seasonal'])

    def test_different_periods(self):
        """Test different seasonal periods."""
        periods = [10, 20, 50]

        for period in periods:
            data = generate_synthetic_data(
                n=200,
                trend_type='linear',
                seasonality_type='fixed',
                seasonal_params={'period': period, 'amplitude': 5}
            )

            assert data is not None
            assert 'seasonal' in data

            # Check approximate periodicity
            if len(data['seasonal']) >= 2 * period:
                first = data['seasonal'][:period]
                second = data['seasonal'][period:2*period]
                corr = np.corrcoef(first, second)[0, 1]
                assert corr > 0.9

    def test_trend_dominance(self):
        """Test data with dominant trend."""
        data = generate_synthetic_data(
            n=200,
            trend_type='linear',
            seasonality_type='fixed',
            seasonal_params={'period': 20, 'amplitude': 1},  # Small seasonal
            residual_std=0.1
        )

        # Trend range should be large relative to seasonal
        trend_range = np.ptp(data['trend'])
        seasonal_range = np.ptp(data['seasonal'])

        # For 200 points with default slope, trend should dominate
        assert trend_range > seasonal_range

    def test_seasonal_dominance(self):
        """Test data with dominant seasonality."""
        data = generate_synthetic_data(
            n=200,
            trend_type='linear',
            seasonality_type='fixed',
            seasonal_params={'period': 20, 'amplitude': 50},  # Large seasonal
            residual_std=0.1
        )

        # Seasonal range should be large relative to trend
        seasonal_range = np.ptp(data['seasonal'])
        trend_range = np.ptp(data['trend'])

        assert seasonal_range > trend_range

    def test_high_noise_scenario(self):
        """Test data with high noise."""
        data = generate_synthetic_data(
            n=200,
            trend_type='linear',
            seasonality_type='fixed',
            seasonal_params={'period': 20, 'amplitude': 5},
            residual_std=10.0  # Very high noise
        )

        # Noise should be significant
        noise_std = np.std(data['residual'])
        assert noise_std > 5.0

    def test_all_components_present(self):
        """Test all components are non-zero."""
        data = generate_synthetic_data(
            n=200,
            trend_type='linear',
            seasonality_type='fixed',
            seasonal_params={'period': 20, 'amplitude': 5},
            residual_std=1.0
        )

        # All components should have some variation
        assert np.std(data['trend']) > 0
        assert np.std(data['seasonal']) > 0
        # Noise might be small but should exist
        assert 'residual' in data


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
