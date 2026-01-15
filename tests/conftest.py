"""
Pytest configuration and shared fixtures for LGTD tests.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ============================================================================
# Shared Fixtures
# ============================================================================

@pytest.fixture
def simple_timeseries():
    """Generate simple time series with known components."""
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
        'period': 20,
        'n': n
    }


@pytest.fixture
def complex_timeseries():
    """Generate complex time series with multiple seasonal components."""
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
        'seasonal1': seasonal1,
        'seasonal2': seasonal2,
        'noise': noise,
        'n': n
    }


@pytest.fixture
def noisy_timeseries():
    """Generate time series with high noise."""
    np.random.seed(999)
    n = 150
    t = np.arange(n)

    trend = 0.3 * t
    seasonal = 3 * np.sin(2 * np.pi * t / 12)
    noise = np.random.normal(0, 5, n)  # High noise

    return {
        'y': trend + seasonal + noise,
        'trend': trend,
        'seasonal': seasonal,
        'noise': noise,
        'period': 12,
        'n': n
    }


@pytest.fixture
def short_timeseries():
    """Generate short time series."""
    np.random.seed(42)
    n = 50
    t = np.arange(n)

    trend = 0.2 * t + 5
    seasonal = 2 * np.sin(2 * np.pi * t / 10)
    noise = np.random.normal(0, 0.3, n)

    return {
        'y': trend + seasonal + noise,
        'trend': trend,
        'seasonal': seasonal,
        'noise': noise,
        'period': 10,
        'n': n
    }


@pytest.fixture
def constant_timeseries():
    """Generate constant time series."""
    n = 100
    value = 42.0
    return {
        'y': np.full(n, value),
        'trend': np.full(n, value),
        'seasonal': np.zeros(n),
        'noise': np.zeros(n),
        'n': n
    }


@pytest.fixture
def ground_truth_components():
    """Generate ground truth components for metrics testing."""
    np.random.seed(42)
    n = 100

    return {
        'trend': np.linspace(0, 10, n),
        'seasonal': 2 * np.sin(2 * np.pi * np.arange(n) / 12),
        'residual': np.random.normal(0, 0.5, n)
    }


# ============================================================================
# Pytest Hooks
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom settings."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow to run"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Auto-mark integration tests
        if "test_experiments" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Auto-mark unit tests
        if "test_modules" in str(item.fspath):
            item.add_marker(pytest.mark.unit)

        # Auto-mark slow tests
        if "comprehensive" in item.nodeid:
            item.add_marker(pytest.mark.slow)
