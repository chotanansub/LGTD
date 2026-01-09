# LGTD Test Suite

Comprehensive test suite for the Linear-Guided Trend Decomposition (LGTD) project.

## Structure

```
tests/
├── test_experiments/       # Experiment infrastructure tests
│   ├── test_baselines.py   # Baseline decomposition methods
│   └── test_runners.py     # Experiment runners and workflows
└── test_modules/           # Core module tests
    ├── test_lgtd.py                      # Original LGTD tests
    ├── test_lgtd_comprehensive.py        # Comprehensive LGTD tests
    ├── test_metrics.py                   # Original metrics tests
    ├── test_metrics_comprehensive.py     # Comprehensive metrics tests
    ├── test_generators.py                # Original generator tests
    └── test_generators_comprehensive.py  # Comprehensive generator tests
```

## Test Coverage

### Experiments (test_experiments/)

**test_baselines.py** - Tests for all baseline decomposition methods:
- STL (Seasonal-Trend decomposition using Loess)
- RobustSTL (Robust STL with outlier handling)
- ASTD (Automated Seasonal-Trend Decomposition)
- STR (Seasonal-Trend decomposition using Regression)
- FastRobustSTL (Fast version of Robust STL)
- OnlineSTL (Online streaming STL)

**test_runners.py** - Tests for experiment infrastructure:
- ExperimentRunner initialization and configuration
- Config loading and validation
- Running single and multiple models
- Results structure and saving
- Error handling and edge cases

### Modules (test_modules/)

**test_lgtd_comprehensive.py** - Comprehensive LGTD algorithm tests:
- Initialization and parameter validation
- Basic decomposition functionality
- Trend recovery (linear and LOWESS)
- Trend selection (auto, manual)
- Seasonal extraction and periodicity
- Edge cases (short series, constants, NaNs, high noise)
- Window size effects
- Error percentile effects
- Complex data scenarios (non-linear trends, multiple seasonalities)

**test_metrics_comprehensive.py** - Comprehensive evaluation metrics tests:
- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- Correlation coefficient
- PSNR (Peak Signal-to-Noise Ratio)
- Edge cases (empty arrays, single values, mismatched lengths)
- Metric properties and relationships
- Integration tests on realistic data

**test_generators_comprehensive.py** - Comprehensive data generator tests:
- Trend generation (linear, inverted-V, piecewise, constant)
- Seasonal generation (fixed, transitive, variable periods)
- Noise generation (Gaussian, configurable std)
- Full synthetic data pipeline
- Data reconstruction validation
- Different data characteristics
- Reproducibility

## Running Tests

### Run all tests
```bash
pytest tests/
```

### Run specific test category
```bash
# Experiment tests
pytest tests/test_experiments/

# Module tests
pytest tests/test_modules/
```

### Run specific test file
```bash
# LGTD comprehensive tests
pytest tests/test_modules/test_lgtd_comprehensive.py

# Baseline methods tests
pytest tests/test_experiments/test_baselines.py

# Metrics tests
pytest tests/test_modules/test_metrics_comprehensive.py
```

### Run with verbose output
```bash
pytest tests/ -v
```

### Run with coverage report
```bash
# Terminal coverage
pytest tests/ --cov=LGTD --cov=experiments --cov=data

# HTML coverage report
pytest tests/ --cov=LGTD --cov=experiments --cov=data --cov-report=html

# View HTML report
open htmlcov/index.html
```

### Run specific test class or method
```bash
# Run specific test class
pytest tests/test_modules/test_lgtd_comprehensive.py::TestLGTDInitialization

# Run specific test method
pytest tests/test_modules/test_lgtd_comprehensive.py::TestLGTDDecomposition::test_basic_decomposition
```

### Run with markers (if defined)
```bash
# Run only fast tests
pytest tests/ -m fast

# Skip slow tests
pytest tests/ -m "not slow"
```

## Test Statistics

- **Total test files**: 8
- **Total lines of test code**: ~2,150 lines
- **Test categories**:
  - Experiments: 2 files (baselines, runners)
  - Modules: 6 files (LGTD, metrics, generators)

## Writing New Tests

### Test File Template

```python
"""
Brief description of what this test module covers.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from module_to_test import ClassToTest


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return {
        'y': np.random.randn(100),
        'expected_result': 42
    }


class TestFeatureName:
    """Test specific feature."""

    def test_basic_functionality(self, sample_data):
        """Test basic feature works."""
        result = ClassToTest().method(sample_data['y'])
        assert result is not None

    def test_edge_case(self):
        """Test edge case handling."""
        with pytest.raises(ValueError):
            ClassToTest().method([])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

### Best Practices

1. **Descriptive test names**: Use `test_<what_is_being_tested>`
2. **One assertion per test**: Keep tests focused
3. **Use fixtures**: Share common setup across tests
4. **Test edge cases**: Empty inputs, None, extreme values
5. **Test error handling**: Verify appropriate errors are raised
6. **Document expectations**: Add docstrings explaining what's being tested

## Continuous Integration

Tests should be run automatically on:
- Every commit (via pre-commit hook)
- Every pull request (via CI/CD pipeline)
- Before releases

## Troubleshooting

### Import errors
If you get import errors, ensure you're running from the project root:
```bash
cd /path/to/LGTD
pytest tests/
```

### Fixture not found
Ensure fixtures are defined in the same file or in `conftest.py`.

### Slow tests
Use `-k` to run specific tests:
```bash
pytest tests/ -k "test_basic"
```

## Contributing

When adding new features:
1. Write tests first (TDD approach)
2. Ensure all existing tests pass
3. Aim for >80% code coverage
4. Include both positive and negative test cases
5. Add integration tests for complex workflows
