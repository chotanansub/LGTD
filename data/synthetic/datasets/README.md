# Synthetic Datasets for LGTD Experiments

## Overview

This directory contains 9 synthetic time series datasets generated for evaluating the LGTD (Local Global Trend Decomposition) method and comparing it with baseline methods.

## Dataset Specifications

- **Number of datasets**: 9
- **Length**: 2000 time points each
- **Noise level**: σ = 1.0 (Gaussian noise)
- **Random seed**: 69 (for reproducibility)

## Dataset Organization

### Fixed Period Datasets (Synth1-3)
Datasets with constant seasonal period throughout the entire series.

| Dataset | Trend Type | Seasonal Period | Amplitude | Description |
|---------|-----------|-----------------|-----------|-------------|
| synth1  | Linear    | 120             | 50.0      | Linear trend with fixed seasonality |
| synth2  | Inverted-V| 60              | 50.0      | Inverted-V trend with fixed seasonality |
| synth3  | Piecewise | 120             | 80.0      | Piecewise trend with fixed seasonality |

### Transitive Period Datasets (Synth4-6)
Datasets where the seasonal period changes once during the series.

| Dataset | Trend Type | Main Period | Transition Period | Description |
|---------|-----------|-------------|-------------------|-------------|
| synth4  | Linear    | 120         | 60                | 120→60→120 period transition |
| synth5  | Inverted-V| 60          | 120               | 60→120→60 period transition |
| synth6  | Piecewise | 120         | 60                | 120→60→120 period transition |

**Transition timing**:
- Starts at 40% of series (t=800)
- Ends at 60% of series (t=1200)

### Variable Period Datasets (Synth7-9)
Datasets with highly variable seasonal periods.

| Dataset | Trend Type | Period Range | Mean Period | Description |
|---------|-----------|--------------|-------------|-------------|
| synth7  | Linear    | 100-500      | ~260        | Variable periods |
| synth8  | Inverted-V| 100-500      | ~260        | Variable periods |
| synth9  | Piecewise | 100-500      | ~260        | Variable periods |

**Period sequence**: [100, 300, 150, 400, 120, 350, 180, 450, 200, 250, 140, 380, 220, 500, 160]

## Trend Types

### Linear Trend
- Equation: `trend = 0.02 * t`
- Slope: 0.02
- Produces a simple upward linear pattern

### Inverted-V Trend
- Peak position: 60% of series
- Max height: 100.0
- Curve sharpness: 3.0
- Increases to a peak then decreases

### Piecewise Trend
- Number of segments: 4
- Slopes: [0.15, -0.08, 0.20, -0.05]
- Creates a multi-segment linear pattern

## Seasonal Patterns

### Fixed Period
- Equation: `seasonal = amplitude * sin(2π * t / period)`
- Constant period throughout

### Transitive Period
- Switches between two different periods
- Smooth phase accumulation during transition
- Tests adaptability to period changes

### Variable Period
- Multiple different periods in sequence
- Each cycle uses a different period from the list
- Tests robustness to highly variable seasonality

## File Format

Each dataset is saved as a NumPy `.npz` file containing:

```python
{
    'name': str,           # Dataset name (e.g., 'synth1')
    'trend_type': str,     # 'linear', 'inverted_v', or 'piecewise'
    'period_type': str,    # 'fixed', 'transitive', or 'variable'
    'time': ndarray,       # Time index [0, 1, 2, ..., 1999]
    'y': ndarray,          # Observed time series (trend + seasonal + residual)
    'trend': ndarray,      # Ground truth trend component
    'seasonal': ndarray,   # Ground truth seasonal component
    'residual': ndarray    # Ground truth residual (noise)
}
```

## Loading Datasets

### Python
```python
import numpy as np

# Load a dataset
data = np.load('data/synthetic/datasets/synth1_data.npz')

# Access components
time = data['time']
y = data['y']
trend = data['trend']
seasonal = data['seasonal']
residual = data['residual']

# Metadata
name = str(data['name'])
trend_type = str(data['trend_type'])
period_type = str(data['period_type'])
```

## Verification

All datasets satisfy:
- `y = trend + seasonal + residual` (exact decomposition)
- Residual has mean ≈ 0 and std ≈ 1.0
- No NaN or infinite values
- Length = 2000 for all components

## Use Cases

### Research Evaluation
- Compare LGTD with baseline methods (STL, RobustSTL, ASTD)
- Evaluate performance across different trend patterns
- Test robustness to period changes

### Method Development
- Test new decomposition algorithms
- Validate period detection methods
- Benchmark computational performance

### Education
- Demonstrate time series decomposition concepts
- Visualize different trend and seasonal patterns
- Understand impact of noise on decomposition

## Generation

Datasets were generated using:
```bash
python -c "from data.synthetic.generators import generate_synthetic_data; ..."
```

See [../generators.py](../generators.py) for the full generator implementation.

## References

- Generator implementation: `data/synthetic/generators.py`
- Experiment configuration: `experiments/configs/synthetic_experiments.yaml`
- Original specification: `ref/generator.py`

## File Location

All datasets are stored in: `data/synthetic/datasets/`

This is the **source data location**. The `results/` directory is for experiment outputs only.

---

**Generated**: 2025-12-28
**Total size**: ~5 MB (JSON + NPZ files)
**Formats**: JSON (human-readable) and NumPy NPZ (compact)
