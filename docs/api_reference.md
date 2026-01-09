# API Reference

## LGTD Module

### Core Classes

#### `LGTD`

Main decomposition class implementing the Local-Global Trend Decomposition algorithm.

```python
from LGTD import LGTD
```

**Constructor:**

```python
LGTD(
    window_size: int = 3,
    error_percentile: float = 50,
    trend_selection: str = 'auto',
    lowess_frac: float = 0.1,
    threshold_r2: float = 0.9
)
```

**Parameters:**

- `window_size` (int, default=3): Local trend sliding window size. Larger values increase smoothness of local trend detection. Must be positive integer.

- `error_percentile` (float, default=50): Percentile threshold for AutoTrend error filtering. Values in [0, 100]. Higher values retain more local trend segments.

- `trend_selection` (str, default='auto'): Global trend extraction method.
  - `'auto'`: Automatically select based on R² criterion (threshold_r2)
  - `'linear'`: Force linear regression
  - `'lowess'`: Force LOWESS (Locally Weighted Scatterplot Smoothing)

- `lowess_frac` (float, default=0.1): LOWESS smoothing fraction. Fraction of data points used for local regression. Values in (0, 1]. Lower values follow data more closely.

- `threshold_r2` (float, default=0.9): R² threshold for automatic trend selection. If linear regression achieves R² ≥ threshold, linear trend is selected; otherwise LOWESS is used. Values in [0, 1].

**Methods:**

##### `fit_transform(y: np.ndarray) -> LGTDResult`

Decompose time series into trend, seasonal, and residual components.

**Parameters:**
- `y` (np.ndarray): Input time series of shape (n_samples,). Must be 1-dimensional array.

**Returns:**
- `LGTDResult`: Dataclass containing decomposition results.

**Raises:**
- `ValueError`: If input is not 1-dimensional or contains invalid values (NaN, Inf).

**Example:**

```python
import numpy as np
from LGTD import LGTD

# Generate synthetic time series
t = np.linspace(0, 4*np.pi, 200)
trend = 0.5 * t
seasonal = 10 * np.sin(t)
noise = np.random.normal(0, 1, 200)
y = trend + seasonal + noise

# Decompose with default parameters
model = LGTD()
result = model.fit_transform(y)

# Access components
print(f"Trend shape: {result.trend.shape}")
print(f"Seasonal shape: {result.seasonal.shape}")
print(f"Residual shape: {result.residual.shape}")
print(f"Detected periods: {result.detected_periods}")
```

##### `fit(y: np.ndarray) -> 'LGTD'`

Fit decomposition model to time series.

**Parameters:**
- `y` (np.ndarray): Input time series of shape (n_samples,).

**Returns:**
- `self`: Fitted LGTD instance.

##### `transform(y: np.ndarray) -> LGTDResult`

Apply decomposition using fitted parameters. Currently equivalent to `fit_transform` as LGTD is non-parametric.

**Parameters:**
- `y` (np.ndarray): Input time series of shape (n_samples,).

**Returns:**
- `LGTDResult`: Decomposition results.

---

### Result Classes

#### `LGTDResult`

Dataclass containing decomposition results and metadata.

**Attributes:**

- `trend` (np.ndarray): Global trend component, shape (n_samples,). Represents long-term movement in the series.

- `seasonal` (np.ndarray): Seasonal component, shape (n_samples,). Represents periodic patterns extracted via local trend analysis.

- `residual` (np.ndarray): Residual component, shape (n_samples,). Represents unexplained variation: `residual = y - trend - seasonal`.

- `y` (np.ndarray): Reconstructed time series, shape (n_samples,). Should approximately equal input: `y ≈ trend + seasonal + residual`.

- `detected_periods` (List[int]): List of detected period lengths from local trend analysis. May be empty if no clear periodicity detected.

- `trend_info` (dict): Metadata about trend extraction:
  - `'method'` (str): Selected trend method ('linear' or 'lowess')
  - `'r2'` (float): R² score of trend fit
  - `'parameters'` (dict): Method-specific parameters used

**Example:**

```python
result = model.fit_transform(y)

# Verify additive decomposition
reconstruction_error = np.mean((result.y - y)**2)
print(f"Reconstruction MSE: {reconstruction_error:.6f}")

# Inspect trend selection
print(f"Trend method: {result.trend_info['method']}")
print(f"Trend R²: {result.trend_info['r2']:.4f}")

# Check detected periods
if result.detected_periods:
    print(f"Detected periods: {result.detected_periods}")
else:
    print("No clear periodicity detected")
```

---

## Evaluation Module

### Metrics Functions

Located in `LGTD.evaluation.metrics`.

#### `compute_mse(ground_truth: dict, prediction: dict) -> dict`

Compute Mean Squared Error for each component.

**Parameters:**
- `ground_truth` (dict): Dictionary with keys `'trend'`, `'seasonal'`, `'residual'` containing ground truth arrays.
- `prediction` (dict): Dictionary with same structure containing predicted arrays.

**Returns:**
- `dict`: MSE for each component: `{'trend': float, 'seasonal': float, 'residual': float, 'total': float}`.

**Example:**

```python
from LGTD.evaluation.metrics import compute_mse

ground_truth = {
    'trend': y_true_trend,
    'seasonal': y_true_seasonal,
    'residual': y_true_residual
}
prediction = {
    'trend': result.trend,
    'seasonal': result.seasonal,
    'residual': result.residual
}

mse = compute_mse(ground_truth, prediction)
print(f"Trend MSE: {mse['trend']:.4f}")
print(f"Seasonal MSE: {mse['seasonal']:.4f}")
print(f"Total MSE: {mse['total']:.4f}")
```

#### `compute_mae(ground_truth: dict, prediction: dict) -> dict`

Compute Mean Absolute Error for each component. Similar interface to `compute_mse`.

#### `compute_correlation(ground_truth: dict, prediction: dict) -> dict`

Compute Pearson correlation coefficient for each component.

**Returns:**
- `dict`: Correlation for each component: `{'trend': float, 'seasonal': float, 'residual': float}`. Values in [-1, 1].

#### `compute_psnr(ground_truth: np.ndarray, prediction: np.ndarray, data_range: float = None) -> float`

Compute Peak Signal-to-Noise Ratio.

**Parameters:**
- `ground_truth` (np.ndarray): Ground truth signal.
- `prediction` (np.ndarray): Predicted signal.
- `data_range` (float, optional): Data range. If None, computed as `max(ground_truth) - min(ground_truth)`.

**Returns:**
- `float`: PSNR in decibels (dB). Higher values indicate better reconstruction.

---

### Visualization Functions

Located in `LGTD.evaluation.visualization`.

#### `plot_decomposition(y: np.ndarray, result: LGTDResult, title: str = None, figsize: tuple = (12, 8)) -> Figure`

Plot decomposition results in a stacked layout.

**Parameters:**
- `y` (np.ndarray): Original time series.
- `result` (LGTDResult): Decomposition result.
- `title` (str, optional): Figure title.
- `figsize` (tuple, default=(12, 8)): Figure size in inches.

**Returns:**
- `matplotlib.figure.Figure`: Figure object containing four subplots (original, trend, seasonal, residual).

**Example:**

```python
from LGTD.evaluation.visualization import plot_decomposition
import matplotlib.pyplot as plt

fig = plot_decomposition(y, result, title="LGTD Decomposition")
plt.savefig("decomposition.png", dpi=300, bbox_inches='tight')
plt.show()
```

#### `plot_comparison(ground_truth: dict, predictions: dict, labels: list, figsize: tuple = (15, 10)) -> Figure`

Compare multiple decomposition methods against ground truth.

**Parameters:**
- `ground_truth` (dict): Ground truth components.
- `predictions` (dict): Dictionary mapping method names to prediction dictionaries.
- `labels` (list): Method names for legend.
- `figsize` (tuple, default=(15, 10)): Figure size.

**Returns:**
- `matplotlib.figure.Figure`: Figure with comparison plots for each component.

---

## Utility Functions

### Data Generators

Located in `data.generators`.

#### `generate_synthetic_data(trend_type: str, period_type: str, n_samples: int, noise_level: float, **kwargs) -> dict`

Generate synthetic time series with known ground truth components.

**Parameters:**
- `trend_type` (str): Type of trend ('linear', 'polynomial', 'exponential', 'none').
- `period_type` (str): Type of periodicity ('fixed', 'variable', 'multiple', 'none').
- `n_samples` (int): Number of time steps.
- `noise_level` (float): Standard deviation of Gaussian noise.
- `**kwargs`: Additional parameters specific to trend/period types.

**Returns:**
- `dict`: Contains keys `'y'`, `'trend'`, `'seasonal'`, `'residual'`, `'period'`.

**Example:**

```python
from data.generators import generate_synthetic_data

data = generate_synthetic_data(
    trend_type='linear',
    period_type='fixed',
    n_samples=500,
    noise_level=1.0,
    period=24,
    amplitude=10.0
)

y = data['y']
true_trend = data['trend']
true_seasonal = data['seasonal']
```

---

## Type Hints

All functions support type hints for improved IDE integration:

```python
from typing import Dict, List, Tuple
import numpy as np
from numpy.typing import NDArray

def fit_transform(self, y: NDArray[np.float64]) -> LGTDResult:
    ...
```
