# API Reference

## LGTD Module

### Core Classes

#### `LGTD`

Main decomposition class implementing the Local-Global Trend Decomposition algorithm.

```python
from lgtd import lgtd
```

**Constructor:**

```python
LGTD(
    window_size: int = 3,
    error_percentile: int = 50,
    trend_selection: str = 'auto',
    lowess_frac: float = 0.1,
    threshold_r2: float = 0.92,
    verbose: bool = False
)
```

**Parameters:**

- `window_size` (int, default=3): Local trend sliding window size. Larger values increase smoothness of local trend detection. Must be positive integer.

- `error_percentile` (int, default=50): Percentile threshold for AutoTrend error filtering. Values in [0, 100]. Higher values retain more local trend segments.

- `trend_selection` (str, default='auto'): Global trend extraction method.
  - `'auto'`: Automatically select based on R² criterion (threshold_r2)
  - `'linear'`: Force linear regression
  - `'lowess'`: Force LOWESS (Locally Weighted Scatterplot Smoothing)

- `lowess_frac` (float, default=0.1): LOWESS smoothing fraction. Fraction of data points used for local regression. Values in (0, 1]. Lower values follow data more closely.

- `threshold_r2` (float, default=0.92): R² threshold for automatic trend selection. If linear regression achieves R² ≥ threshold, linear trend is selected; otherwise LOWESS is used. Values in [0, 1].

- `verbose` (bool, default=False): Print diagnostic information during decomposition process.

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
from lgtd import lgtd

# Generate synthetic time series
t = np.linspace(0, 4*np.pi, 200)
trend = 0.5 * t
seasonal = 10 * np.sin(t)
noise = np.random.normal(0, 1, 200)
y = trend + seasonal + noise

# Decompose with default parameters
model = lgtd()
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

Located in `lgtd.evaluation.metrics`.

#### Core Metric Functions

##### `mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float`

Calculate Mean Squared Error between true and predicted values.

**Parameters:**
- `y_true` (np.ndarray): Ground truth values.
- `y_pred` (np.ndarray): Predicted values.

**Returns:**
- `float`: MSE value.

##### `mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float`

Calculate Mean Absolute Error between true and predicted values.

**Parameters:**
- `y_true` (np.ndarray): Ground truth values.
- `y_pred` (np.ndarray): Predicted values.

**Returns:**
- `float`: MAE value.

##### `root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float`

Calculate Root Mean Squared Error between true and predicted values.

**Parameters:**
- `y_true` (np.ndarray): Ground truth values.
- `y_pred` (np.ndarray): Predicted values.

**Returns:**
- `float`: RMSE value.

##### `correlation_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float`

Calculate Pearson correlation coefficient between true and predicted values.

**Parameters:**
- `y_true` (np.ndarray): Ground truth values.
- `y_pred` (np.ndarray): Predicted values.

**Returns:**
- `float`: Correlation coefficient (between -1 and 1).

##### `peak_signal_noise_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float`

Calculate Peak Signal-to-Noise Ratio.

**Parameters:**
- `y_true` (np.ndarray): Ground truth values.
- `y_pred` (np.ndarray): Predicted values.

**Returns:**
- `float`: PSNR value in dB. Returns `inf` if MSE is 0.

##### `align_component(gt: np.ndarray, pred: np.ndarray) -> np.ndarray`

Align predicted component with ground truth by removing mean difference.

Decomposition components are unique only up to a constant (you can add c to trend and subtract c from seasonal and still get the same y). This function aligns the predicted component to have the same mean as ground truth.

**Parameters:**
- `gt` (np.ndarray): Ground truth component.
- `pred` (np.ndarray): Predicted component.

**Returns:**
- `np.ndarray`: Aligned predicted component.

#### Component-wise Metric Functions

##### `compute_decomposition_metrics(ground_truth: dict, result: dict, align_components: bool = True) -> dict`

Compute comprehensive metrics for decomposition quality.

**Parameters:**
- `ground_truth` (dict): Dictionary with `'trend'`, `'seasonal'`, `'residual'` ground truth arrays.
- `result` (dict): Dictionary with `'trend'`, `'seasonal'`, `'residual'` predictions.
- `align_components` (bool, default=True): Whether to align components before computing metrics.

**Returns:**
- `dict`: Nested dictionary with metrics for each component. Structure:
  ```python
  {
      'trend': {'mse': float, 'mae': float, 'rmse': float, 'correlation': float, 'psnr': float},
      'seasonal': {...},
      'residual': {...}
  }
  ```

##### `compute_mse(ground_truth: dict, result: dict, align_components: bool = True) -> dict`

Compute Mean Squared Error for each component.

**Parameters:**
- `ground_truth` (dict): Dictionary with keys `'trend'`, `'seasonal'`, `'residual'` containing ground truth arrays.
- `result` (dict): Dictionary with same structure containing predicted arrays.
- `align_components` (bool, default=True): Whether to align components before computing MSE.

**Returns:**
- `dict`: MSE for each component: `{'trend': float, 'seasonal': float, 'residual': float}`.

**Example:**

```python
from lgtd.evaluation.metrics import compute_mse

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

mse = compute_mse(ground_truth, prediction, align_components=True)
print(f"Trend MSE: {mse['trend']:.4f}")
print(f"Seasonal MSE: {mse['seasonal']:.4f}")
print(f"Residual MSE: {mse['residual']:.4f}")
```

##### `compute_mae(ground_truth: dict, result: dict, align_components: bool = True) -> dict`

Compute Mean Absolute Error for each component.

**Parameters:**
- `ground_truth` (dict): Dictionary with ground truth components.
- `result` (dict): Dictionary with predicted components.
- `align_components` (bool, default=True): Whether to align components before computing MAE.

**Returns:**
- `dict`: MAE for each component: `{'trend': float, 'seasonal': float, 'residual': float}`.

##### `compute_rmse(ground_truth: dict, result: dict) -> dict`

Compute Root Mean Squared Error for each component.

**Parameters:**
- `ground_truth` (dict): Dictionary with ground truth components.
- `result` (dict): Dictionary with predicted components.

**Returns:**
- `dict`: RMSE for each component: `{'trend': float, 'seasonal': float, 'residual': float}`.

##### `compute_correlation(ground_truth: dict, result: dict) -> dict`

Compute Pearson correlation coefficient for each component.

**Parameters:**
- `ground_truth` (dict): Dictionary with ground truth components.
- `result` (dict): Dictionary with predicted components.

**Returns:**
- `dict`: Correlation for each component: `{'trend': float, 'seasonal': float, 'residual': float}`. Values in [-1, 1].

##### `compute_psnr(ground_truth: dict, result: dict) -> dict`

Compute Peak Signal-to-Noise Ratio for each component.

**Parameters:**
- `ground_truth` (dict): Dictionary with ground truth components.
- `result` (dict): Dictionary with predicted components.

**Returns:**
- `dict`: PSNR for each component: `{'trend': float, 'seasonal': float, 'residual': float}`. Values in dB.

---

### Visualization Functions

Located in `lgtd.evaluation.visualization`.

#### `plot_decomposition(result: LGTDResult, ground_truth: dict = None, figsize: tuple = (14, 12), title: str = "Time Series Decomposition", show: bool = True, save_path: str = None, model_name: str = "LGTD", init_point: int = 0) -> Figure`

Plot decomposition results with optional ground truth comparison in a stacked layout.

**Parameters:**
- `result` (LGTDResult): LGTD decomposition result containing trend, seasonal, residual, and original series.
- `ground_truth` (dict, optional): Dictionary with ground truth components (`'trend'`, `'seasonal'`, `'residual'`). If provided, plots both ground truth and estimated components.
- `figsize` (tuple, default=(14, 12)): Figure size in inches (width, height).
- `title` (str, default="Time Series Decomposition"): Figure title.
- `show` (bool, default=True): Whether to display the plot immediately.
- `save_path` (str, optional): Path to save the figure. If None, figure is not saved.
- `model_name` (str, default="LGTD"): Name of the model for legend labels.
- `init_point` (int, default=0): Index marking end of initialization period. Highlighted in gray on plots.

**Returns:**
- `matplotlib.figure.Figure`: Figure object containing four subplots (original, trend, seasonal, residual).

**Example:**

```python
from lgtd.evaluation.visualization import plot_decomposition
import matplotlib.pyplot as plt

# Basic usage
fig = plot_decomposition(result, title="LGTD Decomposition")

# With ground truth comparison
ground_truth = {
    'trend': true_trend,
    'seasonal': true_seasonal,
    'residual': true_residual
}
fig = plot_decomposition(
    result,
    ground_truth=ground_truth,
    save_path="decomposition.png",
    show=False
)
```

#### `plot_comparison(ground_truth: dict, results_dict: dict, component: str = 'trend', figsize: tuple = (14, 6), title: str = None, show: bool = True, save_path: str = None) -> Figure`

Plot comparison of multiple decomposition methods for a single component.

**Parameters:**
- `ground_truth` (dict): Dictionary with ground truth components (`'trend'`, `'seasonal'`, `'residual'`).
- `results_dict` (dict): Dictionary mapping method names to result dictionaries. Structure: `{'Method1': {'trend': ..., 'seasonal': ..., 'residual': ...}, 'Method2': {...}}`.
- `component` (str, default='trend'): Component to plot. One of `'trend'`, `'seasonal'`, or `'residual'`.
- `figsize` (tuple, default=(14, 6)): Figure size in inches (width, height).
- `title` (str, optional): Plot title. If None, auto-generated as "{Component} Component Comparison".
- `show` (bool, default=True): Whether to display the plot immediately.
- `save_path` (str, optional): Path to save the figure. If None, figure is not saved.

**Returns:**
- `matplotlib.figure.Figure`: Figure object with comparison plot.

**Example:**

```python
from lgtd.evaluation.visualization import plot_comparison

ground_truth = {
    'trend': true_trend,
    'seasonal': true_seasonal,
    'residual': true_residual
}

results = {
    'LGTD': {
        'trend': lgtd_result.trend,
        'seasonal': lgtd_result.seasonal,
        'residual': lgtd_result.residual
    },
    'STL': {
        'trend': stl_trend,
        'seasonal': stl_seasonal,
        'residual': stl_residual
    }
}

# Compare trend components
fig = plot_comparison(ground_truth, results, component='trend')
```

#### `plot_evaluation_bars(evaluation_df, metric: str = 'MSE', figsize: tuple = (12, 6), title: str = None, show: bool = True, save_path: str = None) -> Figure`

Plot bar chart comparison of evaluation metrics across multiple methods.

**Parameters:**
- `evaluation_df`: Pandas DataFrame with evaluation results. Expected columns: `'model'`, `'metric'`, `'trend'`, `'seasonal'`, `'residual'`.
- `metric` (str, default='MSE'): Metric to plot (e.g., `'MSE'`, `'MAE'`).
- `figsize` (tuple, default=(12, 6)): Figure size in inches (width, height).
- `title` (str, optional): Plot title. If None, auto-generated as "{Metric} Comparison Across Methods".
- `show` (bool, default=True): Whether to display the plot immediately.
- `save_path` (str, optional): Path to save the figure. If None, figure is not saved.

**Returns:**
- `matplotlib.figure.Figure`: Figure object with bar chart.

**Example:**

```python
from lgtd.evaluation.visualization import plot_evaluation_bars
import pandas as pd

# Create evaluation DataFrame
eval_data = {
    'model': ['LGTD', 'STL', 'Prophet'],
    'metric': ['MSE', 'MSE', 'MSE'],
    'trend': [0.15, 0.23, 0.18],
    'seasonal': [0.42, 0.56, 0.48],
    'residual': [0.98, 1.12, 1.05]
}
df = pd.DataFrame(eval_data)

fig = plot_evaluation_bars(df, metric='MSE')
```

---

## Utility Functions

### Data Generators

**Note:** Data generators are located in `data/synthetic/generators.py` and are used for research and testing purposes. They are **not** part of the installed `lgtd` package.

#### `generate_synthetic_data(n: int = 2000, trend_type: str = 'linear', seasonality_type: str = 'fixed', seasonal_params: dict = None, residual_std: float = 1.0, seed: int = None) -> dict`

Unified synthetic data generator for time series with known ground truth components.

**Parameters:**
- `n` (int, default=2000): Number of time points.
- `trend_type` (str, default='linear'): Type of trend. Options:
  - `'linear'`: Linear trend with configurable slope
  - `'inverted_v'`: Inverted-V shaped trend (rises then falls)
  - `'piecewise'`: Piecewise linear trend with multiple segments
- `seasonality_type` (str, default='fixed'): Type of seasonality. Options:
  - `'fixed'`: Fixed-period seasonal pattern
  - `'transitive'`: Period transitions between two values
  - `'variable'`: Variable periods across different cycles
- `seasonal_params` (dict, optional): Dictionary with seasonality parameters:
  - For `'fixed'`: `{'period': int, 'amplitude': float}`
  - For `'transitive'`: `{'main_period': int, 'transition_period': int, 'amplitude': float}`
  - For `'variable'`: `{'periods': List[int], 'amplitude': float}`
- `residual_std` (float, default=1.0): Standard deviation of Gaussian noise.
- `seed` (int, optional): Random seed for reproducibility.

**Returns:**
- `dict`: Dictionary containing:
  - `'time'` (np.ndarray): Time array
  - `'y'` (np.ndarray): Combined time series (trend + seasonal + residual)
  - `'trend'` (np.ndarray): Trend component
  - `'seasonal'` (np.ndarray): Seasonal component
  - `'residual'` (np.ndarray): Noise/residual component
  - `'config'` (dict): Configuration used for generation

**Example:**

```python
import sys
sys.path.append('data/synthetic')
from generators import generate_synthetic_data

# Generate data with fixed period
data = generate_synthetic_data(
    n=500,
    trend_type='linear',
    seasonality_type='fixed',
    seasonal_params={'period': 24, 'amplitude': 10.0},
    residual_std=1.0,
    seed=42
)

y = data['y']
true_trend = data['trend']
true_seasonal = data['seasonal']
true_residual = data['residual']
```

#### Additional Generator Functions

##### `generate_linear_trend(time: np.ndarray, slope: float = 0.02) -> np.ndarray`

Generate linear trend component.

##### `generate_inverted_v_trend(time: np.ndarray, peak_position: float = 0.6, max_height: float = 100.0, curve_sharpness: float = 3.0) -> np.ndarray`

Generate inverted-V shaped trend.

##### `generate_piecewise_trend(time: np.ndarray, n_segments: int = 4, slopes: List[float] = None) -> np.ndarray`

Generate piecewise linear trend with multiple segments.

##### `generate_fixed_period_seasonality(time: np.ndarray, period: int, amplitude: float = 50.0) -> np.ndarray`

Generate fixed-period seasonal component using sine wave.

##### `generate_transitive_period_seasonality(time: np.ndarray, main_period: int, transition_period: int, transition_start_ratio: float = 0.4, transition_end_ratio: float = 0.6, amplitude: float = 50.0) -> np.ndarray`

Generate seasonal component with period transition.

##### `generate_variable_period_seasonality(time: np.ndarray, periods: List[int], amplitude: float = 50.0) -> np.ndarray`

Generate seasonal component with variable periods across cycles.

#### Convenience Functions

##### `generate_trend_series(n: int = 2000, trend_type: str = 'linear', noise_std: float = 1.0, seed: int = None) -> dict`

Generate time series with only trend component (no seasonality).

##### `generate_seasonal_series(n: int = 2000, period: int = 120, amplitude: float = 50.0, noise_std: float = 1.0, seed: int = None) -> dict`

Generate time series with only seasonal component (no trend).

##### `generate_trend_seasonal_series(n: int = 2000, trend_type: str = 'linear', period: int = 120, amplitude: float = 50.0, noise_std: float = 1.0, seed: int = None) -> dict`

Generate time series with both trend and seasonal components.

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
