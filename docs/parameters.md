# Hyperparameter Guide

## Overview

LGTD has five main hyperparameters that control decomposition behavior. This guide provides recommendations for parameter selection based on data characteristics.

## Parameters

### 1. `window_size`

**Type:** Integer
**Default:** 3
**Range:** $[1, \infty)$
**Role:** Controls local trend sliding window size in Step 3 of the algorithm.

#### Effect

- **Small values** (1-3): Capture fine-grained local variations. Suitable for high-frequency or rapidly changing patterns.
- **Medium values** (4-10): Balance between local and global behavior. Recommended for most applications.
- **Large values** (>10): Emphasize longer-term local trends. May miss short periodicities.

#### Selection Guidelines

**Rule of thumb:** Set $w = \lceil P / 4 \rceil$ where $P$ is the expected shortest period length.

**Examples:**
- Hourly data with daily pattern ($P = 24$): $w = 6$
- Daily data with weekly pattern ($P = 7$): $w = 2$
- Monthly data with annual pattern ($P = 12$): $w = 3$

**When period is unknown:** Start with default $w = 3$ for exploratory analysis.

#### Sensitivity

Low to moderate. Performance degrades gracefully for values within factor of 2 from optimal.

---

### 2. `error_percentile`

**Type:** Float
**Default:** 50
**Range:** $[0, 100]$
**Role:** Percentile threshold for filtering local trend windows based on fitting error.

#### Effect

- **Low values** (0-40): Retain only segments with excellent linear fit. More conservative seasonal extraction. May underestimate seasonal amplitude.
- **Medium values** (40-60): Balance between precision and recall. Recommended default.
- **High values** (60-100): Retain more segments including those with moderate fit. More liberal seasonal extraction. May include non-seasonal variations.

#### Selection Guidelines

**Data quality:**
- High signal-to-noise ratio: Lower percentiles (30-50) for precise seasonality.
- Low signal-to-noise ratio: Higher percentiles (50-70) to ensure sufficient segments.

**Pattern characteristics:**
- Strong, regular seasonality: Lower percentiles detect clear patterns.
- Weak or irregular seasonality: Higher percentiles improve sensitivity.

#### Sensitivity

Moderate. Values in [40, 60] typically produce similar results. Extreme values (<20 or >80) significantly affect decomposition.

---

### 3. `trend_selection`

**Type:** String
**Default:** `'auto'`
**Options:** `'auto'`, `'linear'`, `'lowess'`
**Role:** Determines global trend extraction method in Step 1.

#### Options

##### `'auto'` (Recommended)

Automatically select trend method based on $R^2$ threshold:
- If linear $R^2 \geq$ `threshold_r2`: Use linear regression.
- Otherwise: Use LOWESS.

**Advantage:** Adapts to data characteristics without manual intervention.

##### `'linear'`

Force linear trend: $T_t = \beta_0 + \beta_1 t$.

**Use cases:**
- Data exhibits clear linear growth or decline.
- Fast computation required ($O(n)$ vs $O(n^2)$ for LOWESS).
- Interpretability of trend parameters important.

##### `'lowess'`

Force LOWESS (Locally Weighted Scatterplot Smoothing).

**Use cases:**
- Data exhibits non-linear, smooth trend.
- Linear model achieves poor fit ($R^2 < 0.8$).
- Flexibility in trend estimation prioritized over speed.

#### Selection Guidelines

**Recommendation:** Use `'auto'` unless specific requirements dictate otherwise.

**Visual inspection:** Plot data and assess trend curvature:
- Approximately straight line: `'linear'` suitable.
- Smooth curve: `'lowess'` suitable.
- Complex or piecewise behavior: `'auto'` recommended.

---

### 4. `lowess_frac`

**Type:** Float
**Default:** 0.1
**Range:** $(0, 1]$
**Role:** LOWESS smoothing fraction. Fraction of data points used for local regression at each point.

#### Effect

- **Small values** (0.05-0.15): Follow data closely. Less smoothing. May overfit to short-term fluctuations.
- **Medium values** (0.15-0.30): Balanced smoothing. Typical choice.
- **Large values** (0.30-1.0): Heavily smoothed. May underfit and miss important trend features.

#### Selection Guidelines

**Data length:**
- Short series ($n < 100$): Use larger fractions (0.2-0.3) for stability.
- Long series ($n > 1000$): Use smaller fractions (0.05-0.1) for flexibility.

**Noise level:**
- High noise: Increase fraction to avoid overfitting.
- Low noise: Decrease fraction to capture details.

**Rule of thumb:** Set $f$ such that local window size $\lceil f \cdot n \rceil$ covers 1-2 expected trend cycles.

#### Sensitivity

Moderate to high when `trend_selection='lowess'`. Negligible when `trend_selection='linear'` or linear trend selected by `'auto'`.

---

### 5. `threshold_r2`

**Type:** Float
**Default:** 0.9
**Range:** $[0, 1]$
**Role:** $R^2$ threshold for automatic trend selection when `trend_selection='auto'`.

#### Effect

- **High values** (0.9-1.0): Require excellent linear fit. More likely to select LOWESS. Conservative approach.
- **Medium values** (0.8-0.9): Balanced threshold. Recommended.
- **Low values** (0.5-0.8): Accept moderate linear fit. More likely to select linear trend. Faster computation.

#### Selection Guidelines

**Computational constraints:**
- Speed critical: Lower threshold (0.8) to favor linear trend.
- Accuracy critical: Higher threshold (0.95) to ensure appropriate trend model.

**Data characteristics:**
- Clearly linear trend: Lower threshold acceptable (0.85).
- Potentially non-linear: Higher threshold ensures flexibility (0.9-0.95).

#### Sensitivity

Low when trend is clearly linear ($R^2 > 0.95$) or clearly non-linear ($R^2 < 0.85$). Moderate in transitional region (0.85-0.95).

---

## Recommended Configurations

### Default Configuration

Suitable for most applications with unknown characteristics:

```python
model = lgtd(
    window_size=3,
    error_percentile=50,
    trend_selection='auto',
    lowess_frac=0.1,
    threshold_r2=0.9
)
```

### High-Frequency Data

Hourly or minutely measurements with short periods:

```python
model = lgtd(
    window_size=5,
    error_percentile=50,
    trend_selection='auto',
    lowess_frac=0.05,
    threshold_r2=0.9
)
```

### Noisy Data

Low signal-to-noise ratio or irregular patterns:

```python
model = lgtd(
    window_size=3,
    error_percentile=60,
    trend_selection='auto',
    lowess_frac=0.2,
    threshold_r2=0.85
)
```

### Fast Computation

Prioritize speed over flexibility:

```python
model = lgtd(
    window_size=3,
    error_percentile=50,
    trend_selection='linear',
    lowess_frac=0.1,  # Unused
    threshold_r2=0.9   # Unused
)
```

### Non-Linear Trend

Known complex trend behavior:

```python
model = lgtd(
    window_size=3,
    error_percentile=50,
    trend_selection='lowess',
    lowess_frac=0.1,
    threshold_r2=0.9   # Unused
)
```

---

## Parameter Tuning Procedure

### Grid Search

For datasets with ground truth, perform grid search:

```python
from itertools import product
from lgtd import lgtd
from lgtd.evaluation.metrics import compute_mse

# Define parameter grid
window_sizes = [2, 3, 5, 7]
error_percentiles = [40, 50, 60, 70]

best_mse = float('inf')
best_params = None

for w, p in product(window_sizes, error_percentiles):
    model = lgtd(window_size=w, error_percentile=p)
    result = model.fit_transform(y)

    mse = compute_mse(ground_truth, {
        'trend': result.trend,
        'seasonal': result.seasonal,
        'residual': result.residual
    })

    if mse['total'] < best_mse:
        best_mse = mse['total']
        best_params = (w, p)

print(f"Best parameters: window_size={best_params[0]}, error_percentile={best_params[1]}")
```

### Visual Inspection

For real-world data without ground truth:

```python
from lgtd.evaluation.visualization import plot_decomposition

# Try different configurations
configs = [
    {'window_size': 3, 'error_percentile': 50},
    {'window_size': 5, 'error_percentile': 50},
    {'window_size': 3, 'error_percentile': 60}
]

for config in configs:
    model = lgtd(**config)
    result = model.fit_transform(y)
    plot_decomposition(y, result, title=str(config))
```

Evaluate visually:
- **Trend:** Should be smooth without short-term fluctuations.
- **Seasonal:** Should exhibit clear periodic pattern.
- **Residual:** Should appear as white noise (no structure).

### Cross-Validation

For forecasting applications, use time series cross-validation:

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

for train_idx, test_idx in tscv.split(y):
    y_train = y[train_idx]
    model = lgtd(window_size=3, error_percentile=50)
    result = model.fit_transform(y_train)
    # Evaluate decomposition quality on training fold
```

---

## Computational Considerations

### Time Complexity by Configuration

| Configuration | Trend | Window | Total |
|---------------|-------|--------|-------|
| Linear trend | $O(n)$ | $O(n \cdot w)$ | $O(n \cdot w)$ |
| LOWESS trend | $O(n^2)$ | $O(n \cdot w)$ | $O(n^2)$ |

**Recommendation:** For $n > 10^4$, use `trend_selection='linear'` if acceptable.

### Memory Usage

All configurations: $O(n)$ space for components and intermediate arrays.

---

## Common Issues

### Issue: Seasonal component is flat (near zero)

**Diagnosis:** No local linear segments retained.

**Solutions:**
- Increase `error_percentile` to retain more segments.
- Decrease `window_size` if period is shorter than expected.
- Verify data contains actual seasonal patterns (plot ACF).

### Issue: Residual contains clear pattern

**Diagnosis:** Seasonal extraction incomplete.

**Solutions:**
- Increase `error_percentile` to capture more variation.
- Verify `window_size` matches periodicity.
- Try `trend_selection='lowess'` if trend misspecified.

### Issue: Trend oscillates (contains seasonality)

**Diagnosis:** Oversmoothing in trend extraction or inappropriate method.

**Solutions:**
- Decrease `lowess_frac` for more flexible LOWESS.
- Lower `threshold_r2` to favor linear trend.
- Manually set `trend_selection='linear'` if appropriate.

### Issue: Computation too slow

**Solutions:**
- Use `trend_selection='linear'` to avoid $O(n^2)$ LOWESS.
- Reduce `window_size` (minor speedup).
- Downsample data if acceptable (e.g., hourly to daily).

---

## Summary

**Most important parameters:** `window_size` and `error_percentile`
**Safe default:** Use default configuration for initial exploration
**When in doubt:** Perform visual inspection or grid search with ground truth
