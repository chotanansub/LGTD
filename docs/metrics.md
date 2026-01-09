# Evaluation Metrics

This document describes all evaluation metrics used to assess decomposition quality in LGTD experiments.

## Overview

Decomposition methods are evaluated using standard metrics that measure component-wise accuracy (when ground truth available) and reconstruction quality (always computable).

---

## Component-Wise Metrics

These metrics require ground truth components, available only for synthetic datasets.

### Mean Squared Error (MSE)

**Definition:**

For component $C \in \{\text{Trend}, \text{Seasonal}, \text{Residual}\}$:

$$\text{MSE}(C) = \frac{1}{n} \sum_{t=1}^{n} (C_t^{\text{true}} - C_t^{\text{pred}})^2$$

**Interpretation:**
- Lower values indicate better fit
- Sensitive to outliers (squared error)
- Units: Square of original data units

**Component-specific MSE:**

$$\text{MSE}_{\text{trend}} = \frac{1}{n} \sum_{t=1}^{n} (T_t^{\text{true}} - T_t^{\text{pred}})^2$$

$$\text{MSE}_{\text{seasonal}} = \frac{1}{n} \sum_{t=1}^{n} (S_t^{\text{true}} - S_t^{\text{pred}})^2$$

$$\text{MSE}_{\text{residual}} = \frac{1}{n} \sum_{t=1}^{n} (R_t^{\text{true}} - R_t^{\text{pred}})^2$$

**Total MSE:**

$$\text{MSE}_{\text{total}} = \text{MSE}_{\text{trend}} + \text{MSE}_{\text{seasonal}} + \text{MSE}_{\text{residual}}$$

**Usage in paper:**
- Primary metric in Tables 1, 3
- Reported for each dataset-method combination

**Implementation:**

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
print(f"Residual MSE: {mse['residual']:.4f}")
print(f"Total MSE: {mse['total']:.4f}")
```

---

### Mean Absolute Error (MAE)

**Definition:**

For component $C$:

$$\text{MAE}(C) = \frac{1}{n} \sum_{t=1}^{n} |C_t^{\text{true}} - C_t^{\text{pred}}|$$

**Interpretation:**
- Lower values indicate better fit
- More robust to outliers than MSE (absolute vs. squared error)
- Same units as original data

**Total MAE:**

$$\text{MAE}_{\text{total}} = \text{MAE}_{\text{trend}} + \text{MAE}_{\text{seasonal}} + \text{MAE}_{\text{residual}}$$

**Usage in paper:**
- Secondary metric in Table 2
- Complements MSE for robustness assessment

**Implementation:**

```python
from LGTD.evaluation.metrics import compute_mae

mae = compute_mae(ground_truth, prediction)
print(f"Total MAE: {mae['total']:.4f}")
```

---

### Pearson Correlation Coefficient

**Definition:**

For component $C$:

$$\rho(C) = \frac{\sum_{t=1}^{n} (C_t^{\text{true}} - \bar{C}^{\text{true}})(C_t^{\text{pred}} - \bar{C}^{\text{pred}})}{\sqrt{\sum_{t=1}^{n} (C_t^{\text{true}} - \bar{C}^{\text{true}})^2} \sqrt{\sum_{t=1}^{n} (C_t^{\text{pred}} - \bar{C}^{\text{pred}})^2}}$$

where $\bar{C}$ denotes mean of component $C$.

**Interpretation:**
- Range: $[-1, 1]$
- $\rho = 1$: Perfect positive correlation
- $\rho = 0$: No linear correlation
- $\rho = -1$: Perfect negative correlation
- Higher values (closer to 1) indicate better shape preservation

**Usage in paper:**
- Table 3: Component-wise correlation
- Assesses whether decomposition preserves temporal patterns even if absolute values differ

**Implementation:**

```python
from LGTD.evaluation.metrics import compute_correlation

corr = compute_correlation(ground_truth, prediction)
print(f"Trend correlation: {corr['trend']:.4f}")
print(f"Seasonal correlation: {corr['seasonal']:.4f}")
```

**Note:** Correlation insensitive to additive constant or multiplicative scaling. Complements MSE/MAE which measure absolute error.

---

## Reconstruction Metrics

These metrics assess overall decomposition quality and can be computed without ground truth.

### Peak Signal-to-Noise Ratio (PSNR)

**Definition:**

$$\text{PSNR}(y^{\text{true}}, y^{\text{recon}}) = 20 \log_{10} \left( \frac{\max(y^{\text{true}}) - \min(y^{\text{true}})}{\text{RMSE}(y^{\text{true}}, y^{\text{recon}})} \right)$$

where:

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{t=1}^{n} (y_t^{\text{true}} - y_t^{\text{recon}})^2}$$

and reconstructed series:

$$y_t^{\text{recon}} = T_t + S_t + R_t$$

**Interpretation:**
- Units: Decibels (dB)
- Higher values indicate better reconstruction quality
- Typical range: 20-50 dB
  - PSNR < 30 dB: Poor quality
  - 30-40 dB: Acceptable quality
  - PSNR > 40 dB: Excellent quality

**Usage in paper:**
- Table 4: Real-world dataset evaluation
- Primary metric when ground truth unavailable

**Implementation:**

```python
from LGTD.evaluation.metrics import compute_psnr

# Original series
y = data['y']

# Reconstructed series
y_recon = result.trend + result.seasonal + result.residual

psnr = compute_psnr(y, y_recon)
print(f"PSNR: {psnr:.2f} dB")
```

---

### Reconstruction Error

**Definition:**

$$\text{Reconstruction Error} = \frac{1}{n} \sum_{t=1}^{n} (y_t - (T_t + S_t + R_t))^2$$

**Interpretation:**
- Should be near zero for valid additive decomposition
- Non-zero values indicate numerical instability or implementation error
- Typical values: < $10^{-10}$ (machine precision)

**Usage:**
- Sanity check for decomposition validity
- Not reported in paper (expected to be zero)

**Implementation:**

```python
# Verify additive decomposition
y_recon = result.trend + result.seasonal + result.residual
recon_error = np.mean((y - y_recon)**2)
assert recon_error < 1e-8, f"Reconstruction error too large: {recon_error}"
```

---

## Execution Time

**Definition:**

Wall-clock time (seconds) from start to completion of decomposition.

**Measurement:**

```python
import time

start_time = time.time()
result = model.fit_transform(y)
execution_time = time.time() - start_time
```

**Usage in paper:**
- Reported in all tables as secondary consideration
- Assesses computational efficiency
- Important for practical applicability

**Interpretation:**
- Lower is better
- Compare methods on same hardware
- Consider asymptotic complexity for scalability

**Typical values (n=500, reference hardware):**
- Fast methods (STL, LGTD-linear): < 0.1s
- Moderate methods (RobustSTL, ASTD): 0.5-2s
- Slow methods (STR, OneShotSTL): > 2s

---

## Statistical Significance Testing

### Paired t-test

For comparing two methods across multiple datasets:

**Null hypothesis:** $H_0: \mu_{\text{diff}} = 0$ (no difference in mean performance)

**Test statistic:**

$$t = \frac{\bar{d}}{s_d / \sqrt{m}}$$

where:
- $d_i$ = metric difference on dataset $i$
- $\bar{d}$ = mean difference
- $s_d$ = standard deviation of differences
- $m$ = number of datasets

**Implementation:**

```python
from scipy.stats import ttest_rel

# Collect MSE values across datasets
mse_lgtd = [mse_dataset1_lgtd, mse_dataset2_lgtd, ...]
mse_stl = [mse_dataset1_stl, mse_dataset2_stl, ...]

# Perform paired t-test
t_stat, p_value = ttest_rel(mse_lgtd, mse_stl)

if p_value < 0.05:
    print(f"Significant difference (p={p_value:.4f})")
else:
    print(f"No significant difference (p={p_value:.4f})")
```

**Usage:** Determine if LGTD significantly outperforms baselines.

---

### Wilcoxon Signed-Rank Test

Non-parametric alternative to paired t-test when normality assumption violated.

**Implementation:**

```python
from scipy.stats import wilcoxon

w_stat, p_value = wilcoxon(mse_lgtd, mse_stl)
```

**Usage:** Robustness check for significance testing.

---

## Metric Selection Guidelines

### For Synthetic Experiments

**Primary metrics:**
1. **MSE (total)**: Overall performance, penalizes large errors
2. **MAE (total)**: Robustness to outliers
3. **Correlation (component-wise)**: Shape preservation

**Rationale:** Ground truth available, enabling detailed component analysis.

### For Real-World Experiments

**Primary metrics:**
1. **PSNR**: Reconstruction quality without ground truth
2. **Execution time**: Practical applicability

**Secondary metrics:**
- Detected period lengths (for period-free methods)
- Visual inspection of decomposition

**Rationale:** No ground truth, focus on reconstruction and interpretability.

---

## Metric Computation Example

Complete evaluation pipeline:

```python
from LGTD import LGTD
from LGTD.evaluation.metrics import compute_mse, compute_mae, compute_correlation, compute_psnr
import numpy as np
import time

# Load data with ground truth
data = load_synthetic_dataset('synth1')
y = data['y']
true_trend = data['trend']
true_seasonal = data['seasonal']
true_residual = data['residual']

# Decompose with timing
start_time = time.time()
model = LGTD()
result = model.fit_transform(y)
execution_time = time.time() - start_time

# Prepare dictionaries
ground_truth = {
    'trend': true_trend,
    'seasonal': true_seasonal,
    'residual': true_residual
}
prediction = {
    'trend': result.trend,
    'seasonal': result.seasonal,
    'residual': result.residual
}

# Compute all metrics
mse = compute_mse(ground_truth, prediction)
mae = compute_mae(ground_truth, prediction)
corr = compute_correlation(ground_truth, prediction)
y_recon = result.trend + result.seasonal + result.residual
psnr = compute_psnr(y, y_recon)

# Print results
print("=== Evaluation Results ===")
print(f"\nMSE:")
print(f"  Trend: {mse['trend']:.4f}")
print(f"  Seasonal: {mse['seasonal']:.4f}")
print(f"  Residual: {mse['residual']:.4f}")
print(f"  Total: {mse['total']:.4f}")

print(f"\nMAE:")
print(f"  Total: {mae['total']:.4f}")

print(f"\nCorrelation:")
print(f"  Trend: {corr['trend']:.4f}")
print(f"  Seasonal: {corr['seasonal']:.4f}")

print(f"\nReconstruction:")
print(f"  PSNR: {psnr:.2f} dB")

print(f"\nExecution time: {execution_time:.4f} seconds")
```

---

## Metric Normalization

For fair comparison across datasets with different scales:

### Normalized MSE (NMSE)

$$\text{NMSE} = \frac{\text{MSE}}{\text{Var}(y^{\text{true}})}$$

where $\text{Var}(y^{\text{true}})$ is variance of ground truth component.

### Normalized MAE (NMAE)

$$\text{NMAE} = \frac{\text{MAE}}{\text{Range}(y^{\text{true}})}$$

where $\text{Range} = \max(y^{\text{true}}) - \min(y^{\text{true}})$.

**Usage:** Not used in main paper results but available for cross-dataset comparison.

---

## Error Analysis

### Component Error Distribution

Analyze error distribution for each component:

```python
import matplotlib.pyplot as plt

errors_trend = result.trend - true_trend
errors_seasonal = result.seasonal - true_seasonal

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(errors_trend, bins=50)
plt.title('Trend Error Distribution')
plt.xlabel('Error')

plt.subplot(1, 3, 2)
plt.hist(errors_seasonal, bins=50)
plt.title('Seasonal Error Distribution')
plt.xlabel('Error')

plt.subplot(1, 3, 3)
plt.scatter(true_seasonal, result.seasonal, alpha=0.5)
plt.plot([true_seasonal.min(), true_seasonal.max()],
         [true_seasonal.min(), true_seasonal.max()], 'r--')
plt.title('Seasonal: True vs Predicted')
plt.xlabel('True')
plt.ylabel('Predicted')

plt.tight_layout()
plt.show()
```

### Residual Analysis

Check if residuals are white noise (desirable property):

```python
from scipy.stats import normaltest

# Normality test
stat, p_value = normaltest(result.residual)
print(f"Normality test p-value: {p_value:.4f}")

# Autocorrelation
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(result.residual, lags=50)
plt.title('Residual Autocorrelation')
plt.show()
```

**Interpretation:**
- Residuals should be approximately normal (p > 0.05)
- Autocorrelation should be near zero for all lags (white noise)
- Significant autocorrelation indicates unexplained structure

---

## Summary

**Most important metrics:**
- **Synthetic data:** MSE (total), correlation
- **Real-world data:** PSNR, execution time

**Robustness checks:**
- MAE (outlier sensitivity)
- Statistical significance tests

**Diagnostics:**
- Reconstruction error (validity)
- Residual analysis (completeness)

All metrics implemented in `LGTD.evaluation.metrics` module with unit tests ensuring correctness.
