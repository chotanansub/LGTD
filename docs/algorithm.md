# Algorithm Description

## Overview

LGTD (Local-Global Trend Decomposition) is a season-length-free time series decomposition method that does not require prior specification of seasonal period lengths. The algorithm combines global trend extraction with local trend analysis to automatically identify and extract seasonal patterns.

## Mathematical Formulation

### Additive Decomposition Model

LGTD decomposes a univariate time series $y_t$ ($t = 1, \ldots, n$) into three components:

$$y_t = T_t + S_t + R_t$$

where:
- $T_t$: Trend component (long-term movement)
- $S_t$: Seasonal component (periodic patterns)
- $R_t$: Residual component (irregular fluctuations)

## Algorithm Steps

### Step 1: Global Trend Extraction

Extract global trend $T_t$ from input series $y_t$ using either linear regression or LOWESS (Locally Weighted Scatterplot Smoothing).

#### Linear Trend

Fit ordinary least squares regression:

$$T_t = \beta_0 + \beta_1 t$$

where parameters $(\beta_0, \beta_1)$ minimize:

$$\sum_{t=1}^{n} (y_t - \beta_0 - \beta_1 t)^2$$

Compute coefficient of determination:

$$R^2 = 1 - \frac{\sum_{t=1}^{n}(y_t - T_t)^2}{\sum_{t=1}^{n}(y_t - \bar{y})^2}$$

#### LOWESS Trend

Apply locally weighted regression at each point $t$:

$$T_t = \sum_{i=1}^{n} w_i(t) y_i$$

where weights $w_i(t)$ are determined by tricube kernel:

$$w_i(t) = \left(1 - \left|\frac{t - i}{\Delta}\right|^3\right)^3 \mathbb{1}_{|t-i| \leq \Delta}$$

with bandwidth $\Delta = \lceil f \cdot n \rceil$ for smoothing fraction $f \in (0, 1]$.

#### Automatic Selection

Select trend method based on $R^2$ threshold $\tau$:

$$\text{method} = \begin{cases}
\text{linear} & \text{if } R^2_{\text{linear}} \geq \tau \\
\text{lowess} & \text{otherwise}
\end{cases}$$

Default: $\tau = 0.9$.

---

### Step 2: Detrending

Compute detrended series:

$$d_t = y_t - T_t$$

The detrended series $d_t$ contains seasonal patterns and irregular fluctuations.

---

### Step 3: Local Trend Analysis

Apply sliding-window local trend detection using AutoTrend algorithm to identify linear segments in $d_t$.

#### Sliding Window

For window size $w \geq 1$, extract windows:

$$W_i = \{d_{i}, d_{i+1}, \ldots, d_{i+w-1}\}, \quad i = 1, \ldots, n-w+1$$

#### Local Linear Fitting

For each window $W_i$, fit local linear trend:

$$\hat{d}_t^{(i)} = \alpha_i + \beta_i (t - i), \quad t \in \{i, \ldots, i+w-1\}$$

Compute fitting error:

$$e_i = \frac{1}{w} \sum_{t=i}^{i+w-1} |d_t - \hat{d}_t^{(i)}|$$

#### Error Thresholding

Compute error threshold as $p$-th percentile of all window errors:

$$\theta = \text{percentile}(\{e_1, \ldots, e_{n-w+1}\}, p)$$

where $p \in [0, 100]$ is the error percentile parameter (default: $p = 50$).

#### Segment Selection

Retain windows with error below threshold:

$$\mathcal{S} = \{W_i : e_i \leq \theta\}$$

These windows represent local linear segments where detrended series follows approximate linear pattern.

---

### Step 4: Seasonal Extraction

Aggregate local deviations from retained segments to construct seasonal pattern.

#### Local Deviation

For each retained segment $W_i \in \mathcal{S}$, compute local deviation:

$$\delta_t^{(i)} = d_t - \hat{d}_t^{(i)}, \quad t \in \{i, \ldots, i+w-1\}$$

#### Aggregation

For each time point $t$, collect all local deviations:

$$\Delta_t = \{\delta_t^{(i)} : W_i \in \mathcal{S}, i \leq t \leq i+w-1\}$$

Compute seasonal component as median of local deviations:

$$S_t = \text{median}(\Delta_t)$$

If $\Delta_t = \emptyset$ (no segments cover $t$), set $S_t = 0$ or interpolate from nearest values.

#### Period Detection

Detect dominant periods by applying autocorrelation or Fourier analysis to $S_t$:

$$r(k) = \frac{\sum_{t=1}^{n-k} S_t S_{t+k}}{\sum_{t=1}^{n} S_t^2}, \quad k = 1, \ldots, \lfloor n/2 \rfloor$$

Identify peaks in $r(k)$ as detected periods.

---

### Step 5: Residual Computation

Compute residual as unexplained variation:

$$R_t = y_t - T_t - S_t$$

Verify additive decomposition:

$$y_t = T_t + S_t + R_t$$

---

## Computational Complexity

**Time Complexity:**

- Global trend extraction: $O(n)$ (linear) or $O(n^2)$ (LOWESS)
- Sliding window analysis: $O((n-w+1) \cdot w)$
- Seasonal aggregation: $O(n \cdot (n-w+1))$ worst case, typically $O(n)$
- **Overall:** $O(n^2)$ dominated by LOWESS or seasonal aggregation

**Space Complexity:** $O(n)$ for storing components and intermediate results.

For typical time series ($n \leq 10^4$), computation completes within seconds on standard hardware.

---

## Advantages

1. **Season-Length-Free:** No prior knowledge of seasonal period required.

2. **Automatic Trend Selection:** Adapts to linear or non-linear trends based on data characteristics.

3. **Robust to Noise:** Median aggregation and percentile thresholding provide robustness to outliers.

4. **Multiple Periodicities:** Can detect and extract multiple overlapping seasonal patterns.

5. **Interpretable Components:** Additive decomposition provides clear interpretation of trend, seasonality, and residuals.

---

## Limitations

1. **Computational Cost:** $O(n^2)$ complexity may be prohibitive for very long series ($n > 10^5$).

2. **Stationarity Assumption:** Assumes seasonal patterns are approximately stationary (constant amplitude and phase).

3. **Minimum Length Requirement:** Requires sufficient data points ($n \geq 2w$) for meaningful local trend analysis.

4. **Parameter Sensitivity:** Performance depends on appropriate selection of `window_size` and `error_percentile` (see [parameters.md](parameters.md)).

---

## Comparison with STL

| Aspect | STL | LGTD |
|--------|-----|------|
| Period specification | Required | Not required |
| Trend estimation | LOESS with fixed span | Linear or LOWESS (automatic) |
| Seasonal extraction | Cyclic subseries smoothing | Local trend deviation aggregation |
| Multiple periods | Requires MSTL extension | Native support |
| Computational cost | $O(n \log n)$ | $O(n^2)$ |

---

## References

For detailed mathematical derivations and empirical validation, refer to the LGTD paper (under review).

Related methods:
- Cleveland, R. B., et al. (1990). STL: A seasonal-trend decomposition procedure based on loess. *Journal of Official Statistics*, 6(1), 3-73.
- Wen, Q., et al. (2019). RobustSTL: A robust seasonal-trend decomposition algorithm for long time series. *AAAI*.
- Phungtua-eng, T., & Yamamoto, Y. (2024). ASTD: Adaptive seasonal-trend decomposition. *ECML PKDD*.
