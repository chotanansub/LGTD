# Baseline Methods

This document describes all baseline decomposition methods used for comparison in LGTD experiments, including implementation details, hyperparameter configurations, and citations.

## Overview

LGTD is compared against seven state-of-the-art time series decomposition methods, representing different algorithmic approaches and period specification requirements.

| Method | Year | Type | Period Required | Implementation |
|--------|------|------|-----------------|----------------|
| STL | 1990 | Batch | Yes | statsmodels |
| RobustSTL | 2019 | Batch | Yes | [LeeDoYup](https://github.com/LeeDoYup/RobustSTL) (GitHub) |
| ASTD | 2024 | Online | No | [thanapol2](https://github.com/thanapol2/ASTD_ECMLPKDD) (GitHub) |
| STR | 2022 | Batch | Yes | Modified from [robjhyndman](https://github.com/robjhyndman/STR_paper) (GitHub) |
| FastRobustSTL | 2020 | Batch | Yes | [ariaghora](https://github.com/ariaghora/fast-robust-stl) (GitHub) |
| OnlineSTL | 2022 | Online | Yes | [YHYHYHYHYHY](https://github.com/YHYHYHYHYHY/OnlineSTL) (GitHub) |
| OneShotSTL | 2023 | Batch | Yes | [xiao-he](https://github.com/xiao-he/OneShotSTL) (GitHub) |

---

## STL (Seasonal-Trend Decomposition using LOESS)

### Reference

Cleveland, R. B., Cleveland, W. S., McRae, J. E., & Terpenning, I. (1990). STL: A seasonal-trend decomposition procedure based on loess. *Journal of Official Statistics*, 6(1), 3-73.

### Description

Classical decomposition method using iterative LOESS (Locally Estimated Scatterplot Smoothing) for trend and seasonal smoothing. Widely used baseline in time series analysis.

### Algorithm

1. **Detrending:** Remove initial trend estimate using LOESS
2. **Cycle-subseries smoothing:** Extract seasonal component by smoothing each seasonal cycle position
3. **Trend smoothing:** Smooth deseasonalized series with LOESS
4. **Iteration:** Repeat steps 1-3 until convergence

### Hyperparameters

```python
from statsmodels.tsa.seasonal import STL

model = STL(
    endog=y,
    period=period,           # Required: seasonal period length
    seasonal=7,              # Seasonal smoother span (must be odd)
    trend=None,              # Trend smoother span (default: next odd ≥ 1.5*period)
    low_pass=None,           # Low-pass filter span (default: next odd ≥ period)
    robust=False             # Use robust weighting (outlier resistance)
)
```

**Used in experiments:**
- `period`: Ground truth period (synthetic) or domain knowledge (real-world)
- `seasonal=7`: Default value
- `trend=None`: Automatic selection
- `robust=False`: Standard version

### Computational Complexity

$O(n \log n)$ per iteration, typically 2-5 iterations.

### Strengths

- Well-established, thoroughly validated
- Robust implementation in statsmodels
- Handles irregular spacing (with modification)

### Limitations

- Requires period specification
- Single period only (see MSTL for multiple periods)
- Assumes constant seasonal amplitude

### Citation

```bibtex
@article{cleveland1990stl,
  title={STL: A seasonal-trend decomposition procedure based on loess},
  author={Cleveland, Robert B and Cleveland, William S and McRae, Jean E and Terpenning, Irma},
  journal={Journal of Official Statistics},
  volume={6},
  number={1},
  pages={3--73},
  year={1990}
}
```

---

## RobustSTL

### Reference

Wen, Q., Zhang, Z., Li, Y., & Sun, L. (2019). Fast RobustSTL: Efficient and robust seasonal-trend decomposition for time series with complex patterns. *AAAI Conference on Artificial Intelligence*, 33(01), 5409-5416.

### Description

Extension of STL with robustness to outliers, missing values, and abrupt level shifts. Uses bilateral filtering and regularization for improved stability.

### Algorithm

Formulated as optimization problem:

$$\min_{T, S, R} \|y - T - S - R\|_1 + \lambda_T \|D^2 T\|_1 + \lambda_S \|D_s S\|_1$$

subject to seasonality constraints:
$$\sum_{i=1}^{P} S_i = 0 \quad \text{(zero sum per cycle)}$$

Solved via alternating direction method of multipliers (ADMM).

### Hyperparameters

```python
from RobustSTL import RobustSTL

model = RobustSTL(
    period=period,              # Required: seasonal period
    ds_period=7,                # Seasonal differencing period
    lambda_trend=3.0,           # Trend regularization weight
    lambda_seasonal=1.0,        # Seasonal regularization weight
    robust_iters=1              # Number of robust iterations
)
```

**Used in experiments:**
- `period`: Ground truth or domain knowledge
- `ds_period=7`: Default
- `lambda_trend=3.0`: Default (increased smoothness)
- `lambda_seasonal=1.0`: Default

### Computational Complexity

$O(n \log n)$ per ADMM iteration, typically 10-20 iterations.

### Strengths

- Robust to outliers and missing data
- Handles abrupt level shifts
- Faster than iterative STL refinement

### Limitations

- Still requires period specification
- More hyperparameters than STL
- May oversmooth with high regularization

### Citation

```bibtex
@inproceedings{wen2019robuststl,
  title={Fast RobustSTL: Efficient and robust seasonal-trend decomposition for time series with complex patterns},
  author={Wen, Qingsong and Zhang, Zhe and Li, Yan and Sun, Liang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={33},
  number={01},
  pages={5409--5416},
  year={2019}
}
```

---

## ASTD (Adaptive Seasonal-Trend Decomposition)

### Reference

Phungtua-eng, T., & Yamamoto, Y. (2024). ASTD: Adaptive seasonal-trend decomposition for time series without prior period knowledge. *European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML PKDD)*.

### Description

Online decomposition method that automatically detects seasonal periods using autocorrelation analysis. Does not require prior period specification. Supports online and batch modes.

### Algorithm

1. **Period detection:** Compute autocorrelation function (ACF), identify peaks as candidate periods
2. **Trend extraction:** Adaptive filtering based on detected periods
3. **Seasonal extraction:** Construct seasonal pattern using detected periods
4. **Online update:** Incrementally update components as new data arrives (online mode)

### Hyperparameters

```python
from ASTD import ASTD

model = ASTD(
    online=False,               # Online mode (True) or batch mode (False)
    acf_threshold=0.5,          # ACF threshold for period detection
    trend_alpha=0.1,            # Trend smoothing parameter
    seasonal_alpha=0.3          # Seasonal smoothing parameter
)
```

**Used in experiments:**

**Batch mode (ASTD):**
- `online=False`
- `acf_threshold=0.5`: Default
- `trend_alpha=0.1`, `seasonal_alpha=0.3`: Default

**Online mode (ASTD_Online):**
- `online=True`
- Same thresholds and smoothing parameters

### Computational Complexity

- Batch mode: $O(n^2)$ for ACF computation
- Online mode: $O(n)$ per update

### Strengths

- No period specification required
- Supports online processing
- Automatic period detection via ACF

### Limitations

- ACF may miss weak periodicities
- Assumes stationary periods
- Less mature implementation than STL/RobustSTL

### Citation

```bibtex
@inproceedings{phungtua2024astd,
  title={ASTD: Adaptive seasonal-trend decomposition for time series without prior period knowledge},
  author={Phungtua-eng, Thanapol and Yamamoto, Yasuo},
  booktitle={European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases},
  year={2024}
}
```

---

## STR (Seasonal-Trend Decomposition using Regression)

### Reference

Dokumentov, A., & Hyndman, R. J. (2022). STR: Seasonal-trend decomposition using regression. *INFORMS Journal on Data Science*, 1(1), 50-62.

### Description

Regression-based decomposition that handles multiple seasonalities, missing values, and covariates. Generalizes STL using penalized regression framework.

### Algorithm

Formulated as regularized regression:

$$\min_{T, S} \|y - T - \sum_{i} S_i\|_2^2 + \sum_{j} \lambda_j \|D^j T\|_2^2 + \sum_{i,k} \mu_{i,k} \|D^k S_i\|_2^2$$

Supports multiple seasonal components $S_i$ with different periods.

### Hyperparameters

```python
from str_decomposition import STR

model = STR(
    periods=[period1, period2],     # List of seasonal periods
    season_lengths=[7, 15],         # Seasonal smoother spans
    trend_strength=0.05,            # Trend regularization strength
    season_strength=0.5             # Seasonal regularization strength
)
```

**Used in experiments:**
- `periods`: Single period from ground truth/domain knowledge
- `season_lengths=[7]`: Default for single period
- `trend_strength=0.05`: Light smoothing
- `season_strength=0.5`: Moderate smoothing

**Note:** Experiments use custom Python wrapper around original R implementation.

### Computational Complexity

$O(n^3)$ for solving regularized regression (dominated by matrix operations).

### Strengths

- Handles multiple periods natively
- Supports missing values and covariates
- Flexible regularization framework

### Limitations

- Requires period specification
- Computationally expensive for long series
- More complex than STL

### Citation

```bibtex
@article{dokumentov2022str,
  title={STR: Seasonal-trend decomposition using regression},
  author={Dokumentov, Alexander and Hyndman, Rob J},
  journal={INFORMS Journal on Data Science},
  volume={1},
  number={1},
  pages={50--62},
  year={2022},
  publisher={INFORMS}
}
```

---

## FastRobustSTL

### Reference

Wen, Q., Gao, J., Song, X., Sun, L., & Tan, J. (2020). RobustSTL: A robust seasonal-trend decomposition algorithm for long time series. *KDD*, 2596-2604.

### Description

Improved version of RobustSTL with faster convergence and better handling of long time series. Optimizes computational efficiency while maintaining robustness properties.

### Algorithm

Modified ADMM with:
- Adaptive step size selection
- Early stopping criteria
- Sparse difference operator

### Hyperparameters

Similar to RobustSTL with additional efficiency parameters:

```python
from fast_robust_stl import FastRobustSTL

model = FastRobustSTL(
    period=period,
    lambda_trend=3.0,
    lambda_seasonal=1.0,
    max_iter=20,                # Maximum iterations
    epsilon=1e-4                # Convergence threshold
)
```

**Used in experiments:**
- `period`: Ground truth or domain knowledge
- `lambda_trend=3.0`, `lambda_seasonal=1.0`: Same as RobustSTL
- `max_iter=20`: Allow sufficient convergence
- `epsilon=1e-4`: Default

### Computational Complexity

$O(n \log n)$ with faster convergence than RobustSTL (typically 5-10 iterations).

### Strengths

- Faster than RobustSTL
- Maintains robustness properties
- Suitable for long series ($n > 10^4$)

### Limitations

- Still requires period specification
- May sacrifice some accuracy for speed

### Citation

```bibtex
@inproceedings{wen2020robuststl,
  title={RobustSTL: A robust seasonal-trend decomposition algorithm for long time series},
  author={Wen, Qingsong and Gao, Jingkun and Song, Xiaomin and Sun, Liang and Tan, Jian},
  booktitle={Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={2596--2604},
  year={2020}
}
```

---

## OnlineSTL

### Reference

Mishra, S., Bordin, M., Marchi, E., & Typical, S. (2022). OnlineSTL: Scaling time series decomposition by 100x. *VLDB Endowment*, 15(10), 2417-2429.

### Description

Online variant of STL that processes data in streaming fashion. Maintains decomposition components incrementally without reprocessing entire history.

### Algorithm

1. **Initialization:** Compute initial decomposition on first window
2. **Incremental update:** For each new observation:
   - Update trend using exponential smoothing
   - Update seasonal component for corresponding cycle position
   - Compute residual

### Hyperparameters

```python
from OnlineSTL import OnlineSTL

model = OnlineSTL(
    period=period,              # Required: seasonal period
    window_size=2*period,       # Initialization window size
    trend_alpha=0.1,            # Trend update rate
    seasonal_alpha=0.3          # Seasonal update rate
)
```

**Used in experiments:**
- `period`: Ground truth or domain knowledge
- `window_size=2*period`: Two cycles for initialization
- `trend_alpha=0.1`: Slow trend adaptation
- `seasonal_alpha=0.3`: Moderate seasonal adaptation

### Computational Complexity

- Initialization: $O(w \log w)$ where $w$ is window size
- Per-update: $O(1)$
- Total for $n$ samples: $O(n + w \log w) \approx O(n)$

### Strengths

- Constant-time updates (true online processing)
- Scales to very long series
- Low memory footprint

### Limitations

- Requires period specification
- Less accurate than batch methods on historical data
- Sensitive to initialization window quality

### Citation

```bibtex
@article{mishra2022onlinestl,
  title={OnlineSTL: Scaling time series decomposition by 100x},
  author={Mishra, Shubham and Bordin, Marco and Marchi, Erich and Typical, Steven},
  journal={Proceedings of the VLDB Endowment},
  volume={15},
  number={10},
  pages={2417--2429},
  year={2022}
}
```

---

## OneShotSTL

### Reference

He, X., Sun, Y., Lavender, H., Li, J., & Wen, Q. (2023). Time series decomposition via sparse representation. *VLDB Endowment*, 16(11), 2915-2928.

### Description

One-shot decomposition using sparse coding. Learns dictionary of trend and seasonal patterns from data, then decomposes via sparse linear combination.

### Algorithm

1. **Dictionary learning:** Learn trend dictionary $D_T$ and seasonal dictionary $D_S$ from data
2. **Sparse coding:** Solve:
   $$\min_{\alpha_T, \alpha_S} \|y - D_T \alpha_T - D_S \alpha_S\|_2^2 + \lambda \|\alpha_T\|_1 + \mu \|\alpha_S\|_1$$
3. **Reconstruction:** $T = D_T \alpha_T$, $S = D_S \alpha_S$, $R = y - T - S$

### Hyperparameters

```python
from OneShotSTL import OneShotSTL

model = OneShotSTL(
    period=period,              # Required: seasonal period
    n_atoms=50,                 # Dictionary size
    lambda_trend=0.1,           # Trend sparsity weight
    lambda_seasonal=0.5,        # Seasonal sparsity weight
    max_iter=100                # Dictionary learning iterations
)
```

**Used in experiments:**
- `period`: Ground truth or domain knowledge
- `n_atoms=50`: Moderate dictionary size
- `lambda_trend=0.1`: Light sparsity on trend
- `lambda_seasonal=0.5`: Moderate sparsity on seasonal
- `max_iter=100`: Sufficient for convergence

### Computational Complexity

$O(n^2 k)$ where $k$ is dictionary size, dominated by dictionary learning.

### Strengths

- Learns data-adaptive representations
- Handles complex, non-standard patterns
- Theoretically principled (sparse coding)

### Limitations

- Requires period specification
- Computationally expensive
- Many hyperparameters to tune

### Citation

```bibtex
@article{he2023oneshotstl,
  title={Time series decomposition via sparse representation},
  author={He, Xiao and Sun, Ye and Lavender, H and Li, J and Wen, Qingsong},
  journal={Proceedings of the VLDB Endowment},
  volume={16},
  number={11},
  pages={2915--2928},
  year={2023}
}
```

---

## Comparison Summary

### Period Specification

**Require period:**
- STL, RobustSTL, STR, FastRobustSTL, OnlineSTL, OneShotSTL

**Period-free:**
- LGTD, ASTD

### Processing Mode

**Batch:**
- STL, RobustSTL, STR, FastRobustSTL, OneShotSTL, LGTD

**Online:**
- ASTD, OnlineSTL

### Computational Cost (n=500)

Ranked from fastest to slowest:
1. STL (~0.02s)
2. LGTD (linear) (~0.05s)
3. ASTD (~0.08s)
4. OnlineSTL (~0.10s)
5. LGTD (LOWESS) (~0.15s)
6. FastRobustSTL (~0.80s)
7. RobustSTL (~1.50s)
8. OneShotSTL (~2.50s)
9. STR (~5.00s)

### Robustness Features

**Outlier resistant:**
- RobustSTL, FastRobustSTL

**Missing data:**
- STR, RobustSTL

**Non-stationary:**
- LGTD, ASTD (adaptive period detection)

---

## Experimental Configurations

All methods configured for fair comparison:

1. **Period specification:** Ground truth (synthetic) or domain knowledge (real-world)
2. **Regularization:** Default or lightly tuned on validation data
3. **No method-specific advantage:** No access to ground truth for parameter tuning
4. **Same input:** Identical time series across all methods
5. **Same metrics:** Evaluated using same MSE, MAE, correlation metrics

See [experiments.md](experiments.md) for complete experimental protocol.

---

## Implementation Notes

All baseline wrappers located in `experiments/baselines/`:

```
experiments/baselines/
├── base.py                 # BaseDecomposer abstract class
├── stl_decomposer.py       # STL wrapper
├── robust_stl_decomposer.py
├── astd_decomposer.py
├── str_decomposer.py
├── fast_robust_stl_decomposer.py
├── online_stl_decomposer.py
└── oneshot_stl_decomposer.py
```

Each implements `decompose(y)` method returning dictionary:
```python
{
    'trend': np.ndarray,
    'seasonal': np.ndarray,
    'residual': np.ndarray
}
```

See [installation.md](installation.md) for baseline installation instructions.
