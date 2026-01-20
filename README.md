# LGTD: Local–Global Trend Decomposition for Season-Length–Free Time Series Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/lgtd.svg)](https://pypi.org/project/lgtd/)
[![GitHub Repo](https://img.shields.io/badge/GitHub-LGTD-black?logo=github)](https://github.com/chotanansub/LGTD)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1hOH2Be9w15yEvIA6YZXYSEbWZVI3ZoEN?usp=sharing)
[![arXiv](https://img.shields.io/badge/arXiv-2601.04820-b31b1b.svg)](https://arxiv.org/abs/2601.04820)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Season-length-free time series decomposition.**
LGTD (Local–Global Trend Decomposition) is a principled method for decomposing a univariate time series into **trend**, **seasonal**, and **residual** components *without requiring prior specification of seasonal periods*. In contrast to classical decomposition techniques—such as STL, X-11, and MSTL—which assume fixed or user-specified seasonal lengths, LGTD automatically identifies and aggregates repeating structures of arbitrary and potentially time-varying scales.

![](assets/pipeline.png)
> *Figure 1. LGTD global–local decomposition pipeline.*

---

## Motivation

Many real-world time series exhibit seasonality that is non-stationary, intermittent, or composed of multiple overlapping cycles. Period-dependent methods are brittle in such settings, as incorrect or misspecified periods can substantially degrade decomposition quality. LGTD is designed to address this limitation by eliminating the need for explicit period selection.

```python
# STL requires an explicit seasonal period
from statsmodels.tsa.seasonal import STL
stl = STL(y, seasonal=24)  # ❌ requires prior knowledge of the period
result = stl.fit()

# LGTD does not require period specification
from lgtd import lgtd
model = lgtd()
result = model.fit_transform(y)  # ✅ accommodates variable and unknown periods
```

---

## Installation

```bash
pip install lgtd
```

---

## Quick Start

```python
from lgtd import lgtd
import numpy as np

# Synthetic time series with trend and seasonality
t = np.arange(500)
y = 0.05 * t + 10 * np.sin(2 * np.pi * t / 24) + np.random.normal(0, 1, 500)

# Decomposition
model = lgtd()
result = model.fit_transform(y)

# Decomposed components
print("Detected periods:", result.detected_periods)
print("Trend:", result.trend)
print("Seasonal:", result.seasonal)
print("Residual:", result.residual)
```

---

## Key Properties

* **Season-length-free** — No assumption of known or fixed seasonal periods
* **Multiple and time-varying periodicities** — Supports overlapping and drifting cycles
* **Adaptive trend modeling** — Automatically selects between linear and LOWESS trends
* **Noise robustness** — Stable performance under irregular and noisy observations
* **Minimal interface** — Simple, scikit-learn–style API

---

## Parameters

```python
model = lgtd(
    window_size=3,            # Local window size for pattern extraction
    error_percentile=50,      # Threshold for seasonal pattern aggregation
    trend_selection='auto',   # {'auto', 'linear', 'lowess'}
    lowess_frac=0.1,          # LOWESS smoothing fraction
    threshold_r2=0.9          # R² threshold for trend selection
)
```

Refer to the **[Parameter Guide](docs/parameters.md)** for detailed tuning guidelines.

---

## Visualization

```python
from lgtd.evaluation.visualization import plot_decomposition

result = model.fit_transform(y)
plot_decomposition(result, title="LGTD Decomposition")
```

---

## Evaluation Metrics

```python
from lgtd.evaluation.metrics import compute_mse, compute_mae

mse = compute_mse(ground_truth, {
    'trend': result.trend,
    'seasonal': result.seasonal,
    'residual': result.residual
})
```

---

## Reproducibility

This repository contains a complete experimental pipeline for reproducing the results reported in the accompanying paper. LGTD is evaluated against seven baseline decomposition methods on both synthetic and real-world datasets.

```bash
python experiments/scripts/run_synthetic_experiments.py
python experiments/scripts/run_realworld_experiments.py

python experiments/scripts/generate_tables.py
python experiments/scripts/generate_figures.py
```

See the **[Experiments Guide](docs/experiments.md)** for full details.

---

## Documentation

* **[API Reference](docs/api_reference.md)** — Complete API specification
* **[Algorithm Details](docs/algorithm.md)** — Mathematical formulation
* **[Parameter Guide](docs/parameters.md)** — Hyperparameter analysis
* **[Experiments](docs/experiments.md)** — Reproducibility instructions
* **[Datasets](docs/datasets.md)** — Dataset descriptions
* **[Baselines](docs/baselines.md)** — Comparative methods

---

## Method Overview

LGTD decomposes a time series ( y_t ) as

[
y_t = T_t + S_t + R_t,
]

where ( T_t ) denotes the global trend, ( S_t ) the aggregated seasonal structure, and ( R_t ) the residual component.

**High-level procedure:**

1. Estimate a global trend (linear or LOWESS, selected adaptively)
2. Remove the estimated trend
3. Identify local linear patterns using sliding windows
4. Aggregate consistent patterns into a seasonal component
5. Compute residuals

See **[Algorithm Documentation](docs/algorithm.md)** for formal definitions and analysis.

---

## Citation

```bibtex
@article{lgtd2026arxiv,
  title        = {LGTD: Local--Global Trend Decomposition for Season-Length-Free Time Series Analysis},
  author       = {Sophaken, Chotanan and Rattanakornphan, Thanadej and Charoenpoonpanich, Piyanon and Phungtua-eng, Thanapol and Amornbunchornvej, Chainarong},
  journal      = {arXiv preprint},
  year         = {2026},
  eprint       = {2601.04820},
  archivePrefix= {arXiv},
  primaryClass = {cs.DB}
}
```

---

## License

Released under the MIT License. See **[LICENSE](LICENSE)** for details.

---

## Links

* **GitHub**: [https://github.com/chotanansub/LGTD](https://github.com/chotanansub/LGTD)
* **PyPI**: [https://pypi.org/project/lgtd/](https://pypi.org/project/lgtd/)
* **Issues**: [https://github.com/chotanansub/LGTD/issues](https://github.com/chotanansub/LGTD/issues)
* **Documentation**: docs/
