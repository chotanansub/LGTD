# LGTD: Local–Global Trend Decomposition for Season-Length–Free Time Series Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-118%20passed-brightgreen.svg)]()

Official implementation of **LGTD** (Local-Global Trend Decomposition), a season-length-free time series decomposition method.

## Overview

LGTD addresses the limitation of traditional decomposition methods (STL, X-11, MSTL) that require prior knowledge of seasonal period lengths. The method combines global trend extraction with local trend analysis to automatically identify and extract seasonal patterns without period specification.

**Key Features:**
- No prior period specification required
- Automatic trend model selection (linear/LOWESS)
- Handles multiple periodicities
- Robust to noise and irregular patterns

## Repository Contents

### 1. LGTD Module

Core implementation of the LGTD decomposition algorithm.

**Structure:**
```
LGTD/
├── decomposition/       # Core decomposition algorithms
│   ├── lgtd.py         # Main LGTD implementation
│   ├── local_trend.py  # Local trend analysis
│   └── seasonal.py     # Seasonal extraction
└── evaluation/         # Metrics and visualization
    ├── metrics.py      # MSE, MAE, correlation, PSNR
    └── visualization.py
```

**Documentation:**
- [Installation Guide](docs/installation.md) - Setup instructions
- [API Reference](docs/api_reference.md) - Complete API documentation
- [Algorithm Description](docs/algorithm.md) - Mathematical formulation
- [Parameter Guide](docs/parameters.md) - Hyperparameter tuning

### 2. Experiment Framework

Complete experimental setup to reproduce all tables and figures from the paper.

**Structure:**
```
experiments/
├── baselines/          # Seven baseline method implementations
├── configs/            # Dataset and experiment configurations
├── scripts/            # Execution scripts for experiments
└── results/            # Output directory for tables and figures

data/
├── generators/         # Synthetic data generation
├── synthetic/          # Eight synthetic datasets (synth1-synth8)
└── real_world/         # ETTh1, ETTh2, Sunspot datasets

tests/                  # Unit tests (118 tests)
```

**Documentation:**
- [Experiment Guide](docs/experiments.md) - Reproducing paper results
- [Dataset Descriptions](docs/datasets.md) - Synthetic and real-world datasets
- [Baseline Methods](docs/baselines.md) - Seven comparison methods
- [Evaluation Metrics](docs/metrics.md) - MSE, MAE, correlation, PSNR

## Quick Start

### Installation

```bash
# Core module only
pip install -e .

# With experiment framework
pip install -e ".[experiments]"
```

See [docs/installation.md](docs/installation.md) for baseline method installation.

### Basic Usage

```python
from LGTD import LGTD
import numpy as np

# Load time series data
y = np.random.randn(500) + np.sin(np.linspace(0, 20*np.pi, 500))

# Decompose without specifying period
model = LGTD()
result = model.fit_transform(y)

# Access components
trend = result.trend
seasonal = result.seasonal
residual = result.residual
print(f"Detected periods: {result.detected_periods}")
```

For detailed usage examples, see [docs/api_reference.md](docs/api_reference.md).

## Reproducing Paper Results

### Run All Experiments

```bash
# Synthetic experiments (Tables 1-3, Figures 2-4)
python experiments/scripts/run_synthetic_experiments.py

# Real-world experiments (Table 4, Figure 5)
python experiments/scripts/run_realworld_experiments.py

# Generate tables and figures
python experiments/scripts/generate_tables.py
python experiments/scripts/generate_figures.py
```

### Datasets

**Synthetic (8 datasets):**
- synth1-synth8: Controlled experiments with known ground truth
- Vary trend type, period structure, noise level, series length

**Real-world (3 datasets):**
- ETTh1/ETTh2: Electricity Transformer Temperature (hourly)
- Sunspot: Monthly sunspot numbers (SILSO)

See [docs/datasets.md](docs/datasets.md) and [data/real_world/DATASETS.md](data/real_world/DATASETS.md) for complete descriptions.

### Baseline Methods

Comparison against seven state-of-the-art methods:

| Method | Year | Period Required | Type | Reference |
|--------|------|-----------------|------|-----------|
| STL (Cleveland et al.) | 1990 | Yes | Batch | [statsmodels](https://www.statsmodels.org/) |
| RobustSTL (Wen et al.) | 2019 | Yes | Batch | [LeeDoYup](https://github.com/LeeDoYup/RobustSTL) (GitHub) |
| FastRobustSTL (Wen et al.) | 2020 | Yes | Batch | [ariaghora](https://github.com/ariaghora/fast-robust-stl) (GitHub) |
| OnlineSTL (Mishra et al.) | 2022 | Yes | Online | [YHYHYHYHYHY](https://github.com/YHYHYHYHYHY/OnlineSTL) (GitHub) |
| STR (Dokumentov & Hyndman) | 2022 | Yes | Batch | Modified from [robjhyndman](https://github.com/robjhyndman/STR_paper) (GitHub) |
| OneShotSTL (He et al.) | 2023 | Yes | Batch | [xiao-he](https://github.com/xiao-he/OneShotSTL) (GitHub) |
| ASTD (Phungtua-eng & Yamamoto) | 2024 | No | Online | [thanapol2](https://github.com/thanapol2/ASTD_ECMLPKDD) (GitHub) |

See [docs/baselines.md](docs/baselines.md) for implementation details and citations.

## Method Overview

LGTD decomposes time series $y_t$ into trend, seasonal, and residual components:

$$y_t = T_t + S_t + R_t$$

**Algorithm:**
1. **Global Trend Extraction** - Linear regression or LOWESS (automatic selection)
2. **Detrending** - Compute detrended series $d_t = y_t - T_t$
3. **Local Trend Analysis** - Sliding-window detection of local linear segments
4. **Seasonal Extraction** - Aggregate local deviations to form seasonal pattern
5. **Residual Computation** - Calculate $R_t = y_t - T_t - S_t$

See [docs/algorithm.md](docs/algorithm.md) for mathematical formulation.

## Documentation

**LGTD Module:**
- [Installation](docs/installation.md) - Core module and dependencies
- [API Reference](docs/api_reference.md) - Classes, methods, functions
- [Algorithm](docs/algorithm.md) - Mathematical description
- [Parameters](docs/parameters.md) - Hyperparameter guide

**Experiments:**
- [Experiments](docs/experiments.md) - Reproducing paper results
- [Datasets](docs/datasets.md) - Dataset descriptions
- [Baselines](docs/baselines.md) - Comparison methods
- [Metrics](docs/metrics.md) - Evaluation metrics

**Entry Point:** [docs/README.md](docs/README.md)

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window_size` | 3 | Local trend sliding window size |
| `error_percentile` | 50 | AutoTrend error threshold percentile |

See [docs/parameters.md](docs/parameters.md) for tuning guidance.

## Citation

This repository accompanies a paper currently under review. Full citation will be provided upon publication.

**Datasets:**
- ETT: Zhou et al., Informer (AAAI 2021)
- Sunspot: SILSO, Royal Observatory of Belgium

See [data/real_world/DATASETS.md](data/real_world/DATASETS.md) for complete citations.

## Contact

For questions or issues, please open a GitHub issue.
