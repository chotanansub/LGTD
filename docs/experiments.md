# Reproducing Paper Experiments

This guide provides complete instructions for reproducing all experimental results reported in the LGTD paper.

## Overview

The experiment framework evaluates LGTD against seven baseline methods on synthetic and real-world datasets. All experiments use standardized metrics and configurations to ensure fair comparison.

## Prerequisites

Install full experiment framework:

```bash
pip install -e ".[experiments]"
```

Install baseline methods (see [installation.md](installation.md) for details):
- STL (via statsmodels)
- RobustSTL
- ASTD
- STR
- FastRobustSTL
- OnlineSTL
- OneShotSTL

## Experiment Structure

```
experiments/
├── configs/                        # YAML configuration files
│   ├── synthetic_experiments.yaml  # Synthetic dataset configs
│   └── realworld_experiments.yaml  # Real-world dataset configs
├── scripts/                        # Execution scripts
│   ├── run_synthetic_experiments.py
│   ├── run_realworld_experiments.py
│   ├── generate_tables.py
│   └── generate_figures.py
└── results/                        # Output directory
    ├── synthetic/                  # Synthetic results
    │   ├── decompositions/         # Saved decomposition results
    │   └── metrics/                # Computed metrics
    ├── realworld/                  # Real-world results
    └── tables/                     # Generated tables
    └── figures/                    # Generated figures
```

## Synthetic Experiments

Synthetic experiments use generated time series with known ground truth components.

### Dataset Configurations

Eight synthetic datasets (`synth1` through `synth8`) with varying characteristics:

| Dataset | Trend Type | Period Type | Length | Noise Level |
|---------|-----------|-------------|--------|-------------|
| synth1 | Linear | Fixed (24) | 500 | Low (σ=1.0) |
| synth2 | Linear | Multiple (12, 24) | 500 | Low (σ=1.0) |
| synth3 | Polynomial | Fixed (24) | 500 | Low (σ=1.0) |
| synth4 | Exponential | Fixed (24) | 500 | Low (σ=1.0) |
| synth5 | Linear | Fixed (24) | 500 | High (σ=5.0) |
| synth6 | Linear | Variable | 500 | Low (σ=1.0) |
| synth7 | None | Fixed (24) | 500 | Low (σ=1.0) |
| synth8 | Linear | Fixed (24) | 1000 | Low (σ=1.0) |

### Running Experiments

#### All Datasets and Methods

Generate complete results for Tables 1-3 and Figures 2-4:

```bash
python experiments/scripts/run_synthetic_experiments.py
```

**Expected runtime:** 15-30 minutes depending on hardware and available baseline methods.

#### Specific Datasets

Run experiments on subset of datasets:

```bash
python experiments/scripts/run_synthetic_experiments.py --datasets synth1 synth2 synth3
```

#### Specific Methods

Compare selected methods only:

```bash
python experiments/scripts/run_synthetic_experiments.py --models LGTD STL RobustSTL
```

Available method names:
- `LGTD` - Our method
- `LGTD_Linear` - LGTD with forced linear trend
- `LGTD_LOWESS` - LGTD with forced LOWESS trend
- `STL` - Cleveland et al. (1990)
- `RobustSTL` - Wen et al. (2019)
- `ASTD` - Phungtua-eng & Yamamoto (2024)
- `ASTD_Online` - ASTD with online mode
- `STR` - Dokumentov & Hyndman (2022)
- `FastRobustSTL` - Wen et al. (2020)
- `OnlineSTL` - Mishra et al. (2022)
- `OneShotSTL` - He et al. (2023)

#### Custom Configuration

Modify experiment parameters:

```bash
python experiments/scripts/run_synthetic_experiments.py \
    --config experiments/configs/custom_config.yaml \
    --save_results \
    --verbose
```

### Output Files

Decomposition results saved to:
```
experiments/results/synthetic/decompositions/{dataset}/{method}.npz
```

Each `.npz` file contains:
- `trend`: Trend component
- `seasonal`: Seasonal component
- `residual`: Residual component
- `y`: Reconstructed series
- `execution_time`: Computation time in seconds

Metrics saved to:
```
experiments/results/synthetic/metrics/{dataset}.json
```

Format:
```json
{
  "LGTD": {
    "mse_trend": 0.123,
    "mse_seasonal": 0.456,
    "mse_residual": 0.789,
    "mae_trend": 0.234,
    "correlation_trend": 0.999,
    "execution_time": 1.23
  },
  ...
}
```

---

## Real-World Experiments

Real-world experiments use publicly available datasets without ground truth.

### Datasets

Three real-world datasets:

1. **ETTh1** - Electricity Transformer Temperature (hourly)
   - Length: 17,420 samples
   - Features: Oil temperature (OT variable used)
   - Expected patterns: Daily (24h), weekly (168h)

2. **ETTh2** - Electricity Transformer Temperature (hourly)
   - Length: 17,420 samples
   - Similar characteristics to ETTh1

3. **Sunspot** - Monthly sunspot numbers
   - Length: Variable (historical data)
   - Expected pattern: ~11-year solar cycle (132 months)

### Data Preparation

Download datasets:

```bash
# ETT datasets
wget https://github.com/zhouhaoyi/ETDataset/raw/main/ETT-small/ETTh1.csv -P data/real_world/raw/
wget https://github.com/zhouhaoyi/ETDataset/raw/main/ETT-small/ETTh2.csv -P data/real_world/raw/

# Sunspot data - download from SILSO website
# https://www.sidc.be/SILSO/INFO/snmtotcsv.php
# Place in: data/real_world/raw/sunspot.csv
```

### Running Experiments

Generate results for Table 4 and Figure 5:

```bash
python experiments/scripts/run_realworld_experiments.py
```

Optional arguments:
```bash
python experiments/scripts/run_realworld_experiments.py \
    --datasets ETTh1 Sunspot \
    --models LGTD STL RobustSTL \
    --save_decompositions \
    --verbose
```

### Output Files

Decomposition results:
```
experiments/results/realworld/decompositions/{dataset}/{method}.npz
```

Evaluation metrics (based on reconstruction quality):
```
experiments/results/realworld/metrics/{dataset}.json
```

---

## Generating Tables

Generate all paper tables:

```bash
python experiments/scripts/generate_tables.py
```

### Table Descriptions

**Table 1:** MSE comparison on synthetic datasets (synth1-synth4)
- Rows: Methods
- Columns: Datasets
- Values: Total MSE (trend + seasonal + residual)

**Table 2:** MAE comparison on synthetic datasets (synth1-synth4)
- Similar structure to Table 1 with MAE metric

**Table 3:** Correlation comparison on synthetic datasets
- Shows component-wise Pearson correlation

**Table 4:** Real-world dataset results
- Reconstruction quality metrics (PSNR, MSE)
- Detected period lengths
- Execution time comparison

### Output Format

Tables saved as:
- LaTeX: `experiments/results/tables/table{N}.tex`
- CSV: `experiments/results/tables/table{N}.csv`
- Markdown: `experiments/results/tables/table{N}.md`

Generate specific table:

```bash
python experiments/scripts/generate_tables.py --table 1
```

---

## Generating Figures

Generate all paper figures:

```bash
python experiments/scripts/generate_figures.py
```

### Figure Descriptions

**Figure 2:** Visual comparison of decomposition results on synth1
- Subplots: Original, Trend, Seasonal, Residual
- Methods: LGTD, STL, RobustSTL
- Shows qualitative differences in component extraction

**Figure 3:** Performance heatmap across datasets
- Rows: Methods
- Columns: Datasets
- Color: MSE (lower is better)

**Figure 4:** Parameter sensitivity analysis
- X-axis: Parameter values (window_size, error_percentile)
- Y-axis: Performance metric
- Shows robustness to parameter selection

**Figure 5:** Real-world decomposition results
- Subplots per dataset (ETTh1, ETTh2, Sunspot)
- Comparison of LGTD vs. best baseline
- Highlights period detection capability

### Output Format

Figures saved as:
- PNG: `experiments/results/figures/fig{N}.png` (300 DPI)
- PDF: `experiments/results/figures/fig{N}.pdf` (vector)

Generate specific figure:

```bash
python experiments/scripts/generate_figures.py --figure fig2
```

---

## Baseline Method Configurations

All baseline methods use hyperparameters tuned for fair comparison. See [baselines.md](baselines.md) for details.

### Methods Requiring Period

For STL, RobustSTL, STR, FastRobustSTL, OnlineSTL, OneShotSTL:
- Synthetic experiments: Use ground truth period from dataset configuration
- Real-world experiments: Use domain knowledge (e.g., 24 for hourly ETT, 132 for monthly Sunspot)

### Season-Length-Free Methods

LGTD and ASTD do not require period specification.

---

## Reproducibility

### Random Seeds

All experiments use fixed random seeds for reproducibility:

```python
np.random.seed(42)
random.seed(42)
```

### Software Versions

Tested with:
- Python 3.8.10
- NumPy 1.21.0
- SciPy 1.7.0
- statsmodels 0.13.0

### Hardware

Experiments conducted on:
- CPU: Intel Core i7-9700K (8 cores)
- RAM: 32 GB
- OS: Ubuntu 20.04 LTS

**Note:** Results may vary slightly across platforms due to numerical precision differences in LOWESS and optimization routines.

---

## Troubleshooting

### Missing Baseline Methods

If baseline method unavailable, experiment framework automatically skips that method and prints warning:

```
WARNING: RobustSTL not available, skipping...
```

To include all methods, ensure all baselines installed (see [installation.md](installation.md)).

### Memory Issues

For large datasets or many methods, reduce memory usage:

```bash
python experiments/scripts/run_synthetic_experiments.py \
    --save_results False \
    --datasets synth1  # Run one at a time
```

### Slow Execution

LOWESS trend extraction is $O(n^2)$. For faster experiments:

1. Use LGTD with linear trend only
2. Reduce dataset sizes in configuration
3. Skip computationally expensive baselines (STR, OneShotSTL)

### Different Results

Minor differences (<1% relative error) expected due to:
- Platform-specific numerical libraries
- Random initialization in some baseline methods
- LOWESS implementation variations

For exact reproduction, use identical software versions and random seeds.

---

## Custom Experiments

### Adding New Datasets

1. Create dataset configuration in `experiments/configs/synthetic_experiments.yaml`:

```yaml
synth_custom:
  type: synthetic
  n_samples: 600
  trend_type: polynomial
  trend_params:
    coefficients: [0, 1, 0.01]
  period_type: multiple
  periods: [12, 24, 168]
  amplitudes: [5, 3, 2]
  noise_level: 2.0
```

2. Run experiments:

```bash
python experiments/scripts/run_synthetic_experiments.py --datasets synth_custom
```

### Adding New Methods

1. Implement decomposer class in `experiments/baselines/`:

```python
from experiments.baselines.base import BaseDecomposer

class CustomDecomposer(BaseDecomposer):
    def __init__(self, period=None, **kwargs):
        self.period = period

    def decompose(self, y):
        # Implement decomposition logic
        return {
            'trend': trend,
            'seasonal': seasonal,
            'residual': residual
        }
```

2. Register in experiment runner
3. Run experiments including new method

---

## Performance Benchmarks

Expected execution times (synth1 dataset, n=500):

| Method | Time (seconds) | Complexity |
|--------|---------------|------------|
| LGTD (linear) | 0.05 | $O(n)$ |
| LGTD (LOWESS) | 0.15 | $O(n^2)$ |
| STL | 0.02 | $O(n \log n)$ |
| RobustSTL | 1.50 | $O(n \log n)$ + iterations |
| ASTD | 0.08 | $O(n)$ |
| FastRobustSTL | 0.80 | $O(n \log n)$ |
| OnlineSTL | 0.10 | $O(n)$ |
| OneShotSTL | 2.50 | $O(n^2)$ |
| STR | 5.00 | $O(n^3)$ |

**Note:** Times measured on reference hardware. Actual performance varies by system.

---

## Citation

When using this experimental framework, cite both the LGTD paper and relevant baseline papers (see [baselines.md](baselines.md) for citations).
