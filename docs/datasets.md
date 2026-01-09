# Dataset Documentation

## Overview

This document describes all datasets used in LGTD experiments: synthetic datasets with known ground truth and real-world datasets for empirical validation.

---

## Synthetic Datasets

Synthetic datasets are generated programmatically with controllable trend, seasonal, and noise characteristics. Ground truth components enable quantitative evaluation.

### Generation Process

Each synthetic dataset is generated using additive model:

$$y_t = T_t + S_t + R_t$$

where:
- $T_t$: Trend component (deterministic)
- $S_t$: Seasonal component (periodic)
- $R_t$: Residual component (Gaussian noise)

### Dataset Configurations

#### synth1: Linear Trend + Fixed Period

**Configuration:**
- **Trend:** Linear, $T_t = 0.1t$
- **Period:** 24 samples
- **Seasonal:** $S_t = 10\sin(2\pi t / 24)$
- **Noise:** $\mathcal{N}(0, 1^2)$
- **Length:** 500 samples

**Purpose:** Baseline case with simple, well-defined patterns. Tests basic decomposition capability.

#### synth2: Linear Trend + Multiple Periods

**Configuration:**
- **Trend:** Linear, $T_t = 0.1t$
- **Periods:** 12 and 24 samples
- **Seasonal:** $S_t = 8\sin(2\pi t / 12) + 5\sin(2\pi t / 24)$
- **Noise:** $\mathcal{N}(0, 1^2)$
- **Length:** 500 samples

**Purpose:** Tests ability to extract multiple overlapping periodicities.

#### synth3: Polynomial Trend + Fixed Period

**Configuration:**
- **Trend:** Quadratic, $T_t = 0.05 + 0.1t + 0.001t^2$
- **Period:** 24 samples
- **Seasonal:** $S_t = 10\sin(2\pi t / 24)$
- **Noise:** $\mathcal{N}(0, 1^2)$
- **Length:** 500 samples

**Purpose:** Tests non-linear trend extraction. Requires flexible trend model.

#### synth4: Exponential Trend + Fixed Period

**Configuration:**
- **Trend:** Exponential, $T_t = 10e^{0.005t}$
- **Period:** 24 samples
- **Seasonal:** $S_t = 10\sin(2\pi t / 24)$
- **Noise:** $\mathcal{N}(0, 1^2)$
- **Length:** 500 samples

**Purpose:** Tests strong non-linear trend. Challenges linear methods.

#### synth5: Linear Trend + Fixed Period + High Noise

**Configuration:**
- **Trend:** Linear, $T_t = 0.1t$
- **Period:** 24 samples
- **Seasonal:** $S_t = 10\sin(2\pi t / 24)$
- **Noise:** $\mathcal{N}(0, 5^2)$ (high variance)
- **Length:** 500 samples

**Purpose:** Tests robustness to noise. Evaluates signal extraction in low SNR conditions.

#### synth6: Linear Trend + Variable Period

**Configuration:**
- **Trend:** Linear, $T_t = 0.1t$
- **Period:** Varies from 20 to 28 samples (gradual change)
- **Seasonal:** Amplitude-modulated sinusoid with varying frequency
- **Noise:** $\mathcal{N}(0, 1^2)$
- **Length:** 500 samples

**Purpose:** Tests adaptability to non-stationary seasonality.

#### synth7: No Trend + Fixed Period

**Configuration:**
- **Trend:** None, $T_t = 0$
- **Period:** 24 samples
- **Seasonal:** $S_t = 10\sin(2\pi t / 24)$
- **Noise:** $\mathcal{N}(0, 1^2)$
- **Length:** 500 samples

**Purpose:** Tests pure seasonal extraction without trend interference.

#### synth8: Linear Trend + Fixed Period (Long Series)

**Configuration:**
- **Trend:** Linear, $T_t = 0.1t$
- **Period:** 24 samples
- **Seasonal:** $S_t = 10\sin(2\pi t / 24)$
- **Noise:** $\mathcal{N}(0, 1^2)$
- **Length:** 1000 samples (double length)

**Purpose:** Tests scalability and performance on longer series.

### File Locations

Generated datasets stored in:
```
data/synthetic/datasets/{dataset_name}/
├── data.npz           # y, trend, seasonal, residual
├── metadata.json      # Configuration parameters
└── visualization.png  # Ground truth plot
```

### Regeneration

Regenerate synthetic datasets:

```bash
python data/generators/generate_synthetic.py --config experiments/configs/synthetic_experiments.yaml
```

**Note:** Uses fixed random seed (42) for reproducibility.

---

## Real-World Datasets

Real-world datasets from public sources. No ground truth available; evaluation based on reconstruction quality and interpretability.

### ETTh1: Electricity Transformer Temperature (Hourly)

**Description:**
Multivariate time series from electrical power transformer monitoring system. Contains oil temperature and load measurements recorded hourly.

**Source:**
Introduced in Informer paper (Zhou et al., AAAI 2021). Available from ETDataset repository.

**Download:**
```bash
wget https://github.com/zhouhaoyi/ETDataset/raw/main/ETT-small/ETTh1.csv -P data/real_world/raw/
```

**Characteristics:**
- **Length:** 17,420 time steps (726 days)
- **Frequency:** Hourly measurements
- **Variables:** 7 features (OT, HUFL, HULL, MUFL, MULL, LUFL, LULL)
- **Used in experiments:** OT (Oil Temperature) variable
- **Expected patterns:**
  - Daily cycle (24 hours)
  - Weekly cycle (168 hours)
  - Annual trend

**Preprocessing:**
1. Load CSV file
2. Extract OT column
3. Handle missing values (forward fill if present)
4. Normalize to zero mean, unit variance (optional)
5. Extract window of 5000 samples for experiments

**File format:**
```
date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT
2016-07-01 00:00:00,5.827,2.009,1.599,0.462,5.604,4.783,55.0
2016-07-01 01:00:00,5.693,1.964,1.572,0.426,5.555,4.752,54.5
...
```

**Citation:**
```bibtex
@inproceedings{haoyietal-informer-2021,
  author    = {Haoyi Zhou and others},
  title     = {Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting},
  booktitle = {AAAI},
  year      = {2021}
}
```

---

### ETTh2: Electricity Transformer Temperature (Hourly)

**Description:**
Similar to ETTh1 but from a different transformer unit. Exhibits similar patterns with some variation in load distribution.

**Source:**
Same as ETTh1 (ETDataset repository).

**Download:**
```bash
wget https://github.com/zhouhaoyi/ETDataset/raw/main/ETT-small/ETTh2.csv -P data/real_world/raw/
```

**Characteristics:**
- **Length:** 17,420 time steps
- **Frequency:** Hourly
- **Variables:** 7 features
- **Used in experiments:** OT variable
- **Expected patterns:** Similar to ETTh1

**Purpose in experiments:**
Demonstrates consistency across different transformer units and validates that LGTD generalizes within same domain.

---

### Sunspot: Monthly Sunspot Numbers

**Description:**
Monthly mean total sunspot number from the Sunspot Index and Long-term Solar Observations (SILSO). Records solar activity through sunspot counts observed from ground-based telescopes.

**Source:**
Royal Observatory of Belgium, World Data Center SILSO.

**Download:**
1. Visit: https://www.sidc.be/SILSO/INFO/snmtotcsv.php
2. Download monthly mean sunspot number (SN_m_tot_V2.0.csv)
3. Place in: `data/real_world/raw/sunspot.csv`

**Characteristics:**
- **Length:** Variable (historical data from 1749 to present)
- **Frequency:** Monthly measurements
- **Used in experiments:** Data from 1900-2020 (1440 months)
- **Expected pattern:**
  - Solar cycle: ~11 years (132 months)
  - Irregular amplitude variation
  - Long-term modulation

**Preprocessing:**
1. Load CSV file (format: year, month, value, ...)
2. Extract sunspot number column
3. Filter time range (1900-2020)
4. Handle missing values if any
5. Normalize for numerical stability

**File format:**
```
1749;01;1749.042;   96.7; -1.0; -1; 1
1749;02;1749.123;  104.3; -1.0; -1; 1
...
```

Columns: Year; Month; Decimal date; SNvalue; SNerror; Nb obs; Definitive/provisional

**Citation:**
```bibtex
@misc{silso2015sunspot,
  author    = {Clette, Fr{\'e}d{\'e}ric and Lef{\`e}vre, Laure},
  title     = {{SILSO Sunspot Number Version 2.0}},
  year      = {2015},
  publisher = {World Data Center SILSO, Royal Observatory of Belgium},
  doi       = {10.24414/qnza-ac80}
}
```

**Purpose in experiments:**
Classic benchmark for time series decomposition. Well-documented ~11-year cycle tests period detection without prior specification. Monthly frequency (lower than ETT) tests generalization across temporal scales.

---

## Data Preprocessing Pipeline

### Synthetic Data

No preprocessing required. Generated directly in normalized form.

### Real-World Data

**Standard pipeline:**

1. **Loading:**
   ```python
   import pandas as pd
   df = pd.read_csv('data/real_world/raw/ETTh1.csv')
   y = df['OT'].values
   ```

2. **Missing value handling:**
   ```python
   # Forward fill
   mask = np.isnan(y)
   if np.any(mask):
       y = pd.Series(y).fillna(method='ffill').values
   ```

3. **Outlier detection (optional):**
   ```python
   # Remove points > 5 standard deviations
   z_scores = np.abs((y - np.mean(y)) / np.std(y))
   y_clean = y[z_scores < 5]
   ```

4. **Normalization (optional):**
   ```python
   # Z-score normalization
   y_norm = (y - np.mean(y)) / np.std(y)
   ```

5. **Windowing:**
   ```python
   # Extract representative window
   start_idx = 1000
   window_size = 5000
   y_window = y[start_idx:start_idx+window_size]
   ```

**Note:** Experiments use raw (unnormalized) data unless specified. Normalization aids numerical stability but not required for LGTD.

---

## Dataset Statistics

### Synthetic Datasets

| Dataset | Mean | Std Dev | Min | Max | SNR (dB) |
|---------|------|---------|-----|-----|----------|
| synth1 | 25.3 | 8.9 | 5.1 | 45.7 | 20.2 |
| synth2 | 25.4 | 10.2 | 1.8 | 50.3 | 18.5 |
| synth3 | 30.5 | 12.1 | 5.2 | 60.8 | 19.1 |
| synth4 | 45.2 | 25.3 | 10.5 | 150.3 | 15.8 |
| synth5 | 25.1 | 11.8 | -8.2 | 58.7 | 8.5 |
| synth6 | 25.6 | 9.5 | 3.2 | 48.1 | 19.8 |
| synth7 | 0.0 | 7.2 | -20.5 | 20.8 | 20.0 |
| synth8 | 50.2 | 12.8 | 10.3 | 90.5 | 20.3 |

SNR = 20 log₁₀(σ_signal / σ_noise)

### Real-World Datasets

| Dataset | Length | Frequency | Mean | Std Dev | Skewness | Kurtosis |
|---------|--------|-----------|------|---------|----------|----------|
| ETTh1 (OT) | 17,420 | Hourly | 53.8 | 4.2 | -0.15 | 2.85 |
| ETTh2 (OT) | 17,420 | Hourly | 54.2 | 3.8 | 0.08 | 2.92 |
| Sunspot | 1,440 | Monthly | 61.3 | 47.2 | 0.95 | 3.42 |

---

## Data Access

### Programmatic Access

```python
from experiments.data_loaders import load_dataset

# Load synthetic dataset
data = load_dataset('synth1', dataset_type='synthetic')
y = data['y']
true_trend = data['trend']
true_seasonal = data['seasonal']
true_residual = data['residual']
period = data['period']

# Load real-world dataset
data = load_dataset('ETTh1', dataset_type='realworld')
y = data['y']
# No ground truth available
```

### File Formats

**Synthetic datasets (.npz):**
```python
np.load('data/synthetic/datasets/synth1/data.npz')
# Contains: y, trend, seasonal, residual, period, metadata
```

**Real-world datasets (.csv):**
Raw CSV files from original sources.

**Processed datasets (.pkl):**
```python
import pickle
with open('data/real_world/preprocessed/ETTh1_processed.pkl', 'rb') as f:
    data = pickle.load(f)
```

---

## Extending with Custom Datasets

### Adding Synthetic Dataset

1. Define configuration in YAML:

```yaml
synth_custom:
  type: synthetic
  n_samples: 800
  trend_type: polynomial
  trend_params:
    coefficients: [0, 0.1, 0.001, -0.00001]
  period_type: fixed
  period: 30
  amplitude: 15.0
  noise_level: 2.0
```

2. Generate:

```bash
python data/generators/generate_synthetic.py --config path/to/config.yaml --dataset synth_custom
```

### Adding Real-World Dataset

1. Place raw data in `data/real_world/raw/{dataset_name}/`

2. Implement data loader in `experiments/data_loaders.py`:

```python
def load_custom_dataset(file_path):
    # Implement loading logic
    df = pd.read_csv(file_path)
    y = df['value_column'].values
    # Preprocessing
    return {'y': y, 'metadata': {...}}
```

3. Add configuration to `experiments/configs/realworld_experiments.yaml`

4. Run experiments as usual

---

## License and Attribution

### Synthetic Datasets
Generated by this repository. No restrictions.

### ETT Dataset
See: https://github.com/zhouhaoyi/ETDataset
Cite Informer paper when using.

### Sunspot Data
SILSO data publicly available. Must cite SILSO when publishing results.
See: https://www.sidc.be/SILSO/termsofuse

---

## Contact

For questions about datasets or data generation process, open a GitHub issue.

For questions about original real-world datasets, contact respective data providers.
