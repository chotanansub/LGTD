# Regenerating Decomposition Results

## Overview

This guide explains how to regenerate all decomposition results for the synthetic datasets.

## Quick Start

### Regenerate All Results

```bash
# From project root directory
PYTHONPATH=$PWD:$PYTHONPATH python experiments/runners/experiment_runner.py
```

This will:
- Run all enabled models on all 9 synthetic datasets (synth1-synth9)
- Save decomposition results to `results/synthetic/decompositions/`
- Update `results/synthetic/experiment_results.csv`
- Generate plots for each dataset/model combination in `results/synthetic/plots/`

### Regenerate Specific Datasets

```bash
python experiments/runners/experiment_runner.py --datasets synth1 synth2 synth3
```

### Regenerate Specific Models

```bash
python experiments/runners/experiment_runner.py --models LGTD STL ASTD
```

### Regenerate Comparison Plots

After regenerating decomposition results, create method comparison plots:

```bash
# Generate all 9 comparison plots
python scripts/generate_method_comparison_plots.py

# Generate specific datasets
python scripts/generate_method_comparison_plots.py --datasets synth7 synth8 synth9

# Generate with specific models only
python scripts/generate_method_comparison_plots.py --models LGTD STL OnlineSTL
```

## Directory Structure

```
results/synthetic/
├── decompositions/          # Individual decomposition results (currently regenerating)
├── plots/                   # Individual model plots
│   ├── synth1/
│   │   ├── LGTD_decomposition.png
│   │   ├── STL_decomposition.png
│   │   └── ...
│   └── ...
├── comparison_plots/        # Method comparison plots (one per dataset)
│   ├── synth1_comparison.png
│   ├── synth2_comparison.png
│   └── ...
└── experiment_results.csv   # Consolidated metrics table
```

## What Gets Generated

### 1. Decomposition Results
Location: `results/synthetic/decompositions/`

Each model generates decomposition files (if the experiment runner is configured to save them).

### 2. Individual Plots
Location: `results/synthetic/plots/{dataset}/`

For each dataset and model:
- Original signal
- Trend component (with ground truth)
- Seasonal component (with ground truth)
- Residual component (with ground truth)
- Gray shaded region shows initialization period (30% of data)

### 3. Comparison Plots
Location: `results/synthetic/comparison_plots/`

One plot per dataset showing all methods:
- Rows: Different models (LGTD, STL, ASTD, OnlineSTL, OneShotSTL, etc.)
- Columns: Components (Original, Trend, Seasonal, Residual)
- Each subplot compares model output vs. ground truth

### 4. Results CSV
Location: `results/synthetic/experiment_results.csv`

Consolidated table with:
- Dataset name
- Model name
- MSE, MAE, RMSE for Trend, Seasonal, and Residual
- Correlation and PSNR metrics
- Execution time

## Models Included

### Batch Models
- **LGTD**: Local-Global Trend Decomposition (auto/linear/LOWESS)
- **STL**: Seasonal-Trend decomposition using LOESS
- **STR**: Seasonal-Trend using Regression
- **ASTD**: Adaptive STL (batch mode)
- **FastRobustSTL**: Fast implementation of Robust STL

### Online Models
- **ASTD_Online**: Adaptive STL (online mode)
- **OnlineSTL**: Online Seasonal-Trend decomposition
- **OneShotSTL**: One-shot learning STL

## Configuration

Each dataset has a configuration file in `experiments/configs/dataset_params/`:
- `synth1_params.json` through `synth9_params.json`

Current initialization ratio for online models: **0.3 (30%)**

## Verification

After regeneration, verify results:

```bash
# Check number of result rows (should be 9 datasets × N enabled models)
wc -l results/synthetic/experiment_results.csv

# Check plots were generated
ls -lh results/synthetic/plots/synth1/*.png

# Check comparison plots
ls -lh results/synthetic/comparison_plots/*.png
```

## Current Status

**Regeneration in progress**: Running full experiments to regenerate all decomposition results.

The experiment runner is currently processing all 9 datasets with all enabled models. This may take several minutes depending on:
- Number of enabled models per dataset
- Complexity of each model
- System resources

### Expected Output Files

After completion, you should see:
- ✅ 6 comparison plots already generated (synth1-synth6)
- 🔄 3 comparison plots pending (synth7-synth9)
- 📊 Updated experiment_results.csv
- 📈 Individual plots in `results/synthetic/plots/`

---

**Last Updated**: 2025-12-30

**Status**: Experiments running in background
