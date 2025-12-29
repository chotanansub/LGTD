# LGTD Experiments

Complete experiment framework for evaluating LGTD and baseline decomposition methods.

---

## Quick Start

### 1. Run All Experiments

```bash
source env/bin/activate
python experiments/scripts/run_experiments.py
```

### 2. Run Specific Tests

```bash
# Test LGTD on linear datasets
python experiments/runners/experiment_runner.py \
    --datasets synth1 synth4 synth7 \
    --models LGTD

# Compare LGTD variants
python experiments/runners/experiment_runner.py \
    --models LGTD LGTD_Linear LGTD_LOWESS

# Run single experiment
python experiments/runners/experiment_runner.py \
    --datasets synth1 \
    --models LGTD

# Run STR on all datasets
python experiments/runners/experiment_runner.py --models STR

# Run all baselines
python experiments/runners/experiment_runner.py \
    --models STL STR FastRobustSTL ASTD
```

### 3. Analyze Results

```bash
# Generate LaTeX tables
python scripts/results_to_latex.py              # Long format (vertical)
python scripts/results_to_latex_wide.py         # Wide format (landscape)

# Analyze results programmatically
python experiments/scripts/analyze_results.py results/synthetic/experiment_results.csv
```

---

## Features

### ✅ JSON-Based Configuration

Each dataset has a dedicated parameter file:
- `experiments/configs/dataset_params/synth1_params.json`
- `experiments/configs/dataset_params/synth2_params.json`
- etc.

### ✅ Automatic Config Synchronization

Generate new datasets and sync configurations automatically:
```bash
python experiments/scripts/sync_dataset_configs.py
```
See [CONFIG_SYNC_GUIDE.md](CONFIG_SYNC_GUIDE.md) for details.

### ✅ Flexible Execution

Run:
- All models on all datasets
- Specific model(s) on all datasets
- All models on specific dataset(s)
- Specific model(s) on specific dataset(s)

### ✅ Comprehensive Metrics

- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- Correlation
- PSNR (Peak Signal-to-Noise Ratio)

### ✅ Automatic Visualization

Plots are automatically saved to `results/synthetic/plots/{dataset}/`

### ✅ Results Tracking

- **Single master CSV**: `results/synthetic/experiment_results.csv`
- Automatically updates existing rows or inserts new results
- Sorted by dataset and model for easy reading
- Historical timestamped CSVs archived in `results/synthetic/records/`

---

## Available Models

| Model | Description | Status |
|-------|-------------|--------|
| **LGTD** | Auto-selection (linear/LOWESS) | ✅ Ready |
| **LGTD_Linear** | Force linear trend | ✅ Ready |
| **LGTD_LOWESS** | Force LOWESS trend | ✅ Ready |
| **STL** | Seasonal-Trend Decomposition (statsmodels) | ✅ Ready |
| **STR** | Seasonal-Trend decomposition using Regression | ✅ Ready |
| **FastRobustSTL** | Fast Robust STL variant (frstl) | ✅ Ready |
| **ASTD** | Adaptive Seasonal-Trend Decomposition | ✅ Ready |
| **RobustSTL** | Robust STL (RobustSTL package) | ✅ Available locally |

---

## Available Datasets

| Dataset | Trend | Period Type | Period | Status |
|---------|-------|-------------|--------|--------|
| synth1 | Linear | Fixed | 120 | ✅ |
| synth2 | Inverted-V | Fixed | 60 | ✅ |
| synth3 | Piecewise | Fixed | 120 | ✅ |
| synth4 | Linear | Transitive | 80→100→120 | ✅ |
| synth5 | Inverted-V | Transitive | 120→100→80 | ✅ |
| synth6 | Piecewise | Transitive | 50→80→110 | ✅ |
| synth7 | Linear | Variable | Variable | ✅ |
| synth8 | Inverted-V | Variable | Variable | ✅ |
| synth9 | Piecewise | Variable | Variable | ✅ |

---

## Directory Structure

```
experiments/
├── README.md                      # This file
├── EXPERIMENT_GUIDE.md            # Detailed guide
├── CONFIG_SYNC_GUIDE.md           # Dataset config sync guide
├── configs/
│   ├── dataset_params/            # JSON parameter files
│   │   ├── synth1_params.json
│   │   ├── synth2_params.json
│   │   └── ...
│   └── synthetic_experiments.yaml # Legacy config
├── runners/
│   ├── experiment_runner.py       # Main experiment engine ⭐
│   ├── synthetic_runner.py        # Legacy runner
│   └── base_experiment.py
├── scripts/
│   ├── run_experiments.py         # CLI entry point
│   ├── analyze_results.py         # Results analysis
│   ├── sync_dataset_configs.py    # Config synchronization
│   └── run_synthetic.py           # Legacy script
├── baselines/
│   ├── STR/                       # STR implementation (built-in)
│   ├── str_decomposer.py          # STR wrapper
│   ├── fast_robust_stl.py         # FastRobustSTL wrapper
│   ├── astd.py                    # ASTD wrapper
│   ├── robust_stl.py              # RobustSTL wrapper
│   ├── RobustSTL/                 # RobustSTL (cloned, not tracked)
│   └── ASTD/                      # ASTD (cloned, not tracked)
└── utils/
    └── reproducibility.py         # Reproducibility utilities
```

---

## Usage Examples

### Example 1: Validate LGTD Auto-Selection

```bash
# Run LGTD on all datasets
python experiments/runners/experiment_runner.py --models LGTD

# Check results
python -c "
import pandas as pd
df = pd.read_csv('results/synthetic/experiment_results.csv')
print(df[['dataset', 'trend_type', 'selected_method', 'mse_trend']])
"
```

Expected output: LGTD should select 'linear' for synth1, synth4, synth7

### Example 2: Compare LGTD vs Baselines

```bash
# Run comparison
python experiments/runners/experiment_runner.py --models LGTD STL STR ASTD

# Generate LaTeX table
python scripts/results_to_latex_wide.py

# Analyze
python experiments/scripts/analyze_results.py results/synthetic/experiment_results.csv
```

### Example 3: Test on Linear Datasets Only

```bash
python experiments/runners/experiment_runner.py \
    --datasets synth1 synth4 synth7 \
    --models LGTD LGTD_Linear
```

### Example 4: Full Experiment Suite

```bash
# Run all models on all datasets
python experiments/runners/experiment_runner.py \
    --models LGTD STL STR FastRobustSTL ASTD

# This will:
# - Test 5 models × 9 datasets = 45 experiments
# - Save 45 plots
# - Update experiment_results.csv with all results
# - Automatic reproducibility with GLOBAL_SEED=69
```

### Example 5: Run STR on All Datasets

```bash
# Run STR experiments
python experiments/runners/experiment_runner.py --models STR

# Generate wide-format table
python scripts/results_to_latex_wide.py
```

---

## Modifying Parameters

### Enable/Disable Models

Edit `experiments/configs/dataset_params/synth1_params.json`:

```json
{
  "models": {
    "LGTD": {
      "enabled": true,  // Set to false to skip
      "params": {...}
    }
  }
}
```

### Change Model Parameters

```json
{
  "models": {
    "LGTD": {
      "enabled": true,
      "params": {
        "window_size": 5,        // Changed from default 3
        "threshold_r2": 0.95,    // Changed from default 0.92
        "trend_selection": "auto"
      }
    }
  }
}
```

### Customize Evaluation

```json
{
  "evaluation": {
    "metrics": ["mse", "mae"],  // Only compute these metrics
    "save_plots": false,        // Disable plotting
    "plot_dir": "custom/path"   // Custom plot directory
  }
}
```

---

## Output Files

### Results CSV

Location: `results/synthetic/experiment_results.csv`

Columns:
- `dataset`: Dataset name
- `trend_type`: linear, inverted_v, or piecewise
- `period_type`: fixed, transitive, or variable
- `model`: Model name
- `time`: Execution time (seconds)
- `mse_trend`, `mse_seasonal`, `mse_residual`: MSE for each component
- `mae_trend`, `mae_seasonal`, `mae_residual`: MAE for each component
- `rmse_trend`, etc.: RMSE for each component
- `corr_trend`, etc.: Correlation for each component
- `psnr_trend`, etc.: PSNR for each component
- `selected_method`: For LGTD, shows 'linear' or 'lowess'

### Plots

Location: `results/synthetic/plots/{dataset}/{model}_decomposition.png`

Each plot shows:
- Original data
- Ground truth vs predicted components (trend, seasonal, residual)
- Decomposition quality visualization

---

## Advanced Usage

### Programmatic Access

```python
from experiments.runners.experiment_runner import ExperimentRunner

# Create runner
runner = ExperimentRunner()

# Run experiments
results = runner.run_experiment(
    datasets=['synth1', 'synth2'],
    models=['LGTD', 'STL'],
    save_results=True,
    verbose=True
)

# Access DataFrame
print(results.head())
print(results.groupby('model')['mse_trend'].mean())
```

### Batch Parameter Testing

```python
import json
from pathlib import Path
from experiments.runners.experiment_runner import ExperimentRunner

# Test different thresholds
thresholds = [0.85, 0.90, 0.92, 0.95]
runner = ExperimentRunner()

for thresh in thresholds:
    # Modify config
    config_path = Path('experiments/configs/dataset_params/synth1_params.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    config['models']['LGTD']['params']['threshold_r2'] = thresh

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Run
    results = runner.run_experiment(
        datasets=['synth1'],
        models=['LGTD'],
        verbose=False
    )

    print(f"Threshold {thresh}: MSE = {results['mse_trend'].iloc[0]:.4f}")
```

---

## Troubleshooting

### Import Errors

```bash
# Make sure you're in project root
cd /path/to/LGTD
source env/bin/activate
python experiments/runners/experiment_runner.py
```

### No Results Saved

Check that `save_results=True` (default) and results directory exists:
```bash
mkdir -p results/synthetic
```

### Plots Not Generated

Check `save_plots` in parameter file:
```json
{
  "evaluation": {
    "save_plots": true,
    "plot_dir": "results/synthetic/plots/synth1"
  }
}
```

### Model Not Available

Check available models:
```python
from experiments.runners.experiment_runner import ExperimentRunner
runner = ExperimentRunner()
print(runner.available_models)
```

---

## Performance Tips

1. **Start Small**: Test on one dataset first
2. **Disable Plots**: Set `save_plots: false` for faster execution
3. **Use Quiet Mode**: Add `--quiet` flag to reduce output
4. **Parallel Execution**: Run multiple experiments in separate terminals
5. **Selective Metrics**: Only compute needed metrics in config

---

## Next Steps

1. **See [EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md)** for detailed documentation
2. **See [CONFIG_SYNC_GUIDE.md](CONFIG_SYNC_GUIDE.md)** for dataset synchronization
3. **View example parameter files** in `configs/dataset_params/`
4. **Run demo experiment**: `python experiments/runners/experiment_runner.py --datasets synth1 --models LGTD`
5. **Analyze results**: `python experiments/scripts/analyze_results.py results/synthetic/experiment_results.csv`

---

**Created**: 2025-12-28
**Version**: 1.0
**Compatible with**: LGTD v3.0 (Fully Optimized)
