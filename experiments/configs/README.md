# Experiment Configuration - Model-Based Structure

This directory contains the new model-based configuration structure for LGTD experiments.

## Directory Structure

```
experiments/configs/
├── datasets.yaml              # Dataset definitions
├── experiment_settings.yaml   # General experiment settings
├── models/                    # Model-specific configurations
│   ├── lgtd.yaml
│   ├── lgtd_linear.yaml
│   ├── lgtd_lowess.yaml
│   ├── stl.yaml
│   ├── fast_robust_stl.yaml
│   ├── str.yaml
│   ├── online_stl.yaml
│   ├── oneshot_stl.yaml
│   ├── astd.yaml
│   └── astd_online.yaml
└── old_configs_backup/        # Archived old configuration files
```

## Configuration Files

### 1. datasets.yaml

Defines all available datasets (both synthetic and real-world).

**Example:**
```yaml
datasets:
- name: synth1
  path: data/synthetic/datasets/synth1_data.json
  trend_type: linear
  period_type: fixed
  period: 120
- name: ETTh1
  type: real_world
  loader: load_ett_dataset
  loader_params:
    column: OT
    subset: all
```

### 2. experiment_settings.yaml

General experiment settings including evaluation metrics.

**Example:**
```yaml
experiment:
  name: lgtd_synthetic_evaluation
  description: Synthetic data experiments for LGTD method validation
  output_dir: experiments/results/synthetic
evaluation:
  metrics:
  - mse
  - mae
  - rmse
  - correlation
  - psnr
  save_plots: true
```

### 3. models/*.yaml

Each model has its own YAML file with default parameters and dataset-specific overrides.

**Example (models/lgtd.yaml):**
```yaml
model_name: lgtd
enabled: true
default_params:
  window_size: 3
  error_percentile: 30
  trend_selection: auto
  lowess_frac: 0.1
  threshold_r2: 0.92
  verbose: false
dataset_params:
  synth2:
    error_percentile: 40  # Override for synth2
  synth7:
    window_size: 5        # Override for synth7
```

## How to Update Parameters

### To change LGTD parameters for specific datasets:

1. Open the relevant model file: `models/lgtd.yaml`
2. Edit the `dataset_params` section:

```yaml
dataset_params:
  synth7:
    window_size: 5
    error_percentile: 30
  synth8:
    window_size: 5
    error_percentile: 30
  synth9:
    window_size: 5
    error_percentile: 30
```

### To change default parameters for a model:

Edit the `default_params` section in the model's YAML file.

## Running Experiments

Use the new experiment runner:

```bash
# Run all experiments
python experiments/scripts/run_experiments_new.py

# Run specific datasets
python experiments/scripts/run_experiments_new.py --datasets synth7 synth8 synth9

# Run specific models
python experiments/scripts/run_experiments_new.py --models lgtd lgtd_linear lgtd_lowess

# Run specific datasets and models
python experiments/scripts/run_experiments_new.py --datasets synth7 synth8 synth9 --models lgtd
```

## Benefits of New Structure

1. **Model-centric organization**: Easier to find and update parameters for each model
2. **Clear parameter inheritance**: Default params + dataset-specific overrides
3. **No redundancy**: Each parameter defined once, overridden only when needed
4. **Easy to maintain**: Single file per model instead of per-dataset configs
5. **Better readability**: YAML format with clear hierarchy

## Migration

Old configuration files have been moved to `old_configs_backup/`:
- `baseline_configs.yaml`
- `synthetic_experiments.yaml`
- `real_world_experiments.yaml`
- `dataset_params/` (all JSON files)

These are kept for reference but are no longer used by the new experiment runner.

## Adding New Datasets

Add to `datasets.yaml`:

```yaml
- name: synth10
  path: data/synthetic/datasets/synth10_data.json
  trend_type: linear
  period_type: variable
  period: variable
```

If this dataset needs specific model parameters, add overrides to the relevant model YAML files.

## Adding New Models

1. Create `models/new_model.yaml`:

```yaml
model_name: new_model
enabled: true
default_params:
  param1: value1
  param2: value2
dataset_params:
  synth1:
    param1: override_value
```

2. Add model implementation to the runner in `experiments/runners/model_based_runner.py`
