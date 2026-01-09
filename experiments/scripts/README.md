# Experiment Scripts

This directory contains all scripts for running experiments, generating tables, creating plots, and utilities.

## Directory Structure

```
experiments/scripts/
├── benchmarks/          # Running experiments and parameter tuning
├── tables/              # LaTeX table generation
├── plots/               # Visualization and plotting
├── utils/               # Utility scripts
└── [legacy scripts]     # Existing experiment runners
```

## Benchmarks

Scripts for running experiments and tuning parameters.

### `benchmarks/run_realworld.py`
Run decomposition experiments on real-world datasets (ETTh1, ETTh2, sunspot).

```bash
# Run all datasets with all models
python -m experiments.scripts.benchmarks.run_realworld

# Run specific datasets
python -m experiments.scripts.benchmarks.run_realworld --datasets ETTh1 ETTh2

# Run specific models
python -m experiments.scripts.benchmarks.run_realworld --models LGTD STL OneShotSTL

# Limit data points
python -m experiments.scripts.benchmarks.run_realworld --n-points 1000
```

### `benchmarks/tune_parameters.py`
Hyperparameter tuning for all decomposition methods.

```bash
# Tune all methods
python -m experiments.scripts.benchmarks.tune_parameters

# Tune specific methods
python -m experiments.scripts.benchmarks.tune_parameters LGTD STL
```

## Tables

LaTeX table generation scripts. All tables are saved to `experiments/results/summary/`.

### `tables/generate_decomposition_tables.py`
Generate decomposition error comparison tables (standard format).

```bash
# Use default benchmarks
python -m experiments.scripts.tables.generate_decomposition_tables

# Use specific CSV file
python -m experiments.scripts.tables.generate_decomposition_tables path/to/results.csv

# Custom output location
python -m experiments.scripts.tables.generate_decomposition_tables -o custom_table.tex
```

### `tables/generate_decomposition_tables_wide.py`
Generate decomposition error comparison tables (wide format).

```bash
python -m experiments.scripts.tables.generate_decomposition_tables_wide
```

### `tables/generate_mae_tables.py`
Generate MAE summary tables by dataset type (all, transitive, variable).

```bash
python -m experiments.scripts.tables.generate_mae_tables
```

**Output:**
- `table_mae_summary_all.tex` - Average MAE across all datasets
- `table_mae_summary_transitive.tex` - Average MAE for transitive period datasets
- `table_mae_summary_variable.tex` - Average MAE for variable period datasets

### `tables/generate_parameter_table.py`
Generate hyperparameter configuration tables for all methods.

```bash
# Synthetic datasets
python -m experiments.scripts.tables.generate_parameter_table --synthetic

# Real-world datasets
python -m experiments.scripts.tables.generate_parameter_table --realworld

# Specific datasets
python -m experiments.scripts.tables.generate_parameter_table --datasets synth1 synth2 synth3

# Custom output
python -m experiments.scripts.tables.generate_parameter_table -o my_params.tex
```

## Plots

Visualization and plotting scripts. Plots are saved to respective results directories.

### `plots/plot_synthetic.py`
Generate comparison plots for synthetic dataset decompositions.

```bash
# Plot by dataset (one plot per dataset with all models)
python -m experiments.scripts.plots.plot_synthetic --mode by-dataset

# Plot by model (one plot per model with all datasets)
python -m experiments.scripts.plots.plot_synthetic --mode by-model

# Generate both
python -m experiments.scripts.plots.plot_synthetic --mode both

# Specific datasets/models
python -m experiments.scripts.plots.plot_synthetic --datasets synth1 synth2 --models LGTD STL
```

### `plots/plot_realworld.py`
Plot real-world dataset decomposition results (4-model comparison).

```bash
# Plot all datasets
python -m experiments.scripts.plots.plot_realworld

# Specific datasets
python -m experiments.scripts.plots.plot_realworld --datasets ETTh1 sunspot

# Specific models
python -m experiments.scripts.plots.plot_realworld --models LGTD STL ASTD OnlineSTL
```

### `plots/plot_realworld_full.py`
Full comparison plots with all available models for real-world datasets.

```bash
python -m experiments.scripts.plots.plot_realworld_full
```

### `plots/plot_method_comparison.py`
Generate method comparison visualizations.

```bash
# Generate all comparison plots
python -m experiments.scripts.plots.plot_method_comparison

# Specific datasets
python -m experiments.scripts.plots.plot_method_comparison --datasets synth1 synth2

# Custom output directory
python -m experiments.scripts.plots.plot_method_comparison --output-dir my_plots/
```

## Utils

Utility scripts for dataset generation, caching, and configuration.

### `utils/populate_cache.py`
Pre-compute and cache decomposition results for faster plotting.

```bash
# Cache all decompositions
python -m experiments.scripts.utils.populate_cache

# Specific datasets
python -m experiments.scripts.utils.populate_cache --datasets synth1 synth2 synth3
```

**Benefits:**
- Speeds up plotting by 10-100x
- Saves decomposition arrays as `.npz` files
- Cached files stored in `experiments/results/synthetic/decompositions/`

### `utils/generate_config.py`
Generate experiment YAML configuration from dataset JSON metadata.

```bash
python -m experiments.scripts.utils.generate_config
```

### `utils/regenerate_datasets.py`
Regenerate datasets with variable periods.

```bash
python -m experiments.scripts.utils.regenerate_datasets
```

## Legacy Scripts

These scripts remain in the root of `experiments/scripts/` for backward compatibility:

- `run_benchmarks.py` - Legacy benchmark runner
- `run_experiments.py` - Legacy experiment runner
- `run_synthetic.py` - Run synthetic experiments
- `analyze_results.py` - Analyze experiment results
- `sync_dataset_configs.py` - Synchronize dataset configurations

## Common Workflows

### Complete Synthetic Experiment Pipeline

```bash
# 1. Run experiments
python -m experiments.runners.experiment_runner

# 2. Generate plots
python -m experiments.scripts.plots.plot_synthetic --mode both

# 3. Create tables
python -m experiments.scripts.tables.generate_mae_tables
python -m experiments.scripts.tables.generate_decomposition_tables
python -m experiments.scripts.tables.generate_parameter_table --synthetic
```

### Complete Real-World Experiment Pipeline

```bash
# 1. Run experiments
python -m experiments.scripts.benchmarks.run_realworld

# 2. Generate plots
python -m experiments.scripts.plots.plot_realworld
python -m experiments.scripts.plots.plot_realworld_full

# 3. Create tables
python -m experiments.scripts.tables.generate_parameter_table --realworld
```

### Speed Up Plotting with Caching

```bash
# 1. Cache decompositions once
python -m experiments.scripts.utils.populate_cache

# 2. Now plotting is much faster
python -m experiments.scripts.plots.plot_synthetic --mode both
```

## Output Locations

- **Benchmarks CSV**: `experiments/results/benchmarks/synthetic_benchmarks.csv`
- **LaTeX Tables**: `experiments/results/summary/*.tex`
- **Synthetic Plots**: `experiments/results/synthetic/figures/`
- **Real-world Plots**: `experiments/results/real_world/figures/`
- **Decomposition Cache**: `experiments/results/synthetic/decompositions/`

## Note on Imports

All scripts should be run as modules from the project root:

```bash
# ✅ Correct
python -m experiments.scripts.benchmarks.run_realworld

# ❌ Incorrect
cd experiments/scripts/benchmarks
python run_realworld.py
```

This ensures proper import paths and file references.
