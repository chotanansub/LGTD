# Quick Start: Fast Plotting with Caching

## The Problem You're Experiencing

When running `generate_method_comparison_plots.py`, **FastRobustSTL takes ~60 seconds per dataset** because it re-runs the full decomposition each time. For 9 datasets, that's ~9 minutes just waiting!

## The Solution (in 2 steps!)

### Step 1: Populate the cache (one-time)

Run this **once** to save all decomposition results:

```bash
# Option A: Run all experiments (recommended if you haven't run them yet)
python experiments/runners/experiment_runner.py

# Option B: Just populate the cache without updating results CSV
python scripts/populate_decomposition_cache.py

# Option C: Only cache FastRobustSTL (if other models are already cached)
python scripts/populate_decomposition_cache.py --models FastRobustSTL
```

This will take time (FastRobustSTL is slow), but you only do it **once**.

### Step 2: Generate plots (now instant!)

```bash
# Generate comparison plots using cached decompositions
python scripts/generate_method_comparison_plots.py
```

Now instead of 9 minutes, it takes **~1 second** to generate all plots! 🚀

## How It Works

### Before (Slow):
```
generate_method_comparison_plots.py
  ↓
  For each dataset:
    For each model:
      Run full decomposition (60s for FastRobustSTL) ❌ SLOW
      Plot results
```

### After (Fast):
```
Step 1 (once):
  experiment_runner.py
    ↓
    Run decompositions
    ↓
    Save to results/synthetic/decompositions/{dataset}/{model}.npz

Step 2 (every time):
  generate_method_comparison_plots.py
    ↓
    For each dataset:
      For each model:
        Load cached .npz file (0.01s) ✅ FAST
        Plot results
```

## Examples

### Example 1: First time setup
```bash
# Run experiments and populate cache
python experiments/runners/experiment_runner.py

# Generate plots (uses cache automatically)
python scripts/generate_method_comparison_plots.py
```

### Example 2: Adding a new model
```bash
# Run experiments for just the new model
python experiments/runners/experiment_runner.py --models NewModel

# Regenerate plots (uses cache for old models, runs NewModel)
python scripts/generate_method_comparison_plots.py
```

### Example 3: Changed model parameters
```bash
# Clear old cache for that model
find results/synthetic/decompositions -name "FastRobustSTL.npz" -delete

# Re-run with new parameters
python experiments/runners/experiment_runner.py --models FastRobustSTL

# Generate plots with updated results
python scripts/generate_method_comparison_plots.py
```

### Example 4: Force bypass cache (for testing)
```bash
# Regenerate everything from scratch (slow!)
python scripts/generate_method_comparison_plots.py --no-cache
```

## Cache Details

**Location**: `results/synthetic/decompositions/`

**Format**:
```
decompositions/
  synth1/
    LGTD.npz          (~50 KB)
    STL.npz           (~50 KB)
    FastRobustSTL.npz (~50 KB)
    ...
  synth2/
    ...
```

**Total size**: ~5 MB for all datasets and models

**What's cached**: Trend, seasonal, and residual components as NumPy arrays

## Tips

1. **Always use cache** (it's on by default) - no reason not to!
2. **Commit `.gitkeep`** in `decompositions/` but add `*.npz` to `.gitignore` (arrays are large)
3. **Regenerate cache** only when model parameters change
4. **Share cache** with team by compressing and sharing the `decompositions/` folder

## Troubleshooting

### "No cached results found"
You haven't run experiments yet. Run:
```bash
python scripts/populate_decomposition_cache.py
```

### "Plots look wrong after changing parameters"
Clear the cache and regenerate:
```bash
rm -rf results/synthetic/decompositions/{dataset}/{model}.npz
python experiments/runners/experiment_runner.py --datasets {dataset} --models {model}
```

### "Want to start fresh"
```bash
rm -rf results/synthetic/decompositions/
python scripts/populate_decomposition_cache.py
```
