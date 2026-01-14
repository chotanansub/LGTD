# Installation Guide

## Prerequisites

**Python Version:** 3.8 or higher

**Required System Libraries:**
- Standard scientific computing stack (NumPy, SciPy)
- Matplotlib for visualization (optional)

## Core Module Installation

The LGTD module can be installed independently without baseline methods.

### Basic Installation

From the project root directory:

```bash
pip install -e .
```

This installs:
- LGTD decomposition algorithm
- Evaluation metrics (MSE, MAE, correlation, PSNR)
- Visualization utilities

### Dependencies

Core dependencies automatically installed:
- `numpy >= 1.19.0`
- `scipy >= 1.5.0`
- `matplotlib >= 3.3.0`
- `statsmodels >= 0.12.0`

## Experiment Framework Installation

For reproducing paper experiments, install with baseline methods.

### Full Installation

```bash
pip install -e ".[experiments]"
```

This adds:
- Synthetic data generators
- Baseline method wrappers
- Experiment runners and configuration loaders
- Table and figure generation scripts

### Additional Dependencies

Experimental framework requires:
- `pandas >= 1.1.0`
- `pyyaml >= 5.3.0`
- `tqdm >= 4.50.0`
- `seaborn >= 0.11.0` (for visualization)

## Baseline Method Installation

Baseline methods require separate installation as they are not pip-installable.

### STL (Cleveland et al., 1990)

Already included via `statsmodels`:

```python
from statsmodels.tsa.seasonal import STL
```

### RobustSTL (Wen et al., 2019)

Clone and install from repository:

```bash
git clone https://github.com/LeeDoYup/RobustSTL.git
cd RobustSTL
pip install -e .
```

### ASTD (Phungtua-eng & Yamamoto, 2024)

Clone and install from repository:

```bash
git clone https://github.com/thanapol2/ASTD_ECMLPKDD.git
cd ASTD_ECMLPKDD
pip install -e .
```

### STR (Dokumentov & Hyndman, 2022)

Custom Python implementation based on R package:

```bash
git clone https://github.com/robjhyndman/STR_paper.git
# Use custom wrapper in experiments/baselines/str_decomposer.py
```

### FastRobustSTL (Wen et al., 2020)

Clone from repository:

```bash
git clone https://github.com/ariaghora/fast-robust-stl.git
cd fast-robust-stl
pip install -e .
```

### OnlineSTL (Mishra et al., 2022)

Clone from repository:

```bash
git clone https://github.com/YHYHYHYHYHY/OnlineSTL.git
cd OnlineSTL
pip install -e .
```

### OneShotSTL (He et al., 2023)

Clone from repository:

```bash
git clone https://github.com/xiao-he/OneShotSTL.git
cd OneShotSTL
pip install -e .
```

## Verification

Verify installation:

```python
# Test core module
from lgtd import LGTD
import numpy as np

y = np.random.randn(100) + np.sin(np.linspace(0, 4*np.pi, 100))
model = LGTD()
result = model.fit_transform(y)
print(f"Decomposition successful: {len(result.trend)} samples")

# Test experiment framework
from experiments.runners.experiment_runner import ExperimentRunner
print("Experiment framework available")
```

## Troubleshooting

### Import Errors

If baseline methods fail to import, the framework automatically skips unavailable methods during experiments. Check `experiments/baselines/` for wrapper implementations.

### Dependency Conflicts

Use virtual environment to isolate dependencies:

```bash
python -m venv lgtd_env
source lgtd_env/bin/activate  # On Windows: lgtd_env\Scripts\activate
pip install -e ".[experiments]"
```

### Platform-Specific Issues

**macOS:** Ensure Xcode Command Line Tools installed for compilation:
```bash
xcode-select --install
```

**Linux:** Install build essentials:
```bash
sudo apt-get install build-essential
```

**Windows:** Install Microsoft C++ Build Tools if compilation errors occur.

## Development Installation

For development with testing dependencies:

```bash
pip install -e ".[dev]"
pytest tests/
```

This adds:
- `pytest >= 6.0.0`
- `pytest-cov >= 2.10.0`
- `black >= 20.8b1` (code formatting)
- `flake8 >= 3.8.0` (linting)
