# LGTD: Local Global Trend Decomposition

A novel time series decomposition method that combines local linear trend detection with seasonal pattern extraction.

## Overview

LGTD (Local Global Trend Decomposition) is designed to decompose time series data into trend and seasonal components. The method integrates:
- **Local Linear Trend Detection**: Using AutoTrend module for identifying local trend patterns
- **Seasonal Decomposition**: Advanced techniques for extracting seasonal patterns

## Installation

### Basic Installation (Core LGTD only)
```bash
git clone https://github.com/chotanansub/lgtd.git
cd lgtd
pip install -e .
```

This installs only the LGTD decomposition method with minimal dependencies.

**Use this if you:**
- Just want to use LGTD for decomposition
- Don't need to compare with other methods
- Want minimal dependencies

### Full Installation (With Experiments and Baselines)
For running experiments and benchmarks:
```bash
pip install -e ".[experiments]"
```

**Use this if you:**
- Want to reproduce research results
- Need to compare LGTD with baseline methods
- Are conducting academic research

### Development Installation
For contributing to the project:
```bash
pip install -e ".[dev]"
```

### All Features
```bash
pip install -e ".[all]"
```

### Dependencies
The project requires AutoTrend from: https://github.com/chotanansub/autotrend

```bash
pip install git+https://github.com/chotanansub/autotrend.git
```

## Quick Start

```python
from lgtd import LGTD
import numpy as np

# Generate sample data
t = np.arange(100)
trend = 0.5 * t
seasonal = 10 * np.sin(2 * np.pi * t / 12)
data = trend + seasonal + np.random.normal(0, 1, 100)

# Decompose
model = LGTD()
result = model.fit_transform(data)

# Access components
print(result.trend)
print(result.seasonal)
print(result.residual)
```

## Project Structure

- `lgtd/`: Core package implementation
- `data/`: Synthetic and real-world datasets
- `experiments/`: Experimental framework and scripts
- `results/`: Experiment outputs
- `notebooks/`: Jupyter notebooks for analysis
- `tests/`: Unit and integration tests

## Experiments

Run synthetic experiments:
```bash
python experiments/scripts/run_synthetic.py
```

Run benchmark comparisons:
```bash
python experiments/scripts/run_benchmarks.py
```

## Citation

If you use LGTD in your research, please cite:
```
[Citation to be added upon publication]
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.
