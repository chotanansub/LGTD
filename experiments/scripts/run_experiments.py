#!/usr/bin/env python
"""
Convenient script to run LGTD experiments.

Examples:
    # Run all models on all datasets
    python experiments/scripts/run_experiments.py

    # Run specific model on all datasets
    python experiments/scripts/run_experiments.py --models LGTD

    # Run all models on specific datasets
    python experiments/scripts/run_experiments.py --datasets synth1 synth2 synth3

    # Run specific model on specific dataset
    python experiments/scripts/run_experiments.py --datasets synth1 --models LGTD STL

    # Run LGTD variants comparison
    python experiments/scripts/run_experiments.py --models LGTD LGTD_Linear LGTD_LOWESS
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.runners.experiment_runner import main

if __name__ == '__main__':
    main()
