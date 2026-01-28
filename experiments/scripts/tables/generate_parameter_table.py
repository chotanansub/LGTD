#!/usr/bin/env python3
"""
Generate LaTeX table of model parameters across synthetic and real-world datasets.

Usage:
    python generate_parameter_table.py
"""

import sys
import yaml
from pathlib import Path
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Configuration
MODELS_DIR = project_root / "experiments" / "configs" / "models"
OUTPUT_DIR = project_root / "experiments" / "results" / "latex_tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Dataset ordering
SYNTHETIC_DATASETS = ['synth1', 'synth2', 'synth3', 'synth4', 'synth5',
                      'synth6', 'synth7', 'synth8', 'synth9']
REALWORLD_DATASETS = ['ETTh1', 'ETTh2', 'Sunspot']
ALL_DATASETS = SYNTHETIC_DATASETS + REALWORLD_DATASETS

# Model display names and ordering
MODEL_ORDER = ['LGTD', 'ASTD', 'ASTD$_{\\text{Online}}$', 'FastRobustSTL',
               'OneShotSTL', 'OnlineSTL', 'RobustSTL', 'STL', 'STR']

MODEL_NAMES = {
    'lgtd': 'LGTD',
    'astd': 'ASTD',
    'astd_online': 'ASTD$_{\\text{Online}}$',
    'fast_robust_stl': 'FastRobustSTL',
    'oneshot_stl': 'OneShotSTL',
    'online_stl': 'OnlineSTL',
    'robust_stl': 'RobustSTL',
    'stl': 'STL',
    'str': 'STR'
}

# Models to exclude from table
EXCLUDE_MODELS = {'lgtd_linear', 'lgtd_lowess'}

# Parameters to show for specific models (if None, show all)
MODEL_PARAMS_FILTER = {
    'lgtd': {'window_size', 'error_percentile'},  # Only show these for LGTD
}

# Parameter display names
PARAM_NAMES = {
    'window_size': 'Window Size',
    'error_percentile': 'Error Percentile',
    'trend_selection': 'Trend Selection',
    'lowess_frac': 'LOWESS Fraction',
    'threshold_r2': '$R^2$ Threshold',
    'period': 'Period',
    'periods': 'Periods',
    'seasonal': 'Seasonal Window',
    'trend': 'Trend Window',
    'robust': 'Robust',
    'reg1': '$\\lambda_1$ (Trend)',
    'reg2': '$\\lambda_2$ (Seasonal)',
    'K': '$K$ (Bilateral)',
    'H': '$H$ (Trend)',
    'dn1': '$d_{n1}$',
    'dn2': '$d_{n2}$',
    'ds1': '$d_{s1}$',
    'ds2': '$d_{s2}$',
    'max_iter': 'Max Iterations',
    'seasonality_smoothing': 'Seasonality Smoothing',
    'init_window_size': 'Init Window Size',
    'init_ratio': 'Init Ratio',
    'init_window_ratio': 'Init Window Ratio',
    'alpha': 'Alpha',
    'beta': 'Beta',
    'seasonal_periods': 'Seasonal Periods',
    'trend_lambda': 'Trend $\\lambda$',
    'seasonal_lambda': 'Seasonal $\\lambda$',
    'auto_params': 'Auto Params',
    'n_trials': 'N Trials',
    'lam': '$\\lambda$ (Smoothing)',
    'shift_window': 'Shift Window',
    'mode': 'Mode',
}

# Parameter ordering
PARAM_ORDER = [
    'period', 'periods', 'seasonal_periods', 'window_size', 'seasonal', 'trend',
    'seasonality_smoothing', 'lam', 'error_percentile', 'trend_selection',
    'lowess_frac', 'threshold_r2', 'reg1', 'reg2', 'trend_lambda', 'seasonal_lambda',
    'K', 'H', 'dn1', 'dn2', 'ds1', 'ds2', 'init_window_size', 'init_window_ratio',
    'init_ratio', 'shift_window', 'robust', 'auto_params', 'max_iter', 'n_trials'
]

# Default parameter values for each model (for underlining)
MODEL_DEFAULTS = {
    'lgtd': {
        'window_size': 5,
        'error_percentile': 50,
        'trend_selection': 'auto',
        'lowess_frac': 0.1,
        'threshold_r2': 0.92,
    },
    'astd': {
        'seasonality_smoothing': 0.7,
    },
    'astd_online': {
        'seasonality_smoothing': 0.7,
        'init_window_size': 300,
    },
    'fast_robust_stl': {
        'period': 120,
        'reg1': 10.0,
        'reg2': 0.5,
        'K': 2,
        'H': 5,
        'dn1': 1.0,
        'dn2': 1.0,
        'ds1': 50.0,
        'ds2': 1.0,
        'max_iter': 1000,
    },
    'oneshot_stl': {
        'period': 120,
        'init_ratio': 0.3,
        'shift_window': 0,
    },
    'online_stl': {
        'periods': [120],
        'lam': 0.7,
        'init_window_ratio': 0.3,
    },
    'stl': {
        'period': 120,
        'seasonal': 13,
        'trend': None,
        'robust': False,
    },
    'str': {
        'seasonal_periods': [120],
        'trend_lambda': 1000.0,
        'seasonal_lambda': 10.0,
        'robust': False,
        'auto_params': False,
        'n_trials': 10,
    },
}


def format_value(value, is_default=False):
    """Format parameter value for LaTeX."""
    if value is None or value == '':
        formatted = ''
    elif isinstance(value, bool):
        formatted = '\\checkmark' if value else '$\\times$'
    elif isinstance(value, (int, float)):
        if isinstance(value, int):
            formatted = str(value)
        elif value >= 10:
            formatted = f"{value:.2f}"
        else:
            formatted = f"{value:.3f}".rstrip('0').rstrip('.')
    elif isinstance(value, list):
        formatted = '[' + ', '.join(str(v) for v in value) + ']'
    elif isinstance(value, str):
        formatted = value.capitalize() if value.lower() in ['auto', 'linear', 'lowess'] else value
    else:
        formatted = str(value)

    # Underline if it's a default value
    if is_default and formatted:
        formatted = f'\\underline{{{formatted}}}'

    return formatted


def is_default_value(model_name, param_name, value):
    """Check if a value matches the default for this model/parameter."""
    defaults = MODEL_DEFAULTS.get(model_name, {})
    default_val = defaults.get(param_name)

    if default_val is None:
        return False

    # Compare values appropriately
    if isinstance(value, list) and isinstance(default_val, list):
        return value == default_val
    elif isinstance(value, (int, float)) and isinstance(default_val, (int, float)):
        return abs(value - default_val) < 1e-9
    else:
        return value == default_val


def load_all_model_configs():
    """Load all model configuration files."""
    configs = {}
    for yaml_file in MODELS_DIR.glob('*.yaml'):
        with open(yaml_file, 'r') as f:
            config = yaml.safe_load(f)
            model_name = config.get('model_name')
            # Skip excluded models
            if model_name and config.get('enabled', True) and model_name not in EXCLUDE_MODELS:
                configs[model_name] = config
    return configs


def check_if_varies(model_config, param_name, datasets):
    """Check if parameter varies across datasets."""
    values = set()
    for dataset in datasets:
        params = model_config.get('dataset_params', {}).get(dataset, {})
        if param_name in params:
            val = params[param_name]
            # Convert to string for comparison
            values.add(str(val) if not isinstance(val, list) else str(sorted(val)))
    return len(values) > 1


def generate_latex_table():
    """Generate the complete LaTeX table."""
    # Load all configs
    all_configs = load_all_model_configs()

    # Map model file names to display names
    model_map = {}
    for file_model_name, config in all_configs.items():
        display_name = MODEL_NAMES.get(file_model_name, file_model_name.upper())
        model_map[file_model_name] = display_name

    # Start building table
    lines = []
    lines.append(r'\begin{table*}[!htb]')
    lines.append(r'\centering')
    lines.append(r'\setlength{\tabcolsep}{2pt}')
    lines.append(r'\renewcommand{\arraystretch}{1.05}')
    lines.append(r'\caption{Hyperparameter configurations for all models across synthetic and real-world datasets.}')
    lines.append(r'\label{tab:model_parameters}')

    # Build column spec: Model + Parameter + 9 synth + 3 real = 14 cols (no Varies column)
    n_synth = len(SYNTHETIC_DATASETS)
    n_real = len(REALWORLD_DATASETS)
    col_spec = 'll' + 'c' * (n_synth + n_real)
    lines.append(r'\begin{tabular}{' + col_spec + '}')
    lines.append(r'\toprule')

    # Header row 1: Group headers
    header1 = '& & \\multicolumn{' + str(n_synth) + '}{c}{\\textbf{Synthetic Datasets}}'
    header1 += ' & \\multicolumn{' + str(n_real) + '}{c}{\\textbf{Real-World Datasets}} \\\\'
    lines.append(header1)

    # Header row 2: cmidrules
    cmidrule = '\\cmidrule(lr){3-' + str(2 + n_synth) + '} '
    cmidrule += '\\cmidrule(lr){' + str(3 + n_synth) + '-' + str(2 + n_synth + n_real) + '}'
    lines.append(cmidrule)

    # Header row 3: Column names
    header2 = '\\textbf{Model} & \\textbf{Parameter}'
    for ds in SYNTHETIC_DATASETS:
        header2 += f' & \\textbf{{{ds.replace("synth", "s")}}}'
    for ds in REALWORLD_DATASETS:
        header2 += f' & \\textbf{{{ds}}}'
    header2 += ' \\\\'
    lines.append(header2)
    lines.append(r'\midrule')

    # Generate rows for each model
    for model_idx, file_model_name in enumerate(sorted(all_configs.keys(),
                                                        key=lambda x: MODEL_ORDER.index(MODEL_NAMES.get(x, x))
                                                        if MODEL_NAMES.get(x, x) in MODEL_ORDER else 999)):
        config = all_configs[file_model_name]
        display_name = model_map[file_model_name]

        # Collect all parameters used for this model
        all_params = set()
        for dataset in ALL_DATASETS:
            params = config.get('dataset_params', {}).get(dataset, {})
            all_params.update(params.keys())

        # Remove verbose and other unwanted params
        all_params.discard('verbose')
        all_params.discard('mode')

        # Apply parameter filter if specified for this model
        if file_model_name in MODEL_PARAMS_FILTER:
            all_params = all_params.intersection(MODEL_PARAMS_FILTER[file_model_name])

        # Sort parameters
        sorted_params = [p for p in PARAM_ORDER if p in all_params]
        sorted_params.extend(sorted([p for p in all_params if p not in PARAM_ORDER]))

        if not sorted_params:
            continue

        # Add multirow for model name
        for param_idx, param_name in enumerate(sorted_params):
            param_display = PARAM_NAMES.get(param_name, param_name.replace('_', ' ').title())

            # Check if varies
            varies = check_if_varies(config, param_name, ALL_DATASETS)

            # Build row
            if param_idx == 0:
                row = f'\\multirow{{{len(sorted_params)}}}{{*}}{{{display_name}}}\n & {param_display}'
            else:
                row = f' & {param_display}'

            # If doesn't vary, merge all dataset columns into one
            if not varies:
                # Get the single value (from any dataset that has it)
                single_value = None
                for dataset in ALL_DATASETS:
                    params = config.get('dataset_params', {}).get(dataset, {})
                    if param_name in params:
                        single_value = params[param_name]
                        break

                is_def = is_default_value(file_model_name, param_name, single_value)
                formatted_val = format_value(single_value, is_default=is_def)
                row += f' & \\multicolumn{{{len(ALL_DATASETS)}}}{{c}}{{{formatted_val}}}'
            else:
                # Add values for each dataset separately
                for dataset in ALL_DATASETS:
                    params = config.get('dataset_params', {}).get(dataset, {})
                    value = params.get(param_name)
                    is_def = is_default_value(file_model_name, param_name, value)
                    row += f' & {format_value(value, is_default=is_def)}'

            row += ' \\\\'
            lines.append(row)

        # Add separator between models
        if model_idx < len(all_configs) - 1:
            lines.append(r'\midrule')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table*}')

    return '\n'.join(lines)


def main():
    """Generate parameter table."""
    print("=" * 70)
    print("Generating LaTeX Parameter Table")
    print("=" * 70)

    # Generate table
    latex_table = generate_latex_table()

    # Save to file
    output_file = OUTPUT_DIR / 'model_parameters.tex'
    with open(output_file, 'w') as f:
        f.write(latex_table)

    print(f"\n✓ Table saved to: {output_file}")
    print(f"  Synthetic datasets: {len(SYNTHETIC_DATASETS)}")
    print(f"  Real-world datasets: {len(REALWORLD_DATASETS)}")
    print("\nTo use in your LaTeX document:")
    print(f"  \\input{{{output_file}}}")


if __name__ == '__main__':
    main()
