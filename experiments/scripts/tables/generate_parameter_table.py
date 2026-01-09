#!/usr/bin/env python3
"""
Generate LaTeX table of full parameter configurations for all models across datasets.

This script reads the parameter configuration JSON files for each dataset and creates
a comprehensive LaTeX table showing all hyperparameters used for each model.

Usage Examples:
    # All synthetic datasets (default)
    python scripts/generate_parameter_table.py

    # Real-world datasets
    python scripts/generate_parameter_table.py --realworld

    # Specific datasets
    python scripts/generate_parameter_table.py --datasets synth1 synth2 synth3

    # Specific models only
    python scripts/generate_parameter_table.py --models LGTD STL FastRobustSTL

    # Custom output file
    python scripts/generate_parameter_table.py -o my_table.tex

    # Exclude certain parameters
    python scripts/generate_parameter_table.py --exclude-params verbose robust

    # Combine options
    python scripts/generate_parameter_table.py --datasets synth1 synth2 --models LGTD STL -o lgtd_stl_params.tex

Output:
    - Generates a LaTeX table showing all model hyperparameters
    - Includes a "Varies" column indicating which parameters differ across datasets
    - Table uses landscape orientation for readability
    - Ready to compile with: pdflatex <output_file>

See scripts/PARAMETER_TABLE_README.md for detailed documentation.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict


# Model display names
MODEL_NAMES = {
    'LGTD': 'LGTD',
    'LGTD_Linear': 'LGTD$_{\\text{Linear}}$',
    'LGTD_LOWESS': 'LGTD$_{\\text{LOWESS}}$',
    'STL': 'STL',
    'RobustSTL': 'RobustSTL',
    'ASTD': 'ASTD',
    'ASTD_Online': 'ASTD$_{\\text{Online}}$',
    'OnlineSTL': 'OnlineSTL',
    'OneShotSTL': 'OneShotSTL',
    'FastRobustSTL': 'FastRobustSTL',
    'STR': 'STR'
}

# Model-specific parameters to exclude
MODEL_SPECIFIC_EXCLUSIONS = {
    'ASTD': {'period'},  # ASTD doesn't use period parameter
    'ASTD_Online': set(),
    'OnlineSTL': set(),
    'STR': set(),
}

# Default parameter values from model implementations
MODEL_DEFAULTS = {
    'LGTD': {
        'window_size': 3,
        'error_percentile': 50,
        'trend_selection': 'auto',
        'lowess_frac': 0.1,
        'threshold_r2': 0.92,
        'verbose': False
    },
    'LGTD_Linear': {
        'window_size': 3,
        'error_percentile': 50,
        'trend_selection': 'linear',
        'verbose': False
    },
    'LGTD_LOWESS': {
        'window_size': 3,
        'error_percentile': 50,
        'trend_selection': 'lowess',
        'lowess_frac': 0.1,
        'verbose': False
    },
    'STL': {
        'period': 12,
        'seasonal': 7,
        'trend': None,
        'robust': False
    },
    'RobustSTL': {
        'period': 12,
        'reg1': 10.0,
        'reg2': 0.5,
        'K': 2,
        'H': 5,
        'dn1': 1.0,
        'dn2': 1.0,
        'ds1': 50.0,
        'ds2': 1.0
    },
    'FastRobustSTL': {
        'period': 12,
        'reg1': 10.0,
        'reg2': 0.5,
        'K': 2,
        'H': 5,
        'dn1': 1.0,
        'dn2': 1.0,
        'ds1': 50.0,
        'ds2': 1.0,
        'max_iter': 100
    },
    'ASTD': {
        'seasonality_smoothing': 0.7,
        'period': None,
        'alpha': 0.1,
        'beta': 0.1
    },
    'ASTD_Online': {
        'seasonality_smoothing': 0.7,
        'init_window_size': 300,
        'init_ratio': 0.5
    },
    'OnlineSTL': {
        'periods': None,
        'period': None,
        'lam': 0.7,
        'init_window_ratio': 0.5
    },
    'OneShotSTL': {
        'period': 12,
        'shift_window': 0,
        'init_ratio': 0.3
    },
    'STR': {
        'seasonal_periods': None,
        'trend_lambda': 1000.0,
        'seasonal_lambda': 10.0,
        'robust': False,
        'auto_params': False,
        'n_trials': 10
    }
}

# Parameter display names and ordering
PARAM_DISPLAY_NAMES = {
    # LGTD parameters
    'window_size': 'Window Size',
    'error_percentile': 'Error Percentile',
    'trend_selection': 'Trend Selection',
    'lowess_frac': 'LOWESS Fraction',
    'threshold_r2': '$R^2$ Threshold',

    # STL parameters
    'period': 'Period',
    'periods': 'Periods',
    'seasonal': 'Seasonal Window',
    'trend': 'Trend Window',
    'robust': 'Robust',

    # RobustSTL / FastRobustSTL parameters
    'reg1': '$\\lambda_1$ (Trend)',
    'reg2': '$\\lambda_2$ (Seasonal)',
    'K': '$K$ (Bilateral)',
    'H': '$H$ (Trend)',
    'dn1': '$d_{n1}$',
    'dn2': '$d_{n2}$',
    'ds1': '$d_{s1}$',
    'ds2': '$d_{s2}$',
    'max_iter': 'Max Iterations',

    # ASTD parameters
    'seasonality_smoothing': 'Seasonality Smoothing',
    'init_window_size': 'Init Window Size',
    'init_ratio': 'Init Ratio',
    'alpha': 'Alpha',
    'beta': 'Beta',

    # STR parameters
    'seasonal_periods': 'Seasonal Periods',
    'trend_lambda': 'Trend $\\lambda$',
    'seasonal_lambda': 'Seasonal $\\lambda$',
    'auto_params': 'Auto Params',
    'n_trials': 'N Trials',

    # OnlineSTL parameters
    'lam': '$\\lambda$ (Smoothing)',
    'init_window_ratio': 'Init Window Ratio',

    # OneShotSTL parameters
    'shift_window': 'Shift Window',

    # General
    'verbose': 'Verbose'
}


def format_param_value(value):
    """Format parameter value for display in LaTeX."""
    if value is None:
        return ''  # Empty cell instead of dashes
    elif isinstance(value, bool):
        # Use checkmark for True, cross for False
        return '\\checkmark' if value else '$\\times$'
    elif isinstance(value, (int, float)):
        # Format numbers appropriately
        if isinstance(value, int):
            return str(value)
        elif value >= 100:
            return f"{value:.1f}"
        elif value >= 10:
            return f"{value:.2f}"
        else:
            return f"{value:.3f}".rstrip('0').rstrip('.')
    elif isinstance(value, list):
        # Format lists (e.g., periods)
        return '[' + ', '.join(str(v) for v in value) + ']'
    elif isinstance(value, str):
        return value.capitalize()
    else:
        return str(value)


def load_dataset_params(dataset_name):
    """Load parameter configuration for a specific dataset."""
    # Try synthetic dataset params first
    param_file = Path(f'experiments/configs/dataset_params/{dataset_name}_params.json')

    # If not found, try real-world params
    if not param_file.exists():
        param_file = Path(f'experiments/configs/realworld_params/{dataset_name}_params.json')

    if not param_file.exists():
        print(f"⚠️  Warning: Parameter file not found for {dataset_name}")
        return None

    with open(param_file, 'r') as f:
        return json.load(f)


def extract_all_parameters(datasets):
    """
    Extract all unique parameters used across all models and datasets.
    Returns a dict mapping model -> set of parameter names.
    """
    model_params = defaultdict(set)

    for dataset in datasets:
        config = load_dataset_params(dataset)
        if not config:
            continue

        models = config.get('models', {})
        for model_name, model_config in models.items():
            # If 'enabled' key exists, check it; otherwise assume enabled if params exist
            enabled = model_config.get('enabled', True)
            if not enabled:
                continue

            params = model_config.get('params', {})
            if not params:  # Skip if no parameters defined
                continue

            for param_name in params.keys():
                model_params[model_name].add(param_name)

    return model_params


def get_param_order(param_names):
    """Return parameter names sorted by importance/logical order."""
    # Define priority order for common parameters
    priority_params = [
        'period', 'periods', 'seasonal_periods',
        'window_size', 'seasonal', 'trend',
        'seasonality_smoothing', 'lam',
        'error_percentile', 'trend_selection',
        'lowess_frac', 'threshold_r2',
        'reg1', 'reg2', 'trend_lambda', 'seasonal_lambda',
        'K', 'H', 'dn1', 'dn2', 'ds1', 'ds2',
        'init_window_size', 'init_window_ratio', 'init_ratio',
        'shift_window', 'robust', 'auto_params',
        'max_iter', 'n_trials', 'verbose'
    ]

    # Sort: priority params first, then alphabetically
    sorted_params = []
    for p in priority_params:
        if p in param_names:
            sorted_params.append(p)

    # Add remaining params alphabetically
    remaining = sorted(set(param_names) - set(sorted_params))
    sorted_params.extend(remaining)

    return sorted_params


def generate_parameter_table(datasets, output_file='table_model_parameters.tex',
                            models_to_include=None, exclude_params=None):
    """
    Generate comprehensive LaTeX table of model parameters.

    Args:
        datasets: List of dataset names
        output_file: Output file path
        models_to_include: List of models to include (None = all enabled models)
        exclude_params: Set of parameter names to exclude (e.g., {'verbose'})
    """
    if exclude_params is None:
        exclude_params = {'verbose'}  # Typically not interesting for paper

    # Load all dataset configurations
    all_configs = {}
    for dataset in datasets:
        config = load_dataset_params(dataset)
        if config:
            all_configs[dataset] = config

    if not all_configs:
        print("❌ No dataset configurations found!")
        return

    # Extract all parameters across models
    model_params_all = extract_all_parameters(datasets)

    # Filter models if specified
    if models_to_include:
        models = [m for m in models_to_include if m in model_params_all]
    else:
        models = sorted(model_params_all.keys())

    # For each model, collect parameters and check if they vary across datasets
    model_param_configs = {}
    param_varies_across_datasets = defaultdict(lambda: defaultdict(bool))

    for model in models:
        model_param_configs[model] = {}
        # Apply both global and model-specific exclusions
        model_exclusions = MODEL_SPECIFIC_EXCLUSIONS.get(model, set())
        params = sorted(model_params_all[model] - exclude_params - model_exclusions)

        for param in params:
            # Collect values across all datasets
            values = []
            for dataset in datasets:
                if dataset not in all_configs:
                    continue

                model_config = all_configs[dataset]['models'].get(model, {})
                # If 'enabled' key exists, check it; otherwise assume enabled if params exist
                enabled = model_config.get('enabled', True)
                if not enabled:
                    continue

                param_value = model_config.get('params', {}).get(param)
                if param_value is not None:  # Only add if parameter is defined
                    values.append((dataset, param_value))

            # Check if parameter has any non-None values or a non-None default
            has_values = len(values) > 0
            default_value = MODEL_DEFAULTS.get(model, {}).get(param)
            has_default = default_value is not None

            # Only include parameter if it has values OR a non-None default
            if has_values or has_default:
                model_param_configs[model][param] = values

                # Check if parameter varies across datasets
                unique_values = set(str(v) for _, v in values if v is not None)
                param_varies_across_datasets[model][param] = len(unique_values) > 1

    # Generate LaTeX table
    latex_lines = []
    latex_lines.append(r'\begin{table*}[htbp]')
    latex_lines.append(r'\centering')
    latex_lines.append(r'\small')
    latex_lines.append(r'\setlength{\tabcolsep}{3pt}')
    latex_lines.append(r'\caption{Hyperparameter Configurations for All Models Across Datasets}')
    latex_lines.append(r'\label{tab:model_parameters}')

    # Build column specification dynamically
    n_datasets = len(datasets)
    col_spec = 'llc' + 'c' * n_datasets
    latex_lines.append(r'\begin{tabular}{' + col_spec + '}')
    latex_lines.append(r'\toprule')

    # Header row
    header = r'\textbf{Model} & \textbf{Parameter} & \textbf{Varies}'
    for dataset in datasets:
        header += f' & \\textbf{{{dataset}}}'
    header += r' \\'
    latex_lines.append(header)
    latex_lines.append(r'\midrule')

    # Generate rows for each model
    for model_idx, model in enumerate(models):
        model_display = MODEL_NAMES.get(model, model)
        params = get_param_order(model_param_configs[model].keys())

        if not params:
            continue

        for param_idx, param in enumerate(params):
            values = model_param_configs[model][param]
            varies = param_varies_across_datasets[model][param]

            # Create value lookup
            dataset_values = {ds: val for ds, val in values}

            # Format parameter name
            param_display = PARAM_DISPLAY_NAMES.get(param, param.replace('_', ' ').title())

            # Build row
            if param_idx == 0:
                # First parameter: show model name with multirow on separate line
                row = f'\\multirow{{{len(params)}}}{{*}}{{{model_display}}} \n & {param_display}'
            else:
                # Subsequent parameters: empty model column
                row = f' & {param_display}'

            # Add "Varies" indicator
            varies_symbol = '\\checkmark' if varies else '---'
            row += f' & {varies_symbol}'

            # Add values for each dataset
            for dataset in datasets:
                value = dataset_values.get(dataset)
                # If value is not in dataset config, use default from model
                if value is None:
                    default_value = MODEL_DEFAULTS.get(model, {}).get(param)
                    formatted_value = format_param_value(default_value)
                else:
                    formatted_value = format_param_value(value)
                row += f' & {formatted_value}'

            row += r' \\'
            latex_lines.append(row)

        # Add separator between models
        if model_idx < len(models) - 1:
            latex_lines.append(r'\midrule')

    latex_lines.append(r'\bottomrule')
    latex_lines.append(r'\end{tabular}')
    latex_lines.append(r'\end{table*}')

    # Write full LaTeX document
    latex_document = r"""\documentclass{article}

\usepackage{booktabs}
\usepackage{multirow}
\usepackage[margin=0.5in,landscape]{geometry}
\usepackage{amssymb}   % for \checkmark
\usepackage{textcomp}  % extra symbols if needed

\begin{document}

""" + '\n'.join(latex_lines) + r"""

\end{document}
"""

    # Save to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        f.write(latex_document)

    print(f"✅ Parameter table saved to: {output_file}")
    print(f"   Models: {len(models)}")
    print(f"   Datasets: {len(datasets)}")
    print(f"\n📄 To compile:")
    print(f"   pdflatex {output_file}")

    return latex_document


def main():
    parser = argparse.ArgumentParser(
        description='Generate LaTeX table of model parameter configurations'
    )
    parser.add_argument(
        '-o', '--output',
        default='experiments/results/summary/table_model_parameters.tex',
        help='Output LaTeX file (default: experiments/results/summary/table_model_parameters.tex)'
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        help='Datasets to include (default: synth1-synth9)'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        help='Models to include (default: all enabled models)'
    )
    parser.add_argument(
        '--exclude-params',
        nargs='+',
        default=['verbose'],
        help='Parameters to exclude from table (default: verbose)'
    )
    parser.add_argument(
        '--synthetic',
        action='store_true',
        help='Use synthetic datasets (synth1-synth9)'
    )
    parser.add_argument(
        '--realworld',
        action='store_true',
        help='Use real-world datasets (ETTh1, ETTh2, sunspot)'
    )

    args = parser.parse_args()

    # Determine datasets
    if args.datasets:
        datasets = args.datasets
    elif args.realworld:
        datasets = ['ETTh1', 'ETTh2', 'sunspot']
    else:
        # Default: synthetic datasets
        datasets = [f'synth{i}' for i in range(1, 10)]

    print(f"📊 Generating parameter table for datasets: {', '.join(datasets)}")

    # Generate table
    try:
        generate_parameter_table(
            datasets=datasets,
            output_file=args.output,
            models_to_include=args.models,
            exclude_params=set(args.exclude_params) if args.exclude_params else None
        )
    except Exception as e:
        print(f"❌ Error generating table: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
