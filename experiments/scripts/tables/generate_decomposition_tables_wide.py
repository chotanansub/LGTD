#!/usr/bin/env python3
"""
Convert experiment results CSV to wide-format LaTeX table.

This generates a landscape table organized by:
- Rows: Trend types (Linear, Inverted-V, Piecewise) with models as sub-rows
- Columns: Period types (Fixed, Transitive, Variable) with components (Trend, Seasonal, Residual)

Usage:
    python scripts/results_to_latex_wide.py
    python scripts/results_to_latex_wide.py results/synthetic/experiment_results.csv
"""

import pandas as pd
import sys
import json
from pathlib import Path
import argparse


# -----------------------------
# Helper Functions - Load from JSON
# -----------------------------

def load_dataset_properties_from_json():
    """Load dataset properties from JSON metadata files as single source of truth."""
    datasets_dir = Path('data/synthetic/datasets')
    properties = {}

    # Mapping from JSON values to display values
    trend_type_map = {
        'linear': 'Linear',
        'inverted_v': 'Inverted-V',
        'piecewise': 'Piecewise'
    }

    period_type_map = {
        'fixed': 'Fixed',
        'transitive': 'Transitive',
        'variable': 'Variable'
    }

    for i in range(1, 10):
        json_file = datasets_dir / f'synth{i}_data.json'
        if json_file.exists():
            with open(json_file, 'r') as f:
                data = json.load(f)
                meta = data.get('meta', {})

                trend_type = trend_type_map.get(meta.get('trend_type', ''), '?')
                period_type = period_type_map.get(meta.get('period_type', ''), '?')

                properties[f'synth{i}'] = {
                    'trend_type': trend_type,
                    'period_type': period_type,
                }

    return properties


# Load dataset properties from JSON files (single source of truth)
DATASET_PROPERTIES = load_dataset_properties_from_json()

# -----------------------------
# Configuration
# -----------------------------

# Model display names and order
MODEL_ORDER = ['STL', 'STR', 'FastRobustSTL', 'ASTD', 'ASTD_Online', 'OnlineSTL', 'OneShotSTL', 'LGTD']

# Model display names for LaTeX
MODEL_DISPLAY_NAMES = {
    'LGTD': 'LGTD',
    'LGTD_Linear': 'LGTD\\(_{Linear}\\)',
    'LGTD_LOWESS': 'LGTD\\(_{LOWESS}\\)',
    'STL': 'STL',
    'RobustSTL': 'RobustSTL',
    'ASTD': 'ASTD',
    'ASTD_Online': 'ASTD\\(_{Online}\\)',
    'OnlineSTL': 'OnlineSTL',
    'OneShotSTL': 'OneShotSTL',
    'FastRobustSTL': 'FastRobustSTL',
    'STR': 'STR'
}

# Model types for display
MODEL_TYPES_DISPLAY = {
    'LGTD': 'batch',
    'STL': 'batch',
    'RobustSTL': 'batch',
    'ASTD': 'batch',
    'ASTD_Online': 'online',
    'OnlineSTL': 'online',
    'OneShotSTL': 'online',
    'FastRobustSTL': 'batch',
    'STR': 'batch'
}

# Models to exclude from the table
EXCLUDED_MODELS = ['LGTD_Linear', 'LGTD_LOWESS']

# Trend types in order
TREND_TYPES = ['Linear', 'Inverted-V', 'Piecewise']

# Period types in order
PERIOD_TYPES = ['Fixed', 'Transitive', 'Variable']

# Components to report
COMPONENTS = ['trend', 'seasonal', 'residual']

# Metrics to show
METRICS = ['MSE', 'MAE']


# -----------------------------
# Helper Functions
# -----------------------------

def format_number(x, precision=2):
    """Format number with specified precision."""
    if pd.isna(x):
        return '-'
    return f"{x:.{precision}f}"


def compute_best_values_per_period(df):
    """Compute best (minimum) values for each (trend_type, period_type, component, metric)."""
    best = {}
    for (trend, period, comp), group in df.groupby(['trend_type', 'period_type', 'component']):
        for metric in METRICS:
            col_name = f'{metric.lower()}_{comp}'
            if col_name in group.columns:
                best[(trend, period, comp, metric)] = group[col_name].min()
    return best


def reshape_to_wide_format(df):
    """Reshape data to wide format with one row per model per trend type."""
    # Filter out excluded models
    df = df[~df['model'].isin(EXCLUDED_MODELS)].copy()

    # Normalize trend and period types to match display format
    trend_map = {'linear': 'Linear', 'inverted_v': 'Inverted-V', 'piecewise': 'Piecewise'}
    period_map = {'fixed': 'Fixed', 'transitive': 'Transitive', 'variable': 'Variable'}
    df['trend_type'] = df['trend_type'].map(trend_map)
    df['period_type'] = df['period_type'].map(period_map)

    # Create rows: one per (trend_type, model)
    rows = []

    for trend_type in TREND_TYPES:
        trend_df = df[df['trend_type'] == trend_type]

        if trend_df.empty:
            continue

        # Get models for this trend type
        models = [m for m in MODEL_ORDER if m in trend_df['model'].unique()]

        for model in models:
            model_df = trend_df[trend_df['model'] == model]

            row = {
                'trend_type': trend_type,
                'model': model
            }

            # Add data for each period type
            for period_type in PERIOD_TYPES:
                period_df = model_df[model_df['period_type'] == period_type]

                if not period_df.empty:
                    # Get the first (and should be only) row for this combination
                    data = period_df.iloc[0]

                    for component in COMPONENTS:
                        for metric in METRICS:
                            col_name = f'{metric.lower()}_{component}'
                            key = f'{period_type}_{component}_{metric}'
                            row[key] = data.get(col_name, float('nan'))
                else:
                    # No data for this period type
                    for component in COMPONENTS:
                        for metric in METRICS:
                            key = f'{period_type}_{component}_{metric}'
                            row[key] = float('nan')

            rows.append(row)

    return pd.DataFrame(rows)


def generate_latex_table_wide(
    df,
    output_file='table_decomposition_results_wide.tex',
    caption='Decomposition Error Comparison Across Synthetic Datasets',
    bold_best=True,
    precision=2
):
    """
    Generate wide-format LaTeX table.

    Args:
        df: DataFrame with experiment results
        output_file: Output LaTeX file path
        caption: Table caption
        bold_best: Whether to bold the best values
        precision: Number of decimal places
    """

    # Reshape to wide format
    df_wide = reshape_to_wide_format(df)

    if df_wide.empty:
        print("No data to generate table")
        return

    # Compute best values for bolding
    best_values = {}
    if bold_best:
        for trend_type in TREND_TYPES:
            for period_type in PERIOD_TYPES:
                for component in COMPONENTS:
                    for metric in METRICS:
                        # Find minimum value across all models for this combination
                        col_key = f'{period_type}_{component}_{metric}'
                        trend_data = df_wide[df_wide['trend_type'] == trend_type]
                        if col_key in trend_data.columns:
                            min_val = trend_data[col_key].min()
                            if not pd.isna(min_val):
                                best_values[(trend_type, period_type, component, metric)] = min_val

    # Generate LaTeX body
    latex_lines = []

    for trend_idx, trend_type in enumerate(TREND_TYPES):
        trend_df = df_wide[df_wide['trend_type'] == trend_type]

        if trend_df.empty:
            continue

        models = trend_df['model'].tolist()
        n_models = len(models)

        for model_idx, (_, row) in enumerate(trend_df.iterrows()):
            model = row['model']
            model_display = MODEL_DISPLAY_NAMES.get(model, model)

            # Build line
            if model_idx == 0:
                # First model row: include rotated trend label
                line_parts = [f"\\multirow{{{n_models}}}{{*}}{{\\rotatebox{{90}}{{\\textbf{{{trend_type}}}}}}}"]
            else:
                line_parts = ['']

            # Add model name
            line_parts.append(model_display)

            # Add data for each period type
            for period_type in PERIOD_TYPES:
                for component in COMPONENTS:
                    for metric in METRICS:
                        col_key = f'{period_type}_{component}_{metric}'
                        value = row[col_key]

                        # Format value
                        val_str = format_number(value, precision=precision)

                        # Bold if best
                        if bold_best and not pd.isna(value):
                            best_key = (trend_type, period_type, component, metric)
                            if best_key in best_values and value == best_values[best_key]:
                                val_str = f"\\textbf{{{val_str}}}"

                        line_parts.append(val_str)

            # Join with &
            line = ' & '.join(line_parts) + ' \\\\'
            latex_lines.append(line)

        # Add midrule between trend types
        if trend_idx < len(TREND_TYPES) - 1:
            latex_lines.append('\\midrule')

    # -----------------------------
    # Full LaTeX document
    # -----------------------------

    latex_document = r"""
\documentclass{article}

\usepackage{booktabs}
\usepackage{graphicx}
\usepackage[margin=0.3in, landscape]{geometry}
\usepackage{multirow}

\begin{document}

\begin{table*}[htbp]
\centering
\footnotesize
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.2}
\caption{""" + caption + r"""}
\label{tab:decomposition_results}
\begin{tabular}{llcccccccccccccccccc}
\toprule
& & \multicolumn{6}{c}{\textbf{Fixed Period}} & \multicolumn{6}{c}{\textbf{Transitive Period}} & \multicolumn{6}{c}{\textbf{Variable Period}} \\
\cmidrule(lr){3-8} \cmidrule(lr){9-14} \cmidrule(lr){15-20}
\textbf{Trend} & \textbf{Model} & \multicolumn{2}{c}{\textbf{Trend}} & \multicolumn{2}{c}{\textbf{Seasonal}} & \multicolumn{2}{c}{\textbf{Residual}} & \multicolumn{2}{c}{\textbf{Trend}} & \multicolumn{2}{c}{\textbf{Seasonal}} & \multicolumn{2}{c}{\textbf{Residual}} & \multicolumn{2}{c}{\textbf{Trend}} & \multicolumn{2}{c}{\textbf{Seasonal}} & \multicolumn{2}{c}{\textbf{Residual}} \\
\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8} \cmidrule(lr){9-10} \cmidrule(lr){11-12} \cmidrule(lr){13-14} \cmidrule(lr){15-16} \cmidrule(lr){17-18} \cmidrule(lr){19-20}
& & \textbf{MSE} & \textbf{MAE} & \textbf{MSE} & \textbf{MAE} & \textbf{MSE} & \textbf{MAE} & \textbf{MSE} & \textbf{MAE} & \textbf{MSE} & \textbf{MAE} & \textbf{MSE} & \textbf{MAE} & \textbf{MSE} & \textbf{MAE} & \textbf{MSE} & \textbf{MAE} & \textbf{MSE} & \textbf{MAE} \\
\midrule
"""

    latex_document += '\n'.join(latex_lines)

    latex_document += r"""
\bottomrule
\end{tabular}
\end{table*}

\end{document}
"""

    # Save to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        f.write(latex_document)

    print(f"✅ Wide-format LaTeX table saved to: {output_file}")
    return latex_document


# -----------------------------
# Helper Functions
# -----------------------------

def load_metrics_from_individual_files(metrics_dir='experiments/results/accuracy/synthetic'):
    """
    Load and combine individual metric CSV files into a single DataFrame.

    Args:
        metrics_dir: Directory containing individual metric CSV files

    Returns:
        Combined DataFrame with all metrics
    """
    metrics_path = Path(metrics_dir)

    if not metrics_path.exists():
        return None

    all_data = []
    csv_files = list(metrics_path.glob('*_metrics.csv'))

    for csv_file in csv_files:
        # Extract dataset name from filename (e.g., synth1_linear_fixed_metrics.csv -> synth1)
        dataset_name = csv_file.stem.split('_')[0]  # Get 'synth1' from 'synth1_linear_fixed_metrics'

        # Load the CSV
        df = pd.read_csv(csv_file)

        # Pivot the data to get mse_trend, mse_seasonal, mae_trend, etc.
        for model in df['model'].unique():
            model_df = df[df['model'] == model]

            row_data = {
                'dataset': dataset_name,
                'model': model
            }

            # Extract MSE values
            mse_row = model_df[model_df['metric'] == 'MSE']
            if not mse_row.empty:
                mse_row = mse_row.iloc[0]
                row_data['mse_trend'] = mse_row.get('trend', float('nan'))
                row_data['mse_seasonal'] = mse_row.get('seasonal', float('nan'))
                row_data['mse_residual'] = mse_row.get('residual', float('nan'))

            # Extract MAE values
            mae_row = model_df[model_df['metric'] == 'MAE']
            if not mae_row.empty:
                mae_row = mae_row.iloc[0]
                row_data['mae_trend'] = mae_row.get('trend', float('nan'))
                row_data['mae_seasonal'] = mae_row.get('seasonal', float('nan'))
                row_data['mae_residual'] = mae_row.get('residual', float('nan'))

            # Add dataset properties from JSON
            if dataset_name in DATASET_PROPERTIES:
                props = DATASET_PROPERTIES[dataset_name]
                row_data['trend_type'] = props['trend_type']
                row_data['period_type'] = props['period_type']

            all_data.append(row_data)

    if not all_data:
        return None

    return pd.DataFrame(all_data)

# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Convert experiment results CSV to wide-format LaTeX table'
    )
    parser.add_argument(
        'csv_file',
        nargs='?',
        help='Path to CSV file or accuracy directory (default: experiments/results/synthetic/accuracy/)'
    )
    parser.add_argument(
        '-o', '--output',
        default='experiments/results/latex_tables/synthetic/table_decomposition_results_wide.tex',
        help='Output LaTeX file (default: experiments/results/latex_tables/synthetic/table_decomposition_results_wide.tex)'
    )
    parser.add_argument(
        '--no-bold',
        action='store_true',
        help='Do not bold best values'
    )
    parser.add_argument(
        '--precision',
        type=int,
        default=2,
        help='Decimal precision (default: 2)'
    )
    parser.add_argument(
        '--caption',
        default='Decomposition Error Comparison Across Synthetic Datasets',
        help='Table caption'
    )

    args = parser.parse_args()

    # Load data
    df = None

    if args.csv_file:
        csv_path = Path(args.csv_file)
        if csv_path.is_file():
            # Load single CSV file
            try:
                df = pd.read_csv(csv_path)
                print(f"✅ Loaded {len(df)} rows from {csv_path.name}")
            except Exception as e:
                print(f"❌ Error loading CSV: {e}")
                sys.exit(1)
        else:
            print(f"❌ File not found: {csv_path}")
            sys.exit(1)
    else:
        # Load from individual metric files
        print("📊 Loading metrics from individual CSV files...")
        df = load_metrics_from_individual_files('experiments/results/accuracy/synthetic')

        if df is None:
            print("❌ No metric files found in experiments/results/synthetic/accuracy/")
            print("   Please run experiments first.")
            sys.exit(1)

        print(f"✅ Loaded {len(df)} rows from individual metric files")

    # Display info
    print(f"   Datasets: {sorted(df['dataset'].unique())}")
    print(f"   Models: {sorted(df['model'].unique())}")

    # Generate LaTeX table
    try:
        generate_latex_table_wide(
            df,
            output_file=args.output,
            caption=args.caption,
            bold_best=not args.no_bold,
            precision=args.precision
        )

        print(f"\n📄 To compile the LaTeX:")
        print(f"   pdflatex {args.output}")

    except Exception as e:
        print(f"❌ Error generating LaTeX: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
