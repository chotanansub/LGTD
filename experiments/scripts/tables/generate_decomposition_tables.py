#!/usr/bin/env python3
"""
Convert experiment results CSV to LaTeX table.

Usage:
    python scripts/results_to_latex.py results/synthetic/experiment_results_YYYYMMDD_HHMMSS.csv

Or use the latest results file:
    python scripts/results_to_latex.py
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

                # Get period value based on period type
                if period_type == 'Fixed':
                    period_value = meta.get('period', '?')
                elif period_type == 'Transitive':
                    main = meta.get('main_period', '?')
                    trans = meta.get('transition_period', '?')
                    period_value = f'{trans}→{main}'
                elif period_type == 'Variable':
                    n_periods = meta.get('n_periods', '?')
                    period_value = f'{n_periods} periods'
                else:
                    period_value = '?'

                properties[f'synth{i}'] = {
                    'trend_type': trend_type,
                    'period': period_type,
                    'period_value': period_value
                }

    return properties

# Load dataset properties from JSON files (single source of truth)
DATASET_PROPERTIES = load_dataset_properties_from_json()

# -----------------------------
# Configuration
# -----------------------------

# Model display names
MODEL_NAMES = {
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
    'LGTD_Linear': 'batch',
    'LGTD_LOWESS': 'batch',
    'STL': 'batch',
    'RobustSTL': 'batch',
    'ASTD': 'batch',
    'ASTD_Online': 'online',
    'OnlineSTL': 'online',
    'OneShotSTL': 'online',
    'FastRobustSTL': 'batch',
    'STR': 'batch'
}

# Default model order (can be overridden)
DEFAULT_MODEL_ORDER = ['STL', 'STR', 'FastRobustSTL', 'ASTD', 'ASTD_Online', 'OnlineSTL', 'OneShotSTL', 'RobustSTL', 'LGTD']

# Models to exclude from the table
EXCLUDED_MODELS = ['LGTD_Linear', 'LGTD_LOWESS']

# Metrics to include
METRIC_ORDER = ['MSE', 'MAE']

# Components to report
VALUE_COLUMNS = ['trend', 'seasonal', 'residual']

# -----------------------------
# Helper Functions
# -----------------------------

def format_number(x, precision=2):
    """Format number with specified precision."""
    if pd.isna(x):
        return '-'
    return f"{x:.{precision}f}"

def compute_best_values(df, metric_cols):
    """Compute best (minimum) values for each metric and component."""
    best = {}
    for (dataset, metric), group in df.groupby(['dataset', 'metric']):
        for col in metric_cols:
            if col in group.columns:
                best[(dataset, metric, col)] = group[col].min()
    return best

def reshape_results(df):
    """Reshape results to have metric as a column."""
    # Melt the dataframe to have separate rows for each metric
    rows = []

    for _, row in df.iterrows():
        # MSE row
        rows.append({
            'dataset': row['dataset'],
            'trend_type': row['trend_type'],
            'period_type': row['period_type'],
            'model': row['model'],
            'metric': 'MSE',
            'trend': row.get('mse_trend', float('nan')),
            'seasonal': row.get('mse_seasonal', float('nan')),
            'residual': row.get('mse_residual', float('nan')),
        })

        # MAE row
        rows.append({
            'dataset': row['dataset'],
            'trend_type': row['trend_type'],
            'period_type': row['period_type'],
            'model': row['model'],
            'metric': 'MAE',
            'trend': row.get('mae_trend', float('nan')),
            'seasonal': row.get('mae_seasonal', float('nan')),
            'residual': row.get('mae_residual', float('nan')),
        })

    return pd.DataFrame(rows)

# -----------------------------
# LaTeX Generation
# -----------------------------

def generate_latex_table(df, output_file='table_decomposition_results.tex',
                        dataset_order=None, model_order=None,
                        caption='Decomposition Error Comparison Across Synthetic Datasets',
                        bold_best=True, precision=2):
    """
    Generate LaTeX table from experiment results.

    Args:
        df: DataFrame with experiment results
        output_file: Output LaTeX file path
        dataset_order: List of datasets in desired order (default: synth1-9)
        model_order: List of models in desired order (default: DEFAULT_MODEL_ORDER)
        caption: Table caption
        bold_best: Whether to bold the best values
        precision: Number of decimal places
    """

    # Reshape data
    df_reshaped = reshape_results(df)

    # Filter out excluded models
    df_reshaped = df_reshaped[~df_reshaped['model'].isin(EXCLUDED_MODELS)]

    # Set default orders
    if dataset_order is None:
        dataset_order = sorted([d for d in df_reshaped['dataset'].unique() if d in DATASET_PROPERTIES])

    if model_order is None:
        model_order = [m for m in DEFAULT_MODEL_ORDER if m in df_reshaped['model'].unique()]

    # Compute best values for bolding
    best_values = compute_best_values(df_reshaped, VALUE_COLUMNS) if bold_best else {}

    # Generate LaTeX body
    latex_lines = []

    for dataset_idx, dataset in enumerate(dataset_order):
        dataset_df = df_reshaped[df_reshaped['dataset'] == dataset]

        if dataset_df.empty:
            continue

        # Get models present in this dataset
        models = [m for m in model_order if m in dataset_df['model'].unique()]
        metrics = METRIC_ORDER

        rows_per_model = len(metrics)
        total_rows = len(models) * rows_per_model

        # Get dataset properties
        props = DATASET_PROPERTIES.get(dataset, {'trend_type': '?', 'period': '?'})
        trend_type = props['trend_type']
        period = props['period']

        row_idx = 0

        for model_idx, model in enumerate(models):
            model_df = dataset_df[dataset_df['model'] == model]
            model_display = MODEL_NAMES.get(model, model)

            for metric_idx, metric in enumerate(metrics):
                row = model_df[model_df['metric'] == metric]

                if row.empty:
                    continue

                # Format values
                values = {}
                for col in VALUE_COLUMNS:
                    val = row[col].values[0]
                    val_str = format_number(val, precision=precision)

                    # Bold if best value
                    if bold_best and not pd.isna(val):
                        key = (dataset, metric, col)
                        if key in best_values and val == best_values[key]:
                            val_str = f"\\textbf{{{val_str}}}"

                    values[col] = val_str

                # Build LaTeX line (matching reference format)
                if row_idx == 0:
                    # First row: include dataset, trend_type, period
                    line = (
                        f"\\multirow{{{total_rows}}}*{{\\textbf{{{dataset}}}}} & "
                        f"\\multirow{{{total_rows}}}*{{{trend_type}}} & "
                        f"\\multirow{{{total_rows}}}*{{{period}}} \n"
                        f"    & \\multirow{{{rows_per_model}}}*{{{model_display}}} & "
                        f"{metric} & {values['trend']} & {values['seasonal']} & {values['residual']} \\\\"
                    )
                elif metric_idx == 0:
                    # First metric of new model: include model name
                    line = (
                        f"    &  &  & \\multirow{{{rows_per_model}}}*{{{model_display}}} & "
                        f"{metric} & {values['trend']} & {values['seasonal']} & {values['residual']} \\\\"
                    )
                else:
                    # Continuation rows
                    line = (
                        f"    &  &  &  & "
                        f"{metric} & {values['trend']} & {values['seasonal']} & {values['residual']} \\\\"
                    )

                latex_lines.append(line)

                # Add horizontal rule between models
                if metric_idx == rows_per_model - 1 and model_idx < len(models) - 1:
                    latex_lines.append("\\cmidrule{4-8}")

                row_idx += 1

        # Add midrule between datasets
        if dataset_idx < len(dataset_order) - 1:
            latex_lines.append("\\midrule")

    # -----------------------------
    # Full LaTeX document
    # -----------------------------

    latex_document = r"""
\documentclass{article}

\usepackage{booktabs}
\usepackage{graphicx}
\usepackage[margin=0.3in]{geometry}
\usepackage{multirow}

\begin{document}

\begin{table*}[htbp]
\centering
\scriptsize
\setlength{\tabcolsep}{2.5pt}
\renewcommand{\arraystretch}{0.75}
\caption{""" + caption + r"""}
\label{tab:decomposition_results}
\begin{tabular}{lcclcrrr}
\toprule
\textbf{Dataset} & \textbf{Trend Type} & \textbf{Period} & \textbf{Model} & \textbf{Metric} & \textbf{Trend} & \textbf{Seasonal} & \textbf{Residual} \\
\midrule
"""

    latex_document += "\n".join(latex_lines)

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

    print(f"✅ LaTeX table saved to: {output_file}")
    return latex_document

# -----------------------------
# Main
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
                row_data['period_type'] = props['period']

            all_data.append(row_data)

    if not all_data:
        return None

    return pd.DataFrame(all_data)

def main():
    parser = argparse.ArgumentParser(
        description='Convert experiment results CSV to LaTeX table'
    )
    parser.add_argument(
        'csv_file',
        nargs='?',
        help='Path to CSV file or accuracy directory (default: experiments/results/accuracy/synthetic/)'
    )
    parser.add_argument(
        '-o', '--output',
        default='experiments/results/latex_tables/synthetic/table_decomposition_results.tex',
        help='Output LaTeX file (default: experiments/results/latex_tables/synthetic/table_decomposition_results.tex)'
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
            print("❌ No metric files found in experiments/results/accuracy/synthetic/")
            print("   Please run experiments first.")
            sys.exit(1)

        print(f"✅ Loaded {len(df)} rows from individual metric files")

    # Display info
    print(f"   Datasets: {sorted(df['dataset'].unique())}")
    print(f"   Models: {sorted(df['model'].unique())}")

    # Generate LaTeX table
    try:
        generate_latex_table(
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
