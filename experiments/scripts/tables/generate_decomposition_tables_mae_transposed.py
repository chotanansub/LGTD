#!/usr/bin/env python3
"""
Convert experiment results to transposed MAE-only LaTeX table.

This generates a table organized by:
- Rows: Trend types (Linear, Inverted-V, Piecewise) × Period types (Fixed, Transitive, Variable)
- Columns: Models (STL, STR, FastRobustSTL, ASTD, ASTD_Online, OnlineSTL, OneShotSTL, LGTD)
- Metrics: MAE only for components (Trend, Seasonal, Residual)

Usage:
    python scripts/tables/generate_decomposition_tables_mae_transposed.py
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
MODEL_ORDER = ['LGTD', 'STL', 'STR', 'FastRobustSTL', 'ASTD', 'ASTD_Online', 'OnlineSTL', 'OneShotSTL']

# Model display names for LaTeX
MODEL_DISPLAY_NAMES = {
    'LGTD': 'LGTD',
    'LGTD_Linear': 'LGTD\\(_{Linear}\\)',
    'LGTD_LOWESS': 'LGTD\\(_{LOWESS}\\)',
    'STL': 'STL',
    'RobustSTL': 'RobustSTL',
    'ASTD': '$ASTD$',
    'ASTD_Online': '$ASTD_{Online}$',
    'OnlineSTL': 'OnlineSTL',
    'OneShotSTL': 'OneShotSTL',
    'FastRobustSTL': 'FastRobustSTL',
    'STR': 'STR'
}

# Mapping from file model names to standardized names
MODEL_NAME_MAPPING = {
    'lgtd': 'LGTD',
    'LGTD': 'LGTD',
    'lgtd_linear': 'LGTD_Linear',
    'LGTD_LINEAR': 'LGTD_Linear',
    'lgtd_lowess': 'LGTD_LOWESS',
    'LGTD_LOWESS': 'LGTD_LOWESS',
    'stl': 'STL',
    'STL': 'STL',
    'str': 'STR',
    'STR': 'STR',
    'fast_robust_stl': 'FastRobustSTL',
    'FAST_ROBUST_STL': 'FastRobustSTL',
    'FastRobustSTL': 'FastRobustSTL',
    'astd': 'ASTD',
    'ASTD': 'ASTD',
    'astd_online': 'ASTD_Online',
    'ASTD_ONLINE': 'ASTD_Online',
    'ASTD_Online': 'ASTD_Online',
    'online_stl': 'OnlineSTL',
    'ONLINE_STL': 'OnlineSTL',
    'OnlineSTL': 'OnlineSTL',
    'oneshot_stl': 'OneShotSTL',
    'ONESHOT_STL': 'OneShotSTL',
    'OneShotSTL': 'OneShotSTL',
}

# Models to exclude from the table
EXCLUDED_MODELS = ['LGTD_Linear', 'LGTD_LOWESS']

# Trend types in order
TREND_TYPES = ['Linear', 'Inverted-V', 'Piecewise']

# Period types in order
PERIOD_TYPES = ['Fixed', 'Transitive', 'Variable']

# Components to report
COMPONENTS = ['trend', 'seasonal', 'residual']


# -----------------------------
# Helper Functions
# -----------------------------

def format_number(x, precision=2):
    """Format number with specified precision."""
    if pd.isna(x):
        return '-'
    return f"{x:.{precision}f}"


def reshape_to_transposed_format(df, include_mse=True):
    """Reshape data to transposed format with models as columns."""
    # Filter out excluded models
    df = df[~df['model'].isin(EXCLUDED_MODELS)].copy()

    # Normalize trend and period types to match display format (handle both lowercase and already-capitalized)
    trend_map = {
        'linear': 'Linear', 'Linear': 'Linear',
        'inverted_v': 'Inverted-V', 'Inverted-V': 'Inverted-V',
        'piecewise': 'Piecewise', 'Piecewise': 'Piecewise'
    }
    period_map = {
        'fixed': 'Fixed', 'Fixed': 'Fixed',
        'transitive': 'Transitive', 'Transitive': 'Transitive',
        'variable': 'Variable', 'Variable': 'Variable'
    }
    df['trend_type'] = df['trend_type'].map(trend_map)
    df['period_type'] = df['period_type'].map(period_map)

    # Create rows: one per (trend_type, period_type, component, metric)
    rows = []

    metrics_to_include = ['MSE', 'MAE'] if include_mse else ['MAE']

    for trend_type in TREND_TYPES:
        for period_type in PERIOD_TYPES:
            for component in COMPONENTS:
                for metric in metrics_to_include:
                    row = {
                        'trend_type': trend_type,
                        'period_type': period_type,
                        'component': component,
                        'metric': metric
                    }

                    # Add data for each model
                    for model in MODEL_ORDER:
                        # Find the data for this combination
                        model_data = df[
                            (df['trend_type'] == trend_type) &
                            (df['period_type'] == period_type) &
                            (df['model'] == model)
                        ]

                        if not model_data.empty:
                            metric_col = f'{metric.lower()}_{component}'
                            row[model] = model_data.iloc[0].get(metric_col, float('nan'))
                        else:
                            row[model] = float('nan')

                    rows.append(row)

    return pd.DataFrame(rows)


def generate_latex_table_transposed(
    df,
    output_file='table_decomposition_mae_transposed.tex',
    caption='Decomposition errors (MSE/MAE) across synthetic datasets',
    bold_best=True,
    precision=2,
    include_mse=True
):
    """
    Generate transposed LaTeX table with models as columns.

    Args:
        df: DataFrame with experiment results
        output_file: Output LaTeX file path
        caption: Table caption
        bold_best: Whether to bold the best values
        precision: Number of decimal places
        include_mse: Whether to include MSE in addition to MAE
    """

    # Reshape to transposed format
    df_transposed = reshape_to_transposed_format(df, include_mse=include_mse)

    if df_transposed.empty:
        print("No data to generate table")
        return

    # Compute best and second-best values for bolding/underlining (minimum for each row)
    best_values = {}
    second_best_values = {}
    if bold_best:
        for idx, row in df_transposed.iterrows():
            trend = row['trend_type']
            period = row['period_type']
            comp = row['component']
            metric = row['metric']

            # Get all model values for this row
            model_values = [row[model] for model in MODEL_ORDER if model in row]
            model_values = [v for v in model_values if not pd.isna(v)]

            if model_values:
                # Sort to get best and second best
                sorted_values = sorted(model_values)
                best_values[(trend, period, comp, metric)] = sorted_values[0]
                # Only set second best if there are at least 2 unique values
                if len(sorted_values) >= 2 and sorted_values[0] != sorted_values[1]:
                    second_best_values[(trend, period, comp, metric)] = sorted_values[1]

    # Generate LaTeX body
    latex_lines = []

    for trend_idx, trend_type in enumerate(TREND_TYPES):
        trend_df = df_transposed[df_transposed['trend_type'] == trend_type]

        if trend_df.empty:
            continue

        # Count rows for this trend type
        n_rows = len(trend_df)
        rows_per_period = len(COMPONENTS) * (2 if include_mse else 1)

        for row_idx, (_, row) in enumerate(trend_df.iterrows()):
            period_type = row['period_type']
            component = row['component']
            metric = row['metric']

            # Build line
            if row_idx == 0:
                # First row for this trend: include rotated trend label
                line_parts = [f"\\multirow{{{n_rows}}}{{*}}{{\\rotatebox{{90}}{{\\textbf{{{trend_type}}}}}}}"]
            else:
                line_parts = ['']

            # Add period type (only show once per period group)
            # Calculate if this is the first row of a period
            period_start = row_idx % rows_per_period == 0
            if period_start:
                line_parts.append(f"\\multirow{{{rows_per_period}}}{{*}}{{\\textbf{{{period_type}}}}}")
            else:
                line_parts.append('')

            # Add component name (show once per component group)
            comp_start = row_idx % (2 if include_mse else 1) == 0
            if comp_start and include_mse:
                component_display = component.capitalize()
                line_parts.append(f"\\multirow{{2}}{{*}}{{{component_display}}}")
            elif not include_mse:
                component_display = component.capitalize()
                line_parts.append(component_display)
            else:
                line_parts.append('')

            # Add metric name
            line_parts.append(metric)

            # Add data for each model
            for model in MODEL_ORDER:
                value = row[model]

                # Format value
                val_str = format_number(value, precision=precision)

                # Bold if best, underline if second best
                if bold_best and not pd.isna(value):
                    best_key = (trend_type, period_type, component, metric)
                    if best_key in best_values and value == best_values[best_key]:
                        val_str = f"\\textbf{{{val_str}}}"
                    elif best_key in second_best_values and value == second_best_values[best_key]:
                        val_str = f"\\underline{{{val_str}}}"

                line_parts.append(val_str)

            # Join with &
            line = ' & '.join(line_parts) + ' \\\\'

            # Add spacing after MAE rows (except the last component in each period)
            is_mae_row = metric == 'MAE'
            component_position = (row_idx % rows_per_period) // (2 if include_mse else 1)
            is_last_component_in_period = component_position == len(COMPONENTS) - 1

            if is_mae_row and not is_last_component_in_period:
                line += ' \\addlinespace[2pt]'

            latex_lines.append(line)

            # Add cmidrule between periods within the same trend
            if is_mae_row and is_last_component_in_period and row_idx < n_rows - 1:
                # Count number of columns: 4 fixed + n_models
                n_cols = 4 + len(MODEL_ORDER)
                latex_lines.append(f' \\cmidrule(lr){{2-{n_cols}}}')

        # Add midrule between trend types
        if trend_idx < len(TREND_TYPES) - 1:
            latex_lines.append('\\midrule')

    # -----------------------------
    # Full LaTeX document
    # -----------------------------

    # Build column spec: l (trend) + l (period) + l (component) + l (metric) + Y for each model
    n_models = len(MODEL_ORDER)
    col_spec = 'llll' + 'Y' * n_models

    # Build model header
    model_headers = [MODEL_DISPLAY_NAMES.get(m, m) for m in MODEL_ORDER]
    model_header_line = ' & '.join(['\\textbf{Trend}', '\\textbf{Period}', '\\textbf{Comp.}', '\\textbf{Metric}'] + model_headers)

    latex_document = r"""
\begin{table*}[!p]
\centering
\footnotesize
\setlength{\tabcolsep}{2pt}
\renewcommand{\arraystretch}{1.0}
\caption{""" + caption + r"""}
\label{tab:decomposition_transposed}
\begin{tabularx}{\textwidth}{""" + col_spec + r"""}
\toprule
""" + model_header_line + r""" \\
\midrule
"""

    latex_document += '\n'.join(latex_lines)

    latex_document += r"""
\bottomrule
\end{tabularx}
\end{table*}
"""

    # Save to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        f.write(latex_document)

    print(f"✅ Transposed LaTeX table saved to: {output_file}")
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

            # Normalize model name
            normalized_model = MODEL_NAME_MAPPING.get(model, model)

            row_data = {
                'dataset': dataset_name,
                'model': normalized_model
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
        description='Convert experiment results to transposed MAE-only LaTeX table'
    )
    parser.add_argument(
        'csv_file',
        nargs='?',
        help='Path to CSV file or accuracy directory (default: experiments/results/accuracy/synthetic/)'
    )
    parser.add_argument(
        '-o', '--output',
        default='experiments/results/latex_tables/table_decomposition_mae_transposed.tex',
        help='Output LaTeX file (default: experiments/results/latex_tables/table_decomposition_mae_transposed.tex)'
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
        default='Decomposition errors (MSE/MAE) across synthetic datasets.',
        help='Table caption'
    )
    parser.add_argument(
        '--mae-only',
        action='store_true',
        help='Include only MAE (exclude MSE)'
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
        generate_latex_table_transposed(
            df,
            output_file=args.output,
            caption=args.caption,
            bold_best=not args.no_bold,
            precision=args.precision,
            include_mse=not args.mae_only
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
