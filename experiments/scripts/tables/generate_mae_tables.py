#!/usr/bin/env python3
"""
Generate summary tables with average MAE across datasets.

Creates 3 LaTeX tables:
1. Average MAE across all datasets
2. Average MAE for transitive period datasets only
3. Average MAE for variable period datasets only
"""

import pandas as pd
import numpy as np
from pathlib import Path


def format_value(val, bold_val, is_bold):
    """Format value with optional bold formatting."""
    if pd.isna(val):
        return '--'
    if is_bold and abs(val - bold_val) < 1e-6:
        return f'\\textbf{{{val:.2f}}}'
    return f'{val:.2f}'


def generate_mae_table(
    df: pd.DataFrame,
    period_filter: str = None,
    output_file: str = None,
    table_title: str = None
):
    """
    Generate LaTeX table with average MAE values across all trend types.

    Args:
        df: Experiment results dataframe
        period_filter: 'transitive', 'variable', or None for all
        output_file: Output LaTeX file path
        table_title: Table caption
    """
    # Filter by period type if specified
    if period_filter:
        df_filtered = df[df['period_type'] == period_filter].copy()
    else:
        df_filtered = df.copy()

    # Define model order (matching the wide table)
    model_order = ['STL', 'STR', 'FastRobustSTL', 'ASTD', 'ASTD_Online',
                   'OnlineSTL', 'OneShotSTL', 'LGTD']

    # Model display names for LaTeX
    model_display_names = {
        'ASTD_Online': 'ASTD\\(_{Online}\\)',
        'LGTD_Linear': 'LGTD\\(_{Linear}\\)',
        'LGTD_LOWESS': 'LGTD\\(_{LOWESS}\\)',
    }

    # Columns to average (MAE only)
    mae_columns = {
        'mae_trend': 'Trend',
        'mae_seasonal': 'Seasonal',
        'mae_residual': 'Residual'
    }

    # Build results table - average across ALL trend types for each model
    results = []

    for model in model_order:
        model_data = df_filtered[df_filtered['model'] == model]

        if model_data.empty:
            continue

        row = {
            'model': model,
        }

        # Calculate average MAE for each component across ALL trend types
        for col, label in mae_columns.items():
            if col in model_data.columns:
                row[col] = model_data[col].mean()
            else:
                row[col] = np.nan

        # Calculate overall MAE (average of trend, seasonal, residual)
        row['mae_overall'] = (row['mae_trend'] + row['mae_seasonal'] + row['mae_residual']) / 3.0

        results.append(row)

    results_df = pd.DataFrame(results)

    if results_df.empty:
        print(f"No data found for period_filter={period_filter}")
        return

    # Find best (minimum) values for each column
    best_trend = results_df['mae_trend'].min()
    best_seasonal = results_df['mae_seasonal'].min()
    best_residual = results_df['mae_residual'].min()
    best_overall = results_df['mae_overall'].min()

    # Generate LaTeX table
    latex_lines = []
    latex_lines.append('\\documentclass{article}')
    latex_lines.append('')
    latex_lines.append('\\usepackage{booktabs}')
    latex_lines.append('\\usepackage{multirow}')
    latex_lines.append('\\usepackage{graphicx}')
    latex_lines.append('')
    latex_lines.append('\\begin{document}')
    latex_lines.append('')
    latex_lines.append('\\begin{table}[htbp]')
    latex_lines.append('\\centering')
    latex_lines.append('\\footnotesize')
    latex_lines.append('\\setlength{\\tabcolsep}{6pt}')
    latex_lines.append('\\renewcommand{\\arraystretch}{1.2}')
    latex_lines.append(f'\\caption*{{{table_title}}}')
    latex_lines.append('\\begin{tabular}{lcccc}')
    latex_lines.append('\\toprule')
    latex_lines.append('\\textbf{Model} & \\textbf{Trend MAE} & \\textbf{Seasonal MAE} & \\textbf{Residual MAE} & \\textbf{Overall MAE} \\\\')
    latex_lines.append('\\midrule')

    # Add rows for each model
    for _, row in results_df.iterrows():
        model_name = row['model']

        # Get display name (with subscripts for variants)
        model_display = model_display_names.get(model_name, model_name)

        # Format values with bold for best
        trend_val = format_value(row['mae_trend'], best_trend,
                                row['mae_trend'] == best_trend)
        seasonal_val = format_value(row['mae_seasonal'], best_seasonal,
                                   row['mae_seasonal'] == best_seasonal)
        residual_val = format_value(row['mae_residual'], best_residual,
                                   row['mae_residual'] == best_residual)
        overall_val = format_value(row['mae_overall'], best_overall,
                                   row['mae_overall'] == best_overall)

        latex_lines.append(f'{model_display} & {trend_val} & {seasonal_val} & {residual_val} & {overall_val} \\\\')

    latex_lines.append('\\bottomrule')
    latex_lines.append('\\end{tabular}')
    latex_lines.append('\\end{table}')
    latex_lines.append('')
    latex_lines.append('\\end{document}')

    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(latex_lines))

    print(f"Generated: {output_file}")


def main():
    # Load experiment results
    results_file = Path('experiments/results/benchmarks/synthetic_benchmarks.csv')

    if not results_file.exists():
        print(f"Error: {results_file} not found!")
        print("Please run experiments first: python experiments/runners/experiment_runner.py")
        return

    df = pd.read_csv(results_file)

    print("="*70)
    print("Generating MAE Summary Tables")
    print("="*70)
    print(f"\nLoaded {len(df)} experiment results from {results_file}")
    print(f"Datasets: {df['dataset'].nunique()}")
    print(f"Models: {df['model'].nunique()}")
    print()

    # Generate 3 tables
    print("Generating tables...")
    print()

    # Create output directory
    output_dir = Path('experiments/results/summary')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Table 1: All datasets
    generate_mae_table(
        df=df,
        period_filter=None,
        output_file=str(output_dir / 'table_mae_summary_all.tex'),
        table_title='Average MAE Across All Datasets'
    )

    # Table 2: Transitive period only
    generate_mae_table(
        df=df,
        period_filter='transitive',
        output_file=str(output_dir / 'table_mae_summary_transitive.tex'),
        table_title='Average MAE for Transitive Period Datasets'
    )

    # Table 3: Variable period only
    generate_mae_table(
        df=df,
        period_filter='variable',
        output_file=str(output_dir / 'table_mae_summary_variable.tex'),
        table_title='Average MAE for Variable Period Datasets'
    )

    print()
    print("="*70)
    print("Summary Tables Generated!")
    print("="*70)
    print(f"\nGenerated files in {output_dir}/:")
    print("  1. table_mae_summary_all.tex        - Average MAE across all datasets")
    print("  2. table_mae_summary_transitive.tex - Average MAE for transitive period")
    print("  3. table_mae_summary_variable.tex   - Average MAE for variable period")
    print()


if __name__ == '__main__':
    main()
