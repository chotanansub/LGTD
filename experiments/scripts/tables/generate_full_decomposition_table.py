#!/usr/bin/env python3
"""
Generate comprehensive LaTeX table of MAE decomposition errors for synthetic datasets.

This script loads decomposition error metrics from CSV files and generates
a comprehensive LaTeX table organized by:
- Trend type (Linear, Inverse-V, Piecewise)
- Period regime (Fixed, Transitive, Variable)
- Component (Trend, Seasonal, Residual)
- Metric (MAE only)

Usage:
    python generate_full_decomposition_table.py
"""

import sys
import pandas as pd
from pathlib import Path
from typing import Dict, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Configuration
DATA_DIR = project_root / "experiments" / "results" / "accuracy" / "synthetic"
OUTPUT_DIR = project_root / "experiments" / "results" / "latex_tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Dataset mapping: (dataset_file_base, trend_type, period_regime)
DATASET_MAP = {
    'synth1_linear_fixed': ('Linear', 'Fixed'),
    'synth2_inverted_v_fixed': ('Inverse-V', 'Fixed'),
    'synth3_piecewise_fixed': ('Piecewise', 'Fixed'),
    'synth4_linear_transitive': ('Linear', 'Transitive'),
    'synth5_inverted_v_transitive': ('Inverse-V', 'Transitive'),
    'synth6_piecewise_transitive': ('Piecewise', 'Transitive'),
    'synth7_linear_variable': ('Linear', 'Variable'),
    'synth8_inverted_v_variable': ('Inverse-V', 'Variable'),
    'synth9_piecewise_variable': ('Piecewise', 'Variable'),
}

# Model display names and ordering
MODEL_ORDER = ['STL', 'STR', 'FAST_ROBUST_STL', 'ASTD', 'ASTD_ONLINE',
               'ONLINE_STL', 'ONESHOT_STL', 'LGTD']

MODEL_DISPLAY_NAMES = {
    'STL': 'STL',
    'STR': 'STR',
    'FAST_ROBUST_STL': 'FastRobustSTL',
    'ASTD': 'ASTD',
    'ASTD_ONLINE': 'ASTD$_{Online}$',
    'ONLINE_STL': 'OnlineSTL',
    'ONESHOT_STL': 'OneShotSTL',
    'LGTD': 'LGTD',
}

# Component ordering
COMPONENT_ORDER = ['trend', 'seasonal', 'residual']
COMPONENT_DISPLAY_NAMES = {
    'trend': 'Trend',
    'seasonal': 'Seasonal',
    'residual': 'Residual',
}


def load_metrics(dataset_name: str) -> Optional[pd.DataFrame]:
    """Load metrics CSV file for a dataset."""
    csv_file = DATA_DIR / f"{dataset_name}_metrics.csv"
    if not csv_file.exists():
        print(f"Warning: {csv_file} not found")
        return None

    df = pd.read_csv(csv_file)
    return df


def extract_model_mae(df: Optional[pd.DataFrame], model_name: str) -> Optional[Dict[str, float]]:
    """
    Extract MAE metrics for a specific model.

    Returns:
        Dict with structure: {component: mae_value}
    """
    if df is None:
        return None

    model_data = df[df['model'] == model_name]
    if model_data.empty:
        return None

    # Get MAE row
    mae_row = model_data[model_data['metric'] == 'MAE']
    if mae_row.empty:
        return None

    result = {}
    for component in COMPONENT_ORDER:
        if component in mae_row.columns:
            result[component] = mae_row[component].values[0]

    return result


def format_value(value: Optional[float]) -> str:
    """Format a numeric value for LaTeX table."""
    if value is None or pd.isna(value):
        return '{-}'

    # Format with 2 decimal places
    return f"{value:.2f}"


def generate_latex_table() -> str:
    """Generate the complete LaTeX table (MAE only)."""

    # Organize data by trend type -> period regime -> dataset
    organized_data = {}
    for dataset_name, (trend_type, period_regime) in DATASET_MAP.items():
        if trend_type not in organized_data:
            organized_data[trend_type] = {}
        if period_regime not in organized_data[trend_type]:
            organized_data[trend_type][period_regime] = {}

        # Load metrics
        df = load_metrics(dataset_name)
        if df is not None:
            organized_data[trend_type][period_regime] = df

    # Start building LaTeX
    latex_lines = []
    latex_lines.append(r"\begin{table*}[t]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\small")
    latex_lines.append(r"\setlength{\tabcolsep}{4pt}")
    latex_lines.append(r"\begin{tabular}{ll l *{8}{S}}")
    latex_lines.append(r"\toprule")

    # Header
    header_parts = [
        r"\textbf{Trend} & \textbf{Period} & \textbf{Component}"
    ]
    for model in MODEL_ORDER:
        header_parts.append(f"& \\textbf{{{MODEL_DISPLAY_NAMES[model]}}}")
    header_parts.append(r"\\")
    latex_lines.append(" ".join(header_parts))
    latex_lines.append(r"\midrule")
    latex_lines.append("")

    # Trend type ordering
    trend_order = ['Linear', 'Inverse-V', 'Piecewise']
    period_order = ['Fixed', 'Transitive', 'Variable']

    for trend_idx, trend_type in enumerate(trend_order):
        if trend_idx > 0:
            latex_lines.append(r"\midrule")

        latex_lines.append(f"% ===================== {trend_type.upper()} =====================")
        latex_lines.append(r"\multirow{9}{*}{\rotatebox{90}{\textbf{" + trend_type + "}}}")
        latex_lines.append("")

        for period_idx, period_regime in enumerate(period_order):
            if period_idx > 0:
                latex_lines.append(r"\cmidrule(lr){2-11}")
                latex_lines.append("")

            # Period regime row
            latex_lines.append(f"& \\multirow{{3}}{{*}}{{{period_regime}}}")

            # Get data for this combination
            df = organized_data.get(trend_type, {}).get(period_regime, None)

            for comp_idx, component in enumerate(COMPONENT_ORDER):
                if comp_idx > 0:
                    latex_lines.append("&")  # Empty cell for trend column

                comp_display = COMPONENT_DISPLAY_NAMES[component]

                # MAE row
                mae_parts = [f"& {comp_display}"]

                for model in MODEL_ORDER:
                    model_metrics = extract_model_mae(df, model) if df is not None else None
                    if model_metrics and component in model_metrics:
                        value = model_metrics[component]
                        mae_parts.append(f"& {format_value(value)}")
                    else:
                        mae_parts.append("& {-}")

                mae_parts.append(r"\\")
                latex_lines.append(" ".join(mae_parts))

            latex_lines.append("")

    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\caption{MAE decomposition errors for all trend types and period regimes. Lower values indicate better performance.}")
    latex_lines.append(r"\label{tab:appendix_mae}")
    latex_lines.append(r"\end{table*}")

    return "\n".join(latex_lines)


def main():
    """Main function to generate and save the LaTeX table."""
    print("=" * 80)
    print("Generating MAE Decomposition Error Table")
    print("=" * 80)
    print(f"\nLoading data from: {DATA_DIR}")

    # Generate table
    latex_table = generate_latex_table()

    # Save to file
    output_file = OUTPUT_DIR / "synthetic_full_decomposition_table.tex"
    with open(output_file, 'w') as f:
        f.write(latex_table)

    print(f"\nTable saved to: {output_file}")
    print("\nPreview (first 30 lines):")
    print("=" * 80)
    for line in latex_table.split('\n')[:30]:
        print(line)
    print("...")
    print("=" * 80)
    print(f"\nTotal lines: {len(latex_table.split(chr(10)))}")


if __name__ == "__main__":
    main()
