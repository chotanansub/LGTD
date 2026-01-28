"""
Script to compute accuracy metrics for decomposition methods.
Loads ground truth from synthetic datasets and compares with decomposition results.
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def load_ground_truth(dataset_path: Path) -> dict:
    """
    Load ground truth components from synthetic dataset.

    Args:
        dataset_path: Path to dataset JSON file

    Returns:
        Dictionary with ground truth components
    """
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    return {
        'trend': np.array(data['data']['trend']),
        'seasonal': np.array(data['data']['seasonal']),
        'residual': np.array(data['data']['residual']),
        'y': np.array(data['data']['y'])
    }


def load_decomposition(decomp_path: Path) -> dict:
    """
    Load decomposition results.

    Args:
        decomp_path: Path to decomposition JSON file

    Returns:
        Dictionary with decomposed components
    """
    with open(decomp_path, 'r') as f:
        data = json.load(f)

    return {
        'trend': np.array(data['trend']),
        'seasonal': np.array(data['seasonal']),
        'residual': np.array(data['residual']),
        'y': np.array(data['y'])
    }


def compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Squared Error.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        MSE value
    """
    return np.mean((y_true - y_pred) ** 2)


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Absolute Error.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        MAE value
    """
    return np.mean(np.abs(y_true - y_pred))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Root Mean Squared Error.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        RMSE value
    """
    return np.sqrt(compute_mse(y_true, y_pred))


def compute_accuracy_metrics(ground_truth: dict, decomposition: dict) -> dict:
    """
    Compute accuracy metrics for all components.

    Args:
        ground_truth: Ground truth components
        decomposition: Decomposed components

    Returns:
        Dictionary with metrics for each component
    """
    metrics = {}

    for component in ['trend', 'seasonal', 'residual']:
        y_true = ground_truth[component]
        y_pred = decomposition[component]

        metrics[component] = {
            'MSE': compute_mse(y_true, y_pred),
            'MAE': compute_mae(y_true, y_pred),
            'RMSE': compute_rmse(y_true, y_pred)
        }

    return metrics


def process_dataset(dataset_name: str,
                   data_dir: Path,
                   decomp_dir: Path) -> pd.DataFrame:
    """
    Process a single dataset and compute metrics for all methods.

    Args:
        dataset_name: Name of the dataset (e.g., 'synth1')
        data_dir: Directory containing synthetic datasets
        decomp_dir: Directory containing decomposition results

    Returns:
        DataFrame with results for this dataset
    """
    # Load ground truth
    dataset_path = data_dir / f"{dataset_name}_data.json"
    if not dataset_path.exists():
        print(f"⚠ Dataset not found: {dataset_path}")
        return pd.DataFrame()

    ground_truth = load_ground_truth(dataset_path)

    # Process all decomposition methods
    results = []
    decomp_dataset_dir = decomp_dir / dataset_name

    if not decomp_dataset_dir.exists():
        print(f"⚠ Decomposition directory not found: {decomp_dataset_dir}")
        return pd.DataFrame()

    for decomp_file in decomp_dataset_dir.glob("*.json"):
        model_name = decomp_file.stem

        try:
            decomposition = load_decomposition(decomp_file)
            metrics = compute_accuracy_metrics(ground_truth, decomposition)

            # Flatten metrics into rows
            for component in ['trend', 'seasonal', 'residual']:
                for metric_name, metric_value in metrics[component].items():
                    results.append({
                        'dataset': dataset_name,
                        'model': model_name,
                        'component': component,
                        'metric': metric_name,
                        'value': metric_value
                    })

        except Exception as e:
            print(f"⚠ Error processing {model_name} for {dataset_name}: {e}")

    return pd.DataFrame(results)


def print_summary_table(df: pd.DataFrame) -> None:
    """
    Print formatted summary table.

    Args:
        df: Results DataFrame
    """
    print("\n" + "="*100)
    print("ACCURACY METRICS SUMMARY")
    print("="*100)

    for metric in ['MSE', 'MAE', 'RMSE']:
        print(f"\n{metric}:")
        print("-"*100)

        metric_df = df[df['metric'] == metric]

        if not metric_df.empty:
            pivot = metric_df.pivot_table(
                index=['dataset', 'model'],
                columns='component',
                values='value',
                aggfunc='first'
            )
            print(pivot.to_string())

    print("\n" + "="*100)


def main():
    """Main execution function."""
    print("="*70)
    print("ACCURACY TEST - SYNTHETIC DATASETS")
    print("="*70)

    # Define paths
    data_dir = project_root / "data" / "synthetic" / "datasets"
    decomp_dir = project_root / "experiments" / "results" / "decompositions" / "synthetic"
    output_dir = project_root / "experiments" / "results" / "accuracy" / "synthetic"

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process all datasets
    all_results = []

    for dataset_path in sorted(data_dir.glob("synth*_data.json")):
        dataset_name = dataset_path.stem.replace('_data', '')
        print(f"\nProcessing {dataset_name}...")

        df = process_dataset(dataset_name, data_dir, decomp_dir)
        if not df.empty:
            all_results.append(df)
            print(f"✓ {dataset_name}: {len(df)} metric entries computed")

    # Combine all results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)

        # Print summary
        print_summary_table(combined_df)

        # Save results
        output_file = output_dir / "accuracy_results.csv"
        combined_df.to_csv(output_file, index=False)
        print(f"\n✓ Results saved to: {output_file}")

        # Save pivot tables for each metric
        for metric in ['MSE', 'MAE', 'RMSE']:
            metric_df = combined_df[combined_df['metric'] == metric]
            if not metric_df.empty:
                pivot = metric_df.pivot_table(
                    index=['dataset', 'model'],
                    columns='component',
                    values='value',
                    aggfunc='first'
                )
                metric_file = output_dir / f"accuracy_{metric.lower()}.csv"
                pivot.to_csv(metric_file)
                print(f"✓ {metric} table saved to: {metric_file}")

    else:
        print("\n⚠ No results to process")

    print("\n" + "="*70)
    print("ACCURACY TEST COMPLETED")
    print("="*70)


if __name__ == "__main__":
    main()
