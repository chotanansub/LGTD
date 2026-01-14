"""
Evaluation metrics for decomposition quality assessment.
"""

import numpy as np
from typing import Dict, Union


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Squared Error.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        MSE value
    """
    return np.mean((y_true - y_pred) ** 2)


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        MAE value
    """
    return np.mean(np.abs(y_true - y_pred))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        RMSE value
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def correlation_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Pearson correlation coefficient.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        Correlation coefficient
    """
    return np.corrcoef(y_true, y_pred)[0, 1]


def peak_signal_noise_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        PSNR value in dB
    """
    mse = mean_squared_error(y_true, y_pred)
    if mse == 0:
        return float('inf')

    max_val = np.max(np.abs(y_true))
    psnr = 20 * np.log10(max_val / np.sqrt(mse))
    return psnr


def align_component(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """
    Align predicted component with ground truth by removing mean difference.

    Decomposition components are unique only up to a constant (you can add c to trend
    and subtract c from seasonal and still get the same y). This function aligns
    the predicted component to have the same mean as ground truth.

    Args:
        gt: Ground truth component
        pred: Predicted component

    Returns:
        Aligned predicted component
    """
    return pred - np.mean(pred) + np.mean(gt)


def compute_decomposition_metrics(
    ground_truth: Dict[str, np.ndarray],
    result: Dict[str, np.ndarray],
    align_components: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Compute comprehensive metrics for decomposition quality.

    Args:
        ground_truth: Dictionary with 'trend', 'seasonal', 'residual' ground truth
        result: Dictionary with 'trend', 'seasonal', 'residual' predictions
        align_components: Whether to align components before computing metrics (default: True)

    Returns:
        Dictionary of metrics for each component
    """
    metrics = {}

    for component in ['trend', 'seasonal', 'residual']:
        if component in ground_truth and component in result:
            gt = ground_truth[component]
            pred = result[component]

            # Align components if requested (decomposition is unique up to a constant)
            if align_components:
                pred = align_component(gt, pred)

            metrics[component] = {
                'mse': mean_squared_error(gt, pred),
                'mae': mean_absolute_error(gt, pred),
                'rmse': root_mean_squared_error(gt, pred),
                'correlation': correlation_coefficient(gt, pred),
                'psnr': peak_signal_noise_ratio(gt, pred)
            }

    return metrics


def compute_mse(ground_truth: Dict[str, np.ndarray], result: Dict[str, np.ndarray], align_components: bool = True) -> Dict[str, float]:
    """
    Compute MSE for each component.

    Args:
        ground_truth: Dictionary with ground truth components
        result: Dictionary with predicted components
        align_components: Whether to align components before computing MSE (default: True)

    Returns:
        Dictionary with MSE for each component
    """
    mse = {}
    for component in ['trend', 'seasonal', 'residual']:
        if component in ground_truth and component in result:
            gt = ground_truth[component]
            pred = result[component]

            # Align if requested
            if align_components:
                pred = align_component(gt, pred)

            mse[component] = mean_squared_error(gt, pred)
    return mse


def compute_mae(ground_truth: Dict[str, np.ndarray], result: Dict[str, np.ndarray], align_components: bool = True) -> Dict[str, float]:
    """
    Compute MAE for each component.

    Args:
        ground_truth: Dictionary with ground truth components
        result: Dictionary with predicted components
        align_components: Whether to align components before computing MAE (default: True)

    Returns:
        Dictionary with MAE for each component
    """
    mae = {}
    for component in ['trend', 'seasonal', 'residual']:
        if component in ground_truth and component in result:
            gt = ground_truth[component]
            pred = result[component]

            # Align if requested
            if align_components:
                pred = align_component(gt, pred)

            mae[component] = mean_absolute_error(gt, pred)
    return mae


def compute_rmse(ground_truth: Dict[str, np.ndarray], result: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Compute RMSE for each component.

    Args:
        ground_truth: Dictionary with ground truth components
        result: Dictionary with predicted components

    Returns:
        Dictionary with RMSE for each component
    """
    rmse = {}
    for component in ['trend', 'seasonal', 'residual']:
        if component in ground_truth and component in result:
            rmse[component] = root_mean_squared_error(
                ground_truth[component],
                result[component]
            )
    return rmse


def compute_correlation(ground_truth: Dict[str, np.ndarray], result: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Compute correlation for each component.

    Args:
        ground_truth: Dictionary with ground truth components
        result: Dictionary with predicted components

    Returns:
        Dictionary with correlation for each component
    """
    corr = {}
    for component in ['trend', 'seasonal', 'residual']:
        if component in ground_truth and component in result:
            corr[component] = correlation_coefficient(
                ground_truth[component],
                result[component]
            )
    return corr


def compute_psnr(ground_truth: Dict[str, np.ndarray], result: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Compute PSNR for each component.

    Args:
        ground_truth: Dictionary with ground truth components
        result: Dictionary with predicted components

    Returns:
        Dictionary with PSNR for each component
    """
    psnr = {}
    for component in ['trend', 'seasonal', 'residual']:
        if component in ground_truth and component in result:
            psnr[component] = peak_signal_noise_ratio(
                ground_truth[component],
                result[component]
            )
    return psnr
