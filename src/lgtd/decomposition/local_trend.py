"""
Local trend detection using AutoTrend module.
"""

import numpy as np
from typing import Dict, Any
from autotrend import decompose_llt


class LocalTrendDetector:
    """
    Wrapper for AutoTrend local linear trend detection.

    This class provides a unified interface for detecting local linear trends
    in time series data using the AutoTrend package.
    """

    def __init__(
        self,
        window_size: int = 3,
        error_percentile: int = 50,
        verbose: int = 0
    ):
        """
        Initialize Local Trend Detector.

        Args:
            window_size: Window size for LLT detection
            error_percentile: Error percentile for LLT
            verbose: Verbosity level (0=quiet, 1=verbose)
        """
        self.window_size = window_size
        self.error_percentile = error_percentile
        self.verbose = verbose

    def detect(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Detect local linear trends in time series data.

        Args:
            data: Input time series array

        Returns:
            Dictionary containing:
                - prediction_marks: Local trend predictions
                - segments: Detected segments
                - Other AutoTrend outputs
        """
        result = decompose_llt(
            data,
            window_size=self.window_size,
            error_percentile=self.error_percentile,
            verbose=self.verbose
        )

        return result

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Fit and transform data to extract local trend.

        Args:
            data: Input time series array

        Returns:
            Local trend array (prediction marks from AutoTrend)
        """
        result = self.detect(data)
        return result.prediction_marks
