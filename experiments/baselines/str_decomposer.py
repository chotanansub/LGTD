"""
STR (Seasonal-Trend decomposition using Regression) baseline method.

STR is a flexible method that recasts time series decomposition as a
regularized regression problem, allowing for multiple seasonal patterns
and time-varying coefficients.

Reference: Dokumentov & Hyndman (2022), INFORMS Journal on Data Science
Implementation: https://github.com/chotanansub/strpy
"""

import numpy as np
from typing import Dict, Optional, List


class STRDecomposer:
    """
    Wrapper for STR (Seasonal-Trend decomposition using Regression).

    STR decomposes time series using regularized regression with:
    - Smoothed trend component
    - Multiple seasonal components (Fourier basis)
    - Flexible regularization parameters
    """

    def __init__(
        self,
        seasonal_periods: Optional[List[int]] = None,
        trend_lambda: float = 1000.0,
        seasonal_lambda: float = 100.0,
        robust: bool = False,
        auto_params: bool = False,
        n_trials: int = 10
    ):
        """
        Initialize STR decomposer.

        Args:
            seasonal_periods: List of seasonal periods (e.g., [12] for monthly data)
            trend_lambda: Smoothing parameter for trend (higher = smoother, default: 1000.0)
            seasonal_lambda: Smoothing parameter for seasonal components (default: 100.0)
            robust: Use robust regression (not yet implemented in strpy, default: False)
            auto_params: Use automatic parameter selection via cross-validation (default: False)
            n_trials: Number of trials for automatic parameter selection (default: 10)
        """
        self.seasonal_periods = seasonal_periods or [12]
        self.trend_lambda = trend_lambda
        self.seasonal_lambda = seasonal_lambda
        self.robust = robust
        self.auto_params = auto_params
        self.n_trials = n_trials
        self._str_available = self._check_availability()

    def _check_availability(self) -> bool:
        """Check if STR is available."""
        try:
            from experiments.baselines.STR import STR_decompose, AutoSTR_simple
            return True
        except ImportError:
            return False

    def decompose(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Perform STR decomposition.

        Args:
            data: Input time series array

        Returns:
            Dictionary with 'trend', 'seasonal', 'residual', 'y' components
        """
        if not self._str_available:
            raise ImportError(
                "STR is not available. "
                "Check experiments/baselines/STR/ module"
            )

        from experiments.baselines.STR import STR_decompose, AutoSTR_simple

        try:
            if self.auto_params:
                # Use automatic parameter selection
                result = AutoSTR_simple(
                    data,
                    seasonal_periods=self.seasonal_periods,
                    n_trials=self.n_trials
                )
            else:
                # Use manual parameters
                result = STR_decompose(
                    data,
                    seasonal_periods=self.seasonal_periods,
                    trend_lambda=self.trend_lambda,
                    seasonal_lambda=self.seasonal_lambda,
                    robust=self.robust
                )

            # Convert result to our standard format
            # Sum all seasonal components if multiple
            seasonal = result.seasonal[0] if len(result.seasonal) == 1 else sum(result.seasonal)

            return {
                "time": np.arange(len(data)),
                "y": data,
                "trend": result.trend,
                "seasonal": seasonal,
                "residual": result.remainder
            }
        except Exception as e:
            raise RuntimeError(f"STR decomposition failed: {str(e)}")

    def fit_transform(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Fit and transform data.

        Args:
            data: Input time series array

        Returns:
            Dictionary with decomposition components
        """
        return self.decompose(data)
