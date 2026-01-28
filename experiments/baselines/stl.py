"""
STL (Seasonal-Trend decomposition using Loess) baseline method.
"""

import numpy as np
import pandas as pd
from typing import Dict


class STLDecomposer:
    """
    Wrapper for STL decomposition from statsmodels.

    STL is a versatile and robust method for decomposing time series.
    It uses LOESS (locally weighted regression) for trend and seasonal extraction.
    """

    def __init__(
        self,
        period: int = None,
        seasonal: int = 7,
        trend: int = None,
        low_pass: int = None,
        seasonal_deg: int = 1,
        trend_deg: int = 1,
        low_pass_deg: int = 1,
        robust: bool = False,
        seasonal_jump: int = 1,
        trend_jump: int = 1,
        low_pass_jump: int = 1
    ):
        """
        Initialize STL decomposer.

        Args:
            period: Seasonal period (required for STL)
            seasonal: Length of the seasonal smoother (must be odd, default: 7)
            trend: Length of the trend smoother (must be odd, default: None)
            low_pass: Length of the low-pass filter (must be odd >=3, default: None)
            seasonal_deg: Degree of seasonal LOESS (0 or 1, default: 1)
            trend_deg: Degree of trend LOESS (0 or 1, default: 1)
            low_pass_deg: Degree of low pass LOESS (0 or 1, default: 1)
            robust: Whether to use robust fitting (default: False)
            seasonal_jump: Linear interpolation step for seasonal (default: 1)
            trend_jump: Linear interpolation step for trend (default: 1)
            low_pass_jump: Linear interpolation step for low-pass (default: 1)
        """
        self.period = period
        self.seasonal = seasonal
        self.trend = trend
        self.low_pass = low_pass
        self.seasonal_deg = seasonal_deg
        self.trend_deg = trend_deg
        self.low_pass_deg = low_pass_deg
        self.robust = robust
        self.seasonal_jump = seasonal_jump
        self.trend_jump = trend_jump
        self.low_pass_jump = low_pass_jump

    def decompose(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Perform STL decomposition.

        Args:
            data: Input time series array

        Returns:
            Dictionary with 'trend', 'seasonal', 'residual', 'y' components
        """
        from statsmodels.tsa.seasonal import STL

        # Convert to pandas Series for STL
        ts = pd.Series(data, index=pd.RangeIndex(len(data)))

        # Perform STL decomposition
        stl = STL(
            ts,
            period=self.period,
            seasonal=self.seasonal,
            trend=self.trend,
            low_pass=self.low_pass,
            seasonal_deg=self.seasonal_deg,
            trend_deg=self.trend_deg,
            low_pass_deg=self.low_pass_deg,
            robust=self.robust,
            seasonal_jump=self.seasonal_jump,
            trend_jump=self.trend_jump,
            low_pass_jump=self.low_pass_jump
        )
        components = stl.fit()

        return {
            "time": np.arange(len(data)),
            "y": data,
            "trend": components.trend.values,
            "seasonal": components.seasonal.values,
            "residual": components.resid.values
        }

    def fit_transform(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Fit and transform data.

        Args:
            data: Input time series array

        Returns:
            Dictionary with decomposition components
        """
        return self.decompose(data)
