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

    def __init__(self, period: int = 12, seasonal: int = 7, trend: int = None, robust: bool = False):
        """
        Initialize STL decomposer.

        Args:
            period: Seasonal period
            seasonal: Length of the seasonal smoother (must be odd)
            trend: Length of the trend smoother (must be odd)
            robust: Whether to use robust fitting
        """
        self.period = period
        self.seasonal = seasonal
        self.trend = trend
        self.robust = robust

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
        stl = STL(ts, period=self.period, seasonal=self.seasonal, trend=self.trend, robust=self.robust)
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
