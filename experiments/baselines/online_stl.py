"""
OnlineSTL baseline method.

Online Seasonal-Trend decomposition that processes data incrementally.
"""

import numpy as np
from typing import Dict
import sys
from pathlib import Path


class OnlineSTLDecomposer:
    """
    Wrapper for OnlineSTL decomposition.

    OnlineSTL is an online decomposition method that requires an initialization
    window (4*period) and then processes new points incrementally.

    Model type: 'online'
    """

    def __init__(
        self,
        periods: list = None,
        lam: float = 0.7,
        init_window_ratio: float = 0.5
    ):
        """
        Initialize OnlineSTL decomposer.

        Args:
            periods: List of seasonal periods (default: inferred from data)
            lam: Smoothing parameter for seasonality (0-1)
            init_window_ratio: Ratio of data to use for initialization (default: 0.5)
        """
        self.periods = periods
        self.lam = lam
        self.init_window_ratio = init_window_ratio
        self._online_stl_available = self._check_availability()

    def _check_availability(self) -> bool:
        """Check if OnlineSTL is available."""
        try:
            # Add OnlineSTL directory to path
            online_stl_dir = Path(__file__).parent / "OnlineSTL"
            if str(online_stl_dir) not in sys.path:
                sys.path.insert(0, str(online_stl_dir))

            from OnlineSTL import OnlineSTL
            return True
        except ImportError:
            return False

    def decompose(self, data: np.ndarray, period: int = None) -> Dict[str, np.ndarray]:
        """
        Perform OnlineSTL decomposition.

        Args:
            data: Input time series array
            period: Seasonal period (if None, uses self.periods or infers from data)

        Returns:
            Dictionary with 'trend', 'seasonal', 'residual', 'y' components
        """
        if not self._online_stl_available:
            raise ImportError(
                "OnlineSTL is not available. "
                "Ensure OnlineSTL files are in experiments/baselines/OnlineSTL/"
            )

        from OnlineSTL import OnlineSTL

        # Determine periods
        if self.periods is None:
            if period is None:
                # Default to detecting period (simple heuristic)
                period = min(len(data) // 4, 120)
            periods = [period]
        else:
            periods = self.periods

        # Ensure we have enough data for initialization
        max_period = max(periods)
        min_init_length = 4 * max_period

        if len(data) < min_init_length:
            raise ValueError(
                f"Data length ({len(data)}) must be at least 4*max_period ({min_init_length}) "
                f"for OnlineSTL initialization."
            )

        # Determine initialization window size
        init_size = max(min_init_length, int(len(data) * self.init_window_ratio))
        init_size = min(init_size, len(data))  # Don't exceed data length

        # Split data
        init_data = data[:init_size]
        online_data = data[init_size:]

        try:
            # Initialize OnlineSTL with first portion of data
            model = OnlineSTL(init_data, periods, lam=self.lam)

            # Store decomposition results
            trend_list = []
            seasonal_list = []
            residual_list = []

            # Process online data point-by-point
            for x in online_data:
                T, S, R = model.update(x)
                trend_list.append(T)

                # Handle multiple seasonal components by summing them
                if isinstance(S, np.ndarray) and len(S) > 0:
                    seasonal_list.append(np.sum(S))
                else:
                    seasonal_list.append(S)

                residual_list.append(R)

            # For the initialization window, we need to decompose it retroactively
            # Use a simple approach: extend the first online point backwards
            if len(trend_list) > 0:
                # Replicate first online decomposition for init window
                init_trend = np.full(init_size, trend_list[0])
                init_seasonal = np.full(init_size, seasonal_list[0])
                init_residual = np.full(init_size, residual_list[0])
            else:
                # Fallback if no online updates
                init_trend = np.zeros(init_size)
                init_seasonal = np.zeros(init_size)
                init_residual = init_data.copy()

            # Combine initialization and online results
            trend = np.concatenate([init_trend, np.array(trend_list)])
            seasonal = np.concatenate([init_seasonal, np.array(seasonal_list)])
            residual = np.concatenate([init_residual, np.array(residual_list)])

            return {
                "time": np.arange(len(data)),
                "y": data,
                "trend": trend,
                "seasonal": seasonal,
                "residual": residual
            }

        except Exception as e:
            raise RuntimeError(f"OnlineSTL decomposition failed: {str(e)}")

    def fit_transform(self, data: np.ndarray, period: int = None) -> Dict[str, np.ndarray]:
        """
        Fit and transform data.

        Args:
            data: Input time series array
            period: Seasonal period

        Returns:
            Dictionary with decomposition components
        """
        return self.decompose(data, period=period)
