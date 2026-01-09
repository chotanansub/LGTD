"""
ASTD (Adaptive Seasonal Trend Decomposition) baseline method.

Supports two modes:
- Batch mode: Treats entire dataset as initialization window (default behavior)
- Online mode: Uses initial window for initialization, then updates online point-by-point
"""

import numpy as np
from typing import Dict


class ASTDDecomposer:
    """
    Wrapper for ASTD decomposition.

    ASTD is an adaptive online decomposition method that can handle
    time-varying seasonality and trend.

    Model type: Can operate in both 'batch' and 'online' modes.
    """

    def __init__(
        self,
        seasonality_smoothing: float = 0.7,
        mode: str = 'batch',
        init_window_size: int = 300
    ):
        """
        Initialize ASTD decomposer.

        Args:
            seasonality_smoothing: Smoothing parameter for seasonality (0-1)
            mode: Decomposition mode - 'batch' or 'online'
            init_window_size: Initial window size for online mode (ignored in batch mode)
        """
        self.seasonality_smoothing = seasonality_smoothing
        self.mode = mode
        self.init_window_size = init_window_size
        self._astd_available = self._check_availability()

    def _check_availability(self) -> bool:
        """Check if ASTD is available."""
        try:
            from experiments.baselines.ASTD.online_decomposition.ASTD import ASTD
            return True
        except ImportError:
            return False

    def decompose(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Perform ASTD decomposition.

        Args:
            data: Input time series array

        Returns:
            Dictionary with 'trend', 'seasonal', 'residual', 'y' components
        """
        if not self._astd_available:
            raise ImportError(
                "ASTD is not available. "
                "Clone from: https://github.com/thanapol2/ASTD_ECMLPKDD.git"
            )

        if self.mode == 'batch':
            return self._decompose_batch(data)
        elif self.mode == 'online':
            return self._decompose_online(data)
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Must be 'batch' or 'online'.")

    def _decompose_batch(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Batch mode: Use entire dataset as initialization window.

        Args:
            data: Input time series array

        Returns:
            Dictionary with decomposition components
        """
        from experiments.baselines.ASTD.online_decomposition.ASTD import ASTD

        window_size = len(data)

        try:
            core_STL = ASTD(
                window_size=window_size,
                seasonality_smoothing=self.seasonality_smoothing
            )
            astd_trend, astd_seasonal, astd_residual = core_STL.initialization_phase(data)

            return {
                "time": np.arange(len(data)),
                "y": data,
                "trend": astd_trend,
                "seasonal": astd_seasonal,
                "residual": astd_residual
            }
        except Exception as e:
            raise RuntimeError(f"ASTD batch decomposition failed: {str(e)}")

    def _decompose_online(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Online mode: Initialize with window, then update point-by-point.

        Args:
            data: Input time series array

        Returns:
            Dictionary with decomposition components
        """
        from experiments.baselines.ASTD.online_decomposition.ASTD import ASTD

        # Determine actual window size (minimum of init_window_size and data length)
        window_size = min(self.init_window_size, len(data))

        # Split data into offline and online phases
        ts_offline_phase = data[:window_size]
        ts_online_phase = data[window_size:]

        try:
            core_STL = ASTD(
                window_size=window_size,
                seasonality_smoothing=self.seasonality_smoothing
            )

            # Initialization phase
            astd_trend, astd_seasonal, astd_residual = core_STL.initialization_phase(ts_offline_phase)

            # Online updating phase
            for y_t in ts_online_phase:
                t_t, s_t, r_t = core_STL.update_phase(y_t)
                astd_trend = np.append(astd_trend, t_t)
                astd_seasonal = np.append(astd_seasonal, s_t)
                astd_residual = np.append(astd_residual, r_t)

            return {
                "time": np.arange(len(data)),
                "y": data,
                "trend": astd_trend,
                "seasonal": astd_seasonal,
                "residual": astd_residual
            }
        except Exception as e:
            raise RuntimeError(f"ASTD online decomposition failed: {str(e)}")

    def fit_transform(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Fit and transform data.

        Args:
            data: Input time series array

        Returns:
            Dictionary with decomposition components
        """
        return self.decompose(data)
