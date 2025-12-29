"""
ASTD (Adaptive Seasonal Trend Decomposition) baseline method.
"""

import numpy as np
from typing import Dict


class ASTDDecomposer:
    """
    Wrapper for ASTD decomposition.

    ASTD is an adaptive online decomposition method that can handle
    time-varying seasonality and trend.
    """

    def __init__(self, seasonality_smoothing: float = 0.7):
        """
        Initialize ASTD decomposer.

        Args:
            seasonality_smoothing: Smoothing parameter for seasonality (0-1)
        """
        self.seasonality_smoothing = seasonality_smoothing
        self._astd_available = self._check_availability()

    def _check_availability(self) -> bool:
        """Check if ASTD is available."""
        try:
            from experiments.baselines.ASTD.ASTD import ASTD
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

        from experiments.baselines.ASTD.ASTD import ASTD

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
            raise RuntimeError(f"ASTD decomposition failed: {str(e)}")

    def fit_transform(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Fit and transform data.

        Args:
            data: Input time series array

        Returns:
            Dictionary with decomposition components
        """
        return self.decompose(data)
