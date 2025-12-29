"""
RobustSTL baseline method.
"""

import numpy as np
from typing import Dict, Optional


class RobustSTLDecomposer:
    """
    Wrapper for RobustSTL decomposition.

    RobustSTL is a robust version of STL that handles outliers better.
    It uses bilateral filtering and L1 optimization.
    """

    def __init__(
        self,
        period: int = 12,
        reg1: float = 10.0,
        reg2: float = 0.5,
        K: int = 2,
        H: int = 5,
        dn1: float = 1.0,
        dn2: float = 1.0,
        ds1: float = 50.0,
        ds2: float = 1.0
    ):
        """
        Initialize RobustSTL decomposer.

        Args:
            period: Seasonal period
            reg1: Regularization parameter for trend
            reg2: Regularization parameter for seasonal
            K: Number of iterations
            H: Bandwidth parameter
            dn1, dn2: Bilateral filter parameters for trend
            ds1, ds2: Bilateral filter parameters for seasonal
        """
        self.period = period
        self.reg1 = reg1
        self.reg2 = reg2
        self.K = K
        self.H = H
        self.dn1 = dn1
        self.dn2 = dn2
        self.ds1 = ds1
        self.ds2 = ds2
        self._robust_stl_available = self._check_availability()

    def _check_availability(self) -> bool:
        """Check if RobustSTL is available."""
        try:
            from experiments.baselines.RobustSTL.RobustSTL import RobustSTL
            return True
        except ImportError:
            return False

    def decompose(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Perform RobustSTL decomposition.

        Args:
            data: Input time series array

        Returns:
            Dictionary with 'trend', 'seasonal', 'residual', 'y' components
        """
        if not self._robust_stl_available:
            raise ImportError(
                "RobustSTL is not available. "
                "Clone from: https://github.com/LeeDoYup/RobustSTL.git"
            )

        from experiments.baselines.RobustSTL.RobustSTL import RobustSTL

        try:
            result = RobustSTL(
                data, self.period,
                reg1=self.reg1, reg2=self.reg2, K=self.K, H=self.H,
                dn1=self.dn1, dn2=self.dn2, ds1=self.ds1, ds2=self.ds2
            )

            return {
                "time": np.arange(len(data)),
                "y": data,
                "trend": result[1],
                "seasonal": result[2],
                "residual": result[3]
            }
        except Exception as e:
            raise RuntimeError(f"RobustSTL decomposition failed: {str(e)}")

    def fit_transform(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Fit and transform data.

        Args:
            data: Input time series array

        Returns:
            Dictionary with decomposition components
        """
        return self.decompose(data)
