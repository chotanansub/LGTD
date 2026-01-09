"""
Fast Robust STL baseline method.

Fast implementation of Robust STL that uses PyTorch for optimization.
Reference: https://github.com/ariaghora/fast-robust-stl
"""

import numpy as np
from typing import Dict, Optional
import torch


class FastRobustSTLDecomposer:
    """
    Wrapper for Fast Robust STL decomposition.

    Fast Robust STL is an efficient implementation of Robust STL that uses
    PyTorch optimization for faster computation while maintaining robustness
    to outliers and handling multiple seasonal patterns.
    """

    def __init__(
        self,
        period: int = 12,
        reg1: float = 1.0,
        reg2: float = 10.0,
        K: int = 2,
        H: int = 5,
        dn1: float = 1.0,
        dn2: float = 1.0,
        ds1: float = 50.0,
        ds2: float = 1.0,
        max_iter: int = 1000
    ):
        """
        Initialize Fast Robust STL decomposer.

        Args:
            period: Seasonal period
            reg1: Regularization parameter for trend (smaller = smoother trend)
            reg2: Regularization parameter for trend derivative
            K: Seasonality bandwidth factor
            H: Bandwidth parameter for bilateral filter
            dn1: Bilateral filter parameter for denoising (temporal)
            dn2: Bilateral filter parameter for denoising (value)
            ds1: Bilateral filter parameter for seasonality (temporal)
            ds2: Bilateral filter parameter for seasonality (value)
            max_iter: Maximum iterations for PyTorch optimization (only for multiple seasonalities)
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
        self.max_iter = max_iter
        self._fast_robust_stl_available = self._check_availability()

    def _check_availability(self) -> bool:
        """Check if Fast Robust STL is available."""
        try:
            from experiments.baselines.frstl.frstl import fast_robustSTL
            import torch
            return True
        except ImportError:
            return False

    def decompose(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Perform Fast Robust STL decomposition.

        Args:
            data: Input time series array

        Returns:
            Dictionary with 'trend', 'seasonal', 'residual', 'y' components
        """
        if not self._fast_robust_stl_available:
            raise ImportError(
                "Fast Robust STL is not available. "
                "Check experiments/baselines/frstl/ module"
            )

        from experiments.baselines.frstl.frstl import fast_robustSTL
        from cvxopt import solvers

        # Suppress cvxopt verbose output
        solvers.options['show_progress'] = False

        try:
            # Configure for single seasonality (simple case)
            season_lens = [self.period]
            trend_regs = [self.reg1, self.reg2]

            # Simplified season_regs for single seasonality
            # [reg1, reg2, reg3] as in the paper
            season_regs = [[0.001, 1.0, 10.0]]

            alphas = [1.0]  # Single season weight
            z = 1.0  # Normalization factor
            denoise_ds = [self.dn1, self.dn2]
            season_ds = [self.ds1, self.ds2]

            # Run Fast Robust STL
            result = fast_robustSTL(
                data,
                season_lens=season_lens,
                trend_regs=trend_regs,
                season_regs=season_regs,
                alphas=alphas,
                z=z,
                denoise_ds=denoise_ds,
                season_ds=season_ds,
                K=self.K,
                H=self.H,
                max_iter=self.max_iter
            )

            # Unpack result: [input, trend, seasonal, residual]
            _, trend, seasonal, residual = result

            # Handle seasonal component (could be 2D for multiple seasonalities)
            if seasonal.ndim > 1:
                # Sum all seasonal components if multiple
                seasonal = seasonal.sum(axis=1)

            return {
                "time": np.arange(len(data)),
                "y": data,
                "trend": trend,
                "seasonal": seasonal,
                "residual": residual
            }
        except Exception as e:
            raise RuntimeError(f"Fast Robust STL decomposition failed: {str(e)}")

    def fit_transform(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Fit and transform data.

        Args:
            data: Input time series array

        Returns:
            Dictionary with decomposition components
        """
        return self.decompose(data)
