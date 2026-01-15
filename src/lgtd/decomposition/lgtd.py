"""
Main LGTD (Local Global Trend Decomposition) implementation.
"""

import numpy as np
from typing import Optional, Literal, Dict, Any
from dataclasses import dataclass

from lgtd.decomposition.local_trend import LocalTrendDetector
from lgtd.decomposition.seasonal import SeasonalExtractor
from lgtd.decomposition.utils import (
    select_trend_method,
    extract_linear_trend,
    extract_lowess_trend,
    detrend
)


@dataclass
class LGTDResult:
    """
    Result object for LGTD decomposition.

    Attributes:
        trend: Trend component
        seasonal: Seasonal component
        residual: Residual component
        y: Original time series (reconstructed)
        detected_periods: List of detected period lengths
        trend_info: Dictionary with trend extraction information
    """
    trend: np.ndarray
    seasonal: np.ndarray
    residual: np.ndarray
    y: np.ndarray
    detected_periods: list
    trend_info: Dict[str, Any]


class lgtd:
    """
    Local Global Trend Decomposition.

    lgtd decomposes time series into trend, seasonal, and residual components
    by combining:
    1. Global trend extraction (linear or LOWESS)
    2. Local linear trend analysis for seasonality (using AutoTrend)
    3. Residual calculation

    Example:
        >>> from lgtd import lgtd
        >>> import numpy as np
        >>>
        >>> # Generate sample data
        >>> t = np.arange(100)
        >>> data = 0.5 * t + 10 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 1, 100)
        >>>
        >>> # Decompose
        >>> model = lgtd()
        >>> result = model.fit_transform(data)
        >>>
        >>> # Access components
        >>> print(result.trend)
        >>> print(result.seasonal)
        >>> print(result.residual)
    """

    def __init__(
        self,
        window_size: int = 3,
        error_percentile: int = 50,
        trend_selection: Literal['auto', 'linear', 'lowess'] = 'auto',
        lowess_frac: float = 0.1,
        threshold_r2: float = 0.92,
        verbose: bool = False
    ):
        """
        Initialize lgtd decomposition model.

        Args:
            window_size: Window size for local linear trend detection
            error_percentile: Error percentile for AutoTrend
            trend_selection: Method for trend extraction ('auto', 'linear', 'lowess')
            lowess_frac: Fraction for LOWESS smoothing
            threshold_r2: R² threshold for auto trend selection
            verbose: Print diagnostic information
        """
        # Input validation
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        if not (0 <= error_percentile <= 100):
            raise ValueError(f"error_percentile must be between 0 and 100, got {error_percentile}")
        if trend_selection not in ['auto', 'linear', 'lowess']:
            raise ValueError(f"trend_selection must be 'auto', 'linear', or 'lowess', got {trend_selection}")
        if not (0 < lowess_frac <= 1):
            raise ValueError(f"lowess_frac must be between 0 and 1, got {lowess_frac}")
        if not (0 <= threshold_r2 <= 1):
            raise ValueError(f"threshold_r2 must be between 0 and 1, got {threshold_r2}")

        self.window_size = window_size
        self.error_percentile = error_percentile
        self.trend_selection = trend_selection
        self.lowess_frac = lowess_frac
        self.threshold_r2 = threshold_r2
        self.verbose = verbose

        # Initialize components
        self.local_trend_detector = LocalTrendDetector(
            window_size=window_size,
            error_percentile=error_percentile,
            verbose=1 if verbose else 0
        )
        self.seasonal_extractor = SeasonalExtractor()

        # Store results
        self.result_: Optional[LGTDResult] = None

    def _extract_global_trend(self, seq: np.ndarray) -> tuple:
        """
        Extract global trend from time series.

        Args:
            seq: Input time series

        Returns:
            Tuple of (trend array, trend info dict)
        """
        # Determine trend method
        if self.trend_selection == 'auto':
            trend_method = select_trend_method(seq, self.threshold_r2)
            if self.verbose:
                print(f"Auto-selected trend method: {trend_method.upper()}")
        else:
            trend_method = self.trend_selection

        # Extract trend
        if trend_method == 'linear':
            global_trend, r2 = extract_linear_trend(seq)
            trend_info = {'method': 'linear', 'r2': r2}
        else:  # lowess
            global_trend, r2 = extract_lowess_trend(seq, self.lowess_frac)
            trend_info = {'method': 'lowess', 'frac': self.lowess_frac, 'r2': r2}

        if self.verbose:
            print(f"Trend extraction: method={trend_info['method']}, R²={r2:.4f}")

        return global_trend, trend_info

    def fit(self, data: np.ndarray) -> 'lgtd':
        """
        Fit lgtd model to data.

        Args:
            data: Input time series array

        Returns:
            self
        """
        self.fit_transform(data)
        return self

    def transform(self, data: np.ndarray) -> LGTDResult:
        """
        Transform data using fitted model.

        Note: Currently, lgtd requires fitting for each dataset.
        This method is provided for API consistency.

        Args:
            data: Input time series array

        Returns:
            LGTDResult object
        """
        return self.fit_transform(data)

    def fit_transform(self, data: np.ndarray) -> LGTDResult:
        """
        Fit and transform data in one step.

        Args:
            data: Input time series array (1D numpy array)

        Returns:
            LGTDResult object containing decomposition components

        Raises:
            ValueError: If data is not 1D array
        """
        # Validate input
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        if data.ndim != 1:
            raise ValueError("Data must be a 1D array")

        seq = data.copy()

        if self.verbose:
            print("="*60)
            print("lgtd Decomposition")
            print("="*60)
            print(f"Data length: {len(seq)}")
            print(f"Window size: {self.window_size}")
            print(f"Error percentile: {self.error_percentile}")

        # Step 1: Extract global trend
        if self.verbose:
            print("\nStep 1: Extracting global trend...")

        global_trend, trend_info = self._extract_global_trend(seq)

        # Step 2: Detrend
        if self.verbose:
            print("\nStep 2: Detrending series...")

        detrended = detrend(seq, global_trend)

        # Step 3: Apply local linear trend for seasonal extraction
        if self.verbose:
            print("\nStep 3: Extracting seasonal component using LLT...")

        seasonal = self.local_trend_detector.fit_transform(detrended)

        # Step 4: Process seasonal component (handle NaNs)
        seasonal = self.seasonal_extractor.fit_transform(seasonal)

        # Step 5: Compute residual
        if self.verbose:
            print("\nStep 4: Computing residual...")

        residual = seq - global_trend - seasonal

        # Store result
        self.result_ = LGTDResult(
            trend=global_trend,
            seasonal=seasonal,
            residual=residual,
            y=seq,
            detected_periods=self.seasonal_extractor.detected_periods_ or [],
            trend_info=trend_info
        )

        if self.verbose:
            print("\n" + "="*60)
            print("Decomposition Complete!")
            print("="*60)
            if self.result_.detected_periods:
                print(f"Detected periods: {self.result_.detected_periods}")
            print(f"Mean residual: {np.mean(residual):.6f}")
            print(f"Std residual: {np.std(residual):.6f}")

        return self.result_

    @property
    def trend(self) -> Optional[np.ndarray]:
        """Get trend component from last decomposition."""
        return self.result_.trend if self.result_ is not None else None

    @property
    def seasonal(self) -> Optional[np.ndarray]:
        """Get seasonal component from last decomposition."""
        return self.result_.seasonal if self.result_ is not None else None

    @property
    def residual(self) -> Optional[np.ndarray]:
        """Get residual component from last decomposition."""
        return self.result_.residual if self.result_ is not None else None

    @property
    def detected_periods(self) -> Optional[list]:
        """Get detected periods from last decomposition."""
        return self.result_.detected_periods if self.result_ is not None else None
