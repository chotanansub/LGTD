"""
Seasonal component extraction for LGTD.
"""

import numpy as np
from typing import Optional, List
from scipy.signal import find_peaks
from scipy.interpolate import interp1d


class SeasonalExtractor:
    """
    Seasonal component extractor for LGTD.

    This class handles the extraction and processing of seasonal patterns
    from detrended time series data.
    """

    def __init__(self, min_period: int = 2, max_period: Optional[int] = None):
        """
        Initialize Seasonal Extractor.

        Args:
            min_period: Minimum expected period length
            max_period: Maximum expected period length (None for auto)
        """
        self.min_period = min_period
        self.max_period = max_period
        self.detected_periods_: Optional[List[int]] = None

    def interpolate_nans(self, seasonal: np.ndarray) -> np.ndarray:
        """
        Interpolate NaN values in seasonal component.

        Args:
            seasonal: Seasonal array possibly containing NaNs

        Returns:
            Seasonal array with NaNs interpolated
        """
        if not np.any(np.isnan(seasonal)):
            return seasonal

        valid_mask = ~np.isnan(seasonal)
        valid_indices = np.where(valid_mask)[0]
        valid_values = seasonal[valid_mask]

        if len(valid_indices) > 1:
            interpolator = interp1d(
                valid_indices,
                valid_values,
                kind='linear',
                fill_value='extrapolate'
            )
            seasonal = interpolator(np.arange(len(seasonal)))
        else:
            # If not enough valid points, return zeros
            seasonal = np.zeros_like(seasonal)

        return seasonal

    def detect_periods(self, seasonal: np.ndarray) -> List[int]:
        """
        Detect periods from seasonal component using peak detection.

        Args:
            seasonal: Seasonal component array

        Returns:
            List of detected period lengths
        """
        peaks, _ = find_peaks(seasonal)

        if len(peaks) > 1:
            periods = np.diff(peaks).tolist()
            # Filter by min/max period constraints
            if self.max_period is not None:
                periods = [p for p in periods if self.min_period <= p <= self.max_period]
            else:
                periods = [p for p in periods if p >= self.min_period]
            return periods
        else:
            return []

    def extract(self, seasonal: np.ndarray) -> np.ndarray:
        """
        Extract and process seasonal component.

        Args:
            seasonal: Raw seasonal component (possibly with NaNs)

        Returns:
            Processed seasonal component
        """
        # Interpolate NaN values
        seasonal_clean = self.interpolate_nans(seasonal)

        # Detect periods
        self.detected_periods_ = self.detect_periods(seasonal_clean)

        return seasonal_clean

    def fit_transform(self, seasonal: np.ndarray) -> np.ndarray:
        """
        Fit and transform seasonal component.

        Args:
            seasonal: Raw seasonal component

        Returns:
            Processed seasonal component
        """
        return self.extract(seasonal)
