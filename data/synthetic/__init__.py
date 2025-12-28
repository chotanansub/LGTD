"""Synthetic data generation for LGTD experiments."""

from data.synthetic.generators import (
    generate_trend_series,
    generate_seasonal_series,
    generate_trend_seasonal_series,
    generate_synthetic_data,
)

__all__ = [
    "generate_trend_series",
    "generate_seasonal_series",
    "generate_trend_seasonal_series",
    "generate_synthetic_data",
]
