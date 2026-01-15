"""
lgtd: Local Global Trend Decomposition

A time series decomposition method combining local linear trend detection
with seasonal pattern extraction.
"""

__version__ = "0.1.0"
__author__ = "Chotanan Subscription"

from lgtd.decomposition.lgtd import lgtd

__all__ = ["lgtd"]
