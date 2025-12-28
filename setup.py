"""
Setup script for LGTD package.

For modern Python packaging, use pyproject.toml.
This file is kept for backward compatibility.
"""

from setuptools import setup, find_packages

setup(
    name="lgtd",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.8",
)
