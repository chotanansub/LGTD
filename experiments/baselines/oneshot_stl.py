"""
OneShotSTL baseline method.

OneShotSTL is an online seasonal-trend decomposition method with O(1) update complexity.
It requires a Java JAR file to run and uses specific JSON input/output format.
"""

import numpy as np
import json
import os
import subprocess
import tempfile
from typing import Dict, Optional
from pathlib import Path


class OneShotSTLDecomposer:
    """
    Wrapper for OneShotSTL decomposition.

    OneShotSTL is an online decomposition method with O(1) update complexity.
    It uses a Java implementation and requires specific JSON format for input/output.

    Model type: 'online'
    """

    def __init__(
        self,
        period: int = None,
        shift_window: int = 0,
        init_ratio: float = 0.5,
        jar_path: Optional[str] = None
    ):
        """
        Initialize OneShotSTL decomposer.

        Args:
            period: Seasonal period (required for OneShotSTL)
            shift_window: Shift window parameter (default: 0)
            init_ratio: Ratio of data to use for training/initialization (default: 0.5)
            jar_path: Path to OneShotSTL.jar (auto-detected if None)
        """
        self.period = period
        self.shift_window = shift_window
        self.init_ratio = init_ratio

        # Auto-detect JAR path if not provided
        if jar_path is None:
            baseline_dir = Path(__file__).parent
            self.jar_path = baseline_dir / "OneShotSTL" / "java" / "OneShotSTL" / "OneShotSTL.jar"
        else:
            self.jar_path = Path(jar_path)

        self._check_availability()

    def _check_availability(self) -> bool:
        """Check if OneShotSTL JAR file is available and Java is installed."""
        # Check if JAR exists
        if not self.jar_path.exists():
            raise FileNotFoundError(
                f"OneShotSTL JAR not found at {self.jar_path}. "
                "Please ensure OneShotSTL is properly installed in experiments/baselines/OneShotSTL/"
            )

        # Check if Java is available
        try:
            result = subprocess.run(
                ['java', '-version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                raise RuntimeError("Java is not available or not working properly")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            raise RuntimeError(
                "Java is not installed or not in PATH. "
                "OneShotSTL requires Java to run. Please install Java."
            )

        return True

    def _prepare_input_json(self, data: np.ndarray, period: int) -> tuple:
        """
        Prepare input JSON in OneShotSTL format.

        OneShotSTL expects:
        {
            "period": int,
            "trainTestSplit": int,  # Index where test data starts
            "ts": list[float]
        }

        OneShotSTL trains on data[0:trainTestSplit] and returns decomposition
        for data[trainTestSplit:].

        Args:
            data: Full input time series
            period: Seasonal period

        Returns:
            Tuple of (input_dict, train_test_split_index)
        """
        # Use init_ratio portion for training
        train_test_split = int(len(data) * self.init_ratio)

        # Ensure we have enough data for both train and test
        min_train_size = 2 * period  # At least 2 periods for training
        min_test_size = period  # At least 1 period for testing

        if train_test_split < min_train_size:
            train_test_split = min_train_size
        if len(data) - train_test_split < min_test_size:
            train_test_split = len(data) - min_test_size

        if train_test_split < 0 or train_test_split >= len(data):
            raise ValueError(
                f"Insufficient data length ({len(data)}) for period ({period}). "
                f"Need at least {min_train_size + min_test_size} points."
            )

        input_data = {
            "period": period,
            "trainTestSplit": train_test_split,
            "ts": data.tolist()
        }
        return input_data, train_test_split

    def _parse_output_json(self, output_data: Dict) -> Dict[str, np.ndarray]:
        """
        Parse OneShotSTL output JSON.

        OneShotSTL returns:
        {
            "trend": list[float],
            "seasonal": list[float],
            "residual": list[float]
        }

        Note: Output length may be shorter than input due to initialization window.
        """
        trend = np.array(output_data['trend'])
        seasonal = np.array(output_data['seasonal'])
        residual = np.array(output_data['residual'])

        return {
            'trend': trend,
            'seasonal': seasonal,
            'residual': residual
        }

    def decompose(self, data: np.ndarray, period: int = None) -> Dict[str, np.ndarray]:
        """
        Perform OneShotSTL decomposition.

        Args:
            data: Input time series array
            period: Seasonal period (uses self.period if None)

        Returns:
            Dictionary with 'trend', 'seasonal', 'residual', 'y' components

        Note:
            OneShotSTL may return shorter arrays than input due to initialization.
            The wrapper handles this by padding with the first values.
        """
        # Determine period
        if period is None:
            if self.period is None:
                # Default to detecting period (simple heuristic)
                period = min(len(data) // 4, 120)
            else:
                period = self.period

        # Ensure period is valid
        if period <= 0:
            raise ValueError(f"Period must be positive, got {period}")

        # Create temporary files for input/output
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            input_file = temp_dir / "input.json"
            output_file = temp_dir / "output.json"

            # Prepare and write input JSON
            input_data, train_test_split = self._prepare_input_json(data, period)
            with open(input_file, 'w') as f:
                json.dump(input_data, f)

            # Run OneShotSTL JAR
            cmd = [
                'java', '-jar', str(self.jar_path),
                '--method', 'OneShotSTL',
                '--task', 'decompose',
                '--shiftWindow', str(self.shift_window),
                '--in', str(input_file),
                '--out', str(output_file)
            ]

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minute timeout
                    check=True
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    f"OneShotSTL failed with return code {e.returncode}.\n"
                    f"stdout: {e.stdout}\n"
                    f"stderr: {e.stderr}"
                )
            except subprocess.TimeoutExpired:
                raise RuntimeError("OneShotSTL timed out after 5 minutes")

            # Read output JSON
            if not output_file.exists():
                raise RuntimeError(
                    f"OneShotSTL did not produce output file.\n"
                    f"stdout: {result.stdout}\n"
                    f"stderr: {result.stderr}"
                )

            with open(output_file, 'r') as f:
                output_data = json.load(f)

            # Parse output
            components = self._parse_output_json(output_data)

        # OneShotSTL returns decomposition only for test portion (data[train_test_split:])
        # We need to pad the beginning to match full data length
        output_len = len(components['trend'])
        expected_test_len = len(data) - train_test_split

        if output_len == 0:
            raise RuntimeError(
                f"OneShotSTL returned empty output. "
                f"Data length: {len(data)}, Period: {period}, "
                f"Train/test split: {train_test_split}"
            )

        if output_len != expected_test_len:
            # OneShotSTL might return slightly different length
            # Adjust by padding or truncating
            if output_len < expected_test_len:
                # Pad with last values
                pad_len = expected_test_len - output_len
                for key in ['trend', 'seasonal', 'residual']:
                    last_val = components[key][-1] if len(components[key]) > 0 else 0
                    padding = np.full(pad_len, last_val)
                    components[key] = np.concatenate([components[key], padding])
            else:
                # Truncate to expected length
                for key in ['trend', 'seasonal', 'residual']:
                    components[key] = components[key][:expected_test_len]

        # Pad the training portion at the beginning
        # Use simple strategy: extend first test value backwards
        for key in ['trend', 'seasonal', 'residual']:
            first_val = components[key][0]
            padding = np.full(train_test_split, first_val)
            components[key] = np.concatenate([padding, components[key]])

        # Add time and original data
        return {
            "time": np.arange(len(data)),
            "y": data,
            "trend": components['trend'],
            "seasonal": components['seasonal'],
            "residual": components['residual']
        }

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
