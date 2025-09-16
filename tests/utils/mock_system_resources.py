"""
Mock System Resources for Testing

This module provides utilities to mock system resources like memory, disk space,
and time for comprehensive testing of data manager implementations.
"""

import os
import time
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import shutil


class MockSystemResources:
    """Mock system resources for testing data manager behavior."""

    def __init__(self):
        self._mock_memory_usage = 50.0  # Default 50% memory usage
        self._mock_disk_usage = 50.0  # Default 50% disk usage
        self._mock_disk_total = 100 * 1024 * 1024 * 1024  # 100GB
        self._mock_memory_total = 8 * 1024 * 1024 * 1024  # 8GB
        self._current_time = time.time()
        self._time_advance = 0.0

    def set_memory_usage(self, percent: float) -> None:
        """Set mock memory usage percentage."""
        self._mock_memory_usage = percent

    def set_disk_usage(self, percent: float) -> None:
        """Set mock disk usage percentage."""
        self._mock_disk_usage = percent

    def set_memory_total(self, total_bytes: int) -> None:
        """Set mock total memory size."""
        self._mock_memory_total = total_bytes

    def set_disk_total(self, total_bytes: int) -> None:
        """Set mock total disk size."""
        self._mock_disk_total = total_bytes

    def advance_time(self, seconds: float) -> None:
        """Advance mock time by specified seconds."""
        self._time_advance += seconds

    def get_current_time(self) -> float:
        """Get current mock time."""
        return self._current_time + self._time_advance

    def reset_time(self) -> None:
        """Reset mock time to initial value."""
        self._time_advance = 0.0

    def create_mock_psutil(self) -> Mock:
        """Create mock psutil module."""
        mock_psutil = Mock()

        # Mock virtual_memory
        mock_memory = Mock()
        mock_memory.percent = self._mock_memory_usage
        mock_memory.total = self._mock_memory_total
        mock_memory.available = int(
            self._mock_memory_total * (100 - self._mock_memory_usage) / 100
        )
        mock_psutil.virtual_memory.return_value = mock_memory

        # Mock disk_usage
        mock_disk = Mock()
        mock_disk.percent = self._mock_disk_usage
        mock_disk.total = self._mock_disk_total
        mock_disk.used = int(self._mock_disk_total * self._mock_disk_usage / 100)
        mock_disk.free = int(
            self._mock_disk_total * (100 - self._mock_disk_usage) / 100
        )
        mock_psutil.disk_usage.return_value = mock_disk

        return mock_psutil

    def create_mock_time(self) -> Mock:
        """Create mock time module."""
        mock_time = Mock()
        mock_time.time.return_value = self.get_current_time()
        return mock_time


class MockTempDirectory:
    """Mock temporary directory for filesystem testing."""

    def __init__(self):
        self._temp_dir = None
        self._original_tempdir = None

    def __enter__(self):
        self._temp_dir = tempfile.mkdtemp()
        self._original_tempdir = tempfile.gettempdir()
        return Path(self._temp_dir)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._temp_dir and os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir)
        if self._original_tempdir:
            tempfile.tempdir = self._original_tempdir


def create_mock_dataframe(size_mb: float = 1.0) -> "pd.DataFrame":
    """Create a mock DataFrame of specified size."""
    import pandas as pd

    # Calculate number of rows needed for approximate size
    # Rough estimate: 100 bytes per row
    rows = int(size_mb * 1024 * 1024 / 100)

    return pd.DataFrame(
        {
            "id": range(rows),
            "data": [f"data_{i}" for i in range(rows)],
            "value": [i * 0.1 for i in range(rows)],
        }
    )


def create_mock_data(size_bytes: int) -> bytes:
    """Create mock data of specified size."""
    return b"x" * size_bytes


class TestConfig:
    """Configuration for test scenarios."""

    # Memory thresholds
    MEMORY_THRESHOLD_50_PERCENT = 50.0
    MEMORY_THRESHOLD_90_PERCENT = 90.0
    MEMORY_THRESHOLD_95_PERCENT = 95.0

    # Disk thresholds
    DISK_THRESHOLD_50_PERCENT = 50.0
    DISK_THRESHOLD_90_PERCENT = 90.0
    DISK_THRESHOLD_95_PERCENT = 95.0

    # Time configurations for testing
    SHORT_TTL_SECONDS = 10  # 10 seconds for fast testing
    MEDIUM_TTL_SECONDS = 30  # 30 seconds for medium testing
    LONG_TTL_SECONDS = 60  # 60 seconds for longer testing

    # Memory sizes for testing
    SMALL_MEMORY_TOTAL = 100 * 1024 * 1024  # 100MB
    MEDIUM_MEMORY_TOTAL = 500 * 1024 * 1024  # 500MB
    LARGE_MEMORY_TOTAL = 1024 * 1024 * 1024  # 1GB

    # Disk sizes for testing
    SMALL_DISK_TOTAL = 1024 * 1024 * 1024  # 1GB
    MEDIUM_DISK_TOTAL = 10 * 1024 * 1024 * 1024  # 10GB
    LARGE_DISK_TOTAL = 100 * 1024 * 1024 * 1024  # 100GB

    # Data sizes for testing
    SMALL_DATA_SIZE = 1024  # 1KB
    MEDIUM_DATA_SIZE = 1024 * 1024  # 1MB
    LARGE_DATA_SIZE = 10 * 1024 * 1024  # 10MB


def patch_system_resources(mock_resources: MockSystemResources):
    """Context manager to patch system resources."""
    return patch.multiple(
        "psutil",
        virtual_memory=mock_resources.create_mock_psutil().virtual_memory,
        disk_usage=mock_resources.create_mock_psutil().disk_usage,
    )
