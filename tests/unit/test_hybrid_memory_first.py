"""
Unit tests for memory-first access behavior in HybridDataManager.

Covers:
- Immediate return from memory without touching disk
- Disk fallback when not present in memory
- Not-found returns None without raising
"""

from unittest.mock import patch
import pandas as pd

from mcp_server_ds.hybrid_data_manager import HybridDataManager
from tests.utils.mock_system_resources import (
    MockSystemResources,
    TestConfig,
    patch_system_resources,
    MockTempDirectory,
)


def test_memory_first_immediate_return_no_disk_access():
    with MockTempDirectory() as temp_dir:
        mock_resources = MockSystemResources()
        with patch_system_resources(mock_resources):
            manager = HybridDataManager(
                memory_ttl_seconds=TestConfig.MEDIUM_TTL_SECONDS,
                filesystem_ttl_seconds=TestConfig.MEDIUM_TTL_SECONDS,
                memory_max_sessions=5,
                memory_max_items_per_session=5,
                memory_threshold_percent=90.0,
                cache_dir=str(temp_dir),
                use_parquet=True,
                max_disk_usage_percent=90.0,
            )
            try:
                df = pd.DataFrame({"A": [1, 2]})
                manager.set_dataframe("s", "df", df)

                # Ensure it is in memory
                assert manager._memory_manager.has_session("s")

                # Disk path should not be called at all when data is in memory
                with patch.object(
                    manager._filesystem_manager, "get_dataframe"
                ) as disk_get:
                    out = manager.get_dataframe("s", "df")
                    pd.testing.assert_frame_equal(out, df)
                    disk_get.assert_not_called()
            finally:
                manager.close()


def test_disk_fallback_when_not_in_memory():
    with MockTempDirectory() as temp_dir:
        mock_resources = MockSystemResources()
        with patch_system_resources(mock_resources):
            manager = HybridDataManager(
                memory_ttl_seconds=TestConfig.MEDIUM_TTL_SECONDS,
                filesystem_ttl_seconds=TestConfig.MEDIUM_TTL_SECONDS,
                memory_max_sessions=5,
                memory_max_items_per_session=5,
                memory_threshold_percent=90.0,
                cache_dir=str(temp_dir),
                use_parquet=True,
                max_disk_usage_percent=90.0,
            )
            try:
                df = pd.DataFrame({"A": [3, 4]})
                manager.set_dataframe("s2", "df", df)

                # Remove from memory to force disk fallback
                manager._memory_manager.remove_session("s2")
                assert not manager._memory_manager.has_session("s2")

                out = manager.get_dataframe("s2", "df")
                assert out is not None
                pd.testing.assert_frame_equal(out, df)
            finally:
                manager.close()


def test_not_found_returns_none_without_raising():
    with MockTempDirectory() as temp_dir:
        mock_resources = MockSystemResources()
        with patch_system_resources(mock_resources):
            manager = HybridDataManager(cache_dir=str(temp_dir))
            try:
                out = manager.get_dataframe("missing_session", "missing_df")
                assert out is None
            finally:
                manager.close()
