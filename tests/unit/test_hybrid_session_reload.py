"""
Unit tests for session reload behavior in HybridDataManager.

Verifies that sessions removed from memory are loaded back from disk on access,
and that eviction under constraints is recoverable by lazy loading.
"""

import pandas as pd

from mcp_server_ds.hybrid_data_manager import HybridDataManager
from tests.utils.mock_system_resources import (
    MockSystemResources,
    TestConfig,
    patch_system_resources,
    MockTempDirectory,
    create_mock_dataframe,
)


def test_reload_from_disk_after_manual_memory_removal():
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
                df = pd.DataFrame({"A": [1, 2, 3]})
                manager.set_dataframe("s1", "df", df)

                # Ensure present in memory, then remove from memory only
                assert manager._memory_manager.has_session("s1")
                manager._memory_manager.remove_session("s1")
                assert not manager._memory_manager.has_session("s1")

                # Access should reload from disk to memory
                out = manager.get_dataframe("s1", "df")
                assert out is not None
                assert manager._memory_manager.has_session("s1")
            finally:
                manager.close()


def test_reload_after_eviction_due_to_memory_constraints():
    with MockTempDirectory() as temp_dir:
        mock_resources = MockSystemResources()
        # Keep memory usage moderate; rely on max_sessions to trigger eviction
        mock_resources.set_memory_usage(50.0)
        with patch_system_resources(mock_resources):
            manager = HybridDataManager(
                memory_ttl_seconds=TestConfig.MEDIUM_TTL_SECONDS,
                filesystem_ttl_seconds=TestConfig.MEDIUM_TTL_SECONDS,
                memory_max_sessions=1,  # force eviction when adding second session
                memory_max_items_per_session=5,
                memory_threshold_percent=90.0,
                cache_dir=str(temp_dir),
                use_parquet=True,
                max_disk_usage_percent=90.0,
            )
            try:
                df1 = create_mock_dataframe(0.1)
                manager.set_dataframe("s_ev1", "df", df1)
                assert manager._memory_manager.has_session("s_ev1")

                # Adding a second session should evict the oldest since max_sessions=1
                df2 = create_mock_dataframe(0.1)
                manager.set_dataframe("s_ev2", "df", df2)
                assert manager._memory_manager.has_session("s_ev2")

                # s_ev1 may have been evicted from memory but remains on disk; access should reload it
                out = manager.get_dataframe("s_ev1", "df")
                assert out is not None
                assert manager._memory_manager.has_session("s_ev1")
            finally:
                manager.close()
