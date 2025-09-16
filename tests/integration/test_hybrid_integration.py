"""
Integration tests for HybridDataManager

Tests the complete hybrid storage system with real-world scenarios including
memory pressure, session eviction, lazy loading, and data persistence.
"""

import tempfile
import threading
from unittest.mock import patch

import pandas as pd
import pytest

from mcp_server_ds.hybrid_data_manager import HybridDataManager
from mcp_server_ds.storage_types import StorageTier


class TestHybridIntegration:
    """Integration test suite for HybridDataManager."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary cache directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def hybrid_manager(self, temp_cache_dir):
        """Create a HybridDataManager instance for testing."""
        return HybridDataManager(
            memory_ttl_seconds=60,  # 1 minute for testing
            filesystem_ttl_seconds=300,  # 5 minutes for testing
            memory_max_sessions=5,  # Small limit for testing
            memory_max_items_per_session=3,  # Small limit for testing
            memory_threshold_percent=80.0,  # Lower threshold for testing
            cache_dir=temp_cache_dir,
            use_parquet=True,
            max_disk_usage_percent=90.0,
        )

    def test_complete_session_lifecycle(self, hybrid_manager):
        """Test complete session lifecycle from creation to eviction."""
        session_id = "lifecycle_session"

        # Phase 1: Create session with data
        data1 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        data2 = pd.DataFrame({"C": [7, 8, 9], "D": [10, 11, 12]})

        hybrid_manager.set_dataframe(session_id, "df1", data1)
        hybrid_manager.set_dataframe(session_id, "df2", data2)

        # Verify data is in both memory and filesystem
        assert hybrid_manager._memory_manager.has_session(session_id)
        assert hybrid_manager._filesystem_manager.has_session(session_id)

        # Phase 2: Access data (should be fast from memory)
        retrieved_data1 = hybrid_manager.get_dataframe(session_id, "df1")
        retrieved_data2 = hybrid_manager.get_dataframe(session_id, "df2")

        pd.testing.assert_frame_equal(retrieved_data1, data1)
        pd.testing.assert_frame_equal(retrieved_data2, data2)

        # Phase 3: Force eviction from memory (simulate memory pressure)
        hybrid_manager._memory_manager.remove_session(session_id)
        assert not hybrid_manager._memory_manager.has_session(session_id)
        assert hybrid_manager._filesystem_manager.has_session(session_id)

        # Phase 4: Access data (should trigger lazy loading)
        retrieved_data1 = hybrid_manager.get_dataframe(session_id, "df1")
        retrieved_data2 = hybrid_manager.get_dataframe(session_id, "df2")

        pd.testing.assert_frame_equal(retrieved_data1, data1)
        pd.testing.assert_frame_equal(retrieved_data2, data2)

        # Verify data is back in memory
        assert hybrid_manager._memory_manager.has_session(session_id)

        # Phase 5: Complete removal
        hybrid_manager.remove_session(session_id)
        assert not hybrid_manager._memory_manager.has_session(session_id)
        assert not hybrid_manager._filesystem_manager.has_session(session_id)

    def test_memory_pressure_scenario(self, hybrid_manager):
        """Test memory pressure scenario with multiple sessions."""
        # Mock high memory usage
        with patch("psutil.virtual_memory") as mock_memory:
            mock_memory.return_value.percent = 95.0  # Above threshold

            # Create multiple sessions to fill memory
            sessions = []
            for i in range(8):  # More than memory_max_sessions
                session_id = f"session_{i}"
                sessions.append(session_id)

                # Add data that will exceed memory limits
                for j in range(5):  # More than memory_max_items_per_session
                    df_name = f"df_{j}"
                    data = pd.DataFrame(
                        {"A": [i, j, i + j], "B": [i * j, i + j, i - j]}
                    )
                    hybrid_manager.set_dataframe(session_id, df_name, data)

            # Verify some sessions are evicted from memory due to pressure
            memory_sessions = hybrid_manager.get_memory_sessions()
            assert len(memory_sessions) <= hybrid_manager._memory_manager._max_sessions

            # Verify all sessions still exist on disk
            for session_id in sessions:
                assert hybrid_manager._filesystem_manager.has_session(session_id)

            # Verify data can still be accessed (lazy loading)
            for session_id in sessions:
                retrieved_data = hybrid_manager.get_dataframe(session_id, "df_0")
                assert retrieved_data is not None

    def test_concurrent_access_scenario(self, hybrid_manager):
        """Test concurrent access to the hybrid storage system."""
        session_id = "concurrent_session"
        data = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

        # Set initial data
        hybrid_manager.set_dataframe(session_id, "df", data)

        results = []
        errors = []

        def worker(worker_id):
            """Worker function for concurrent access."""
            try:
                for i in range(10):
                    # Read data
                    retrieved_data = hybrid_manager.get_dataframe(session_id, "df")
                    assert retrieved_data is not None

                    # Write new data
                    new_data = pd.DataFrame({"A": [worker_id, i, worker_id + i]})
                    hybrid_manager.set_dataframe(
                        session_id, f"df_{worker_id}_{i}", new_data
                    )

                    results.append((worker_id, i))
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify no errors occurred
        assert len(errors) == 0, f"Errors occurred: {errors}"

        # Verify data integrity
        session_data = hybrid_manager.get_session_data(session_id)
        assert len(session_data) > 0

        # Verify original data is still accessible
        original_data = hybrid_manager.get_dataframe(session_id, "df")
        pd.testing.assert_frame_equal(original_data, data)

    def test_persistence_across_restarts(self, hybrid_manager, temp_cache_dir):
        """Test data persistence across manager restarts."""
        session_id = "persistent_session"
        data = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

        # Add data
        hybrid_manager.set_dataframe(session_id, "df", data)

        # Verify data is in both storage tiers
        assert hybrid_manager._memory_manager.has_session(session_id)
        assert hybrid_manager._filesystem_manager.has_session(session_id)

        # Simulate restart by creating new manager instance
        new_manager = HybridDataManager(
            memory_ttl_seconds=60,
            filesystem_ttl_seconds=300,
            memory_max_sessions=5,
            memory_max_items_per_session=3,
            memory_threshold_percent=80.0,
            cache_dir=temp_cache_dir,
            use_parquet=True,
            max_disk_usage_percent=90.0,
        )

        # Data should not be in memory (fresh start)
        assert not new_manager._memory_manager.has_session(session_id)

        # Data should be on disk
        assert new_manager._filesystem_manager.has_session(session_id)

        # Accessing data should trigger lazy loading
        retrieved_data = new_manager.get_dataframe(session_id, "df")
        pd.testing.assert_frame_equal(retrieved_data, data)

        # Data should now be in memory
        assert new_manager._memory_manager.has_session(session_id)

    def test_mixed_data_types(self, hybrid_manager):
        """Test storage of mixed data types (DataFrames and other objects)."""
        session_id = "mixed_session"

        # DataFrame (should use parquet)
        df_data = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        hybrid_manager.set_dataframe(session_id, "df", df_data)

        # Dictionary (should use pickle)
        dict_data = {"key": "value", "numbers": [1, 2, 3]}
        hybrid_manager.set_dataframe(session_id, "dict", dict_data)

        # List (should use pickle)
        list_data = [1, 2, 3, "string", {"nested": "dict"}]
        hybrid_manager.set_dataframe(session_id, "list", list_data)

        # Verify all data can be retrieved correctly
        retrieved_df = hybrid_manager.get_dataframe(session_id, "df")
        retrieved_dict = hybrid_manager.get_dataframe(session_id, "dict")
        retrieved_list = hybrid_manager.get_dataframe(session_id, "list")

        pd.testing.assert_frame_equal(retrieved_df, df_data)
        assert retrieved_dict == dict_data
        assert retrieved_list == list_data

        # Verify files were created with correct extensions
        parquet_path = hybrid_manager._filesystem_manager._get_parquet_file_path(
            session_id, "df"
        )
        pickle_path = hybrid_manager._filesystem_manager._get_data_file_path(
            session_id, "dict"
        )

        assert parquet_path.exists()
        assert pickle_path.exists()

    def test_size_aware_memory_management(self, hybrid_manager):
        """Test size-aware memory management."""
        session_id = "size_test_session"

        # Create large DataFrame
        large_data = pd.DataFrame(
            {
                "A": list(range(1000)),
                "B": list(range(1000, 2000)),
                "C": [f"string_{i}" for i in range(1000)],
            }
        )

        # Check if we can fit it in memory
        can_fit = hybrid_manager.can_fit_in_memory(session_id, 1024 * 1024)  # 1MB
        assert isinstance(can_fit, bool)

        # Add the large DataFrame
        hybrid_manager.set_dataframe(session_id, "large_df", large_data)

        # Verify size tracking
        df_size = hybrid_manager.get_dataframe_size(session_id, "large_df")
        session_size = hybrid_manager.get_session_size(session_id)

        assert df_size > 0
        assert session_size >= df_size

        # Verify storage stats reflect the size
        stats = hybrid_manager.get_storage_stats()
        assert stats.total_size_bytes >= df_size

    def test_emergency_cleanup_scenario(self, hybrid_manager):
        """Test emergency cleanup when disk usage is high."""
        # Add many sessions to fill disk
        for i in range(20):
            session_id = f"disk_session_{i}"
            data = pd.DataFrame({"A": [i, i + 1, i + 2], "B": [i * 2, i * 3, i * 4]})
            hybrid_manager.set_dataframe(session_id, "df", data)

        # Mock high disk usage
        with patch("psutil.disk_usage") as mock_disk:
            mock_disk.return_value.percent = 95.0  # Above threshold

            # Trigger emergency cleanup
            hybrid_manager._filesystem_manager._emergency_cleanup()

            # Some sessions should have been removed
            # (exact number depends on implementation details)
            assert len(hybrid_manager._filesystem_manager._metadata) < 20

    def test_ttl_expiration_scenario(self, hybrid_manager):
        """Test TTL expiration for both memory and filesystem."""
        session_id = "ttl_session"
        data = pd.DataFrame({"A": [1, 2, 3]})

        # Add data
        hybrid_manager.set_dataframe(session_id, "df", data)

        # Verify data is in both tiers
        assert hybrid_manager._memory_manager.has_session(session_id)
        assert hybrid_manager._filesystem_manager.has_session(session_id)

        # Manually expire memory session
        hybrid_manager._memory_manager._sessions.delete(session_id)
        assert not hybrid_manager._memory_manager.has_session(session_id)

        # Data should still be accessible from disk
        retrieved_data = hybrid_manager.get_dataframe(session_id, "df")
        pd.testing.assert_frame_equal(retrieved_data, data)

        # Data should be back in memory after access
        assert hybrid_manager._memory_manager.has_session(session_id)

    def test_force_load_scenario(self, hybrid_manager):
        """Test forcing a session to load into memory."""
        session_id = "force_load_session"
        data = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

        # Add data
        hybrid_manager.set_dataframe(session_id, "df", data)

        # Remove from memory
        hybrid_manager._memory_manager.remove_session(session_id)
        assert not hybrid_manager._memory_manager.has_session(session_id)

        # Force load to memory
        success = hybrid_manager.force_load_session_to_memory(session_id)
        assert success
        assert hybrid_manager._memory_manager.has_session(session_id)

        # Verify data is accessible
        retrieved_data = hybrid_manager.get_dataframe(session_id, "df")
        pd.testing.assert_frame_equal(retrieved_data, data)

    def test_storage_tier_distribution(self, hybrid_manager):
        """Test storage tier distribution in statistics."""
        # Add some sessions
        for i in range(3):
            session_id = f"stats_session_{i}"
            data = pd.DataFrame({"A": [i, i + 1, i + 2]})
            hybrid_manager.set_dataframe(session_id, "df", data)

        # Get storage stats
        stats = hybrid_manager.get_storage_stats()

        # Verify tier distribution
        assert StorageTier.MEMORY in stats.tier_distribution
        assert StorageTier.FILESYSTEM in stats.tier_distribution

        # Both tiers should have data
        assert stats.tier_distribution[StorageTier.MEMORY] > 0
        assert stats.tier_distribution[StorageTier.FILESYSTEM] > 0

        # Total items should match
        total_items = (
            stats.tier_distribution[StorageTier.MEMORY]
            + stats.tier_distribution[StorageTier.FILESYSTEM]
        )
        assert total_items >= 3  # At least our test data

    def test_error_recovery_scenario(self, hybrid_manager):
        """Test error recovery scenarios."""
        session_id = "error_session"
        data = pd.DataFrame({"A": [1, 2, 3]})

        # Add data
        hybrid_manager.set_dataframe(session_id, "df", data)

        # Simulate filesystem error by corrupting the file
        parquet_path = hybrid_manager._filesystem_manager._get_parquet_file_path(
            session_id, "df"
        )
        with open(parquet_path, "w") as f:
            f.write("corrupted data")

        # Data should still be accessible from memory
        retrieved_data = hybrid_manager.get_dataframe(session_id, "df")
        pd.testing.assert_frame_equal(retrieved_data, data)

        # Remove from memory to test filesystem fallback
        hybrid_manager._memory_manager.remove_session(session_id)

        # Should return None for corrupted file
        retrieved_data = hybrid_manager.get_dataframe(session_id, "df")
        assert retrieved_data is None
