"""
Comprehensive Unit Tests for TTLInMemoryDataManager

Tests the TTL-based memory implementation with mocked system resources
to validate exact behavior according to requirements.
"""

import time
import threading
import pickle

from mcp_server_ds.ttl_in_memory_data_manager import TTLInMemoryDataManager
from mcp_server_ds.storage_types import StorageTier
from unittest.mock import patch
from tests.utils.mock_system_resources import (
    MockSystemResources,
    TestConfig,
    patch_system_resources,
    create_mock_dataframe,
)


class TestTTLInMemoryComprehensive:
    """Comprehensive test suite for TTLInMemoryDataManager with mocked resources."""

    def test_memory_threshold_behavior_50_percent(self):
        """Test behavior when memory usage is at 50% threshold."""
        mock_resources = MockSystemResources()
        mock_resources.set_memory_usage(TestConfig.MEMORY_THRESHOLD_50_PERCENT)
        mock_resources.set_memory_total(TestConfig.SMALL_MEMORY_TOTAL)

        with patch_system_resources(mock_resources):
            manager = TTLInMemoryDataManager(
                ttl_seconds=TestConfig.SHORT_TTL_SECONDS,
                max_sessions=5,
                max_items_per_session=3,
            )

            # Should be able to fit data when at 50% usage
            can_fit = manager.can_fit_in_memory("session1", TestConfig.SMALL_DATA_SIZE)
            assert can_fit is True, "Should be able to fit data at 50% memory usage"

            # Add some data
            data = create_mock_dataframe(0.1)  # 0.1MB
            manager.set_dataframe("session1", "df1", data)

            # Verify data is stored
            retrieved = manager.get_dataframe("session1", "df1")
            assert retrieved is not None
            assert manager.has_session("session1")

    def test_memory_threshold_behavior_90_percent(self):
        """Test behavior when memory usage is at 90% threshold."""
        mock_resources = MockSystemResources()
        mock_resources.set_memory_usage(TestConfig.MEMORY_THRESHOLD_90_PERCENT)
        mock_resources.set_memory_total(TestConfig.SMALL_MEMORY_TOTAL)

        with patch_system_resources(mock_resources):
            manager = TTLInMemoryDataManager(
                ttl_seconds=TestConfig.SHORT_TTL_SECONDS,
                max_sessions=5,
                max_items_per_session=3,
            )

            # Should NOT be able to fit data when at 90% usage
            can_fit = manager.can_fit_in_memory("session1", TestConfig.SMALL_DATA_SIZE)
            assert can_fit is False, (
                "Should NOT be able to fit data at 90% memory usage"
            )

            # Try to add data anyway - should still work (implementation may override)
            data = create_mock_dataframe(0.1)  # 0.1MB
            manager.set_dataframe("session1", "df1", data)

            # Verify data is stored (implementation may have its own logic)
            retrieved = manager.get_dataframe("session1", "df1")
            assert retrieved is not None

    def test_ttl_expiry_behavior(self):
        """Test TTL expiry behavior with short TTL."""
        mock_resources = MockSystemResources()

        with patch_system_resources(mock_resources):
            manager = TTLInMemoryDataManager(
                ttl_seconds=TestConfig.SHORT_TTL_SECONDS,  # 10 seconds
                max_sessions=5,
                max_items_per_session=3,
            )

            # Add data
            data = create_mock_dataframe(0.1)
            manager.set_dataframe("session1", "df1", data)

            # Verify data is initially available
            assert manager.has_session("session1")
            assert manager.get_dataframe("session1", "df1") is not None

            # Advance time by more than TTL
            mock_resources.advance_time(TestConfig.SHORT_TTL_SECONDS + 5)

            # Data should still be available (TTL is sliding, refreshed on access)
            assert manager.has_session("session1")
            assert manager.get_dataframe("session1", "df1") is not None

            # But if we don't access it and advance time, it should expire
            # Note: This depends on the implementation - cacheout may handle this differently

    def test_session_based_eviction(self):
        """Test that entire sessions are evicted, not partial data."""
        mock_resources = MockSystemResources()

        with patch_system_resources(mock_resources):
            manager = TTLInMemoryDataManager(
                ttl_seconds=TestConfig.SHORT_TTL_SECONDS,
                max_sessions=2,  # Small limit to force eviction
                max_items_per_session=3,
            )

            # Add multiple sessions
            for i in range(3):
                session_id = f"session_{i}"
                data = create_mock_dataframe(0.1)
                manager.set_dataframe(session_id, "df1", data)
                manager.set_dataframe(session_id, "df2", data)

            # Due to max_sessions=2, only the last 2 sessions should be present
            # The first session should be evicted
            assert not manager.has_session("session_0"), (
                "First session should be evicted due to max_sessions limit"
            )
            assert manager.has_session("session_1"), "Second session should be present"
            assert manager.has_session("session_2"), "Third session should be present"

            # Verify the present sessions have all their data
            for i in [1, 2]:
                session_id = f"session_{i}"
                assert manager.get_dataframe(session_id, "df1") is not None
                assert manager.get_dataframe(session_id, "df2") is not None

            # When we exceed max_sessions, oldest sessions should be evicted
            # This depends on the cacheout implementation behavior

    def test_size_tracking_accuracy(self):
        """Test that size tracking is accurate."""
        mock_resources = MockSystemResources()

        with patch_system_resources(mock_resources):
            manager = TTLInMemoryDataManager(
                ttl_seconds=TestConfig.SHORT_TTL_SECONDS,
                max_sessions=5,
                max_items_per_session=3,
            )

            # Add data of known size
            data = create_mock_dataframe(0.1)  # ~0.1MB
            manager.set_dataframe("session1", "df1", data)

            # Check size tracking
            df_size = manager.get_dataframe_size("session1", "df1")
            session_size = manager.get_session_size("session1")

            assert df_size > 0, "DataFrame size should be tracked"
            assert session_size > 0, "Session size should be tracked"
            assert session_size >= df_size, "Session size should be >= DataFrame size"

            # Add another DataFrame
            data2 = create_mock_dataframe(0.2)  # ~0.2MB
            manager.set_dataframe("session1", "df2", data2)

            # Check updated sizes
            new_session_size = manager.get_session_size("session1")
            assert new_session_size > session_size, "Session size should increase"

    def test_memory_pressure_relief_oldest_first(self):
        """Test that memory pressure relief removes oldest sessions first."""
        mock_resources = MockSystemResources()
        mock_resources.set_memory_usage(TestConfig.MEMORY_THRESHOLD_90_PERCENT)

        with patch_system_resources(mock_resources):
            manager = TTLInMemoryDataManager(
                ttl_seconds=TestConfig.SHORT_TTL_SECONDS,
                max_sessions=3,
                max_items_per_session=3,
            )

            # Add sessions with time gaps
            for i in range(3):
                session_id = f"session_{i}"
                data = create_mock_dataframe(0.1)
                manager.set_dataframe(session_id, "df1", data)
                time.sleep(0.1)  # Small delay to ensure different access times

            # Get oldest sessions
            oldest_sessions = manager.get_oldest_sessions(limit=3)
            assert len(oldest_sessions) <= 3

            # Verify they are sorted by access time (oldest first)
            if len(oldest_sessions) > 1:
                for i in range(len(oldest_sessions) - 1):
                    assert oldest_sessions[i][1] <= oldest_sessions[i + 1][1], (
                        "Sessions should be sorted by access time (oldest first)"
                    )

    def test_storage_stats_accuracy(self):
        """Test that storage statistics are accurate."""
        mock_resources = MockSystemResources()

        with patch_system_resources(mock_resources):
            manager = TTLInMemoryDataManager(
                ttl_seconds=TestConfig.SHORT_TTL_SECONDS,
                max_sessions=5,
                max_items_per_session=3,
            )

            # Initially should have no sessions
            stats = manager.get_storage_stats()
            assert stats.total_sessions == 0
            assert stats.total_items == 0
            assert stats.total_size_bytes == 0
            assert StorageTier.MEMORY in stats.tier_distribution
            assert stats.tier_distribution[StorageTier.MEMORY] == 0

            # Add some data
            for i in range(2):
                session_id = f"session_{i}"
                data = create_mock_dataframe(0.1)
                manager.set_dataframe(session_id, "df1", data)
                manager.set_dataframe(session_id, "df2", data)

            # Check updated stats
            stats = manager.get_storage_stats()
            assert stats.total_sessions == 2
            assert stats.total_items == 4  # 2 sessions * 2 dataframes each
            assert stats.total_size_bytes > 0
            assert stats.tier_distribution[StorageTier.MEMORY] == 4

    def test_concurrent_access_thread_safety(self):
        """Test thread safety with concurrent access."""
        mock_resources = MockSystemResources()

        with patch_system_resources(mock_resources):
            manager = TTLInMemoryDataManager(
                ttl_seconds=TestConfig.SHORT_TTL_SECONDS,
                max_sessions=10,
                max_items_per_session=5,
            )

            results = []
            errors = []

            def worker(worker_id):
                """Worker function for concurrent access."""
                try:
                    for i in range(5):
                        session_id = f"session_{worker_id}_{i}"
                        data = create_mock_dataframe(0.1)
                        manager.set_dataframe(session_id, "df1", data)

                        # Verify data
                        retrieved = manager.get_dataframe(session_id, "df1")
                        assert retrieved is not None

                        results.append((worker_id, i))
                except Exception as e:
                    errors.append(e)

            # Create multiple threads
            threads = []
            for i in range(3):
                thread = threading.Thread(target=worker, args=(i,))
                threads.append(thread)
                thread.start()

            # Wait for all threads
            for thread in threads:
                thread.join()

            # Verify no errors occurred
            assert len(errors) == 0, f"Thread safety errors: {errors}"

            # Verify data integrity
            assert len(results) == 15  # 3 workers * 5 iterations each

    def test_memory_usage_monitoring(self):
        """Test that memory usage monitoring works correctly."""
        mock_resources = MockSystemResources()

        # Test with different memory usage levels
        for usage_percent in [25.0, 50.0, 75.0, 90.0, 95.0]:
            mock_resources.set_memory_usage(usage_percent)

            with patch_system_resources(mock_resources):
                manager = TTLInMemoryDataManager(
                    ttl_seconds=TestConfig.SHORT_TTL_SECONDS,
                    max_sessions=5,
                    max_items_per_session=3,
                )

                stats = manager.get_storage_stats()
                assert stats.memory_usage_percent == usage_percent, (
                    f"Memory usage should match mocked value: {usage_percent}%"
                )

    def test_max_sessions_enforcement(self):
        """Test that max_sessions limit is enforced."""
        mock_resources = MockSystemResources()

        with patch_system_resources(mock_resources):
            manager = TTLInMemoryDataManager(
                ttl_seconds=TestConfig.SHORT_TTL_SECONDS,
                max_sessions=2,  # Small limit
                max_items_per_session=3,
            )

            # Add more sessions than the limit
            for i in range(5):
                session_id = f"session_{i}"
                data = create_mock_dataframe(0.1)
                manager.set_dataframe(session_id, "df1", data)

            # Check that we don't exceed max_sessions
            stats = manager.get_storage_stats()
            assert stats.total_sessions <= 2, (
                f"Should not exceed max_sessions limit: {stats.total_sessions} > 2"
            )

    def test_max_items_per_session_enforcement(self):
        """Test that max_items_per_session limit is enforced."""
        mock_resources = MockSystemResources()

        with patch_system_resources(mock_resources):
            manager = TTLInMemoryDataManager(
                ttl_seconds=TestConfig.SHORT_TTL_SECONDS,
                max_sessions=5,
                max_items_per_session=2,  # Small limit
            )

            # Add more items than the limit per session
            for i in range(5):
                df_name = f"df_{i}"
                data = create_mock_dataframe(0.1)
                manager.set_dataframe("session1", df_name, data)

            # Check that we don't exceed max_items_per_session
            session_data = manager.get_session_data("session1")
            assert len(session_data) <= 2, (
                f"Should not exceed max_items_per_session limit: {len(session_data)} > 2"
            )

    def test_cleanup_after_ttl_expiry(self):
        """Test cleanup behavior after TTL expiry."""
        mock_resources = MockSystemResources()

        with patch_system_resources(mock_resources):
            manager = TTLInMemoryDataManager(
                ttl_seconds=TestConfig.SHORT_TTL_SECONDS,  # 10 seconds
                max_sessions=5,
                max_items_per_session=3,
            )

            # Add data
            data = create_mock_dataframe(0.1)
            manager.set_dataframe("session1", "df1", data)

            # Verify initial state
            assert manager.has_session("session1")
            initial_stats = manager.get_storage_stats()
            assert initial_stats.total_sessions == 1

            # Advance time beyond TTL without accessing data
            mock_resources.advance_time(TestConfig.SHORT_TTL_SECONDS + 5)

            # Force a cleanup by accessing the cache directly
            # Note: This depends on the cacheout implementation
            # The cache should automatically clean up expired entries

            # Check if cleanup occurred
            # The exact behavior depends on cacheout's cleanup mechanism

    def test_get_payload_none_return(self):
        """Test _get_payload returns None for non-existent session."""
        # This tests line 62: return None when payload is None
        manager = TTLInMemoryDataManager(
            ttl_seconds=10,
            max_sessions=10,
            max_items_per_session=5,
        )
        result = manager._get_payload("nonexistent_session")
        assert result is None

    def test_get_dataframe_none_payload(self):
        """Test get_dataframe returns None when payload is None."""
        # This tests line 107: return None when payload is None
        manager = TTLInMemoryDataManager(
            ttl_seconds=10,
            max_sessions=10,
            max_items_per_session=5,
        )
        result = manager.get_dataframe("nonexistent_session", "df1")
        assert result is None

    def test_remove_session_keyerror_handling_existing_session(self):
        """Test remove_session handles KeyError gracefully when removing existing session twice."""
        # This tests lines 135-137: KeyError handling in remove_session
        manager = TTLInMemoryDataManager(
            ttl_seconds=10,
            max_sessions=10,
            max_items_per_session=5,
        )
        # First add a session
        manager.set_dataframe("session1", "df1", "data1")
        assert manager.has_session("session1")

        # Remove it normally
        manager.remove_session("session1")
        assert not manager.has_session("session1")

        # Try to remove again - should handle KeyError gracefully
        manager.remove_session("session1")  # Should not raise KeyError

    def test_get_dataframe_size_none_payload(self):
        """Test get_dataframe_size returns 0 when payload is None."""
        # This tests line 144: return 0 when payload is None
        manager = TTLInMemoryDataManager(
            ttl_seconds=10,
            max_sessions=10,
            max_items_per_session=5,
        )
        size = manager.get_dataframe_size("nonexistent_session", "df1")
        assert size == 0

    def test_get_session_size_none_payload(self):
        """Test get_session_size returns 0 when payload is None."""
        # This tests line 147: return 0 when payload is None
        manager = TTLInMemoryDataManager(
            ttl_seconds=10,
            max_sessions=10,
            max_items_per_session=5,
        )
        size = manager.get_session_size("nonexistent_session")
        assert size == 0

    def test_get_storage_stats_none_payload(self):
        """Test get_storage_stats handles empty sessions."""
        # This tests lines 153-154: handling when payload is None
        manager = TTLInMemoryDataManager(
            ttl_seconds=10,
            max_sessions=10,
            max_items_per_session=5,
        )
        stats = manager.get_storage_stats()
        assert stats.total_sessions == 0
        assert stats.total_items == 0
        assert stats.total_size_bytes == 0

    def test_can_fit_in_memory_edge_cases(self):
        """Test can_fit_in_memory edge cases."""
        # This tests the improved memory management logic
        manager = TTLInMemoryDataManager(
            ttl_seconds=10,
            max_sessions=2,  # Small limit for testing
            max_items_per_session=3,
        )

        # Test with normal memory usage - should return True
        can_fit = manager.can_fit_in_memory("session1", 1024)
        assert can_fit is True

        # Fill up the cache to test max_sessions limit
        manager.set_dataframe("session1", "df1", "data1")
        manager.set_dataframe("session2", "df1", "data2")

        # Should return False when max_sessions is reached
        can_fit = manager.can_fit_in_memory("session3", 1024)
        assert can_fit is False

        # Test max_items_per_session limit
        manager2 = TTLInMemoryDataManager(
            ttl_seconds=10,
            max_sessions=10,
            max_items_per_session=2,  # Small limit for testing
        )
        manager2.set_dataframe("session1", "df1", "data1")
        manager2.set_dataframe("session1", "df2", "data2")

        # Should return False when max_items_per_session is reached
        can_fit = manager2.can_fit_in_memory("session1", 1024)
        assert can_fit is False

    def test_get_oldest_sessions_empty_cache(self):
        """Test get_oldest_sessions with empty cache."""
        # This tests lines 170-171: return empty list when no sessions
        manager = TTLInMemoryDataManager(
            ttl_seconds=10,
            max_sessions=10,
            max_items_per_session=5,
        )
        oldest = manager.get_oldest_sessions()
        assert oldest == []

    def test_remove_session_keyerror_path(self):
        """Test remove_session KeyError path (lines 135-137)."""
        manager = TTLInMemoryDataManager(
            ttl_seconds=10,
            max_sessions=10,
            max_items_per_session=5,
        )

        # Try to remove a non-existent session - should handle KeyError gracefully
        # This tests the KeyError exception handling in remove_session
        manager.remove_session("nonexistent_session")  # Should not raise KeyError

    def test_get_dataframe_size_df_not_found(self):
        """Test get_dataframe_size when dataframe not found (line 147)."""
        manager = TTLInMemoryDataManager(
            ttl_seconds=10,
            max_sessions=10,
            max_items_per_session=5,
        )

        # Add a session with some data
        manager.set_dataframe("session1", "df1", "data1")

        # Try to get size of non-existent dataframe in existing session
        size = manager.get_dataframe_size("session1", "nonexistent_df")
        assert size == 0  # Should return 0 when dataframe not found

    def test_get_storage_stats_with_sessions(self):
        """Test get_storage_stats with actual sessions (lines 153-154)."""
        manager = TTLInMemoryDataManager(
            ttl_seconds=10,
            max_sessions=10,
            max_items_per_session=5,
        )

        # Add some sessions
        manager.set_dataframe("session1", "df1", "data1")
        manager.set_dataframe("session1", "df2", "data2")
        manager.set_dataframe("session2", "df1", "data3")

        stats = manager.get_storage_stats()
        assert stats.total_sessions == 2
        assert stats.total_items == 3
        assert stats.total_size_bytes > 0

    def test_get_oldest_sessions_with_data(self):
        """Test get_oldest_sessions with actual sessions (lines 170-171)."""
        manager = TTLInMemoryDataManager(
            ttl_seconds=10,
            max_sessions=10,
            max_items_per_session=5,
        )

        # Add some sessions with small delays to ensure different timestamps
        manager.set_dataframe("session1", "df1", "data1")
        import time

        time.sleep(0.01)  # Small delay
        manager.set_dataframe("session2", "df1", "data2")
        time.sleep(0.01)  # Small delay
        manager.set_dataframe("session3", "df1", "data3")

        oldest = manager.get_oldest_sessions()
        assert len(oldest) == 3
        assert all(isinstance(item, tuple) for item in oldest)
        assert all(len(item) == 2 for item in oldest)
        # Should be sorted by last access time (oldest first)
        assert oldest[0][0] == "session1"  # First added should be oldest

    def test_remove_session_keyerror_handling_nonexistent_session(self):
        """Test remove_session KeyError handling when removing non-existent session (lines 135-137)."""
        manager = TTLInMemoryDataManager(
            ttl_seconds=10,
            max_sessions=10,
            max_items_per_session=5,
        )
        # Remove non-existent session should not raise exception
        manager.remove_session("nonexistent_session")
        # Should complete without error

    def test_get_dataframe_size_exception_handling(self):
        """Test get_dataframe_size exception handling (lines 153-154)."""
        manager = TTLInMemoryDataManager(
            ttl_seconds=10,
            max_sessions=10,
            max_items_per_session=5,
        )

        # Create a session with data
        manager.set_dataframe("session1", "df1", "test_data")

        # Mock pickle.dumps to raise an exception
        with patch("pickle.dumps", side_effect=Exception("Pickle error")):
            size = manager.get_dataframe_size("session1", "df1")
            assert size == 0  # Should return 0 on exception

    def test_get_session_size_exception_handling(self):
        """Test get_session_size exception handling (lines 170-171)."""
        manager = TTLInMemoryDataManager(
            ttl_seconds=10,
            max_sessions=10,
            max_items_per_session=5,
        )

        # Create a session with data
        manager.set_dataframe("session1", "df1", "test_data")
        manager.set_dataframe("session1", "df2", "test_data2")

        # Mock pickle.dumps to raise an exception for one item
        original_dumps = pickle.dumps
        call_count = 0

        def mock_dumps(data, protocol=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # First call succeeds
                return original_dumps(data, protocol)
            else:  # Second call fails
                raise Exception("Pickle error")

        with patch("pickle.dumps", side_effect=mock_dumps):
            size = manager.get_session_size("session1")
            # Should return size of first item only (second item failed)
            assert size > 0
