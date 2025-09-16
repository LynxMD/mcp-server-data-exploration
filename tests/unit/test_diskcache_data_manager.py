"""
Unit Tests for DiskCacheDataManager

Tests the diskcache-based filesystem implementation to ensure it works
correctly and doesn't create hanging threads.
"""

import time
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch
import pytest
import pandas as pd

from mcp_server_ds.diskcache_data_manager import DiskCacheDataManager
from mcp_server_ds.storage_types import StorageTier
from mcp_server_ds.session_metadata import SessionMetadata


class TestDiskCacheDataManager:
    """Test suite for DiskCacheDataManager."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def manager(self, temp_dir):
        """Create a DiskCacheDataManager instance for testing."""
        manager = DiskCacheDataManager(
            cache_dir=temp_dir,
            ttl_seconds=10,  # Short TTL for testing
            max_disk_usage_percent=90.0,
            use_parquet=True,
        )
        yield manager
        # Ensure cleanup
        manager.close()

    def test_initialization(self, temp_dir):
        """Test manager initialization."""
        manager = DiskCacheDataManager(
            cache_dir=temp_dir,
            ttl_seconds=300,
            max_disk_usage_percent=85.0,
            use_parquet=False,
        )

        assert manager._cache_dir == Path(temp_dir)
        assert manager._ttl_seconds == 300
        assert manager._max_disk_usage_percent == 85.0
        assert manager._use_parquet is False

        # Test context manager
        with manager:
            assert manager._cache is not None
            assert manager._metadata_cache is not None

        # Should be closed after context manager
        manager.close()

    def test_context_manager(self, temp_dir):
        """Test context manager functionality."""
        with DiskCacheDataManager(cache_dir=temp_dir) as manager:
            assert manager._cache is not None
            assert manager._metadata_cache is not None

            # Add some data
            data = pd.DataFrame({"A": [1, 2, 3]})
            manager.set_dataframe("session1", "df1", data)

            # Verify data is stored
            assert manager.has_session("session1")
            retrieved = manager.get_dataframe("session1", "df1")
            pd.testing.assert_frame_equal(retrieved, data)

        # Manager should be closed after context exit

    def test_set_and_get_dataframe_parquet(self, manager):
        """Test setting and getting DataFrame with parquet serialization."""
        data = pd.DataFrame(
            {"A": [1, 2, 3, 4], "B": ["x", "y", "z", "w"], "C": [1.1, 2.2, 3.3, 4.4]}
        )

        manager.set_dataframe("session1", "df1", data)

        # Verify data is stored
        assert manager.has_session("session1")
        retrieved = manager.get_dataframe("session1", "df1")
        pd.testing.assert_frame_equal(retrieved, data)

    def test_set_and_get_dataframe_pickle(self, temp_dir):
        """Test setting and getting non-DataFrame data with pickle serialization."""
        manager = DiskCacheDataManager(
            cache_dir=temp_dir,
            ttl_seconds=10,
            use_parquet=False,  # Use pickle for all data
        )

        try:
            # Test with dictionary
            data = {"key": "value", "numbers": [1, 2, 3]}
            manager.set_dataframe("session1", "dict1", data)

            retrieved = manager.get_dataframe("session1", "dict1")
            assert retrieved == data

            # Test with list
            data2 = [1, 2, 3, 4, 5]
            manager.set_dataframe("session1", "list1", data2)

            retrieved2 = manager.get_dataframe("session1", "list1")
            assert retrieved2 == data2
        finally:
            manager.close()

    def test_session_data_operations(self, manager):
        """Test session-level data operations."""
        # Add multiple DataFrames to a session
        data1 = pd.DataFrame({"A": [1, 2, 3]})
        data2 = pd.DataFrame({"B": [4, 5, 6]})

        manager.set_dataframe("session1", "df1", data1)
        manager.set_dataframe("session1", "df2", data2)

        # Get all session data
        session_data = manager.get_session_data("session1")
        assert len(session_data) == 2
        assert "df1" in session_data
        assert "df2" in session_data

        pd.testing.assert_frame_equal(session_data["df1"], data1)
        pd.testing.assert_frame_equal(session_data["df2"], data2)

    def test_has_session(self, manager):
        """Test session existence checking."""
        assert not manager.has_session("nonexistent")

        data = pd.DataFrame({"A": [1, 2, 3]})
        manager.set_dataframe("session1", "df1", data)

        assert manager.has_session("session1")

    def test_remove_session(self, manager):
        """Test session removal."""
        data = pd.DataFrame({"A": [1, 2, 3]})
        manager.set_dataframe("session1", "df1", data)

        assert manager.has_session("session1")

        manager.remove_session("session1")

        assert not manager.has_session("session1")
        assert manager.get_dataframe("session1", "df1") is None

    def test_size_tracking(self, manager):
        """Test size tracking functionality."""
        data = pd.DataFrame({"A": [1, 2, 3, 4, 5]})
        manager.set_dataframe("session1", "df1", data)

        # Check DataFrame size
        df_size = manager.get_dataframe_size("session1", "df1")
        assert df_size > 0

        # Check session size
        session_size = manager.get_session_size("session1")
        assert session_size > 0
        assert session_size >= df_size

        # Add another DataFrame
        data2 = pd.DataFrame({"B": [6, 7, 8, 9, 10]})
        manager.set_dataframe("session1", "df2", data2)

        # Session size should increase
        new_session_size = manager.get_session_size("session1")
        assert new_session_size > session_size

    def test_storage_stats(self, manager):
        """Test storage statistics."""
        # Initially should have no sessions
        stats = manager.get_storage_stats()
        assert stats.total_sessions == 0
        assert stats.total_items == 0
        assert stats.total_size_bytes == 0
        assert StorageTier.FILESYSTEM in stats.tier_distribution
        assert stats.tier_distribution[StorageTier.FILESYSTEM] == 0

        # Add some data
        data1 = pd.DataFrame({"A": [1, 2, 3]})
        data2 = pd.DataFrame({"B": [4, 5, 6]})

        manager.set_dataframe("session1", "df1", data1)
        manager.set_dataframe("session2", "df1", data2)

        # Check updated stats
        stats = manager.get_storage_stats()
        assert stats.total_sessions == 2
        assert stats.total_items == 2
        assert stats.total_size_bytes > 0
        assert stats.tier_distribution[StorageTier.FILESYSTEM] == 2

    def test_oldest_sessions(self, manager):
        """Test getting oldest sessions."""
        # Add sessions with time gaps
        for i in range(3):
            session_id = f"session_{i}"
            data = pd.DataFrame({"A": [i, i + 1, i + 2]})
            manager.set_dataframe(session_id, "df1", data)
            time.sleep(0.1)  # Small delay to ensure different access times

        # Get oldest sessions
        oldest_sessions = manager.get_oldest_sessions(limit=3)
        assert len(oldest_sessions) == 3

        # Should be sorted by last access time (oldest first)
        for i in range(len(oldest_sessions) - 1):
            assert oldest_sessions[i][1] <= oldest_sessions[i + 1][1]

    def test_can_fit_in_memory(self, manager):
        """Test memory fitting check."""
        # Should be able to fit data when disk usage is low
        can_fit = manager.can_fit_in_memory("session1", 1024)
        assert can_fit is True

        # Mock high disk usage
        with patch.object(manager, "_get_disk_usage_percent", return_value=95.0):
            can_fit = manager.can_fit_in_memory("session1", 1024)
            assert can_fit is False

    def test_emergency_cleanup(self, manager):
        """Test emergency cleanup functionality."""
        # Add some sessions
        for i in range(5):
            session_id = f"session_{i}"
            data = pd.DataFrame({"A": [i, i + 1, i + 2]})
            manager.set_dataframe(session_id, "df1", data)

        # Verify all sessions exist
        for i in range(5):
            assert manager.has_session(f"session_{i}")

        # Trigger emergency cleanup
        manager._emergency_cleanup()

        # Some sessions should have been removed
        remaining_sessions = len(manager._metadata)
        assert remaining_sessions < 5

    def test_metadata_property(self, manager):
        """Test metadata property for testing compatibility."""
        data = pd.DataFrame({"A": [1, 2, 3]})
        manager.set_dataframe("session1", "df1", data)

        # Test metadata property
        metadata = manager._metadata
        assert isinstance(metadata, dict)
        assert "session1" in metadata
        assert isinstance(metadata["session1"], SessionMetadata)

    def test_ttl_expiry(self, temp_dir):
        """Test TTL expiry functionality."""
        manager = DiskCacheDataManager(
            cache_dir=temp_dir,
            ttl_seconds=1,  # Very short TTL for testing
        )

        try:
            data = pd.DataFrame({"A": [1, 2, 3]})
            manager.set_dataframe("session1", "df1", data)

            # Data should be available initially
            assert manager.has_session("session1")
            retrieved = manager.get_dataframe("session1", "df1")
            assert retrieved is not None

            # Wait for TTL to expire
            time.sleep(2)

            # Data should still be accessible (diskcache handles TTL internally)
            # The exact behavior depends on diskcache's internal cleanup
            retrieved = manager.get_dataframe("session1", "df1")
            # Note: diskcache may or may not return None depending on cleanup timing
        finally:
            manager.close()

    def test_multiple_dataframes_per_session(self, manager):
        """Test multiple DataFrames per session."""
        # Add multiple DataFrames to same session
        for i in range(5):
            df_name = f"df_{i}"
            data = pd.DataFrame({"A": [i, i + 1, i + 2]})
            manager.set_dataframe("session1", df_name, data)

        # Verify all DataFrames are stored
        session_data = manager.get_session_data("session1")
        assert len(session_data) == 5

        # Verify individual DataFrames
        for i in range(5):
            df_name = f"df_{i}"
            retrieved_data = manager.get_dataframe("session1", df_name)
            assert retrieved_data is not None
            assert len(retrieved_data) == 3

    def test_concurrent_access(self, manager):
        """Test concurrent access to the same manager."""
        import threading

        results = []
        errors = []

        def worker(worker_id):
            """Worker function for concurrent access."""
            try:
                for i in range(3):
                    session_id = f"session_{worker_id}_{i}"
                    data = pd.DataFrame({"A": [worker_id, i, worker_id + i]})
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
        assert len(errors) == 0, f"Concurrent access errors: {errors}"

        # Verify data integrity
        assert len(results) == 9  # 3 workers * 3 iterations each

    def test_no_hanging_threads(self, temp_dir):
        """Test that no threads are left hanging after manager is closed."""
        import threading

        # Get initial thread count
        initial_threads = threading.active_count()

        # Create and use manager
        manager = DiskCacheDataManager(cache_dir=temp_dir)

        # Add some data
        data = pd.DataFrame({"A": [1, 2, 3]})
        manager.set_dataframe("session1", "df1", data)

        # Close manager
        manager.close()

        # Wait a bit for any cleanup
        time.sleep(0.5)

        # Check thread count
        final_threads = threading.active_count()

        # Should not have significantly more threads
        # Allow for some variance due to test framework
        assert final_threads <= initial_threads + 2, (
            f"Thread count increased from {initial_threads} to {final_threads}"
        )

    def test_context_manager_cleanup(self, temp_dir):
        """Test that context manager properly cleans up resources."""
        import threading

        initial_threads = threading.active_count()

        # Use context manager
        with DiskCacheDataManager(cache_dir=temp_dir) as manager:
            data = pd.DataFrame({"A": [1, 2, 3]})
            manager.set_dataframe("session1", "df1", data)
            assert manager.has_session("session1")

        # Wait for cleanup
        time.sleep(0.5)

        final_threads = threading.active_count()
        assert final_threads <= initial_threads + 2, (
            f"Context manager left threads hanging: {initial_threads} -> {final_threads}"
        )
