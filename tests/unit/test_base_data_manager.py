"""
Unit tests for Base Data Manager

Tests the abstract DataManager base class interface.
"""

import pytest

from mcp_server_ds.base_data_manager import DataManager
from mcp_server_ds.storage_types import StorageStats


class ConcreteDataManager(DataManager):
    """Concrete implementation of DataManager for testing."""

    def __init__(self):
        self.data = {}
        self.sizes = {}
        self.stats = StorageStats(
            total_sessions=0,
            total_items=0,
            total_size_bytes=0,
            memory_usage_percent=0.0,
            disk_usage_percent=0.0,
            tier_distribution={},
        )

    def get_session_data(self, session_id: str) -> dict:
        return dict(self.data.get(session_id, {}))

    def set_session_data(self, session_id: str, data: dict) -> None:
        self.data[session_id] = data

    def get_dataframe(self, session_id: str, df_name: str):
        session_data = self.data.get(session_id, {})
        return session_data.get(df_name)

    def set_dataframe(self, session_id: str, df_name: str, data) -> None:
        if session_id not in self.data:
            self.data[session_id] = {}
        self.data[session_id][df_name] = data

    def has_session(self, session_id: str) -> bool:
        return session_id in self.data

    def remove_session(self, session_id: str) -> None:
        self.data.pop(session_id, None)

    def get_dataframe_size(self, session_id: str, df_name: str) -> int:
        return int(self.sizes.get(f"{session_id}:{df_name}", 0))

    def get_session_size(self, session_id: str) -> int:
        session_data = self.data.get(session_id, {})
        return sum(
            self.sizes.get(f"{session_id}:{df_name}", 0) for df_name in session_data
        )

    def get_storage_stats(self) -> StorageStats:
        return self.stats

    def can_fit_in_memory(self, session_id: str, additional_size: int) -> bool:
        return additional_size < 1024 * 1024  # 1MB limit for testing

    def get_oldest_sessions(self, limit: int = 10) -> list[tuple[str, float]]:
        return [("session1", 1234567890.0), ("session2", 1234567891.0)][:limit]


class TestDataManager:
    """Test suite for DataManager abstract base class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that DataManager cannot be instantiated directly."""
        with pytest.raises(TypeError):
            DataManager()

    def test_concrete_implementation_works(self):
        """Test that concrete implementation works correctly."""
        manager = ConcreteDataManager()

        # Test basic operations
        manager.set_dataframe("session1", "df1", "data1")
        assert manager.get_dataframe("session1", "df1") == "data1"
        assert manager.has_session("session1")

        # Test session data
        session_data = manager.get_session_data("session1")
        assert session_data == {"df1": "data1"}

    def test_get_session_data_interface(self):
        """Test get_session_data interface."""
        manager = ConcreteDataManager()

        # Test with non-existent session
        result = manager.get_session_data("nonexistent")
        assert result == {}

        # Test with existing session
        manager.set_dataframe("session1", "df1", "data1")
        result = manager.get_session_data("session1")
        assert result == {"df1": "data1"}

    def test_set_session_data_interface(self):
        """Test set_session_data interface."""
        manager = ConcreteDataManager()

        data = {"df1": "data1", "df2": "data2"}
        manager.set_session_data("session1", data)

        result = manager.get_session_data("session1")
        assert result == data

    def test_get_dataframe_interface(self):
        """Test get_dataframe interface."""
        manager = ConcreteDataManager()

        # Test with non-existent session/dataframe
        result = manager.get_dataframe("nonexistent", "df1")
        assert result is None

        # Test with existing dataframe
        manager.set_dataframe("session1", "df1", "data1")
        result = manager.get_dataframe("session1", "df1")
        assert result == "data1"

    def test_set_dataframe_interface(self):
        """Test set_dataframe interface."""
        manager = ConcreteDataManager()

        manager.set_dataframe("session1", "df1", "data1")
        assert manager.get_dataframe("session1", "df1") == "data1"

        # Test overwriting
        manager.set_dataframe("session1", "df1", "new_data")
        assert manager.get_dataframe("session1", "df1") == "new_data"

    def test_has_session_interface(self):
        """Test has_session interface."""
        manager = ConcreteDataManager()

        # Test with non-existent session
        assert not manager.has_session("nonexistent")

        # Test with existing session
        manager.set_dataframe("session1", "df1", "data1")
        assert manager.has_session("session1")

    def test_remove_session_interface(self):
        """Test remove_session interface."""
        manager = ConcreteDataManager()

        # Test removing non-existent session (should not raise error)
        manager.remove_session("nonexistent")

        # Test removing existing session
        manager.set_dataframe("session1", "df1", "data1")
        assert manager.has_session("session1")

        manager.remove_session("session1")
        assert not manager.has_session("session1")

    def test_get_dataframe_size_interface(self):
        """Test get_dataframe_size interface."""
        manager = ConcreteDataManager()

        # Test with non-existent dataframe
        size = manager.get_dataframe_size("nonexistent", "df1")
        assert size == 0

        # Test with existing dataframe
        manager.sizes["session1:df1"] = 1024
        size = manager.get_dataframe_size("session1", "df1")
        assert size == 1024

    def test_get_session_size_interface(self):
        """Test get_session_size interface."""
        manager = ConcreteDataManager()

        # Test with non-existent session
        size = manager.get_session_size("nonexistent")
        assert size == 0

        # Test with existing session
        manager.set_dataframe("session1", "df1", "data1")
        manager.set_dataframe("session1", "df2", "data2")
        manager.sizes["session1:df1"] = 512
        manager.sizes["session1:df2"] = 256

        size = manager.get_session_size("session1")
        assert size == 768  # 512 + 256

    def test_get_storage_stats_interface(self):
        """Test get_storage_stats interface."""
        manager = ConcreteDataManager()

        stats = manager.get_storage_stats()
        assert isinstance(stats, StorageStats)
        assert stats.total_sessions == 0
        assert stats.total_items == 0
        assert stats.total_size_bytes == 0

    def test_can_fit_in_memory_interface(self):
        """Test can_fit_in_memory interface."""
        manager = ConcreteDataManager()

        # Test with small size
        can_fit = manager.can_fit_in_memory("session1", 1024)
        assert can_fit is True

        # Test with large size
        can_fit = manager.can_fit_in_memory("session1", 2 * 1024 * 1024)  # 2MB
        assert can_fit is False

    def test_get_oldest_sessions_interface(self):
        """Test get_oldest_sessions interface."""
        manager = ConcreteDataManager()

        # Test with default limit
        oldest = manager.get_oldest_sessions()
        assert isinstance(oldest, list)
        assert len(oldest) <= 10
        assert all(isinstance(item, tuple) for item in oldest)
        assert all(len(item) == 2 for item in oldest)
        assert all(isinstance(item[0], str) for item in oldest)
        assert all(isinstance(item[1], float) for item in oldest)

        # Test with custom limit
        oldest = manager.get_oldest_sessions(limit=1)
        assert len(oldest) <= 1

    def test_interface_consistency(self):
        """Test that all interface methods work together consistently."""
        manager = ConcreteDataManager()

        # Set up test data
        manager.set_dataframe("session1", "df1", "data1")
        manager.set_dataframe("session1", "df2", "data2")
        manager.sizes["session1:df1"] = 512
        manager.sizes["session1:df2"] = 256

        # Test consistency
        assert manager.has_session("session1")
        assert manager.get_dataframe("session1", "df1") == "data1"
        assert manager.get_dataframe("session1", "df2") == "data2"

        session_data = manager.get_session_data("session1")
        assert len(session_data) == 2
        assert "df1" in session_data
        assert "df2" in session_data

        assert manager.get_dataframe_size("session1", "df1") == 512
        assert manager.get_dataframe_size("session1", "df2") == 256
        assert manager.get_session_size("session1") == 768

        # Test removal
        manager.remove_session("session1")
        assert not manager.has_session("session1")
        assert manager.get_session_data("session1") == {}
        assert manager.get_session_size("session1") == 0

    def test_method_signatures(self):
        """Test that all methods have correct signatures."""
        manager = ConcreteDataManager()

        # Test that methods can be called with correct arguments
        manager.get_session_data("session1")
        manager.set_session_data("session1", {})
        manager.get_dataframe("session1", "df1")
        manager.set_dataframe("session1", "df1", "data1")
        manager.has_session("session1")
        manager.remove_session("session1")
        manager.get_dataframe_size("session1", "df1")
        manager.get_session_size("session1")
        manager.get_storage_stats()
        manager.can_fit_in_memory("session1", 1024)
        manager.get_oldest_sessions(limit=5)

        # All methods should execute without errors
        assert True
