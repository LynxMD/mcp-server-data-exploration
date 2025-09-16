"""
Unit tests for DataManager abstraction and InMemoryDataManager implementation.
"""

import pytest
from mcp_server_ds.data_manager import DataManager
from mcp_server_ds.in_memory_data_manager import InMemoryDataManager


class TestDataManager:
    """Test the abstract DataManager interface."""

    def test_data_manager_is_abstract(self):
        """Test that DataManager cannot be instantiated directly."""
        with pytest.raises(TypeError):
            DataManager()


class TestInMemoryDataManager:
    """Test the InMemoryDataManager implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.data_manager = InMemoryDataManager()
        self.session_id = "test_session_123"
        self.df_name = "test_dataframe"
        self.test_data = {"column1": [1, 2, 3], "column2": ["a", "b", "c"]}

    def test_initialization(self):
        """Test that InMemoryDataManager initializes correctly."""
        assert isinstance(self.data_manager, InMemoryDataManager)
        assert isinstance(self.data_manager, DataManager)

    def test_get_session_data_new_session(self):
        """Test getting data for a new session returns empty dict."""
        result = self.data_manager.get_session_data(self.session_id)
        assert result == {}
        assert isinstance(result, dict)

    def test_set_session_data(self):
        """Test setting session data."""
        session_data = {self.df_name: self.test_data}
        self.data_manager.set_session_data(self.session_id, session_data)

        result = self.data_manager.get_session_data(self.session_id)
        assert result == session_data
        assert result[self.df_name] == self.test_data

    def test_set_session_data_overwrites(self):
        """Test that setting session data overwrites existing data."""
        # Set initial data
        initial_data = {"df1": {"col1": [1, 2]}}
        self.data_manager.set_session_data(self.session_id, initial_data)

        # Overwrite with new data
        new_data = {"df2": {"col2": [3, 4]}}
        self.data_manager.set_session_data(self.session_id, new_data)

        result = self.data_manager.get_session_data(self.session_id)
        assert result == new_data
        assert "df1" not in result
        assert "df2" in result

    def test_get_dataframe_existing(self):
        """Test getting an existing DataFrame."""
        self.data_manager.set_dataframe(self.session_id, self.df_name, self.test_data)

        result = self.data_manager.get_dataframe(self.session_id, self.df_name)
        assert result == self.test_data

    def test_get_dataframe_nonexistent_session(self):
        """Test getting DataFrame from non-existent session returns None."""
        result = self.data_manager.get_dataframe("nonexistent_session", self.df_name)
        assert result is None

    def test_get_dataframe_nonexistent_dataframe(self):
        """Test getting non-existent DataFrame returns None."""
        # Create session but don't add the DataFrame
        self.data_manager.get_session_data(self.session_id)

        result = self.data_manager.get_dataframe(self.session_id, "nonexistent_df")
        assert result is None

    def test_set_dataframe(self):
        """Test setting a DataFrame."""
        self.data_manager.set_dataframe(self.session_id, self.df_name, self.test_data)

        result = self.data_manager.get_dataframe(self.session_id, self.df_name)
        assert result == self.test_data

    def test_set_dataframe_overwrites(self):
        """Test that setting DataFrame overwrites existing DataFrame."""
        # Set initial DataFrame
        initial_data = {"col1": [1, 2]}
        self.data_manager.set_dataframe(self.session_id, self.df_name, initial_data)

        # Overwrite with new data
        new_data = {"col2": [3, 4]}
        self.data_manager.set_dataframe(self.session_id, self.df_name, new_data)

        result = self.data_manager.get_dataframe(self.session_id, self.df_name)
        assert result == new_data
        assert result != initial_data

    def test_has_session_existing(self):
        """Test has_session returns True for existing session."""
        self.data_manager.set_dataframe(self.session_id, self.df_name, self.test_data)

        assert self.data_manager.has_session(self.session_id) is True

    def test_has_session_nonexistent(self):
        """Test has_session returns False for non-existent session."""
        assert self.data_manager.has_session("nonexistent_session") is False

    def test_has_session_empty_session(self):
        """Test has_session returns True for empty session."""
        # Create empty session
        self.data_manager.get_session_data(self.session_id)

        assert self.data_manager.has_session(self.session_id) is True

    def test_remove_session_existing(self):
        """Test removing an existing session."""
        self.data_manager.set_dataframe(self.session_id, self.df_name, self.test_data)
        assert self.data_manager.has_session(self.session_id) is True

        self.data_manager.remove_session(self.session_id)
        assert self.data_manager.has_session(self.session_id) is False
        assert self.data_manager.get_dataframe(self.session_id, self.df_name) is None

    def test_remove_session_nonexistent(self):
        """Test removing a non-existent session doesn't raise error."""
        # Should not raise an exception
        self.data_manager.remove_session("nonexistent_session")

    def test_multiple_sessions_isolation(self):
        """Test that multiple sessions are properly isolated."""
        session1 = "session_1"
        session2 = "session_2"
        df1_name = "df1"
        df2_name = "df2"
        data1 = {"col1": [1, 2, 3]}
        data2 = {"col2": [4, 5, 6]}

        # Set data in different sessions
        self.data_manager.set_dataframe(session1, df1_name, data1)
        self.data_manager.set_dataframe(session2, df2_name, data2)

        # Verify isolation
        assert self.data_manager.get_dataframe(session1, df1_name) == data1
        assert self.data_manager.get_dataframe(session1, df2_name) is None
        assert self.data_manager.get_dataframe(session2, df1_name) is None
        assert self.data_manager.get_dataframe(session2, df2_name) == data2

        # Verify session data isolation
        session1_data = self.data_manager.get_session_data(session1)
        session2_data = self.data_manager.get_session_data(session2)

        assert session1_data == {df1_name: data1}
        assert session2_data == {df2_name: data2}

    def test_multiple_dataframes_per_session(self):
        """Test managing multiple DataFrames in a single session."""
        df1_name = "df1"
        df2_name = "df2"
        data1 = {"col1": [1, 2, 3]}
        data2 = {"col2": [4, 5, 6]}

        # Set multiple DataFrames in same session
        self.data_manager.set_dataframe(self.session_id, df1_name, data1)
        self.data_manager.set_dataframe(self.session_id, df2_name, data2)

        # Verify both DataFrames exist
        assert self.data_manager.get_dataframe(self.session_id, df1_name) == data1
        assert self.data_manager.get_dataframe(self.session_id, df2_name) == data2

        # Verify session data contains both
        session_data = self.data_manager.get_session_data(self.session_id)
        assert session_data == {df1_name: data1, df2_name: data2}

    def test_data_persistence_across_operations(self):
        """Test that data persists across multiple operations."""
        # Set initial data
        self.data_manager.set_dataframe(self.session_id, self.df_name, self.test_data)

        # Perform multiple operations
        session_data = self.data_manager.get_session_data(self.session_id)
        has_session = self.data_manager.has_session(self.session_id)
        dataframe = self.data_manager.get_dataframe(self.session_id, self.df_name)

        # Verify data persists
        assert session_data[self.df_name] == self.test_data
        assert has_session is True
        assert dataframe == self.test_data

    def test_set_session_data_creates_copy(self):
        """Test that set_session_data creates a copy of the input data."""
        original_data = {self.df_name: self.test_data}
        self.data_manager.set_session_data(self.session_id, original_data)

        # Modify original data
        # Keep value types consistent with existing mapping to satisfy typing
        original_data = {**original_data, "new_key": {"dummy": 1}}

        # Verify stored data is unchanged
        stored_data = self.data_manager.get_session_data(self.session_id)
        assert "new_key" not in stored_data
        assert stored_data == {self.df_name: self.test_data}
