"""
Integration tests for ScriptRunner with DataManager abstraction.
"""

import pytest
import pandas as pd
from mcp_server_ds.server import ScriptRunner
from mcp_server_ds.ttl_in_memory_data_manager import TTLInMemoryDataManager
from mcp_server_ds.hybrid_data_manager import HybridDataManager


class TestScriptRunnerDataManagerIntegration:
    """Test ScriptRunner integration with DataManager."""

    def setup_method(self):
        """Set up test fixtures."""
        self.data_manager = TTLInMemoryDataManager()
        self.script_runner = ScriptRunner(data_manager=self.data_manager)
        self.session_id = "test_session_123"

    def test_script_runner_initialization_with_custom_data_manager(self):
        """Test ScriptRunner can be initialized with custom DataManager."""
        assert self.script_runner.data_manager is self.data_manager
        assert isinstance(self.script_runner.data_manager, TTLInMemoryDataManager)

    def test_script_runner_initialization_with_default_data_manager(self):
        """Test ScriptRunner initializes with a default DataManager implementation."""
        default_runner = ScriptRunner()
        assert isinstance(
            default_runner.data_manager, (TTLInMemoryDataManager, HybridDataManager)
        )

    def test_load_csv_uses_data_manager(self):
        """Test that load_csv uses the DataManager for storage."""
        # Create a temporary CSV file for testing
        test_data = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        csv_path = "/tmp/test_data.csv"
        test_data.to_csv(csv_path, index=False)

        try:
            # Load CSV
            result = self.script_runner.load_csv(csv_path, "test_df", self.session_id)

            # Verify success message
            assert "Successfully loaded CSV into dataframe 'test_df'" in result

            # Verify data is stored in DataManager
            stored_data = self.data_manager.get_dataframe(self.session_id, "test_df")
            assert stored_data is not None
            assert isinstance(stored_data, pd.DataFrame)
            assert len(stored_data) == 3
            assert list(stored_data.columns) == ["col1", "col2"]

        finally:
            # Clean up
            import os

            if os.path.exists(csv_path):
                os.remove(csv_path)

    def test_safe_eval_uses_data_manager_for_retrieval(self):
        """Test that safe_eval retrieves data from DataManager."""
        # Pre-populate DataManager with test data
        test_df = pd.DataFrame({"value": [10, 20, 30]})
        self.data_manager.set_dataframe(self.session_id, "test_df", test_df)

        # Run script that uses the DataFrame
        script = "result = test_df['value'].sum()\nprint(f'Sum: {result}')"
        result = self.script_runner.safe_eval(script, session_id=self.session_id)

        # Verify script executed and used the DataFrame
        assert "Sum: 60" in result

    def test_safe_eval_uses_data_manager_for_storage(self):
        """Test that safe_eval saves results to DataManager."""
        # Pre-populate DataManager with test data
        test_df = pd.DataFrame({"value": [1, 2, 3]})
        self.data_manager.set_dataframe(self.session_id, "input_df", test_df)

        # Run script that creates new DataFrame and saves it
        script = """
new_df = input_df.copy()
new_df['doubled'] = new_df['value'] * 2
print("Created new DataFrame")
"""
        result = self.script_runner.safe_eval(
            script, save_to_memory=["new_df"], session_id=self.session_id
        )

        # Verify script executed
        assert "Created new DataFrame" in result

        # Verify new DataFrame was saved to DataManager
        stored_df = self.data_manager.get_dataframe(self.session_id, "new_df")
        assert stored_df is not None
        assert isinstance(stored_df, pd.DataFrame)
        assert "doubled" in stored_df.columns
        assert list(stored_df["doubled"]) == [2, 4, 6]

    def test_session_isolation_with_data_manager(self):
        """Test that session isolation works with DataManager."""
        session1 = "session_1"
        session2 = "session_2"

        # Create test data
        test_data1 = pd.DataFrame({"col1": [1, 2, 3]})
        test_data2 = pd.DataFrame({"col1": [4, 5, 6]})

        # Store data in different sessions
        self.data_manager.set_dataframe(session1, "df1", test_data1)
        self.data_manager.set_dataframe(session2, "df1", test_data2)

        # Verify isolation in ScriptRunner
        session1_data = self.script_runner._get_session_data(session1)
        session2_data = self.script_runner._get_session_data(session2)

        assert session1_data["df1"].equals(test_data1)
        assert session2_data["df1"].equals(test_data2)
        assert not session1_data["df1"].equals(test_data2)
        assert not session2_data["df1"].equals(test_data1)

    def test_data_manager_persistence_across_script_runner_operations(self):
        """Test that data persists in DataManager across multiple ScriptRunner operations."""
        # Create test data
        test_df = pd.DataFrame({"numbers": [1, 2, 3, 4, 5]})
        self.data_manager.set_dataframe(self.session_id, "numbers_df", test_df)

        # Run multiple operations
        script1 = "total = numbers_df['numbers'].sum()\nprint(f'Total: {total}')"
        result1 = self.script_runner.safe_eval(script1, session_id=self.session_id)

        script2 = "average = numbers_df['numbers'].mean()\nprint(f'Average: {average}')"
        result2 = self.script_runner.safe_eval(script2, session_id=self.session_id)

        # Verify both operations worked and data persisted
        assert "Total: 15" in result1
        assert "Average: 3.0" in result2

        # Verify data is still in DataManager
        stored_df = self.data_manager.get_dataframe(self.session_id, "numbers_df")
        assert stored_df is not None
        assert len(stored_df) == 5

    def test_error_handling_with_data_manager(self):
        """Test that errors don't corrupt DataManager state."""
        # Pre-populate with valid data
        test_df = pd.DataFrame({"col1": [1, 2, 3]})
        self.data_manager.set_dataframe(self.session_id, "valid_df", test_df)

        # Run script with error
        error_script = "invalid_syntax_here["
        with pytest.raises(Exception):
            self.script_runner.safe_eval(error_script, session_id=self.session_id)

        # Verify original data is still intact
        stored_df = self.data_manager.get_dataframe(self.session_id, "valid_df")
        assert stored_df is not None
        assert len(stored_df) == 3
        assert list(stored_df["col1"]) == [1, 2, 3]

    def test_data_manager_interface_consistency(self):
        """Test that ScriptRunner uses DataManager interface consistently."""
        # Test that _get_session_data returns the same as DataManager.get_session_data
        test_df = pd.DataFrame({"test": [1, 2, 3]})
        self.data_manager.set_dataframe(self.session_id, "test_df", test_df)

        # Get data through both interfaces
        script_runner_data = self.script_runner._get_session_data(self.session_id)
        data_manager_data = self.data_manager.get_session_data(self.session_id)

        # Verify they return the same data
        assert script_runner_data == data_manager_data
        assert script_runner_data["test_df"].equals(test_df)
        assert data_manager_data["test_df"].equals(test_df)
