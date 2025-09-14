"""Unit tests for ScriptRunner class."""

import pytest
import pandas as pd
from mcp_server_ds.server import ScriptRunner


class TestScriptRunner:
    """Test cases for ScriptRunner class."""

    def test_initialization(self):
        """Test that ScriptRunner initializes correctly."""
        runner = ScriptRunner()
        assert runner.session_data == {}
        assert runner.session_notes == {}
        assert runner.session_df_count == {}

    def test_load_csv_success(self, script_runner, temp_csv_file):
        """Test successful CSV loading."""
        session_id = "test_session_123"
        result = script_runner.load_csv(temp_csv_file, "test_df", session_id)

        assert "Successfully loaded CSV into dataframe 'test_df'" in result
        assert "test_df" in script_runner.session_data[session_id]
        assert isinstance(
            script_runner.session_data[session_id]["test_df"], pd.DataFrame
        )
        assert (
            len(script_runner.session_data[session_id]["test_df"]) == 3
        )  # 3 rows in our test data

    def test_load_csv_auto_name(self, script_runner, temp_csv_file):
        """Test CSV loading with auto-generated name."""
        session_id = "test_session_123"
        result = script_runner.load_csv(temp_csv_file, session_id=session_id)

        assert "Successfully loaded CSV into dataframe 'df_1'" in result
        assert "df_1" in script_runner.session_data[session_id]

    def test_load_csv_nonexistent_file(self, script_runner):
        """Test CSV loading with non-existent file."""
        session_id = "test_session_123"
        with pytest.raises(Exception, match="Error loading CSV"):
            script_runner.load_csv("nonexistent_file.csv", session_id=session_id)

    def test_safe_eval_simple_script(self, script_runner):
        """Test safe_eval with a simple script."""
        session_id = "test_session_123"
        script = "result = 2 + 2\nprint(f'Result: {result}')"
        output = script_runner.safe_eval(script, session_id=session_id)

        assert "Result: 4" in output
        assert (
            "Running script for session" in script_runner.session_notes[session_id][-2]
        )

    def test_safe_eval_with_dataframe(self, script_runner, temp_csv_file):
        """Test safe_eval with DataFrame operations."""
        session_id = "test_session_123"
        # Load CSV first
        script_runner.load_csv(temp_csv_file, "test_df", session_id)

        # Run script that uses the DataFrame
        script = """
print(f"DataFrame shape: {test_df.shape}")
print(f"Columns: {list(test_df.columns)}")
print(f"First name: {test_df['name'].iloc[0]}")
"""
        output = script_runner.safe_eval(script, session_id=session_id)

        assert "DataFrame shape: (3, 3)" in output
        assert "Columns: ['name', 'age', 'city']" in output
        assert "First name: Alice" in output

    def test_safe_eval_save_to_memory(self, script_runner, temp_csv_file):
        """Test safe_eval with save_to_memory parameter."""
        session_id = "test_session_123"
        # Load CSV first
        script_runner.load_csv(temp_csv_file, "original_df", session_id)

        # Run script that creates a new DataFrame
        script = """
new_df = original_df[original_df['age'] > 25].copy()
new_df['age_group'] = 'adult'
"""
        script_runner.safe_eval(
            script, save_to_memory=["new_df"], session_id=session_id
        )

        assert "new_df" in script_runner.session_data[session_id]
        assert (
            len(script_runner.session_data[session_id]["new_df"]) == 2
        )  # 2 rows with age > 25

    def test_safe_eval_error_handling(self, script_runner):
        """Test safe_eval error handling."""
        session_id = "test_session_123"
        script = "undefined_variable + 1"

        with pytest.raises(Exception, match="name 'undefined_variable' is not defined"):
            script_runner.safe_eval(script, session_id=session_id)

    def test_notes_tracking(self, script_runner, temp_csv_file):
        """Test that notes are properly tracked."""
        session_id = "test_session_123"
        initial_notes_count = len(script_runner.session_notes.get(session_id, []))

        script_runner.load_csv(temp_csv_file, "test_df", session_id)
        script_runner.safe_eval("print('test')", session_id=session_id)

        assert (
            len(script_runner.session_notes[session_id]) == initial_notes_count + 3
        )  # load + script + result
        assert "Successfully loaded CSV" in script_runner.session_notes[session_id][-3]
        assert (
            "Running script for session" in script_runner.session_notes[session_id][-2]
        )
        assert "Result for session" in script_runner.session_notes[session_id][-1]
