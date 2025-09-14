"""Unit tests for ScriptRunner class."""

import pytest
import pandas as pd
from mcp_server_ds.server import ScriptRunner


class TestScriptRunner:
    """Test cases for ScriptRunner class."""

    def test_initialization(self):
        """Test that ScriptRunner initializes correctly."""
        runner = ScriptRunner()
        assert runner.data == {}
        assert runner.notes == []
        assert runner.df_count == 0

    def test_load_csv_success(self, script_runner, temp_csv_file):
        """Test successful CSV loading."""
        result = script_runner.load_csv(temp_csv_file, "test_df")

        assert "Successfully loaded CSV into dataframe 'test_df'" in result
        assert "test_df" in script_runner.data
        assert isinstance(script_runner.data["test_df"], pd.DataFrame)
        assert len(script_runner.data["test_df"]) == 3  # 3 rows in our test data

    def test_load_csv_auto_name(self, script_runner, temp_csv_file):
        """Test CSV loading with auto-generated name."""
        result = script_runner.load_csv(temp_csv_file)

        assert "Successfully loaded CSV into dataframe 'df_1'" in result
        assert "df_1" in script_runner.data

    def test_load_csv_nonexistent_file(self, script_runner):
        """Test CSV loading with non-existent file."""
        with pytest.raises(Exception, match="Error loading CSV"):
            script_runner.load_csv("nonexistent_file.csv")

    def test_safe_eval_simple_script(self, script_runner):
        """Test safe_eval with a simple script."""
        script = "result = 2 + 2\nprint(f'Result: {result}')"
        output = script_runner.safe_eval(script)

        assert "Result: 4" in output
        assert "Running script:" in script_runner.notes[-2]

    def test_safe_eval_with_dataframe(self, script_runner, temp_csv_file):
        """Test safe_eval with DataFrame operations."""
        # Load CSV first
        script_runner.load_csv(temp_csv_file, "test_df")

        # Run script that uses the DataFrame
        script = """
print(f"DataFrame shape: {test_df.shape}")
print(f"Columns: {list(test_df.columns)}")
print(f"First name: {test_df['name'].iloc[0]}")
"""
        output = script_runner.safe_eval(script)

        assert "DataFrame shape: (3, 3)" in output
        assert "Columns: ['name', 'age', 'city']" in output
        assert "First name: Alice" in output

    def test_safe_eval_save_to_memory(self, script_runner, temp_csv_file):
        """Test safe_eval with save_to_memory parameter."""
        # Load CSV first
        script_runner.load_csv(temp_csv_file, "original_df")

        # Run script that creates a new DataFrame
        script = """
new_df = original_df[original_df['age'] > 25].copy()
new_df['age_group'] = 'adult'
"""
        script_runner.safe_eval(script, save_to_memory=["new_df"])

        assert "new_df" in script_runner.data
        assert len(script_runner.data["new_df"]) == 2  # 2 rows with age > 25

    def test_safe_eval_error_handling(self, script_runner):
        """Test safe_eval error handling."""
        script = "undefined_variable + 1"

        with pytest.raises(Exception, match="name 'undefined_variable' is not defined"):
            script_runner.safe_eval(script)

    def test_notes_tracking(self, script_runner, temp_csv_file):
        """Test that notes are properly tracked."""
        initial_notes_count = len(script_runner.notes)

        script_runner.load_csv(temp_csv_file, "test_df")
        script_runner.safe_eval("print('test')")

        assert (
            len(script_runner.notes) == initial_notes_count + 3
        )  # load + script + result
        assert "Successfully loaded CSV" in script_runner.notes[-3]
        assert "Running script:" in script_runner.notes[-2]
        assert "Result:" in script_runner.notes[-1]
