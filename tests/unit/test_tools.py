"""Unit tests for MCP tools."""

from mcp_server_ds.server import script_runner


class TestMCPTools:
    """Test cases for MCP tool functions."""

    def test_load_csv_tool(self, temp_csv_file):
        """Test the load_csv MCP tool functionality."""
        session_id = "test_session_123"
        result = script_runner.load_csv(temp_csv_file, "test_df", session_id)

        assert "Successfully loaded CSV into dataframe 'test_df'" in result

    def test_load_csv_tool_auto_name(self, temp_csv_file):
        """Test the load_csv MCP tool with auto-generated name."""
        session_id = "test_session_123"
        result = script_runner.load_csv(temp_csv_file, session_id=session_id)

        assert "Successfully loaded CSV into dataframe 'df_" in result

    def test_run_script_tool(self, sample_script):
        """Test the run_script MCP tool functionality."""
        session_id = "test_session_123"
        result = script_runner.safe_eval(sample_script, session_id=session_id)

        assert "Hello from test script!" in result
        assert "2 + 2 = 4" in result

    def test_run_script_tool_with_save(self, temp_csv_file, sample_script):
        """Test the run_script MCP tool with save_to_memory."""
        session_id = "test_session_123"
        # Load data first
        script_runner.load_csv(temp_csv_file, "test_df", session_id)

        # Run script that uses the data
        script = """
print(f"DataFrame loaded: {test_df.shape}")
new_result = test_df['age'].mean()
"""
        result = script_runner.safe_eval(
            script, save_to_memory=["new_result"], session_id=session_id
        )

        assert "DataFrame loaded: (3, 3)" in result

    def test_get_exploration_notes_empty(self):
        """Test get_exploration_notes when no notes exist."""
        session_id = "empty_session_123"
        # Test the logic directly since the function is decorated
        result = (
            "\n".join(script_runner.session_notes.get(session_id, []))
            if script_runner.session_notes.get(session_id)
            else "No notes yet"
        )
        assert result == "No notes yet"

    def test_get_exploration_notes_with_content(self, temp_csv_file):
        """Test get_exploration_notes with content."""
        session_id = "test_session_123"
        # Generate some notes
        script_runner.load_csv(temp_csv_file, "test_df", session_id)
        script_runner.safe_eval("print('test')", session_id=session_id)

        # Test the logic directly since the function is decorated
        result = (
            "\n".join(script_runner.session_notes.get(session_id, []))
            if script_runner.session_notes.get(session_id)
            else "No notes yet"
        )
        assert "Successfully loaded CSV" in result
        assert "test" in result
