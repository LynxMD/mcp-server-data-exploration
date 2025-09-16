"""
Tests for Session Isolation in MCP Server

Tests the session-based data storage and isolation functionality
to ensure users cannot access each other's data.
"""

import pytest
import pandas as pd
import tempfile
import os
from src.mcp_server_ds.server import (
    ScriptRunner,
)


class TestScriptRunnerSessionIsolation:
    """Test session isolation in ScriptRunner class."""

    def setup_method(self):
        """Set up a fresh ScriptRunner for each test."""
        self.script_runner = ScriptRunner()

    def test_session_data_isolation(self):
        """Test that data is isolated between different sessions."""
        session1 = "user_123_session_456"
        session2 = "user_789_session_012"

        # Create test CSV data
        test_data1 = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        test_data2 = pd.DataFrame({"col1": [4, 5, 6], "col2": ["d", "e", "f"]})

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f1:
            test_data1.to_csv(f1.name, index=False)
            csv_path1 = f1.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f2:
            test_data2.to_csv(f2.name, index=False)
            csv_path2 = f2.name

        try:
            # Load data for session1
            result1 = self.script_runner.load_csv(csv_path1, "df1", session1)
            assert "Successfully loaded CSV into dataframe 'df1'" in result1

            # Load data for session2
            result2 = self.script_runner.load_csv(csv_path2, "df1", session2)
            assert "Successfully loaded CSV into dataframe 'df1'" in result2

            # Verify data isolation
            session1_data = self.script_runner._get_session_data(session1)
            session2_data = self.script_runner._get_session_data(session2)

            assert "df1" in session1_data
            assert "df1" in session2_data

            # Data should be different
            assert not session1_data["df1"].equals(session2_data["df1"])

            # Verify session1 data is correct
            assert session1_data["df1"]["col1"].tolist() == [1, 2, 3]
            assert session1_data["df1"]["col2"].tolist() == ["a", "b", "c"]

            # Verify session2 data is correct
            assert session2_data["df1"]["col1"].tolist() == [4, 5, 6]
            assert session2_data["df1"]["col2"].tolist() == ["d", "e", "f"]

        finally:
            # Clean up temp files
            os.unlink(csv_path1)
            os.unlink(csv_path2)

    def test_session_notes_isolation(self):
        """Test that notes are isolated between different sessions."""
        session1 = "user_123_session_456"
        session2 = "user_789_session_012"

        # Create test CSV data
        test_data = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            test_data.to_csv(f.name, index=False)
            csv_path = f.name

        try:
            # Load data for session1
            self.script_runner.load_csv(csv_path, "df1", session1)

            # Load data for session2
            self.script_runner.load_csv(csv_path, "df1", session2)

            # Get notes for each session
            session1_notes = self.script_runner._get_session_notes(session1)
            session2_notes = self.script_runner._get_session_notes(session2)

            # Notes should be isolated
            assert len(session1_notes) == 1
            assert len(session2_notes) == 1

            assert "user_123_session_456" in session1_notes[0]
            assert "user_789_session_012" in session2_notes[0]

            # Notes should be different
            assert session1_notes[0] != session2_notes[0]

        finally:
            os.unlink(csv_path)

    def test_session_script_execution_isolation(self):
        """Test that script execution is isolated between sessions."""
        session1 = "user_123_session_456"
        session2 = "user_789_session_012"

        # Create test CSV data
        test_data1 = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        test_data2 = pd.DataFrame({"col1": [4, 5, 6], "col2": ["d", "e", "f"]})

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f1:
            test_data1.to_csv(f1.name, index=False)
            csv_path1 = f1.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f2:
            test_data2.to_csv(f2.name, index=False)
            csv_path2 = f2.name

        try:
            # Load data for both sessions
            self.script_runner.load_csv(csv_path1, "df1", session1)
            self.script_runner.load_csv(csv_path2, "df1", session2)

            # Execute script for session1
            script1 = "print(f'Session1 data shape: {df1.shape}'); print(f'Session1 col1 sum: {df1[\"col1\"].sum()}')"
            result1 = self.script_runner.safe_eval(script1, session_id=session1)

            # Execute script for session2
            script2 = "print(f'Session2 data shape: {df1.shape}'); print(f'Session2 col1 sum: {df1[\"col1\"].sum()}')"
            result2 = self.script_runner.safe_eval(script2, session_id=session2)

            # Results should be different
            assert "Session1 data shape: (3, 2)" in result1
            assert "Session1 col1 sum: 6" in result1

            assert "Session2 data shape: (3, 2)" in result2
            assert "Session2 col1 sum: 15" in result2

            # Verify notes are isolated
            session1_notes = self.script_runner._get_session_notes(session1)
            session2_notes = self.script_runner._get_session_notes(session2)

            assert len(session1_notes) == 3  # load_csv + safe_eval + result
            assert len(session2_notes) == 3  # load_csv + safe_eval + result

            assert "user_123_session_456" in session1_notes[1]  # safe_eval note
            assert "user_789_session_012" in session2_notes[1]  # safe_eval note

        finally:
            os.unlink(csv_path1)
            os.unlink(csv_path2)

    def test_session_dataframe_counter_isolation(self):
        """Test that DataFrame counters are isolated between sessions."""
        session1 = "user_123_session_456"
        session2 = "user_789_session_012"

        # Create test CSV data
        test_data = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            test_data.to_csv(f.name, index=False)
            csv_path = f.name

        try:
            # Load data for session1
            self.script_runner.load_csv(csv_path, session_id=session1)
            self.script_runner.load_csv(csv_path, session_id=session1)

            # Load data for session2
            self.script_runner.load_csv(csv_path, session_id=session2)

            # Check counters
            count1 = self.script_runner._get_session_df_count(session1)
            count2 = self.script_runner._get_session_df_count(session2)

            assert count1 == 2  # Two DataFrames loaded for session1
            assert count2 == 1  # One DataFrame loaded for session2

            # Check that auto-generated names are correct
            session1_data = self.script_runner._get_session_data(session1)
            session2_data = self.script_runner._get_session_data(session2)

            assert "df_1" in session1_data
            assert "df_2" in session1_data
            assert "df_1" in session2_data
            assert "df_2" not in session2_data

        finally:
            os.unlink(csv_path)

    def test_session_id_validation(self):
        """Test session ID validation."""
        # Test empty session_id
        with pytest.raises(
            ValueError, match="session_id is required for session isolation"
        ):
            self.script_runner.load_csv("dummy.csv", session_id=None)

        # Test empty string session_id
        with pytest.raises(
            ValueError, match="session_id is required for session isolation"
        ):
            self.script_runner.load_csv("dummy.csv", session_id="")

        # Test whitespace-only session_id
        with pytest.raises(ValueError, match="session_id must be a non-empty string"):
            self.script_runner.load_csv("dummy.csv", session_id="   ")

        # Test non-string session_id
        with pytest.raises(ValueError, match="session_id must be a non-empty string"):
            self.script_runner.load_csv("dummy.csv", session_id=123)  # type: ignore[arg-type]

    def test_session_id_whitespace_handling(self):
        """Test that session IDs are properly stripped of whitespace."""
        session_id = "  user_123_session_456  "
        test_data = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            test_data.to_csv(f.name, index=False)
            csv_path = f.name

        try:
            # Load data with whitespace in session_id
            result = self.script_runner.load_csv(csv_path, "df1", session_id)

            # Should work and strip whitespace
            assert "Successfully loaded CSV into dataframe 'df1'" in result

            # Check that data is stored with stripped session_id
            stripped_session_id = "user_123_session_456"
            session_data = self.script_runner._get_session_data(stripped_session_id)
            assert "df1" in session_data

        finally:
            os.unlink(csv_path)


class TestMCPToolsSessionIsolation:
    """Test session isolation in MCP tools."""

    @pytest.fixture(autouse=True)
    def setup_script_runner(self):
        """Set up script_runner for each test."""
        from mcp_server_ds.server import script_runner

        self.script_runner = script_runner

    def test_load_csv_tool_session_isolation(self):
        """Test that load_csv tool enforces session isolation."""
        session1 = "user_123_session_456"
        session2 = "user_789_session_012"

        # Create test CSV data
        test_data1 = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        test_data2 = pd.DataFrame({"col1": [4, 5, 6], "col2": ["d", "e", "f"]})

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f1:
            test_data1.to_csv(f1.name, index=False)
            csv_path1 = f1.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f2:
            test_data2.to_csv(f2.name, index=False)
            csv_path2 = f2.name

        try:
            # Load data for session1
            result1 = self.script_runner.load_csv(csv_path1, "df1", session1)
            assert "Successfully loaded CSV into dataframe 'df1'" in result1

            # Load data for session2
            result2 = self.script_runner.load_csv(csv_path2, "df1", session2)
            assert "Successfully loaded CSV into dataframe 'df1'" in result2

        finally:
            os.unlink(csv_path1)
            os.unlink(csv_path2)

    def test_load_csv_tool_validation(self):
        """Test that load_csv tool validates session_id."""
        # Test missing session_id
        with pytest.raises(
            ValueError, match="session_id is required for session isolation"
        ):
            self.script_runner.load_csv("dummy.csv", "df1", None)

        # Test empty session_id
        with pytest.raises(
            ValueError, match="session_id is required for session isolation"
        ):
            self.script_runner.load_csv("dummy.csv", "df1", "")

        # Test whitespace-only session_id
        with pytest.raises(ValueError, match="session_id must be a non-empty string"):
            self.script_runner.load_csv("dummy.csv", "df1", "   ")

    def test_run_script_tool_session_isolation(self):
        """Test that run_script tool enforces session isolation."""
        session1 = "user_123_session_456"
        session2 = "user_789_session_012"

        # Create test CSV data
        test_data1 = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        test_data2 = pd.DataFrame({"col1": [4, 5, 6], "col2": ["d", "e", "f"]})

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f1:
            test_data1.to_csv(f1.name, index=False)
            csv_path1 = f1.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f2:
            test_data2.to_csv(f2.name, index=False)
            csv_path2 = f2.name

        try:
            # Load data for both sessions
            self.script_runner.load_csv(csv_path1, "df1", session1)
            self.script_runner.load_csv(csv_path2, "df1", session2)

            # Execute script for session1
            script1 = "print(f'Session1 col1 sum: {df1[\"col1\"].sum()}')"
            result1 = self.script_runner.safe_eval(script1, session_id=session1)
            assert "Session1 col1 sum: 6" in result1

            # Execute script for session2
            script2 = "print(f'Session2 col1 sum: {df1[\"col1\"].sum()}')"
            result2 = self.script_runner.safe_eval(script2, session_id=session2)
            assert "Session2 col1 sum: 15" in result2

        finally:
            os.unlink(csv_path1)
            os.unlink(csv_path2)

    def test_run_script_tool_validation(self):
        """Test that run_script tool validates session_id."""
        # Test missing session_id
        with pytest.raises(
            ValueError, match="session_id is required for session isolation"
        ):
            self.script_runner.safe_eval("print('test')", session_id=None)

        # Test empty session_id
        with pytest.raises(
            ValueError, match="session_id is required for session isolation"
        ):
            self.script_runner.safe_eval("print('test')", session_id="")

        # Test whitespace-only session_id
        with pytest.raises(ValueError, match="session_id must be a non-empty string"):
            self.script_runner.safe_eval("print('test')", session_id="   ")


class TestMCPResourcesSessionIsolation:
    """Test session isolation in MCP resources."""

    @pytest.fixture(autouse=True)
    def setup_script_runner(self):
        """Set up script_runner for each test."""
        from mcp_server_ds.server import script_runner

        self.script_runner = script_runner

    def test_get_exploration_notes_session_isolation(self):
        """Test that get_exploration_notes is session-specific."""
        session1 = "user_123_session_456"
        session2 = "user_789_session_012"

        # Create test CSV data
        test_data = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            test_data.to_csv(f.name, index=False)
            csv_path = f.name

        try:
            # Load data for session1
            self.script_runner.load_csv(csv_path, "df1", session1)

            # Load data for session2
            self.script_runner.load_csv(csv_path, "df1", session2)

            # Get notes for each session
            notes1 = self.script_runner._get_session_notes(session1)
            notes2 = self.script_runner._get_session_notes(session2)

            # Notes should be different and session-specific
            assert any("user_123_session_456" in note for note in notes1)
            assert any("user_789_session_012" in note for note in notes2)
            assert notes1 != notes2

        finally:
            os.unlink(csv_path)

    def test_get_exploration_notes_validation(self):
        """Test that get_exploration_notes validates session_id."""
        # Test empty session_id
        result = self.script_runner._get_session_notes("")
        assert result == []

        # Test whitespace-only session_id
        result = self.script_runner._get_session_notes("   ")
        assert result == []


class TestSessionIsolationEdgeCases:
    """Test edge cases for session isolation."""

    @pytest.fixture(autouse=True)
    def setup_script_runner(self):
        """Set up script_runner for each test."""
        from mcp_server_ds.server import script_runner

        self.script_runner = script_runner

    def test_concurrent_session_access(self):
        """Test that concurrent access to different sessions works correctly."""
        session1 = "user_123_session_456"
        session2 = "user_789_session_012"

        # Create test CSV data
        test_data = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            test_data.to_csv(f.name, index=False)
            csv_path = f.name

        try:
            # Simulate concurrent access by alternating between sessions
            self.script_runner.load_csv(csv_path, "df1", session1)
            self.script_runner.load_csv(csv_path, "df1", session2)
            self.script_runner.load_csv(csv_path, "df2", session1)
            self.script_runner.load_csv(csv_path, "df2", session2)

            # Verify data isolation is maintained
            session1_data = self.script_runner._get_session_data(session1)
            session2_data = self.script_runner._get_session_data(session2)

            assert "df1" in session1_data
            assert "df2" in session1_data
            assert "df1" in session2_data
            assert "df2" in session2_data

            # All data should be identical (same CSV file)
            assert session1_data["df1"].equals(session2_data["df1"])
            assert session1_data["df2"].equals(session2_data["df2"])

        finally:
            os.unlink(csv_path)

    def test_session_data_persistence(self):
        """Test that session data persists across multiple operations."""
        session_id = "user_123_session_456"

        # Create test CSV data
        test_data = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            test_data.to_csv(f.name, index=False)
            csv_path = f.name

        try:
            # Load data
            self.script_runner.load_csv(csv_path, "df1", session_id)

            # Execute script that uses the data
            script = "df1['col3'] = df1['col1'] * 2; print('Data modified')"
            result = self.script_runner.safe_eval(script, session_id=session_id)
            assert "Data modified" in result

            # Execute another script that should see the modified data
            script2 = "print(f'col3 values: {df1[\"col3\"].tolist()}')"
            result2 = self.script_runner.safe_eval(script2, session_id=session_id)
            assert "col3 values: [2, 4, 6]" in result2

        finally:
            os.unlink(csv_path)

    def test_session_cleanup_simulation(self):
        """Test that sessions can be 'cleaned up' by removing from storage."""
        session_id = "user_123_session_456"

        # Create test CSV data
        test_data = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            test_data.to_csv(f.name, index=False)
            csv_path = f.name

        try:
            # Load data
            self.script_runner.load_csv(csv_path, "df1", session_id)

            # Verify data exists
            assert self.script_runner.data_manager.has_session(session_id)
            assert session_id in self.script_runner.session_notes

            # Simulate cleanup by removing session data
            self.script_runner.data_manager.remove_session(session_id)
            del self.script_runner.session_notes[session_id]
            del self.script_runner.session_df_count[session_id]

            # Verify session is cleaned up
            assert not self.script_runner.data_manager.has_session(session_id)
            assert session_id not in self.script_runner.session_notes
            assert session_id not in self.script_runner.session_df_count

        finally:
            os.unlink(csv_path)
