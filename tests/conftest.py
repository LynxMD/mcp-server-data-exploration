"""Shared pytest fixtures for MCP server tests."""

import pytest
import tempfile
import os
import pandas as pd

from mcp_server_ds.server import ScriptRunner


@pytest.fixture
def temp_csv_file():
    """Create a temporary CSV file for testing."""
    # Create sample data
    data = {
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35],
        "city": ["New York", "London", "Tokyo"],
    }
    df = pd.DataFrame(data)

    # Create temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        yield f.name

    # Cleanup
    os.unlink(f.name)


@pytest.fixture
def script_runner():
    """Create a fresh ScriptRunner instance for testing."""
    return ScriptRunner()


@pytest.fixture
def sample_script():
    """Return a simple Python script for testing."""
    return """
import pandas as pd
print("Hello from test script!")
result = 2 + 2
print(f"2 + 2 = {result}")
"""
