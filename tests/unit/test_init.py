"""Unit tests for __init__.py module."""

from unittest.mock import patch
from mcp_server_ds import main, __version__, SERVER_NAME


class TestInit:
    """Test __init__.py functionality."""

    def test_main_function_calls_server_main(self):
        """Test that main() calls server.main() (line 7)."""
        with patch("mcp_server_ds.server.main") as mock_server_main:
            main()
            mock_server_main.assert_called_once()

    def test_version_attribute_exists(self):
        """Test that __version__ attribute is accessible."""
        assert hasattr(__import__("mcp_server_ds"), "__version__")
        assert isinstance(__version__, str)

    def test_server_name_constant(self):
        """Test that SERVER_NAME constant is defined."""
        assert SERVER_NAME == "Data Science Explorer 🔬"

    def test_all_exports(self):
        """Test that __all__ contains expected exports."""
        import mcp_server_ds

        expected_exports = ["main", "server", "__version__", "SERVER_NAME"]
        assert hasattr(mcp_server_ds, "__all__")
        assert set(mcp_server_ds.__all__) == set(expected_exports)
