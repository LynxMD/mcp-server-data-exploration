"""Unit tests for data_manager.py abstract base class."""

import pytest
from abc import ABC
from typing import Any
from mcp_server_ds.data_manager import DataManager


class TestDataManagerAbstract:
    """Test DataManager abstract base class."""

    def test_data_manager_is_abstract(self):
        """Test that DataManager is an abstract base class (lines 9-13)."""
        # Test that DataManager is an ABC
        assert issubclass(DataManager, ABC)

        # Test that DataManager cannot be instantiated directly
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            DataManager()

    def test_data_manager_has_required_abstract_methods(self):
        """Test that DataManager has all required abstract methods."""
        abstract_methods = DataManager.__abstractmethods__
        expected_methods = {
            "get_session_data",
            "set_session_data",
            "get_dataframe",
            "set_dataframe",
            "has_session",
            "remove_session",
        }
        assert abstract_methods == expected_methods

    def test_data_manager_method_signatures(self):
        """Test that DataManager methods have correct signatures."""
        # Test get_session_data signature
        import inspect

        sig = inspect.signature(DataManager.get_session_data)
        assert list(sig.parameters.keys()) == ["self", "session_id"]
        assert sig.return_annotation == dict[str, Any]

        # Test set_session_data signature
        sig = inspect.signature(DataManager.set_session_data)
        assert list(sig.parameters.keys()) == ["self", "session_id", "data"]
        assert sig.return_annotation is None

        # Test get_dataframe signature
        sig = inspect.signature(DataManager.get_dataframe)
        assert list(sig.parameters.keys()) == ["self", "session_id", "df_name"]
        assert sig.return_annotation == Any

        # Test set_dataframe signature
        sig = inspect.signature(DataManager.set_dataframe)
        assert list(sig.parameters.keys()) == ["self", "session_id", "df_name", "data"]
        assert sig.return_annotation is None

        # Test has_session signature
        sig = inspect.signature(DataManager.has_session)
        assert list(sig.parameters.keys()) == ["self", "session_id"]
        assert sig.return_annotation is bool

        # Test remove_session signature
        sig = inspect.signature(DataManager.remove_session)
        assert list(sig.parameters.keys()) == ["self", "session_id"]
        assert sig.return_annotation is None
