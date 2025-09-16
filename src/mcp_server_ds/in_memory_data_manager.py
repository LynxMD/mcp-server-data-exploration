"""
In-Memory Data Manager Implementation

This module provides an in-memory implementation of the DataManager interface.
It replicates the current session_data behavior using a simple dictionary structure.
"""

from typing import Any

from .data_manager import DataManager


class InMemoryDataManager(DataManager):
    """
    In-memory implementation of DataManager.

    This implementation stores all session data in memory using nested dictionaries.
    It replicates the exact behavior of the current session_data implementation.
    """

    def __init__(self) -> None:
        """Initialize the in-memory data manager."""
        # Session-based data storage: {session_id: {df_name: DataFrame}}
        self._session_data: dict[str, dict[str, Any]] = {}

    def get_session_data(self, session_id: str) -> dict[str, Any]:
        """
        Get all data for a specific session.

        Args:
            session_id: The session identifier

        Returns:
            Dictionary mapping DataFrame names to their data
        """
        if session_id not in self._session_data:
            self._session_data[session_id] = {}
        return self._session_data[session_id]

    def set_session_data(self, session_id: str, data: dict[str, Any]) -> None:
        """
        Set all data for a specific session.

        Args:
            session_id: The session identifier
            data: Dictionary mapping DataFrame names to their data
        """
        self._session_data[session_id] = data.copy()

    def get_dataframe(self, session_id: str, df_name: str) -> Any:
        """
        Get a specific DataFrame from a session.

        Args:
            session_id: The session identifier
            df_name: The DataFrame name

        Returns:
            The DataFrame data, or None if not found
        """
        if session_id not in self._session_data:
            return None
        return self._session_data[session_id].get(df_name)

    def set_dataframe(self, session_id: str, df_name: str, data: Any) -> None:
        """
        Set a specific DataFrame in a session.

        Args:
            session_id: The session identifier
            df_name: The DataFrame name
            data: The DataFrame data
        """
        if session_id not in self._session_data:
            self._session_data[session_id] = {}
        self._session_data[session_id][df_name] = data

    def has_session(self, session_id: str) -> bool:
        """
        Check if a session exists.

        Args:
            session_id: The session identifier

        Returns:
            True if session exists, False otherwise
        """
        return session_id in self._session_data

    def remove_session(self, session_id: str) -> None:
        """
        Remove all data for a specific session.

        Args:
            session_id: The session identifier
        """
        if session_id in self._session_data:
            del self._session_data[session_id]
