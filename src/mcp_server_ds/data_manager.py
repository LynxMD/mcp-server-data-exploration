"""
Data Manager Abstraction

This module provides an abstraction for managing session-based data storage.
The abstraction allows different storage implementations (memory, filesystem, Redis, etc.)
while maintaining a consistent interface for the ScriptRunner.
"""

from abc import ABC, abstractmethod
from typing import Any


class DataManager(ABC):
    """
    Abstract base class for session data management.

    This class defines the interface that all data storage implementations
    must follow. The ScriptRunner uses this interface to store and retrieve
    session-specific DataFrames without knowing the underlying storage mechanism.
    """

    @abstractmethod
    def get_session_data(self, session_id: str) -> dict[str, Any]:
        """
        Get all data for a specific session.

        Args:
            session_id: The session identifier

        Returns:
            Dictionary mapping DataFrame names to their data
        """
        pass

    @abstractmethod
    def set_session_data(self, session_id: str, data: dict[str, Any]) -> None:
        """
        Set all data for a specific session.

        Args:
            session_id: The session identifier
            data: Dictionary mapping DataFrame names to their data
        """
        pass

    @abstractmethod
    def get_dataframe(self, session_id: str, df_name: str) -> Any:
        """
        Get a specific DataFrame from a session.

        Args:
            session_id: The session identifier
            df_name: The DataFrame name

        Returns:
            The DataFrame data, or None if not found
        """
        pass

    @abstractmethod
    def set_dataframe(self, session_id: str, df_name: str, data: Any) -> None:
        """
        Set a specific DataFrame in a session.

        Args:
            session_id: The session identifier
            df_name: The DataFrame name
            data: The DataFrame data
        """
        pass

    @abstractmethod
    def has_session(self, session_id: str) -> bool:
        """
        Check if a session exists.

        Args:
            session_id: The session identifier

        Returns:
            True if session exists, False otherwise
        """
        pass

    @abstractmethod
    def remove_session(self, session_id: str) -> None:
        """
        Remove all data for a specific session.

        Args:
            session_id: The session identifier
        """
        pass
