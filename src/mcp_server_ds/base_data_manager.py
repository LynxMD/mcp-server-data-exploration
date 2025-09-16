"""
Abstract Base Data Manager

This module contains the abstract base class that defines the interface
for all data storage implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Tuple

from .storage_types import StorageStats


class DataManager(ABC):
    """
    Abstract base class for session data management.

    This class defines the interface that all data storage implementations
    must follow. The ScriptRunner uses this interface to store and retrieve
    session-specific DataFrames without knowing the underlying storage mechanism.

    The interface is designed to support:
    - Size tracking for memory management
    - Session-based operations
    - Individual DataFrame operations
    - Storage statistics and monitoring
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

    # Enhanced interface for size tracking and memory management
    @abstractmethod
    def get_dataframe_size(self, session_id: str, df_name: str) -> int:
        """
        Get the size in bytes of a specific DataFrame.

        Args:
            session_id: The session identifier
            df_name: The DataFrame name

        Returns:
            Size in bytes, or 0 if not found
        """
        pass

    @abstractmethod
    def get_session_size(self, session_id: str) -> int:
        """
        Get the total size in bytes of all data in a session.

        Args:
            session_id: The session identifier

        Returns:
            Total size in bytes, or 0 if session doesn't exist
        """
        pass

    @abstractmethod
    def get_storage_stats(self) -> StorageStats:
        """
        Get comprehensive storage statistics.

        Returns:
            StorageStats object with detailed metrics
        """
        pass

    @abstractmethod
    def can_fit_in_memory(self, session_id: str, additional_size: int) -> bool:
        """
        Check if additional data can fit in memory without exceeding thresholds.

        Args:
            session_id: The session identifier
            additional_size: Size in bytes of data to be added

        Returns:
            True if data can fit, False otherwise
        """
        pass

    @abstractmethod
    def get_oldest_sessions(self, limit: int = 10) -> list[tuple[str, float]]:
        """
        Get the oldest sessions by last access time.

        Args:
            limit: Maximum number of sessions to return

        Returns:
            List of (session_id, last_access_time) tuples, sorted by oldest first
        """
        pass
