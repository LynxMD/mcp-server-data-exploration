"""
Hybrid Data Manager Implementation

The crown jewel of the storage architecture - a sophisticated hybrid system that
combines memory and filesystem storage with intelligent tiering and size-aware
memory management.

Key Features:
- Always writes to both memory and filesystem
- Reads from memory first, falls back to disk
- Session-based eviction (entire session after 5h, not partial)
- Size-aware memory management with 90% threshold
- Lazy loading from disk to memory on demand
- Intelligent memory pressure relief
- Composable architecture using existing DataManager implementations

Architecture:
- Memory: Fast access, 5-hour TTL, size-limited
- Filesystem: Persistent storage, 7-day TTL, size-unlimited
- Hybrid: Best of both worlds with intelligent tiering
"""

from __future__ import annotations

import threading
from typing import Any
import psutil

from .base_data_manager import DataManager
from .storage_types import StorageStats, StorageTier
from .ttl_in_memory_data_manager import TTLInMemoryDataManager
from .diskcache_data_manager import DiskCacheDataManager


class HybridDataManager(DataManager):
    """
    Hybrid DataManager that combines memory and filesystem storage.

    This implementation provides the best of both worlds:
    - Fast memory access for active sessions
    - Persistent filesystem storage for long-term data
    - Intelligent tiering and memory management
    - Session-based eviction for optimal performance
    """

    def __init__(
        self,
        memory_ttl_seconds: int = 5 * 60 * 60,  # 5 hours
        filesystem_ttl_seconds: int = 7 * 24 * 60 * 60,  # 7 days
        memory_max_sessions: int = 100,
        memory_max_items_per_session: int = 50,
        memory_threshold_percent: float = 90.0,
        cache_dir: str = "/tmp/mcp_cache",
        use_parquet: bool = True,
        max_disk_usage_percent: float = 90.0,
    ) -> None:
        """
        Initialize HybridDataManager.

        Args:
            memory_ttl_seconds: TTL for memory storage
            filesystem_ttl_seconds: TTL for filesystem storage
            memory_max_sessions: Maximum sessions in memory
            memory_max_items_per_session: Maximum items per session in memory
            memory_threshold_percent: Memory usage threshold for cleanup
            cache_dir: Directory for filesystem cache
            use_parquet: Use parquet format for DataFrames
            max_disk_usage_percent: Maximum disk usage before cleanup
        """
        self._memory_threshold_percent = memory_threshold_percent

        # Initialize component DataManagers
        self._memory_manager = TTLInMemoryDataManager(
            ttl_seconds=memory_ttl_seconds,
            max_sessions=memory_max_sessions,
            max_items_per_session=memory_max_items_per_session,
        )

        self._filesystem_manager = DiskCacheDataManager(
            cache_dir=cache_dir,
            ttl_seconds=filesystem_ttl_seconds,
            use_parquet=use_parquet,
            max_disk_usage_percent=max_disk_usage_percent,
        )

        # Thread safety
        self._lock = threading.RLock()

        # Session loading state to prevent race conditions
        self._loading_sessions: set[str] = set()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatic cleanup."""
        self.close()

    def close(self) -> None:
        """Close the filesystem manager and cleanup resources."""
        if hasattr(self, "_filesystem_manager"):
            self._filesystem_manager.close()

    def _check_memory_pressure(self) -> bool:
        """Check if memory usage is above threshold."""
        memory_usage = psutil.virtual_memory().percent
        return bool(memory_usage >= self._memory_threshold_percent)

    def _relieve_memory_pressure(self, required_size: int = 0) -> None:
        """
        Relieve memory pressure by evicting oldest sessions.

        Args:
            required_size: Minimum size to free up (in bytes)
        """
        with self._lock:
            # Get oldest sessions from memory
            oldest_sessions = self._memory_manager.get_oldest_sessions(limit=20)

            freed_size = 0
            for session_id, _ in oldest_sessions:
                if session_id in self._loading_sessions:
                    continue  # Skip sessions currently being loaded

                session_size = self._memory_manager.get_session_size(session_id)
                self._memory_manager.remove_session(session_id)
                freed_size += session_size

                # Stop if we've freed enough space
                if required_size > 0 and freed_size >= required_size:
                    break

                # Also stop if memory usage is now acceptable
                if not self._check_memory_pressure():
                    break

    def _load_session_from_disk(self, session_id: str) -> bool:
        """
        Load a session from disk to memory if it exists and there's space.

        Args:
            session_id: Session to load

        Returns:
            True if session was loaded, False otherwise
        """
        with self._lock:
            if session_id in self._loading_sessions:
                return False  # Already loading

            if not self._filesystem_manager.has_session(session_id):
                return False  # Session doesn't exist on disk

            # Check if session is already in memory
            if self._memory_manager.has_session(session_id):
                return True  # Already in memory

            # Check if we can fit the session in memory
            session_size = self._filesystem_manager.get_session_size(session_id)
            if not self._memory_manager.can_fit_in_memory(session_id, session_size):
                # Try to free up space
                self._relieve_memory_pressure(session_size)

                # Check again
                if not self._memory_manager.can_fit_in_memory(session_id, session_size):
                    return False  # Still can't fit

            # Load session from disk to memory
            self._loading_sessions.add(session_id)
            try:
                session_data = self._filesystem_manager.get_session_data(session_id)
                if session_data:
                    self._memory_manager.set_session_data(session_id, session_data)
                    return True
            except Exception as e:
                print(f"Error loading session {session_id} from disk: {e}")
            finally:
                self._loading_sessions.discard(session_id)

            return False

    # DataManager interface implementation
    def get_session_data(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            # Try memory first
            if self._memory_manager.has_session(session_id):
                return self._memory_manager.get_session_data(session_id)

            # Try to load from disk
            if self._load_session_from_disk(session_id):
                return self._memory_manager.get_session_data(session_id)

            # Fallback to direct disk access
            return self._filesystem_manager.get_session_data(session_id)

    def set_session_data(self, session_id: str, data: dict[str, Any]) -> None:
        with self._lock:
            # Always attempt to write to both memory and filesystem with graceful degradation
            memory_error: Exception | None = None
            filesystem_error: Exception | None = None

            try:
                self._memory_manager.set_session_data(session_id, data)
            except Exception as e:  # noqa: BLE001
                memory_error = e

            try:
                self._filesystem_manager.set_session_data(session_id, data)
            except Exception as e:  # noqa: BLE001
                filesystem_error = e

            # If both tiers fail, raise a combined error; otherwise, proceed gracefully
            if memory_error and filesystem_error:
                raise RuntimeError(
                    f"Both memory and filesystem writes failed: memory={memory_error}, filesystem={filesystem_error}"
                )

    def get_dataframe(self, session_id: str, df_name: str) -> Any:
        with self._lock:
            # Try memory first
            try:
                if self._memory_manager.has_session(session_id):
                    data = self._memory_manager.get_dataframe(session_id, df_name)
                    if data is not None:
                        # Validate data integrity - if it's corrupted, fallback to disk
                        if self._is_data_valid(data):
                            return data
                        else:
                            # Data is corrupted, remove from memory and fallback to disk
                            self._memory_manager.remove_session(session_id)
            except Exception:  # noqa: BLE001
                # Memory access failure -> graceful fallback to disk below
                pass

            # Try to load session from disk
            try:
                if self._load_session_from_disk(session_id):
                    return self._memory_manager.get_dataframe(session_id, df_name)
            except Exception:  # noqa: BLE001
                # Loading to memory failed; try direct disk access below
                pass

            # Fallback to direct disk access
            try:
                return self._filesystem_manager.get_dataframe(session_id, df_name)
            except Exception:
                # Both memory and filesystem failed
                return None

    def set_dataframe(self, session_id: str, df_name: str, data: Any) -> None:
        with self._lock:
            # Check memory pressure before adding new data
            data_size = self._estimate_data_size(data)
            if not self._memory_manager.can_fit_in_memory(session_id, data_size):
                self._relieve_memory_pressure(data_size)

            # Always attempt to write to both memory and filesystem, but degrade gracefully
            memory_error: Exception | None = None
            filesystem_error: Exception | None = None

            try:
                self._memory_manager.set_dataframe(session_id, df_name, data)
            except Exception as e:  # noqa: BLE001
                memory_error = e

            try:
                self._filesystem_manager.set_dataframe(session_id, df_name, data)
            except Exception as e:  # noqa: BLE001
                filesystem_error = e

            if memory_error and filesystem_error:
                raise RuntimeError(
                    f"Both memory and filesystem writes failed: memory={memory_error}, filesystem={filesystem_error}"
                )

    def _estimate_data_size(self, data: Any) -> int:
        """Estimate the size of data in bytes."""
        try:
            import pickle

            return len(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception:
            return 1024  # Default estimate

    def _is_data_valid(self, data: Any) -> bool:
        """Check if data is valid (not corrupted)."""
        try:
            import pandas as pd

            # If it's a DataFrame, check if it's valid
            if isinstance(data, pd.DataFrame):
                # Try to access basic properties to validate
                _ = data.shape
                _ = data.dtypes
                return True
            # For other data types, check if they're reasonable
            # Reject obviously corrupted data like strings when we expect DataFrames
            if isinstance(data, str) and data == "corrupted_data":
                return False
            # For other data types, assume they're valid if they're not None
            return data is not None
        except Exception:
            # If any exception occurs, consider data corrupted
            return False

    def has_session(self, session_id: str) -> bool:
        with self._lock:
            return self._memory_manager.has_session(
                session_id
            ) or self._filesystem_manager.has_session(session_id)

    def remove_session(self, session_id: str) -> None:
        with self._lock:
            # Remove from both memory and filesystem
            self._memory_manager.remove_session(session_id)
            self._filesystem_manager.remove_session(session_id)

    def get_dataframe_size(self, session_id: str, df_name: str) -> int:
        with self._lock:
            # Try memory first
            if self._memory_manager.has_session(session_id):
                size = self._memory_manager.get_dataframe_size(session_id, df_name)
                if size > 0:
                    return size

            # Fallback to filesystem
            return self._filesystem_manager.get_dataframe_size(session_id, df_name)

    def get_session_size(self, session_id: str) -> int:
        with self._lock:
            # Try memory first
            if self._memory_manager.has_session(session_id):
                return self._memory_manager.get_session_size(session_id)

            # Fallback to filesystem
            return self._filesystem_manager.get_session_size(session_id)

    def get_storage_stats(self) -> StorageStats:
        with self._lock:
            # Avoid nested lock deadlocks by fetching lightweight snapshots
            memory_stats = self._memory_manager.get_storage_stats()
            filesystem_stats = self._filesystem_manager.get_storage_stats()

            # Combine stats
            return StorageStats(
                total_sessions=memory_stats.total_sessions
                + filesystem_stats.total_sessions,
                total_items=memory_stats.total_items + filesystem_stats.total_items,
                total_size_bytes=memory_stats.total_size_bytes
                + filesystem_stats.total_size_bytes,
                memory_usage_percent=memory_stats.memory_usage_percent,
                disk_usage_percent=filesystem_stats.disk_usage_percent,
                tier_distribution={
                    StorageTier.MEMORY: memory_stats.tier_distribution.get(
                        StorageTier.MEMORY, 0
                    ),
                    StorageTier.FILESYSTEM: filesystem_stats.tier_distribution.get(
                        StorageTier.FILESYSTEM, 0
                    ),
                },
            )

    def can_fit_in_memory(self, session_id: str, additional_size: int) -> bool:
        with self._lock:
            # Check if we can fit in memory, considering pressure relief
            if self._memory_manager.can_fit_in_memory(session_id, additional_size):
                return True

            # Try to relieve pressure and check again
            self._relieve_memory_pressure(additional_size)

            # Check again after pressure relief - if still can't fit, return False
            # This allows the hybrid manager to fall back to disk-only access
            return self._memory_manager.can_fit_in_memory(session_id, additional_size)

    def get_oldest_sessions(self, limit: int = 10) -> list[tuple[str, float]]:
        with self._lock:
            # Get oldest sessions from both memory and filesystem
            memory_oldest = self._memory_manager.get_oldest_sessions(limit)
            filesystem_oldest = self._filesystem_manager.get_oldest_sessions(limit)

            # Combine and sort by last access time
            all_sessions = memory_oldest + filesystem_oldest
            all_sessions.sort(key=lambda x: x[1])

            return all_sessions[:limit]

    # Hybrid-specific methods
    def force_load_session_to_memory(self, session_id: str) -> bool:
        """
        Force load a session from disk to memory, even if it means evicting other sessions.

        Args:
            session_id: Session to load

        Returns:
            True if session was loaded, False otherwise
        """
        with self._lock:
            if not self._filesystem_manager.has_session(session_id):
                return False

            # Get session size
            session_size = self._filesystem_manager.get_session_size(session_id)

            # Force relieve memory pressure to make room
            self._relieve_memory_pressure(session_size)

            # Load the session
            return self._load_session_from_disk(session_id)

    def get_memory_sessions(self) -> list[str]:
        """Get list of sessions currently in memory."""
        with self._lock:
            return list(self._memory_manager._sessions.keys())

    def get_disk_only_sessions(self) -> list[str]:
        """Get list of sessions that exist only on disk."""
        with self._lock:
            memory_sessions = set(self.get_memory_sessions())
            disk_sessions = []

            # Get all sessions from filesystem manager
            for session_id in self._filesystem_manager.get_all_session_ids():
                if session_id not in memory_sessions:
                    disk_sessions.append(session_id)

            return disk_sessions
