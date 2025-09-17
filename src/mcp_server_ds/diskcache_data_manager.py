"""
DiskCache-based Data Manager Implementation

A filesystem-based data manager using the diskcache library for automatic
background cleanup, TTL management, and proper resource handling.

Key Benefits:
- Automatic background cleanup (no manual thread management)
- Built-in TTL support with automatic expiration
- Context manager support for proper cleanup
- No hanging threads in tests
- High performance with SQLite backend
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any
import pandas as pd
import diskcache

from .base_data_manager import DataManager
from .storage_types import StorageStats, StorageTier
from .session_metadata import SessionMetadata


class DiskCacheDataManager(DataManager):
    """
    Filesystem-based DataManager using diskcache library.

    This implementation provides automatic background cleanup, TTL management,
    and proper resource handling without manual thread management.
    """

    def __init__(
        self,
        cache_dir: str = "/tmp/mcp_cache",
        ttl_seconds: int = 7 * 24 * 60 * 60,  # 7 days
        max_disk_usage_percent: float = 90.0,
        use_parquet: bool = True,
    ) -> None:
        """
        Initialize DiskCacheDataManager.

        Args:
            cache_dir: Directory for cache storage
            ttl_seconds: TTL for cached data
            max_disk_usage_percent: Maximum disk usage before cleanup
            use_parquet: Use parquet format for DataFrames
        """
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._ttl_seconds = ttl_seconds
        self._max_disk_usage_percent = max_disk_usage_percent
        self._use_parquet = use_parquet

        # Initialize diskcache with automatic cleanup
        self._cache = diskcache.Cache(
            directory=str(self._cache_dir),
            eviction_policy="least-recently-used",
            size_limit=int(1024**4),  # 1TB default limit
        )

        # Metadata cache for session information
        self._metadata_cache = diskcache.Cache(
            directory=str(self._cache_dir / "metadata"),
            eviction_policy="least-recently-used",
        )

    def get_all_session_ids(self) -> list[str]:
        """Get all session IDs that have metadata."""
        session_ids = []
        for key in self._metadata_cache:
            if key.startswith("metadata:"):
                session_id = key[9:]  # Remove "metadata:" prefix
                try:
                    # Verify the metadata is accessible
                    _ = self._metadata_cache[key]
                    session_ids.append(session_id)
                except Exception as e:
                    # Self-heal: delete corrupted metadata entries to prevent repeated errors
                    import sys

                    print(
                        f"[MCP-DEBUG] Deleting corrupted metadata in get_all_session_ids {key}: {e}",
                        file=sys.stderr,
                    )
                    try:
                        del self._metadata_cache[key]
                    except Exception:
                        # Best-effort deletion; continue
                        pass
                    continue
        return session_ids

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatic cleanup."""
        self.close()

    def close(self) -> None:
        """Close the cache and cleanup resources."""
        if hasattr(self, "_cache"):
            self._cache.close()
        if hasattr(self, "_metadata_cache"):
            self._metadata_cache.close()

    def _get_data_key(self, session_id: str, df_name: str) -> str:
        """Get cache key for data."""
        return f"data:{session_id}:{df_name}"

    def _get_metadata_key(self, session_id: str) -> str:
        """Get cache key for session metadata."""
        return f"metadata:{session_id}"

    def _serialize_data(self, data: Any) -> bytes:
        """Serialize data for storage."""
        if isinstance(data, pd.DataFrame) and self._use_parquet:
            # Use parquet for DataFrames
            import io

            buffer = io.BytesIO()
            data.to_parquet(buffer, index=False)
            return buffer.getvalue()
        else:
            # Use pickle for other data types
            import pickle

            return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)

    def _deserialize_data(self, data_bytes: bytes, is_dataframe: bool = False) -> Any:
        """Deserialize data from storage."""
        if is_dataframe and self._use_parquet:
            # Deserialize parquet DataFrames
            import io

            buffer = io.BytesIO(data_bytes)
            return pd.read_parquet(buffer)
        else:
            # Deserialize pickle data
            import pickle

            return pickle.loads(data_bytes)

    def _update_session_metadata(
        self, session_id: str, df_name: str, data_size: int
    ) -> None:
        """Update session metadata."""
        metadata_key = self._get_metadata_key(session_id)

        # Get existing metadata or create new
        if metadata_key in self._metadata_cache:
            metadata = self._metadata_cache[metadata_key]
        else:
            metadata = SessionMetadata(
                session_id=session_id,
                created_at=time.time(),
                last_access=time.time(),
                total_size_bytes=0,
                item_count=0,
                item_sizes={},
            )

        # Update metadata
        metadata.last_access = time.time()
        metadata.item_sizes[df_name] = data_size
        metadata.item_count = len(metadata.item_sizes)
        metadata.total_size_bytes = sum(metadata.item_sizes.values())

        # Store updated metadata
        self._metadata_cache[metadata_key] = metadata

    # DataManager interface implementation
    def get_session_data(self, session_id: str) -> dict[str, Any]:
        """Get all data for a session."""
        session_data = {}

        # Get metadata to find all items
        metadata_key = self._get_metadata_key(session_id)
        if metadata_key in self._metadata_cache:
            metadata = self._metadata_cache[metadata_key]

            for df_name in metadata.item_sizes.keys():
                data_key = self._get_data_key(session_id, df_name)
                if data_key in self._cache:
                    data_bytes = self._cache[data_key]
                    # Check if data is parquet by looking at the magic bytes
                    is_dataframe = self._use_parquet and data_bytes.startswith(b"PAR1")
                    session_data[df_name] = self._deserialize_data(
                        data_bytes, is_dataframe
                    )

                    # Sliding TTL: refresh TTL on access and update metadata
                    try:
                        self._cache.touch(data_key, expire=self._ttl_seconds)
                    except AttributeError:
                        # Older diskcache versions may not have touch; fallback to set
                        self._cache.set(data_key, data_bytes, expire=self._ttl_seconds)
                    # Update last access and sizes in metadata
                    self._update_session_metadata(session_id, df_name, len(data_bytes))

        return session_data

    def set_session_data(self, session_id: str, data: dict[str, Any]) -> None:
        """Set all data for a session."""
        for df_name, df_data in data.items():
            self.set_dataframe(session_id, df_name, df_data)

    def get_dataframe(self, session_id: str, df_name: str) -> Any:
        """Get a specific DataFrame from cache."""
        data_key = self._get_data_key(session_id, df_name)

        if data_key in self._cache:
            data_bytes = self._cache[data_key]
            # Check if data is parquet by looking at the magic bytes
            is_dataframe = self._use_parquet and data_bytes.startswith(b"PAR1")
            data = self._deserialize_data(data_bytes, is_dataframe)

            # Sliding TTL: refresh TTL on access and update metadata
            try:
                self._cache.touch(data_key, expire=self._ttl_seconds)
            except AttributeError:
                # Older diskcache versions may not have touch; fallback to set
                self._cache.set(data_key, data_bytes, expire=self._ttl_seconds)
            # Update last access time
            self._update_session_metadata(session_id, df_name, len(data_bytes))

            return data

        return None

    def set_dataframe(self, session_id: str, df_name: str, data: Any) -> None:
        """Set a DataFrame in cache with TTL."""
        data_key = self._get_data_key(session_id, df_name)

        # Serialize data
        data_bytes = self._serialize_data(data)
        data_size = len(data_bytes)

        # Store in cache with TTL
        self._cache.set(data_key, data_bytes, expire=self._ttl_seconds)

        # Update session metadata
        self._update_session_metadata(session_id, df_name, data_size)

    def has_session(self, session_id: str) -> bool:
        """Check if session exists."""
        metadata_key = self._get_metadata_key(session_id)
        return metadata_key in self._metadata_cache

    def remove_session(self, session_id: str) -> None:
        """Remove all data for a session."""
        # Get metadata to find all items
        metadata_key = self._get_metadata_key(session_id)
        if metadata_key in self._metadata_cache:
            metadata = self._metadata_cache[metadata_key]

            # Remove all data items
            for df_name in metadata.item_sizes.keys():
                data_key = self._get_data_key(session_id, df_name)
                if data_key in self._cache:
                    del self._cache[data_key]

            # Remove metadata
            del self._metadata_cache[metadata_key]

    def get_dataframe_size(self, session_id: str, df_name: str) -> int:
        """Get size of a specific DataFrame."""
        metadata_key = self._get_metadata_key(session_id)
        if metadata_key in self._metadata_cache:
            metadata = self._metadata_cache[metadata_key]
            return int(metadata.item_sizes.get(df_name, 0))
        return 0

    def get_session_size(self, session_id: str) -> int:
        """Get total size of a session."""
        metadata_key = self._get_metadata_key(session_id)
        if metadata_key in self._metadata_cache:
            metadata = self._metadata_cache[metadata_key]
            return int(metadata.total_size_bytes)
        return 0

    def get_storage_stats(self) -> StorageStats:
        """Get storage statistics."""
        total_sessions = 0
        total_items = 0
        total_size_bytes = 0

        # Iterate through metadata cache
        for key in self._metadata_cache:
            if key.startswith("metadata:"):
                total_sessions += 1
                try:
                    metadata = self._metadata_cache[key]
                    total_items += len(metadata.item_sizes)
                    total_size_bytes += metadata.total_size_bytes
                except Exception as e:
                    # Self-heal: delete corrupted metadata entries to prevent repeated errors
                    import sys

                    print(
                        f"[MCP-DEBUG] Deleting corrupted metadata {key}: {e}",
                        file=sys.stderr,
                    )
                    try:
                        del self._metadata_cache[key]
                    except Exception:
                        # Best-effort deletion; continue
                        pass
                    continue

        return StorageStats(
            total_sessions=total_sessions,
            total_items=total_items,
            total_size_bytes=total_size_bytes,
            memory_usage_percent=0.0,  # Not applicable for disk cache
            disk_usage_percent=self._get_disk_usage_percent(),
            tier_distribution={StorageTier.FILESYSTEM: total_items},
        )

    def _get_disk_usage_percent(self) -> float:
        """Get current disk usage percentage."""
        try:
            import psutil

            disk_usage = psutil.disk_usage(str(self._cache_dir))
            return float((disk_usage.used / disk_usage.total) * 100)
        except Exception:
            return 0.0

    def can_fit_in_memory(self, session_id: str, additional_size: int) -> bool:
        """Check if data can fit in available disk space."""
        current_usage = self._get_disk_usage_percent()
        return current_usage < self._max_disk_usage_percent

    def get_oldest_sessions(self, limit: int = 10) -> list[tuple[str, float]]:
        """Get oldest sessions by last access time."""
        sessions = []

        for metadata_key in self._metadata_cache:
            if metadata_key.startswith("metadata:"):
                session_id = metadata_key[9:]  # Remove "metadata:" prefix
                metadata = self._metadata_cache[metadata_key]
                sessions.append((session_id, metadata.last_access))

        # Sort by last access time (oldest first)
        sessions.sort(key=lambda x: x[1])
        return sessions[:limit]

    def _emergency_cleanup(self) -> None:
        """Emergency cleanup when disk usage is high."""
        # Get oldest sessions and remove them
        oldest_sessions = self.get_oldest_sessions(limit=10)

        for session_id, _ in oldest_sessions:
            self.remove_session(session_id)

            # Check if we've freed enough space
            try:
                current_usage = self._get_disk_usage_percent()
                if current_usage < self._max_disk_usage_percent:
                    break
            except (TypeError, AttributeError):
                # Handle mock objects in tests
                break
