"""
TTL In-Memory Data Manager Implementation (Cacheout-backed)

Provides a DataManager implementation with sliding TTL and simple per-session
item caps. Sessions are cached with a TTL that is refreshed on access.

Design notes:
- Uses a single Cacheout cache keyed by session_id.
- Each session value is a small dict containing:
  - data: mapping of df_name -> object
  - order: list of df_names in insertion order (for simple per-session eviction)
  - created_at: float epoch seconds
  - last_access: float epoch seconds
- Sliding TTL is achieved by re-setting the same payload on every get/set.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any, Optional, cast, Tuple
import pickle
import psutil

from cacheout import Cache

from .base_data_manager import DataManager
from .storage_types import StorageStats, StorageTier


class TTLInMemoryDataManager(DataManager):
    """In-memory DataManager with sliding TTL and per-session caps."""

    def __init__(
        self,
        ttl_seconds: int = 5 * 60 * 60,
        max_sessions: int = 100,
        max_items_per_session: int = 50,
    ) -> None:
        self._ttl_seconds = ttl_seconds
        self._max_sessions = max_sessions
        self._max_items_per_session = max_items_per_session

        # Cache sessions by id, with TTL and size cap
        self._sessions = Cache(maxsize=max_sessions, ttl=ttl_seconds)
        # Use re-entrant lock to avoid deadlocks when nested methods acquire the same lock
        self._lock = threading.RLock()

    # Internal helpers
    def _now(self) -> float:
        return time.time()

    def _touch(self, session_id: str, payload: dict[str, Any]) -> None:
        payload["last_access"] = self._now()
        # Re-set to refresh TTL (sliding TTL behavior)
        self._sessions.set(session_id, payload, ttl=self._ttl_seconds)

    def _get_payload(self, session_id: str) -> dict[str, Any] | None:
        payload = cast(Optional[dict[str, Any]], self._sessions.get(session_id))
        if payload is None:
            return None
        # Refresh TTL and last_access on read
        self._touch(session_id, payload)
        return payload

    def _ensure_payload(self, session_id: str) -> dict[str, Any]:
        payload = cast(Optional[dict[str, Any]], self._sessions.get(session_id))
        if payload is None:
            payload = {
                "data": OrderedDict(),
                "created_at": self._now(),
                "last_access": self._now(),
            }
            # Initial set with TTL
            self._sessions.set(session_id, payload, ttl=self._ttl_seconds)
        # Refresh TTL on creation or fetch
        self._touch(session_id, payload)
        return payload

    def _enforce_item_cap(self, payload: dict[str, Any]) -> None:
        data: OrderedDict[str, Any] = payload["data"]
        while len(data) > self._max_items_per_session:
            # Evict oldest inserted item
            data.popitem(last=False)

    # DataManager interface
    def get_session_data(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            payload = self._ensure_payload(session_id)
            # Return a regular dict view (copy to avoid external mutation of order)
            return dict(payload["data"])  # shallow copy is fine for mapping

    def set_session_data(self, session_id: str, data: dict[str, Any]) -> None:
        with self._lock:
            payload = self._ensure_payload(session_id)
            # Replace the OrderedDict while preserving insertion order from the provided dict
            ordered = OrderedDict(data.items())
            payload["data"] = ordered
            self._enforce_item_cap(payload)
            self._touch(session_id, payload)

    def get_dataframe(self, session_id: str, df_name: str) -> Any:
        with self._lock:
            payload = self._get_payload(session_id)
            if payload is None:
                return None
            data: OrderedDict[str, Any] = payload["data"]
            return data.get(df_name)

    def set_dataframe(self, session_id: str, df_name: str, data: Any) -> None:
        with self._lock:
            payload = self._ensure_payload(session_id)
            od: OrderedDict[str, Any] = payload["data"]
            # If existing, delete first to re-insert at the end (acts like simple LRU within session)
            if df_name in od:
                del od[df_name]
            od[df_name] = data
            self._enforce_item_cap(payload)
            self._touch(session_id, payload)

    def has_session(self, session_id: str) -> bool:
        with self._lock:
            payload = cast(Optional[dict[str, Any]], self._sessions.get(session_id))
            if payload is None:
                return False
            # Touch to refresh TTL when we confirm existence
            self._touch(session_id, payload)
            return True

    def remove_session(self, session_id: str) -> None:
        with self._lock:
            try:
                self._sessions.delete(session_id)
            except KeyError:
                # Already gone
                return

    def get_dataframe_size(self, session_id: str, df_name: str) -> int:
        """Get the size in bytes of a specific DataFrame."""
        with self._lock:
            payload = self._get_payload(session_id)
            if payload is None:
                return 0
            data: OrderedDict[str, Any] = payload["data"]
            if df_name not in data:
                return 0

            try:
                return len(
                    pickle.dumps(data[df_name], protocol=pickle.HIGHEST_PROTOCOL)
                )
            except Exception:
                return 0

    def get_session_size(self, session_id: str) -> int:
        """Get the total size in bytes of all data in a session."""
        with self._lock:
            payload = self._get_payload(session_id)
            if payload is None:
                return 0

            total_size = 0
            data: OrderedDict[str, Any] = payload["data"]
            for df_name, df_data in data.items():
                try:
                    total_size += len(
                        pickle.dumps(df_data, protocol=pickle.HIGHEST_PROTOCOL)
                    )
                except Exception:
                    continue
            return total_size

    def get_storage_stats(self) -> StorageStats:
        """Get comprehensive storage statistics."""
        with self._lock:
            total_sessions = len(self._sessions)
            total_items = 0
            total_size_bytes = 0

            for session_id in list(self._sessions.keys()):
                payload = self._get_payload(session_id)
                if payload:
                    data: OrderedDict[str, Any] = payload["data"]
                    total_items += len(data)
                    total_size_bytes += self.get_session_size(session_id)

            # Get system stats
            memory_usage = psutil.virtual_memory().percent
            disk_usage = psutil.disk_usage("/").percent

            return StorageStats(
                total_sessions=total_sessions,
                total_items=total_items,
                total_size_bytes=total_size_bytes,
                memory_usage_percent=memory_usage,
                disk_usage_percent=disk_usage,
                tier_distribution={StorageTier.MEMORY: total_items},
            )

    def can_fit_in_memory(self, session_id: str, additional_size: int) -> bool:
        """Check if additional data can fit in memory without exceeding thresholds."""
        with self._lock:
            # Simple heuristic: if we're under 90% memory usage, we can fit more
            memory_usage = psutil.virtual_memory().percent
            return memory_usage < 90.0

    def get_oldest_sessions(self, limit: int = 10) -> list[tuple[str, float]]:
        """Get the oldest sessions by last access time."""
        with self._lock:
            sessions_with_times = []

            for session_id in list(self._sessions.keys()):
                payload = self._get_payload(session_id)
                if payload:
                    sessions_with_times.append((session_id, payload["last_access"]))

            # Sort by last access time (oldest first)
            sessions_with_times.sort(key=lambda x: x[1])
            return sessions_with_times[:limit]
