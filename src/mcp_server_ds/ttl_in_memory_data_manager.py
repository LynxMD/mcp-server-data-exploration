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
from typing import Any, Optional, cast

from cacheout import Cache

from .data_manager import DataManager


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
        self._lock = threading.Lock()

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
