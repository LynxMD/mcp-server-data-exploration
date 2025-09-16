"""
Session Metadata

This module contains the SessionMetadata class used by the DiskCacheDataManager
to track session information and file metadata.
"""

from dataclasses import dataclass


@dataclass
class SessionMetadata:
    """Metadata for a session stored on filesystem."""

    session_id: str
    created_at: float
    last_access: float
    total_size_bytes: int
    item_count: int
    item_sizes: dict[str, int]  # df_name -> size_bytes
