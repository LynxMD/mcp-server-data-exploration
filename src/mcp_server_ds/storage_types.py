"""
Storage Types and Data Classes

This module contains the core data structures and enums used by the storage system.
"""

from dataclasses import dataclass
from enum import Enum


class StorageTier(Enum):
    """Storage tier enumeration for hybrid implementations."""

    MEMORY = "memory"
    FILESYSTEM = "filesystem"
    REDIS = "redis"


@dataclass
class StorageStats:
    """Storage statistics for monitoring and optimization."""

    total_sessions: int
    total_items: int
    total_size_bytes: int
    memory_usage_percent: float
    disk_usage_percent: float
    tier_distribution: dict[StorageTier, int]
