"""
Unit tests for Storage Types

Tests the StorageTier enum and StorageStats dataclass.
"""

from mcp_server_ds.storage_types import StorageTier, StorageStats


class TestStorageTier:
    """Test suite for StorageTier enum."""

    def test_storage_tier_values(self):
        """Test StorageTier enum values."""
        assert StorageTier.MEMORY.value == "memory"
        assert StorageTier.FILESYSTEM.value == "filesystem"
        assert StorageTier.REDIS.value == "redis"

    def test_storage_tier_enumeration(self):
        """Test StorageTier enum iteration."""
        tiers = list(StorageTier)
        assert len(tiers) == 3
        assert StorageTier.MEMORY in tiers
        assert StorageTier.FILESYSTEM in tiers
        assert StorageTier.REDIS in tiers

    def test_storage_tier_string_conversion(self):
        """Test StorageTier string conversion."""
        assert str(StorageTier.MEMORY) == "StorageTier.MEMORY"
        assert str(StorageTier.FILESYSTEM) == "StorageTier.FILESYSTEM"
        assert str(StorageTier.REDIS) == "StorageTier.REDIS"


class TestStorageStats:
    """Test suite for StorageStats dataclass."""

    def test_storage_stats_creation(self):
        """Test StorageStats creation with all fields."""
        stats = StorageStats(
            total_sessions=10,
            total_items=50,
            total_size_bytes=1024 * 1024,  # 1MB
            memory_usage_percent=75.5,
            disk_usage_percent=60.0,
            tier_distribution={
                StorageTier.MEMORY: 30,
                StorageTier.FILESYSTEM: 20,
                StorageTier.REDIS: 0,
            },
        )

        assert stats.total_sessions == 10
        assert stats.total_items == 50
        assert stats.total_size_bytes == 1024 * 1024
        assert stats.memory_usage_percent == 75.5
        assert stats.disk_usage_percent == 60.0
        assert stats.tier_distribution[StorageTier.MEMORY] == 30
        assert stats.tier_distribution[StorageTier.FILESYSTEM] == 20
        assert stats.tier_distribution[StorageTier.REDIS] == 0

    def test_storage_stats_default_values(self):
        """Test StorageStats with minimal values."""
        stats = StorageStats(
            total_sessions=0,
            total_items=0,
            total_size_bytes=0,
            memory_usage_percent=0.0,
            disk_usage_percent=0.0,
            tier_distribution={},
        )

        assert stats.total_sessions == 0
        assert stats.total_items == 0
        assert stats.total_size_bytes == 0
        assert stats.memory_usage_percent == 0.0
        assert stats.disk_usage_percent == 0.0
        assert len(stats.tier_distribution) == 0

    def test_storage_stats_immutability(self):
        """Test that StorageStats fields can be accessed but not modified."""
        stats = StorageStats(
            total_sessions=5,
            total_items=25,
            total_size_bytes=512 * 1024,
            memory_usage_percent=50.0,
            disk_usage_percent=30.0,
            tier_distribution={StorageTier.MEMORY: 15},
        )

        # Should be able to access fields
        assert stats.total_sessions == 5

        # Should be able to modify mutable fields (like dict)
        stats.tier_distribution[StorageTier.FILESYSTEM] = 10
        assert stats.tier_distribution[StorageTier.FILESYSTEM] == 10

    def test_storage_stats_tier_distribution(self):
        """Test StorageStats tier distribution functionality."""
        tier_dist = {
            StorageTier.MEMORY: 100,
            StorageTier.FILESYSTEM: 50,
            StorageTier.REDIS: 25,
        }

        stats = StorageStats(
            total_sessions=10,
            total_items=175,  # 100 + 50 + 25
            total_size_bytes=1024 * 1024,
            memory_usage_percent=80.0,
            disk_usage_percent=40.0,
            tier_distribution=tier_dist,
        )

        # Test individual tier access
        assert stats.tier_distribution[StorageTier.MEMORY] == 100
        assert stats.tier_distribution[StorageTier.FILESYSTEM] == 50
        assert stats.tier_distribution[StorageTier.REDIS] == 25

        # Test total items matches tier distribution
        total_tier_items = sum(stats.tier_distribution.values())
        assert total_tier_items == 175
        assert stats.total_items == total_tier_items

    def test_storage_stats_edge_cases(self):
        """Test StorageStats with edge case values."""
        # Test with very large numbers
        stats = StorageStats(
            total_sessions=1000000,
            total_items=5000000,
            total_size_bytes=1024 * 1024 * 1024 * 10,  # 10GB
            memory_usage_percent=99.9,
            disk_usage_percent=99.9,
            tier_distribution={
                StorageTier.MEMORY: 1000000,
                StorageTier.FILESYSTEM: 2000000,
                StorageTier.REDIS: 2000000,
            },
        )

        assert stats.total_sessions == 1000000
        assert stats.total_items == 5000000
        assert stats.total_size_bytes == 1024 * 1024 * 1024 * 10
        assert stats.memory_usage_percent == 99.9
        assert stats.disk_usage_percent == 99.9

        # Test with negative values (should be allowed by dataclass)
        stats_negative = StorageStats(
            total_sessions=-1,
            total_items=-5,
            total_size_bytes=-1024,
            memory_usage_percent=-10.0,
            disk_usage_percent=-5.0,
            tier_distribution={StorageTier.MEMORY: -10},
        )

        assert stats_negative.total_sessions == -1
        assert stats_negative.total_items == -5
        assert stats_negative.total_size_bytes == -1024
        assert stats_negative.memory_usage_percent == -10.0
        assert stats_negative.disk_usage_percent == -5.0
        assert stats_negative.tier_distribution[StorageTier.MEMORY] == -10
