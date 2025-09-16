"""
Unit tests for Session Metadata

Tests the SessionMetadata dataclass used by DiskCacheDataManager.
"""

from mcp_server_ds.session_metadata import SessionMetadata


class TestSessionMetadata:
    """Test suite for SessionMetadata dataclass."""

    def test_session_metadata_creation(self):
        """Test SessionMetadata creation with all fields."""
        metadata = SessionMetadata(
            session_id="test_session_123",
            created_at=1234567890.0,
            last_access=1234567895.0,
            total_size_bytes=1024 * 1024,  # 1MB
            item_count=5,
            item_sizes={"df1": 512 * 1024, "df2": 256 * 1024, "df3": 256 * 1024},
        )

        assert metadata.session_id == "test_session_123"
        assert metadata.created_at == 1234567890.0
        assert metadata.last_access == 1234567895.0
        assert metadata.total_size_bytes == 1024 * 1024
        assert metadata.item_count == 5
        assert len(metadata.item_sizes) == 3
        assert metadata.item_sizes["df1"] == 512 * 1024
        assert metadata.item_sizes["df2"] == 256 * 1024
        assert metadata.item_sizes["df3"] == 256 * 1024

    def test_session_metadata_empty_session(self):
        """Test SessionMetadata for empty session."""
        metadata = SessionMetadata(
            session_id="empty_session",
            created_at=1234567890.0,
            last_access=1234567890.0,
            total_size_bytes=0,
            item_count=0,
            item_sizes={},
        )

        assert metadata.session_id == "empty_session"
        assert metadata.total_size_bytes == 0
        assert metadata.item_count == 0
        assert len(metadata.item_sizes) == 0

    def test_session_metadata_mutability(self):
        """Test that SessionMetadata fields can be modified."""
        metadata = SessionMetadata(
            session_id="mutable_session",
            created_at=1234567890.0,
            last_access=1234567890.0,
            total_size_bytes=1024,
            item_count=1,
            item_sizes={"df1": 1024},
        )

        # Should be able to modify fields
        metadata.last_access = 1234567900.0
        metadata.total_size_bytes = 2048
        metadata.item_count = 2
        metadata.item_sizes["df2"] = 1024

        assert metadata.last_access == 1234567900.0
        assert metadata.total_size_bytes == 2048
        assert metadata.item_count == 2
        assert len(metadata.item_sizes) == 2
        assert metadata.item_sizes["df2"] == 1024

    def test_session_metadata_item_sizes_consistency(self):
        """Test that item_sizes dictionary is consistent with item_count."""
        metadata = SessionMetadata(
            session_id="consistent_session",
            created_at=1234567890.0,
            last_access=1234567890.0,
            total_size_bytes=600,  # Should match sum of item sizes
            item_count=3,
            item_sizes={"df1": 100, "df2": 200, "df3": 300},
        )

        # item_count should match number of items in item_sizes
        assert metadata.item_count == len(metadata.item_sizes)

        # total_size_bytes should match sum of item sizes
        expected_total = sum(metadata.item_sizes.values())
        assert metadata.total_size_bytes == expected_total

    def test_session_metadata_edge_cases(self):
        """Test SessionMetadata with edge case values."""
        # Test with very large numbers
        metadata = SessionMetadata(
            session_id="large_session",
            created_at=0.0,
            last_access=9999999999.0,
            total_size_bytes=1024 * 1024 * 1024 * 10,  # 10GB
            item_count=1000000,
            item_sizes={f"df_{i}": 1024 * 1024 for i in range(1000000)},
        )

        assert metadata.session_id == "large_session"
        assert metadata.created_at == 0.0
        assert metadata.last_access == 9999999999.0
        assert metadata.total_size_bytes == 1024 * 1024 * 1024 * 10
        assert metadata.item_count == 1000000
        assert len(metadata.item_sizes) == 1000000

        # Test with negative values (should be allowed by dataclass)
        metadata_negative = SessionMetadata(
            session_id="negative_session",
            created_at=-1234567890.0,
            last_access=-1234567895.0,
            total_size_bytes=-1024,
            item_count=-5,
            item_sizes={"df1": -512, "df2": -256},
        )

        assert metadata_negative.session_id == "negative_session"
        assert metadata_negative.created_at == -1234567890.0
        assert metadata_negative.last_access == -1234567895.0
        assert metadata_negative.total_size_bytes == -1024
        assert metadata_negative.item_count == -5
        assert metadata_negative.item_sizes["df1"] == -512
        assert metadata_negative.item_sizes["df2"] == -256

    def test_session_metadata_string_values(self):
        """Test SessionMetadata with various string values."""
        # Test with special characters in session_id
        metadata = SessionMetadata(
            session_id="session-with-dashes_and_underscores.123",
            created_at=1234567890.0,
            last_access=1234567890.0,
            total_size_bytes=1024,
            item_count=1,
            item_sizes={"df-with-special.chars": 1024},
        )

        assert metadata.session_id == "session-with-dashes_and_underscores.123"
        assert "df-with-special.chars" in metadata.item_sizes

        # Test with empty string
        metadata_empty = SessionMetadata(
            session_id="",
            created_at=1234567890.0,
            last_access=1234567890.0,
            total_size_bytes=0,
            item_count=0,
            item_sizes={},
        )

        assert metadata_empty.session_id == ""

    def test_session_metadata_time_relationships(self):
        """Test time relationships in SessionMetadata."""
        base_time = 1234567890.0

        metadata = SessionMetadata(
            session_id="time_test_session",
            created_at=base_time,
            last_access=base_time + 100.0,  # 100 seconds later
            total_size_bytes=1024,
            item_count=1,
            item_sizes={"df1": 1024},
        )

        # last_access should be after created_at
        assert metadata.last_access > metadata.created_at
        assert metadata.last_access - metadata.created_at == 100.0

        # Test with same times
        metadata_same_time = SessionMetadata(
            session_id="same_time_session",
            created_at=base_time,
            last_access=base_time,  # Same time
            total_size_bytes=1024,
            item_count=1,
            item_sizes={"df1": 1024},
        )

        assert metadata_same_time.last_access == metadata_same_time.created_at
