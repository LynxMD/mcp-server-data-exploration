"""
Unit tests for graceful degradation in HybridDataManager.

Covers:
- Memory access failure -> fallback to disk
- Filesystem access failure -> memory still works
- Both write paths fail -> combined error raised
"""

from unittest.mock import patch
import pandas as pd

from mcp_server_ds.hybrid_data_manager import HybridDataManager


def test_memory_read_failure_falls_back_to_disk(tmp_path):
    manager = HybridDataManager(cache_dir=str(tmp_path))
    try:
        df = pd.DataFrame({"A": [1, 2, 3]})
        manager.set_dataframe("s1", "df", df)

        # Ensure data exists on disk by evicting from memory
        manager._memory_manager.remove_session("s1")

        # Patch memory manager to raise on has_session to simulate failure
        with patch.object(
            manager._memory_manager, "has_session", side_effect=Exception("mem failure")
        ):
            out = manager.get_dataframe("s1", "df")
            assert out is not None
            assert isinstance(out, pd.DataFrame)
    finally:
        manager.close()


def test_filesystem_write_failure_allows_memory_write(tmp_path):
    manager = HybridDataManager(cache_dir=str(tmp_path))
    try:
        df = pd.DataFrame({"A": [1, 2, 3]})

        with patch.object(
            manager._filesystem_manager,
            "set_dataframe",
            side_effect=OSError("disk write fail"),
        ):
            # Should not raise since memory write succeeds
            manager.set_dataframe("s2", "df", df)
            # Data should be at least in memory
            got = manager._memory_manager.get_dataframe("s2", "df")
            assert got is not None
    finally:
        manager.close()


def test_both_writes_fail_raises_combined_error(tmp_path):
    manager = HybridDataManager(cache_dir=str(tmp_path))
    try:
        df = pd.DataFrame({"A": [1]})

        with patch.object(
            manager._memory_manager,
            "set_dataframe",
            side_effect=OSError("mem write fail"),
        ):
            with patch.object(
                manager._filesystem_manager,
                "set_dataframe",
                side_effect=OSError("disk write fail"),
            ):
                try:
                    manager.set_dataframe("s3", "df", df)
                    assert False, "Expected RuntimeError when both writes fail"
                except RuntimeError as e:
                    msg = str(e).lower()
                    assert "both memory and filesystem writes failed" in msg
    finally:
        manager.close()


def test_set_session_data_graceful_when_one_tier_fails(tmp_path):
    manager = HybridDataManager(cache_dir=str(tmp_path))
    try:
        data = {"df": pd.DataFrame({"A": [1]})}

        # Filesystem fails, memory succeeds
        with patch.object(
            manager._filesystem_manager,
            "set_session_data",
            side_effect=OSError("disk write fail"),
        ):
            # Should not raise
            manager.set_session_data("s4", data)

        # Memory fails, filesystem succeeds
        with patch.object(
            manager._memory_manager,
            "set_session_data",
            side_effect=OSError("mem write fail"),
        ):
            manager.set_session_data("s5", data)
    finally:
        manager.close()


def test_set_session_data_both_fail_raises(tmp_path):
    manager = HybridDataManager(cache_dir=str(tmp_path))
    try:
        data = {"df": pd.DataFrame({"A": [1]})}

        with patch.object(
            manager._memory_manager,
            "set_session_data",
            side_effect=OSError("mem write fail"),
        ):
            with patch.object(
                manager._filesystem_manager,
                "set_session_data",
                side_effect=OSError("disk write fail"),
            ):
                try:
                    manager.set_session_data("s6", data)
                    assert False, (
                        "Expected RuntimeError when both set_session_data fail"
                    )
                except RuntimeError as e:
                    msg = str(e).lower()
                    assert "both memory and filesystem writes failed" in msg
    finally:
        manager.close()


def test_get_dataframe_returns_none_when_both_tiers_fail(tmp_path):
    manager = HybridDataManager(cache_dir=str(tmp_path))
    try:
        # Ensure memory path errors and loading fails, then filesystem read also fails
        with patch.object(
            manager._memory_manager,
            "has_session",
            side_effect=Exception("mem check fail"),
        ):
            with patch.object(
                manager, "_load_session_from_disk", side_effect=Exception("load fail")
            ):
                with patch.object(
                    manager._filesystem_manager,
                    "get_dataframe",
                    side_effect=Exception("disk read fail"),
                ):
                    out = manager.get_dataframe("s7", "df")
                    assert out is None
    finally:
        manager.close()
