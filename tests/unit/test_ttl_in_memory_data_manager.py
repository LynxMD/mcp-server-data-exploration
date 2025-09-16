"""Unit tests for TTLInMemoryDataManager (Cacheout-backed)."""

import time

import pandas as pd

from mcp_server_ds.ttl_in_memory_data_manager import TTLInMemoryDataManager


class TestTTLInMemoryDataManager:
    def test_basic_set_get(self):
        dm = TTLInMemoryDataManager(
            ttl_seconds=60, max_sessions=10, max_items_per_session=5
        )
        session = "s1"
        df = pd.DataFrame({"a": [1, 2]})
        dm.set_dataframe(session, "df1", df)
        out = dm.get_dataframe(session, "df1")
        assert out is not None
        assert list(out["a"]) == [1, 2]

    def test_ttl_expiry(self):
        dm = TTLInMemoryDataManager(
            ttl_seconds=1, max_sessions=10, max_items_per_session=5
        )
        session = "s2"
        dm.set_dataframe(session, "df", pd.DataFrame({"x": [1]}))
        assert dm.has_session(session)
        time.sleep(1.2)
        # After TTL, session should be gone
        assert not dm.has_session(session)

    def test_touch_on_access_refreshes_ttl(self):
        dm = TTLInMemoryDataManager(
            ttl_seconds=1, max_sessions=10, max_items_per_session=5
        )
        session = "s3"
        dm.set_dataframe(session, "df", pd.DataFrame({"x": [1]}))
        time.sleep(0.6)
        # Access should refresh TTL
        assert dm.get_dataframe(session, "df") is not None
        time.sleep(0.6)
        # Should still exist due to refresh
        assert dm.has_session(session)

    def test_per_session_item_cap(self):
        dm = TTLInMemoryDataManager(
            ttl_seconds=60, max_sessions=10, max_items_per_session=2
        )
        session = "s4"
        dm.set_dataframe(session, "df1", pd.DataFrame({"x": [1]}))
        dm.set_dataframe(session, "df2", pd.DataFrame({"x": [2]}))
        dm.set_dataframe(session, "df3", pd.DataFrame({"x": [3]}))
        # Oldest (df1) should be evicted
        assert dm.get_dataframe(session, "df1") is None
        assert dm.get_dataframe(session, "df2") is not None
        assert dm.get_dataframe(session, "df3") is not None

    def test_max_sessions_eviction(self):
        dm = TTLInMemoryDataManager(
            ttl_seconds=60, max_sessions=2, max_items_per_session=5
        )
        dm.set_dataframe("s1", "df", pd.DataFrame({"x": [1]}))
        dm.set_dataframe("s2", "df", pd.DataFrame({"x": [2]}))
        # Touch s1 to make it most recent
        dm.get_dataframe("s1", "df")
        # Insert s3 should evict least recent (s2)
        dm.set_dataframe("s3", "df", pd.DataFrame({"x": [3]}))
        assert dm.has_session("s1")
        assert not dm.has_session("s2")
        assert dm.has_session("s3")
