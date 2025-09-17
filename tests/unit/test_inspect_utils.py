from __future__ import annotations

import pandas as pd

from mcp_server_ds.utils.inspect_utils import summarize_session_data


def test_summarize_empty_session():
    out = summarize_session_data("s", {}, include_preview=True)
    assert "Items: 0" in out
    assert "No dataframes" in out


def test_summarize_with_dataframes_limits_preview_and_columns():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    out = summarize_session_data(
        "s", {"df": df}, include_preview=True, max_rows=2, max_cols=2
    )
    assert "df" in out
    assert "shape: (3, 3)" in out
    assert "columns: ['a', 'b']" in out
    # No preview expected (privacy); ensure preview header not present
    assert "preview:" not in out


def test_summarize_non_dataframe_object():
    out = summarize_session_data("s", {"x": 123}, include_preview=True)
    assert "type: <class 'int'>" in out


class BadDF:
    def __init__(self):
        self.shape = (1, 1)
        self.columns = ["a"]

        class D:
            def to_dict(self):
                return {"a": "int64"}

        self.dtypes = D()

    def head(self, n):  # noqa: D401
        raise RuntimeError("no preview")


def test_summarize_preview_failure_path():
    out = summarize_session_data("s", {"bad": BadDF()}, include_preview=True)
    # No preview marker present
    assert "preview:" not in out
