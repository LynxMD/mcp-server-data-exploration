from __future__ import annotations

import pandas as pd

from mcp_server_ds.utils.df_info_utils import summarize_dataframe_info


def test_summarize_dataframe_info_numeric_and_categorical():
    df = pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0, None],
            "cat": ["a", "b", "a", "c"],
            "flag": [True, False, True, None],
            "dt": pd.to_datetime(["2020-01-01", "2020-01-02", None, "2020-01-04"]),
        }
    )
    out = summarize_dataframe_info(
        "df",
        df,
        max_cols_report=10,
        max_uniques_per_col=2,
        include_numeric_aggregates=True,
        include_quality_score=True,
        include_recommendations=True,
    )
    assert "DATAFRAME INFO: df" in out
    assert "shape: (4, 4)" in out
    assert "dtypes:" in out
    assert "memory_usage_bytes:" in out
    assert "memory_usage_human:" in out
    assert "missing_counts:" in out
    assert "unique_counts:" in out
    assert "numeric_describe:" in out
    assert "numeric_aggregates:" in out
    assert "categorical_samples:" in out
    assert "cat:" in out
    assert "boolean_columns:" in out
    assert "datetime_ranges:" in out
    assert "quality_score:" in out
    assert "column_quality:" in out
    assert "recommendations:" in out


def test_summarize_dataframe_info_limits_columns_and_uniques():
    df = pd.DataFrame({f"c{i}": [i, i] for i in range(50)})
    out = summarize_dataframe_info("wide", df, max_cols_report=5, max_uniques_per_col=1)
    assert "columns (limited)" in out or "dtypes:" in out
    # Ensure we see ellipsis hints when limiting
    assert "..." in out
