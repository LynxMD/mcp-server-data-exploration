from __future__ import annotations

import pandas as pd


def read_csv_strict(csv_path: str) -> pd.DataFrame:
    """Read a CSV file with pandas and rethrow with normalized message.

    Logic identical to direct pd.read_csv usage; this is a thin wrapper for testability.
    """
    try:
        return pd.read_csv(csv_path)
    except Exception as e:  # noqa: BLE001
        raise Exception(f"Error loading CSV: {e}")
