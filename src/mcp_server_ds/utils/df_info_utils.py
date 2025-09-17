from __future__ import annotations

from typing import Any


def summarize_dataframe_info(
    df_name: str,
    df: Any,
    max_cols_report: int = 20,
    max_uniques_per_col: int = 5,
    include_numeric_aggregates: bool = False,
    include_quality_score: bool = False,
    include_recommendations: bool = False,
) -> str:
    """Create a privacy-safe, human-readable summary of a DataFrame.

    Includes: shape, dtypes, memory usage, missing counts, nunique per column,
    numeric describe() (summary only), and sample unique values for categoricals (bounded).
    No row previews/values are printed beyond bounded unique samples for categorical columns.
    """
    lines: list[str] = []
    lines.append(f"=== DATAFRAME INFO: {df_name} ===")

    # Shape
    shape = getattr(df, "shape", None)
    if shape is not None:
        lines.append(f"shape: {shape}")

    # Dtypes
    dtypes = getattr(df, "dtypes", None)
    if dtypes is not None:
        try:
            dtype_map = {str(k): str(v) for k, v in dtypes.to_dict().items()}
            # Limit number of columns reported
            dtype_items = list(dtype_map.items())[:max_cols_report]
            limited_dtypes = {k: v for k, v in dtype_items}
            more = "..." if len(dtype_map) > max_cols_report else ""
            lines.append(f"dtypes: {limited_dtypes}{more}")
        except Exception:
            pass

    # Memory usage (deep)
    mem_usage = None
    try:
        mem_usage = int(df.memory_usage(index=True, deep=True).sum())
    except Exception:
        try:
            mem_usage = int(df.memory_usage().sum())
        except Exception:
            mem_usage = None
    if mem_usage is not None:
        lines.append(f"memory_usage_bytes: {mem_usage}")

        # Human-readable
        def _humanize_bytes(n: int) -> str:
            units = ["B", "KB", "MB", "GB", "TB"]
            i = 0
            val = float(n)
            while val >= 1024.0 and i < len(units) - 1:
                val /= 1024.0
                i += 1
            return f"{val:.2f} {units[i]}"

        lines.append(f"memory_usage_human: {_humanize_bytes(mem_usage)}")

    # Missing value counts per column
    try:
        na_counts = getattr(df, "isna")().sum()
        na_map = {str(k): int(v) for k, v in na_counts.to_dict().items()}
        na_items = list(na_map.items())[:max_cols_report]
        limited_na_map = {k: v for k, v in na_items}
        more = "..." if len(na_map) > max_cols_report else ""
        lines.append(f"missing_counts: {limited_na_map}{more}")
    except Exception:
        pass

    # Unique counts per column
    try:
        nunique = getattr(df, "nunique")()
        nu_map = {str(k): int(v) for k, v in nunique.to_dict().items()}
        nu_items = list(nu_map.items())[:max_cols_report]
        limited_nu_map = {k: v for k, v in nu_items}
        more = "..." if len(nu_map) > max_cols_report else ""
        lines.append(f"unique_counts: {limited_nu_map}{more}")
    except Exception:
        pass

    # Numeric describe summary
    try:
        desc = df.describe(include=[float, int])
        lines.append("numeric_describe:")
        # Print only the index (stat names) and limit columns
        desc_cols = list(desc.columns)[:max_cols_report]
        lines.append(
            f"  columns (limited): {desc_cols}{'...' if len(desc.columns) > max_cols_report else ''}"
        )
        # Show the first few stats names
        lines.append(f"  stats: {list(desc.index)}")

        # Optional: numeric aggregates preview (privacy-safe aggregates)
        if include_numeric_aggregates and not desc.empty:
            lines.append("numeric_aggregates:")
            for col in list(desc.columns)[:max_cols_report]:
                try:
                    col_stats = desc[col]
                    col_min = col_stats.get("min", None)
                    col_max = col_stats.get("max", None)
                    col_mean = col_stats.get("mean", None)
                    col_std = col_stats.get("std", None)
                    parts: list[str] = []
                    if col_min is not None:
                        parts.append(f"min={col_min}")
                    if col_max is not None:
                        parts.append(f"max={col_max}")
                    if col_mean is not None:
                        parts.append(f"mean={col_mean}")
                    if col_std is not None:
                        parts.append(f"std={col_std}")
                    lines.append(f"  {col}: " + ", ".join(parts))
                except Exception:
                    lines.append(f"  {col}: <unavailable>")
    except Exception:
        pass

    # Categorical sample unique values (bounded)
    try:
        lines.append("categorical_samples:")
        for col in list(df.select_dtypes(include=["object", "category"]).columns)[
            :max_cols_report
        ]:
            try:
                uniques = list(df[col].dropna().unique())[:max_uniques_per_col]
                lines.append(
                    f"  {col}: {uniques}{'...' if df[col].nunique() > max_uniques_per_col else ''}"
                )
            except Exception:
                lines.append(f"  {col}: <unavailable>")
    except Exception:
        pass

    # Boolean columns (list only)
    try:
        bool_cols = list(df.select_dtypes(include=["bool"]).columns)
        # Heuristic: object columns that are effectively boolean (ignoring NaNs)
        try:
            obj_cols = list(df.select_dtypes(include=["object"]).columns)
        except Exception:
            obj_cols = []
        for col in obj_cols:
            try:
                non_null = df[col].dropna()
                if len(non_null) == 0:
                    continue
                # Treat as boolean-like if all values are True/False (bool type or truthy bools)
                if non_null.map(lambda x: isinstance(x, bool)).all():
                    bool_cols.append(col)
            except Exception:
                continue
        if bool_cols:
            # De-duplicate while preserving order
            seen: set[str] = set()
            deduped: list[str] = []
            for c in bool_cols:
                if c not in seen:
                    deduped.append(c)
                    seen.add(c)
            limited_bool = deduped[:max_cols_report]
            more = "..." if len(deduped) > max_cols_report else ""
            lines.append(f"boolean_columns: {limited_bool}{more}")
    except Exception:
        pass

    # Datetime ranges per column
    try:
        dt_cols = list(df.select_dtypes(include=["datetime"]).columns)
        if dt_cols:
            lines.append("datetime_ranges:")
            for col in dt_cols[:max_cols_report]:
                try:
                    col_series = df[col].dropna()
                    if len(col_series) == 0:
                        lines.append(f"  {col}: <no non-null values>")
                    else:
                        dt_min = str(col_series.min())
                        dt_max = str(col_series.max())
                        lines.append(f"  {col}: min={dt_min}, max={dt_max}")
                except Exception:
                    lines.append(f"  {col}: <unavailable>")
    except Exception:
        pass

    # Optional: data quality score (overall and per-column)
    if include_quality_score:
        try:
            total_cells = int(df.shape[0] * df.shape[1]) if hasattr(df, "shape") else 0
            total_missing = int(df.isna().sum().sum())
            quality = (
                100.0
                if total_cells == 0
                else 100.0 * (1.0 - (total_missing / max(total_cells, 1)))
            )
            lines.append(f"quality_score: {quality:.2f}")
            # Interpretation bands for quick reading
            try:
                if quality >= 95.0:
                    interp = "Excellent"
                elif quality >= 85.0:
                    interp = "Good"
                elif quality >= 70.0:
                    interp = "Fair"
                else:
                    interp = "Poor"
                lines.append(f"quality_interpretation: {interp}")
            except Exception:
                pass

            # Per-column quality (missing ratios)
            na_counts = df.isna().sum()
            col_quality: dict[str, float] = {}
            for k, v in na_counts.to_dict().items():
                try:
                    ratio = 0.0 if df.shape[0] == 0 else float(v) / float(df.shape[0])
                    col_quality[str(k)] = ratio
                except Exception:
                    col_quality[str(k)] = 1.0
            # Limit reported columns
            quality_items = list(col_quality.items())[:max_cols_report]
            limited_quality_map = {k: round(v, 4) for k, v in quality_items}
            more = "..." if len(col_quality) > max_cols_report else ""
            lines.append(f"column_quality: {limited_quality_map}{more}")
        except Exception:
            pass

    # Optional: column recommendations (heuristics)
    if include_recommendations:
        try:
            recommendations: dict[str, list[str]] = {
                "group_by_candidates": [],
                "key_like_columns": [],
                "numeric_analysis_candidates": [],
            }

            nunique_series = df.nunique(dropna=True)
            total_rows = int(df.shape[0]) if hasattr(df, "shape") else 0
            # Group-by: categorical/bool with 2-100 uniques and <50% missing
            try:
                cat_cols = list(
                    df.select_dtypes(include=["object", "category", "bool"]).columns
                )
            except Exception:
                cat_cols = []
            na_counts = df.isna().sum()
            for col in cat_cols:
                try:
                    u_cat = int(nunique_series.get(col, 0))
                    miss_ratio = (
                        0.0
                        if total_rows == 0
                        else float(na_counts.get(col, 0)) / float(total_rows)
                    )
                    if 2 <= u_cat <= 100 and miss_ratio <= 0.5:
                        recommendations["group_by_candidates"].append(str(col))
                except Exception:
                    continue

            # Key-like: object/category with uniqueness ratio > 0.8
            try:
                obj_cols = list(
                    df.select_dtypes(include=["object", "category"]).columns
                )
            except Exception:
                obj_cols = []
            for col in obj_cols:
                try:
                    u_obj = float(nunique_series.get(col, 0))
                    ratio = 0.0 if total_rows == 0 else u_obj / float(total_rows)
                    if ratio >= 0.8:
                        recommendations["key_like_columns"].append(str(col))
                except Exception:
                    continue

            # Numeric analysis: numeric with variance > 0 and low missing
            try:
                num_cols = list(df.select_dtypes(include=[float, int]).columns)
            except Exception:
                num_cols = []
            for col in num_cols:
                try:
                    miss_ratio = (
                        0.0
                        if total_rows == 0
                        else float(na_counts.get(col, 0)) / float(total_rows)
                    )
                    if miss_ratio <= 0.5:
                        variance = float(df[col].dropna().var())
                        if variance > 0.0:
                            recommendations["numeric_analysis_candidates"].append(
                                str(col)
                            )
                except Exception:
                    continue

            lines.append("recommendations:")
            for k in [
                "group_by_candidates",
                "key_like_columns",
                "numeric_analysis_candidates",
            ]:
                vals = recommendations.get(k, [])
                limited_list = vals[:max_cols_report]
                more = "..." if len(vals) > max_cols_report else ""
                lines.append(f"  {k}: {limited_list}{more}")
        except Exception:
            pass

    return "\n".join(lines)
