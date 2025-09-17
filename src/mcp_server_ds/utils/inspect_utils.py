from __future__ import annotations

from typing import Any


def summarize_session_data(
    session_id: str,
    session_data: dict[str, Any] | None,
    include_preview: bool = True,
    max_rows: int = 5,
    max_cols: int = 10,
) -> str:
    """Produce a human-readable summary of session data mapping.

    Pure utility (no side effects), suitable for testing.
    """
    data = session_data or {}
    lines: list[str] = []
    lines.append("=== INSPECT MEMORY ===")
    lines.append(f"Items: {len(data)}")

    if not data:
        return "\n".join(lines + ["No dataframes found in this session."])

    for name, obj in data.items():
        lines.append("")
        lines.append(f"- {name}:")
        shape = getattr(obj, "shape", None)
        if shape is not None:
            lines.append(f"  shape: {shape}")
            cols = getattr(obj, "columns", None)
            if cols is not None:
                cols_list = list(cols)[:max_cols]
                more = "..." if len(list(cols)) > max_cols else ""
                lines.append(f"  columns: {cols_list}{more}")
            dtypes = getattr(obj, "dtypes", None)
            if dtypes is not None:
                try:
                    dtype_map = {str(k): str(v) for k, v in dtypes.to_dict().items()}
                    lines.append(f"  dtypes: {dtype_map}")
                except Exception:
                    pass
            # Preview disabled to avoid exposing sensitive data content
        else:
            lines.append(f"  type: {type(obj)}")

    return "\n".join(lines)
