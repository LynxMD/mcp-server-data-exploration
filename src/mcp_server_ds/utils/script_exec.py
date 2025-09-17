from __future__ import annotations

from io import StringIO
from typing import Any
import sys


def build_exec_globals(
    pd, np, scipy, sklearn, sm, pyarrow, Image, pytesseract, pymupdf
) -> dict[str, Any]:
    """Build the globals dict used for exec()."""
    return {
        "pd": pd,
        "np": np,
        "scipy": scipy,
        "sklearn": sklearn,
        "statsmodels": sm,
        "pyarrow": pyarrow,
        "Image": Image,
        "pytesseract": pytesseract,
        "pymupdf": pymupdf,
    }


def capture_stdout_exec(
    script: str, globals_dict: dict[str, Any], locals_dict: dict[str, Any]
) -> str:
    """Execute script capturing stdout and return captured output."""
    stdout_capture = StringIO()
    old_stdout = sys.stdout
    try:
        sys.stdout = stdout_capture
        exec(script, globals_dict, locals_dict)
    finally:
        sys.stdout = old_stdout
    return stdout_capture.getvalue()
