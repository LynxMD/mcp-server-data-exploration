from __future__ import annotations

import pytest

from mcp_server_ds.utils.session_utils import validate_session_id
from mcp_server_ds.utils.notes_utils import append_note
from mcp_server_ds.utils.io_utils import read_csv_strict
from mcp_server_ds.utils.script_exec import build_exec_globals, capture_stdout_exec


def test_validate_session_id_ok():
    assert validate_session_id(" abc ") == "abc"


@pytest.mark.parametrize("bad", [None, "", "   ", 123])
def test_validate_session_id_errors(bad):
    with pytest.raises(ValueError):
        validate_session_id(bad)


def test_append_note():
    notes: list[str] = []
    append_note(notes, "hello")
    assert notes == ["hello"]


def test_read_csv_strict(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("x\n1\n")
    df = read_csv_strict(str(p))
    assert list(df.columns) == ["x"]


def test_read_csv_strict_error(tmp_path):
    with pytest.raises(Exception) as e:
        read_csv_strict(str(tmp_path / "missing.csv"))
    assert "Error loading CSV:" in str(e.value)


def test_capture_stdout_exec_and_globals():
    # Minimal globals
    globals_dict = build_exec_globals(
        pd=object(),
        np=object(),
        scipy=object(),
        sklearn=object(),
        sm=object(),
        pyarrow=object(),
        Image=object(),
        pytesseract=object(),
        pymupdf=object(),
    )
    locals_dict: dict[str, object] = {}
    out = capture_stdout_exec("print('ok')", globals_dict, locals_dict)
    assert out.strip() == "ok"
