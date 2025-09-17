from __future__ import annotations


def validate_session_id(session_id: str | None) -> str:
    """Validate that session_id is a non-empty string and return the stripped value.

    Error message matches existing tests: both None and empty/blank strings
    raise "session_id is required for session isolation".
    """
    if session_id is None:
        raise ValueError("session_id is required for session isolation")
    if not isinstance(session_id, str):
        raise ValueError("session_id must be a non-empty string")
    # Distinguish empty vs whitespace-only to match existing tests
    if session_id == "":
        raise ValueError("session_id is required for session isolation")
    cleaned = session_id.strip()
    if not cleaned:
        raise ValueError("session_id must be a non-empty string")
    return cleaned
