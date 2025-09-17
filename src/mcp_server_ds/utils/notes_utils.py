from __future__ import annotations


def append_note(notes: list[str], message: str) -> None:
    """Append a message to session notes in-place (simple helper)."""
    notes.append(message)
