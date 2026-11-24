"""Small presentation helpers for the Streamlit UI."""

from __future__ import annotations

from datetime import datetime


def human_bytes(n: int | float | None) -> str:
    """Format a byte count as a human-readable string (e.g. ``1.4 MB``)."""
    if not n:
        return "0 B"
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def human_time(ts: float | None) -> str:
    """Format a POSIX timestamp as a local ``YYYY-MM-DD HH:MM`` string."""
    if not ts:
        return "—"
    try:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
    except (OSError, OverflowError, ValueError):
        return "—"


def class_display(label: str | None) -> str:
    """Human label for a class value (flat datasets have no class)."""
    return label if label else "(no class)"
