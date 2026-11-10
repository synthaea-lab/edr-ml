"""The one spelling of a UTC timestamp in a record: `2026-09-11T12:00:00Z`."""

from __future__ import annotations

from datetime import UTC, datetime

UTC_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def format_utc(moment: datetime) -> str:
    """`moment` as UTC to the second with a `Z` suffix, e.g. `2026-09-11T12:00:00Z`.

    Raises:
        ValueError: If `moment` is naive: its zone cannot be guessed.
    """
    if moment.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return moment.astimezone(UTC).strftime(UTC_FORMAT)
