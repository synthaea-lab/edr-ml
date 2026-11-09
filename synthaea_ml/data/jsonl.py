"""Read JSON-Lines files (agent event logs, baselines, site corpora) with one stated policy.

Every reader of a `.jsonl` file goes through `iter_jsonl`, and every caller names what
happens to a line that is not a JSON object (`InvalidLine`): raise, skip silently, or
skip with a logged warning. The policy is a decision of the caller, never an accident of
how its loop was written.

Lines are split on `\\n` only (`\\r\\n` too), not on every Unicode line boundary:
`str.splitlines` would also break a record on a raw U+2028, U+2029 or U+0085 inside a
JSON string, which the agent's serializer does not escape.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from enum import Enum
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

EncodingErrors = Literal["strict", "replace"]


class InvalidLine(Enum):
    """What `iter_jsonl` does with a line that is not a JSON object."""

    RAISE = "raise"
    """Stop with `InvalidJsonlLineError`."""
    SKIP = "skip"
    """Drop the line without a trace."""
    WARN = "warn"
    """Drop the line and log a warning naming the file and line number."""


class InvalidJsonlLineError(ValueError):
    """A line of a JSON-Lines file is not valid JSON, or not a JSON object."""

    def __init__(self, path: Path, line_number: int, reason: str) -> None:
        """Record where the invalid line is and why it was rejected."""
        super().__init__(f"{path} line {line_number} invalid: {reason}")
        self.path = path
        self.line_number = line_number
        self.reason = reason


def iter_jsonl(
    path: Path,
    *,
    on_invalid: InvalidLine = InvalidLine.RAISE,
    encoding_errors: EncodingErrors = "strict",
) -> Iterator[dict[str, Any]]:
    """Yield each JSON object of `path`, in file order.

    Blank (whitespace-only) lines are ignored under every policy. A line that does not
    parse as JSON, or parses to something other than an object (a list, a number,
    `null`), is handled by `on_invalid`. A torn last line, as left by a capture that was
    interrupted mid-write, is such a line.

    Args:
        path: The JSON-Lines file. Opened lazily, on the first `next()`.
        on_invalid: What to do with an invalid line (see `InvalidLine`).
        encoding_errors: How undecodable UTF-8 bytes are handled, as in `open()`.

    Raises:
        FileNotFoundError: `path` does not exist.
        InvalidJsonlLineError: An invalid line under `InvalidLine.RAISE`.
    """
    with path.open(encoding="utf-8", errors=encoding_errors) as lines:
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                record = _parse_object(path, line_number, line)
            except InvalidJsonlLineError as error:
                _handle_invalid(error, on_invalid)
                continue
            yield record


def _parse_object(path: Path, line_number: int, line: str) -> dict[str, Any]:
    try:
        value = json.loads(line)
    except json.JSONDecodeError as error:
        raise InvalidJsonlLineError(path, line_number, str(error)) from error
    if not isinstance(value, dict):
        raise InvalidJsonlLineError(
            path, line_number, f"expected a JSON object, got {type(value).__name__}"
        )
    return value


def _handle_invalid(error: InvalidJsonlLineError, on_invalid: InvalidLine) -> None:
    if on_invalid is InvalidLine.RAISE:
        raise error
    if on_invalid is InvalidLine.WARN:
        logger.warning("%s", error)
