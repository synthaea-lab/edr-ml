"""Write text files whose bytes do not depend on the OS that writes them.

Provenance checks bind artifacts (the model record, the dataset manifest, the baselines it
pins) by the SHA-256 of their bytes. Text mode alone turns every `\\n` into `\\r\\n` on
Windows, so the same content would hash differently there (#8, #57). Every hashed text
artifact is written through this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import TextIO


def open_lf_text(path: Path) -> TextIO:
    """Open `path` for writing UTF-8 text with `\\n` line endings on every platform."""
    return path.open("w", encoding="utf-8", newline="\n")


def write_lf_text(path: Path, text: str) -> None:
    """Write `text` to `path` as UTF-8 with `\\n` line endings on every platform."""
    with open_lf_text(path) as file:
        file.write(text)
