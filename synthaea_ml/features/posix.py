"""POSIX `open(2)` flags and the write-intent rule shared by the feature extractors.

Mirror of `schema::has_write_intent` (Rust), the one definition of write intent: the
access mode is a 2-bit *field* (`O_ACCMODE`), not independent bits, so the invalid `0o3`
combination, which the kernel refuses with `EINVAL`, is NOT a write.
"""

from __future__ import annotations

O_ACCMODE: int = 0o3
O_WRONLY: int = 0o1
O_RDWR: int = 0o2
O_CREAT: int = 0o100
O_TRUNC: int = 0o1000


def has_write_intent(flags: int) -> bool:
    """Whether `open(2)` flags ask to write: access mode `O_WRONLY`/`O_RDWR`, `O_CREAT`, or
    `O_TRUNC` (`O_RDONLY | O_TRUNC` empties the file on Linux, agent#233)."""
    access_mode = flags & O_ACCMODE
    return access_mode in (O_WRONLY, O_RDWR) or bool(flags & (O_CREAT | O_TRUNC))
