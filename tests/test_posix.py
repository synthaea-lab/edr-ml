"""The write-intent rule shared by the behavior and correlation extractors."""

from __future__ import annotations

import pytest

from synthaea_ml.features.posix import O_CREAT, O_RDWR, O_TRUNC, O_WRONLY, has_write_intent


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        (0, False),
        (O_WRONLY, True),
        (O_RDWR, True),
        (0o3, False),  # invalid access mode, refused by the kernel
        (O_CREAT, True),
        (0o3 | O_CREAT, True),
        (0o2000000, False),  # O_CLOEXEC alone
        (O_TRUNC, True),  # O_RDONLY | O_TRUNC empties the file (agent#233)
        (O_TRUNC | 0o2000000, True),
        (0o3 | O_TRUNC, True),  # truncation still counts
    ],
)
def test_write_intent(flags: int, expected: bool) -> None:
    assert has_write_intent(flags) is expected
