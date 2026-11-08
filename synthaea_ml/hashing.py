"""Content hashing shared by the dataset manifest, the training record and the
scenario replays.

Every provenance check in this package binds an artifact by the SHA-256 of its
bytes, so they all go through one helper.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_CHUNK_SIZE = 1 << 16  # 64 KiB: streamed so large baselines need not fit in memory.


def sha256_file(path: Path) -> str:
    """Return the hex-encoded SHA-256 of the file's byte contents.

    The digest depends only on the bytes, not on the chunk size used to read them.
    """
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()
