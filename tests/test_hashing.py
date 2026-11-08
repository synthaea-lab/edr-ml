"""`sha256_file` must equal the one-shot digest of the file's bytes, whatever
the size relative to the read chunk: the training record and the dataset
manifest lock artifacts by this value."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from synthaea_ml.hashing import _CHUNK_SIZE, sha256_file

EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


@pytest.mark.parametrize(
    "size",
    [0, 1, _CHUNK_SIZE - 1, _CHUNK_SIZE, _CHUNK_SIZE + 1, 3 * _CHUNK_SIZE + 17],
    ids=["empty", "one-byte", "chunk-minus-one", "one-chunk", "chunk-plus-one", "multi-chunk"],
)
def test_sha256_file_matches_one_shot_digest(tmp_path: Path, size: int) -> None:
    data = bytes(i % 251 for i in range(size))
    path = tmp_path / "blob.bin"
    path.write_bytes(data)
    assert sha256_file(path) == hashlib.sha256(data).hexdigest()


def test_sha256_file_of_empty_file_is_the_known_empty_digest(tmp_path: Path) -> None:
    path = tmp_path / "empty"
    path.write_bytes(b"")
    assert sha256_file(path) == EMPTY_SHA256
