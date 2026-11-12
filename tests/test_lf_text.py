"""`lf_text`: hashed text artifacts have the same bytes whichever OS writes them."""

from __future__ import annotations

from pathlib import Path

from synthaea_ml.lf_text import open_lf_text, write_lf_text


def test_write_lf_text_writes_lf_line_endings_on_every_platform(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"

    write_lf_text(path, '{\n  "n": 1\n}\n')

    assert path.read_bytes() == b'{\n  "n": 1\n}\n'


def test_write_lf_text_encodes_utf8(tmp_path: Path) -> None:
    path = tmp_path / "artifact.txt"

    write_lf_text(path, "é\n")

    assert path.read_bytes() == b"\xc3\xa9\n"


def test_open_lf_text_writes_lf_line_endings_on_every_platform(tmp_path: Path) -> None:
    path = tmp_path / "artifact.jsonl"

    with open_lf_text(path) as file:
        file.write("a\nb\n")

    assert path.read_bytes() == b"a\nb\n"
