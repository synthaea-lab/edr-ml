"""`iter_jsonl`, the one JSON-Lines reader, and the policy each caller states for bad lines."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from synthaea_ml.data.jsonl import InvalidJsonlLineError, InvalidLine, iter_jsonl


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "data.jsonl"
    path.write_bytes(text.encode("utf-8"))
    return path


def test_yields_objects_in_file_order(tmp_path: Path) -> None:
    path = _write(tmp_path, '{"n": 1}\n{"n": 2}\n{"n": 3}\n')

    assert [r["n"] for r in iter_jsonl(path)] == [1, 2, 3]


@pytest.mark.parametrize("policy", list(InvalidLine))
def test_blank_lines_are_ignored_under_every_policy(
    tmp_path: Path, policy: InvalidLine, caplog: pytest.LogCaptureFixture
) -> None:
    path = _write(tmp_path, '\n{"n": 1}\n   \n\t\n{"n": 2}\n\n')

    with caplog.at_level(logging.WARNING):
        assert list(iter_jsonl(path, on_invalid=policy)) == [{"n": 1}, {"n": 2}]
    assert caplog.records == []


def test_last_line_without_newline_is_read(tmp_path: Path) -> None:
    path = _write(tmp_path, '{"n": 1}\n{"n": 2}')

    assert list(iter_jsonl(path)) == [{"n": 1}, {"n": 2}]


def test_crlf_line_endings_are_read(tmp_path: Path) -> None:
    path = _write(tmp_path, '{"n": 1}\r\n{"n": 2}\r\n')

    assert list(iter_jsonl(path)) == [{"n": 1}, {"n": 2}]


@pytest.mark.parametrize("separator", [" ", " ", "\x85"])
def test_unicode_line_separator_inside_a_string_does_not_split_the_record(
    tmp_path: Path, separator: str
) -> None:
    record = {"cmdline": f"echo a{separator}b"}
    path = _write(tmp_path, json.dumps(record, ensure_ascii=False) + "\n")

    assert list(iter_jsonl(path)) == [record]


def test_torn_last_line_raises_by_default(tmp_path: Path) -> None:
    path = _write(tmp_path, '{"n": 1}\n{"n": 2, "argv": ["l')

    records = iter_jsonl(path)
    assert next(records) == {"n": 1}
    with pytest.raises(InvalidJsonlLineError) as raised:
        next(records)
    assert raised.value.path == path
    assert raised.value.line_number == 2
    assert str(raised.value).startswith(f"{path} line 2 invalid: ")


def test_invalid_line_error_is_a_value_error(tmp_path: Path) -> None:
    path = _write(tmp_path, "not json\n")

    with pytest.raises(ValueError, match="line 1 invalid"):
        list(iter_jsonl(path))


@pytest.mark.parametrize("value", ["[1, 2]", "42", '"text"', "null", "true"])
def test_non_object_json_is_an_invalid_line(tmp_path: Path, value: str) -> None:
    path = _write(tmp_path, value + "\n")

    with pytest.raises(InvalidJsonlLineError, match="expected a JSON object"):
        list(iter_jsonl(path, on_invalid=InvalidLine.RAISE))


def test_skip_drops_invalid_lines_silently(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = _write(tmp_path, '{"n": 1}\nnot json\n[1]\n{"n": 2}\n{"n": 3')

    with caplog.at_level(logging.DEBUG):
        records = list(iter_jsonl(path, on_invalid=InvalidLine.SKIP))
    assert records == [{"n": 1}, {"n": 2}]
    assert caplog.records == []


def test_warn_drops_invalid_lines_and_logs_each_one(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = _write(tmp_path, '{"n": 1}\nnot json\n[1]\n{"n": 2}\n{"n": 3')

    with caplog.at_level(logging.WARNING, logger="synthaea_ml.data.jsonl"):
        records = list(iter_jsonl(path, on_invalid=InvalidLine.WARN))
    assert records == [{"n": 1}, {"n": 2}]
    assert [r.levelno for r in caplog.records] == [logging.WARNING] * 3
    assert [r.getMessage().split(" invalid:")[0] for r in caplog.records] == [
        f"{path} line 2",
        f"{path} line 3",
        f"{path} line 5",
    ]


def test_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        list(iter_jsonl(tmp_path / "absent.jsonl"))


def test_undecodable_bytes_raise_unless_replaced(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_bytes(b'{"cmdline": "a\xffb"}\n')

    with pytest.raises(UnicodeDecodeError):
        list(iter_jsonl(path))
    assert list(iter_jsonl(path, encoding_errors="replace")) == [{"cmdline": "a�b"}]
