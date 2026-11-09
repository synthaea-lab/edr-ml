"""The bad-line policy of each `data/` reader, now stated through `iter_jsonl`.

Each reader kept the policy its hand-written loop had: the baseline reader raises, the T1
capture reader skips silently, and the correlation aggregator and the capture converter
warn and go on. Changing one of these is a decision of its own, not a side effect.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest

from synthaea_ml.data import aggregate_correlation, behavior_dataset, capture_to_baseline
from synthaea_ml.data.baselines import load_cmdlines


def _exec(pid: int, ts_ns: int, argv: list[str]) -> str:
    return json.dumps(
        {
            "type": "exec",
            "meta": {"pid": pid, "timestamp_ns": ts_ns},
            "argv": argv,
            "cmdline": " ".join(argv),
        }
    )


def _write(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_baseline_reader_raises_on_an_invalid_line(tmp_path: Path) -> None:
    _write(tmp_path / "baseline.jsonl", [json.dumps({"argv": ["id"]}), "{not json"])

    with pytest.raises(ValueError, match="line 2 invalid"):
        load_cmdlines(tmp_path, "linux")


def test_capture_reader_skips_invalid_lines_silently(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    capture = _write(
        tmp_path / "events.jsonl",
        [_exec(1, 10, ["id"]), "{torn", "[1, 2]", "", _exec(2, 20, ["ls"])],
    )

    with caplog.at_level(logging.DEBUG):
        events = behavior_dataset.load_events(capture)
    assert [e["pid"] for e in events] == [1, 2]
    assert caplog.records == []


def test_capture_reader_replaces_undecodable_bytes(tmp_path: Path) -> None:
    capture = tmp_path / "events.jsonl"
    capture.write_bytes(_exec(1, 10, ["a\udcff"]).encode().replace(b"\\udcff", b"\xff") + b"\n")

    events = behavior_dataset.load_events(capture)
    assert [e["argv"] for e in events] == [["a�"]]


def test_capture_reader_requires_the_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="capture not found"):
        behavior_dataset.load_events(tmp_path / "events.jsonl")


def test_correlation_aggregator_warns_and_skips_invalid_lines(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    capture = _write(
        tmp_path / "events.jsonl",
        [_exec(1, 10, ["id"]), "{torn", "[1, 2]", _exec(2, 20, ["ls"])],
    )

    with caplog.at_level(logging.WARNING):
        events = aggregate_correlation.charger_events(capture)
    assert [e["pid"] for e in events] == [1, 2]
    assert [r.getMessage().split(" invalid:")[0] for r in caplog.records] == [
        f"{capture} line 2",
        f"{capture} line 3",
    ]


def test_correlation_aggregator_exits_when_the_capture_is_missing(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exited:
        aggregate_correlation.charger_events(tmp_path / "events.jsonl")
    assert exited.value.code == 1


def test_capture_converter_warns_skips_and_dedupes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    capture = _write(
        tmp_path / "events.jsonl",
        [
            _exec(1, 10, ["id"]),
            "{torn",
            json.dumps({"type": "connect", "meta": {"pid": 1, "timestamp_ns": 11}}),
            _exec(2, 20, ["ls", "-la"]),
            _exec(3, 30, ["id"]),
        ],
    )
    out = tmp_path / "out" / "baseline.jsonl"
    monkeypatch.setattr(sys, "argv", ["capture_to_baseline", str(capture), "--out", str(out)])

    with caplog.at_level(logging.WARNING):
        capture_to_baseline.main()

    assert out.read_text(encoding="utf-8").splitlines() == [
        json.dumps({"argv": ["id"]}),
        json.dumps({"argv": ["ls", "-la"]}),
    ]
    assert [r.getMessage().split(" invalid:")[0] for r in caplog.records] == [f"{capture} line 2"]
    assert "4 records read, 3 exec events, 2 unique command lines" in capsys.readouterr().out


def test_capture_converter_exits_when_the_capture_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", ["capture_to_baseline", str(tmp_path / "absent.jsonl")])

    with pytest.raises(SystemExit) as exited:
        capture_to_baseline.main()
    assert exited.value.code == 1
