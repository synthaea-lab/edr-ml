"""Baseline parsing and order-preserving dedup shared by the cmdline trainers.

These pin the behavior the Linux, Windows and per-site trainers had when each one carried
its own copy of the loop: file order is kept, blank lines are skipped, duplicates survive
`load_cmdlines` (so the trainers can log the raw count) and `unique_in_order` keeps the
first occurrence. The trainers split and fit on this list, so order is part of the model.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from synthaea_ml.data.baselines import load_cmdlines, unique_in_order
from synthaea_ml.training import train_linux, train_site_model, train_windows


def _write_baseline(tmp_path: Path, lines: list[str]) -> Path:
    (tmp_path / "baseline.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tmp_path


def test_unique_in_order_keeps_first_occurrence() -> None:
    assert unique_in_order(["b", "a", "b", "c", "a"]) == ["b", "a", "c"]


def test_unique_in_order_accepts_any_iterable() -> None:
    assert unique_in_order(iter([3, 1, 3, 2])) == [3, 1, 2]
    assert unique_in_order([]) == []


def test_linux_records_become_nul_terminated_argv(tmp_path: Path) -> None:
    baseline = _write_baseline(
        tmp_path,
        [
            json.dumps({"argv": ["ls", "-la"]}),
            "",
            "   ",
            json.dumps({"argv": ["whoami"]}),
            json.dumps({"argv": ["ls", "-la"]}),
        ],
    )

    assert load_cmdlines(baseline, "linux") == ["ls\0-la\0", "whoami\0", "ls\0-la\0"]


def test_linux_record_without_argv_falls_back_to_flat_cmdline(tmp_path: Path) -> None:
    baseline = _write_baseline(tmp_path, [json.dumps({"cmdline": "bash -c id"})])

    assert load_cmdlines(baseline, "linux") == ["bash -c id"]


def test_windows_records_use_the_flat_cmdline_verbatim(tmp_path: Path) -> None:
    svchost = "C:\\Windows\\System32\\svchost.exe -k netsvcs"
    baseline = _write_baseline(
        tmp_path,
        [
            json.dumps({"cmdline": svchost}),
            "",
            json.dumps({"cmdline": svchost, "argv": ["ignored"]}),
        ],
    )

    assert load_cmdlines(baseline, "windows") == [svchost, svchost]


def test_windows_record_without_cmdline_is_rejected(tmp_path: Path) -> None:
    baseline = _write_baseline(tmp_path, [json.dumps({"argv": ["whoami"]})])

    with pytest.raises(KeyError):
        load_cmdlines(baseline, "windows")


@pytest.mark.parametrize("platform", ["linux", "windows"])
def test_missing_baseline_raises(tmp_path: Path, platform: str) -> None:
    with pytest.raises(FileNotFoundError, match="baseline not found"):
        load_cmdlines(tmp_path, platform)  # type: ignore[arg-type]


def _info_messages(caplog: pytest.LogCaptureFixture, logger: str) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == logger and r.levelno == logging.INFO]


def test_linux_trainer_dedupes_its_baseline_and_logs_the_counts(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    baseline = _write_baseline(
        tmp_path,
        [json.dumps({"argv": a}) for a in (["id"], ["ls"], ["id"])],
    )

    with caplog.at_level(logging.INFO, logger=train_linux.__name__):
        assert train_linux.load_baseline(baseline) == ["id\0", "ls\0"]

    assert _info_messages(caplog, train_linux.__name__) == [
        "Baseline: 3 entries -> 2 unique after deduplication"
    ]
    assert capsys.readouterr().out == ""


def test_windows_trainer_dedupes_its_baseline_and_logs_the_counts(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    baseline = _write_baseline(
        tmp_path,
        [json.dumps({"cmdline": c}) for c in ("b.exe", "a.exe", "b.exe")],
    )

    with caplog.at_level(logging.INFO, logger=train_windows.__name__):
        assert train_windows.load_baseline(baseline) == ["b.exe", "a.exe"]

    assert _info_messages(caplog, train_windows.__name__) == [
        "Baseline: 3 entries -> 2 unique after deduplication"
    ]
    assert capsys.readouterr().out == ""


def test_site_global_baseline_keeps_its_own_missing_file_message(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Global baseline not found"):
        train_site_model.load_global_baseline(tmp_path)


def test_site_corpus_keeps_exec_events_only_and_dedupes(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        "\n".join(
            json.dumps(r)
            for r in (
                {"event_type": "exec", "event": {"argv": ["cron"]}},
                {"event_type": "net", "event": {"argv": ["skipped"]}},
                {"event_type": "exec"},
                {"event_type": "exec", "event": {"argv": ["cron"]}},
                {"event_type": "exec", "event": {"argv": ["backup", "--all"]}},
            )
        ),
        encoding="utf-8",
    )

    assert train_site_model.load_site_corpus(corpus) == ["cron\0", "backup\0--all\0"]


def test_combine_corpora_puts_global_first_and_dedupes_site_against_it() -> None:
    combined = train_site_model.combine_corpora(["a", "b"], ["c", "a", "d", "c"])

    assert combined == ["a", "b", "c", "d"]


def test_site_corpus_loaders_log_their_counts_and_write_nothing_to_stdout(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    baseline = _write_baseline(
        tmp_path, [json.dumps({"argv": a}) for a in (["id"], ["ls"], ["id"])]
    )
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        json.dumps({"event_type": "exec", "event": {"argv": ["cron"]}}) + "\n", encoding="utf-8"
    )

    with caplog.at_level(logging.INFO, logger=train_site_model.__name__):
        global_corpus = train_site_model.load_global_baseline(baseline)
        site_corpus = train_site_model.load_site_corpus(corpus)
        train_site_model.combine_corpora(global_corpus, site_corpus)

    assert _info_messages(caplog, train_site_model.__name__) == [
        "Global baseline: 3 entries -> 2 unique",
        "Site corpus: 1 entries -> 1 unique",
        "Combined corpus: 3 unique samples (2 global + 1 site)",
    ]
    assert capsys.readouterr().out == ""
