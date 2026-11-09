"""Read benign command-line baselines and deduplicate samples, shared by the trainers.

A baseline is `<baseline_dir>/baseline.jsonl`, one exec record per line. Its shape
depends on the platform the capture came from:

- `linux`: an eBPF exec record whose `argv` is rebuilt into the NUL-terminated
  string `extract_features` expects (`ml_cmdline_from_record`, identical to
  `ExecEvent::ml_cmdline` on the Rust side).
- `windows`: `{"cmdline": "C:\\\\Windows\\\\..."}`, the flat ETW command line. ETW
  does not populate `argv[]`, so the string is used as-is.
"""

import json
from collections.abc import Callable, Hashable, Iterable
from pathlib import Path
from typing import Any, Literal, TypeVar

from synthaea_ml.data.canonical import ml_cmdline_from_record
from synthaea_ml.data.manifest import DEFAULT_BASELINE_FILENAME

Platform = Literal["linux", "windows"]

T = TypeVar("T", bound=Hashable)


def _windows_cmdline(record: dict[str, Any]) -> str:
    cmdline: str = record["cmdline"]
    return cmdline


_CMDLINE_FROM_RECORD: dict[Platform, Callable[[dict[str, Any]], str]] = {
    "linux": ml_cmdline_from_record,
    "windows": _windows_cmdline,
}


def unique_in_order(items: Iterable[T]) -> list[T]:
    """Return `items` without duplicates, keeping the first occurrence of each.

    Order matters downstream: the trainers split and fit on the resulting list, so a
    different order would change the model.
    """
    return list(dict.fromkeys(items))


def load_cmdlines(baseline_dir: Path, platform: Platform) -> list[str]:
    """Return every command line of `baseline_dir/baseline.jsonl`, in file order.

    Blank lines are skipped. Duplicates are kept so callers can report the raw count;
    pass the result through `unique_in_order` to deduplicate.

    Raises `FileNotFoundError` when the baseline file is missing.
    """
    baseline_path = baseline_dir / DEFAULT_BASELINE_FILENAME
    if not baseline_path.exists():
        raise FileNotFoundError(f"baseline not found: {baseline_path}")

    cmdline_from_record = _CMDLINE_FROM_RECORD[platform]
    return [
        cmdline_from_record(json.loads(line))
        for line in baseline_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
