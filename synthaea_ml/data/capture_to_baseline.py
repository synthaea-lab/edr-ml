"""Converts a raw agent events.jsonl capture into a benign JSONL baseline
consumable by train.py (cmdline isolation-forest).

Reads the JSON-Lines format produced by the agent (`crates/schema::Event`):
    {"type": "exec", "meta": {...}, "cmdline": "...", "argv": [...], ...}

argv is the canonical form (see `synthaea_ml.data.canonical`): on Linux/eBPF it is
populated and used directly; on Windows/ETW it is empty and the flat "cmdline"
string is used as one token.

Duplicate command lines (same argv tuple) are dropped so the baseline stays
compact and representative — a process that runs 10 000 times adds one sample.

Usage:
    python -m synthaea_ml.data.capture_to_baseline <events.jsonl> [--out baseline_benign.jsonl]
"""

import argparse
import json
import sys
from pathlib import Path

from synthaea_ml.data.canonical import argv_from_record
from synthaea_ml.data.jsonl import InvalidLine, iter_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events_file", type=Path, help="Path to events.jsonl")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("baseline_benign.jsonl"),
        help="Output path (default: baseline_benign.jsonl)",
    )
    args = parser.parse_args()

    if not args.events_file.exists():
        print(f"[error] {args.events_file} not found.", file=sys.stderr)
        sys.exit(1)

    seen: set[tuple[str, ...]] = set()
    argv_list: list[list[str]] = []
    n_records = 0
    n_exec = 0

    for event in iter_jsonl(
        args.events_file, on_invalid=InvalidLine.WARN, encoding_errors="replace"
    ):
        n_records += 1
        if event.get("type") != "exec":
            continue

        n_exec += 1
        argv = argv_from_record(event)
        if not argv:
            continue

        key = tuple(argv)
        if key in seen:
            continue
        seen.add(key)
        argv_list.append(argv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for argv in argv_list:
            f.write(json.dumps({"argv": argv}) + "\n")

    print(
        f"[info] {n_records} records read, {n_exec} exec events, "
        f"{len(argv_list)} unique command lines → {args.out}"
    )


if __name__ == "__main__":
    main()
