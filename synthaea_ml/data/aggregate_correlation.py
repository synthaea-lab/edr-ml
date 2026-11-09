"""Aggregates `events.jsonl` into a training set for the correlation scorer (2nd ML scorer).

Reads a raw event log (`events.jsonl`, produced by `edr-cli capture-events` or
`edr-cli run`) and produces **one feature vector per PID**: for each PID, the window is
reconstructed as it would be at its last appearance — all of its events within
`[last_ts - window, last_ts]` — mirroring the eviction semantics of
`synthaea_correlator::EventBus` (`cutoff = latest_ns - window_ns`, keep `ts >= cutoff`).

One vector per PID (not one per fixed time step): decision of 2026-09-02 with Nikolas —
the most complete instant for a PID, and the simplest to align with the Rust side.

This script does **not** do a benign/malicious split: the 2nd scorer's Isolation Forest is
unsupervised, it trains on a benign baseline alone (cf. `train_correlation.py`).
`calibrate_llr.py` (Bayesian filter) is the one that needs the split — don't mix them up.

Output: JSON-Lines, one `{"pid", "features": [...], "feature_names": [...]}` object per line.

Usage (from the repository root):
    python -m synthaea_ml.data.aggregate_correlation \
        [--events events.jsonl] --out <dataset-dir>/correlation_train.jsonl
        [--window 60] [--min-events 1]

then `python -m synthaea_ml.data.manifest write --baseline-filename correlation_train.jsonl
...` in `<dataset-dir>` and train with `train_correlation --dataset <dataset-dir>`.

Without `--out`, the vectors go to `LEGACY_OUT_PATH` inside the package, where
`train_correlation` run without options reads them. That default is deprecated.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from synthaea_ml.data.jsonl import InvalidLine, iter_jsonl
from synthaea_ml.data.manifest import CORRELATION_DATASET_FILENAME
from synthaea_ml.data.wire import RAW_EVENT_TYPES, flatten_wire_event
from synthaea_ml.features.correlation import FEATURE_NAMES, extract_features

# Re-exported: callers imported the wire translator from here before `data/wire.py`.
__all__ = [
    "LEGACY_OUT_PATH",
    "RAW_EVENT_TYPES",
    "charger_events",
    "flatten_wire_event",
    "main",
    "vecteurs_par_pid",
]

LEGACY_OUT_PATH = Path(__file__).parents[1] / "training" / "data" / CORRELATION_DATASET_FILENAME
"""Where the vectors go without `--out`, and where `train_correlation` run without options
reads them. Deprecated: it lies inside the package source tree."""


def charger_events(path: Path) -> list[dict]:
    """Loads the usable events from events.jsonl (exec/connect/file_open), flattened by
    `flatten_wire_event` to the shape `correlation.py` expects."""
    events: list[dict] = []
    try:
        for raw in iter_jsonl(path, on_invalid=InvalidLine.WARN):
            e = flatten_wire_event(raw)
            if e is not None:
                events.append(e)
    except FileNotFoundError:
        print(
            f"[error] {path} not found — run `edr-cli capture-events` first.",
            file=sys.stderr,
        )
        sys.exit(1)
    return events


def vecteurs_par_pid(events: list[dict], window_ns: int, min_events: int) -> list[dict]:
    """One `extract_features` per PID, over its window at its last appearance."""
    par_pid: dict[int, list[dict]] = defaultdict(list)
    for e in events:
        par_pid[e["pid"]].append(e)

    out: list[dict] = []
    for pid, pid_events in sorted(par_pid.items()):
        pid_events.sort(key=lambda e: e["ts_ns"])
        last_ts = pid_events[-1]["ts_ns"]
        cutoff = last_ts - window_ns
        fenetre = [e for e in pid_events if e["ts_ns"] >= cutoff]
        if len(fenetre) < min_events:
            continue
        feats = extract_features(fenetre, pid)
        out.append({"pid": pid, "features": feats, "feature_names": list(FEATURE_NAMES)})
    return out


@dataclass(frozen=True)
class AggregateCorrelationArgs:
    """The parsed command line of a run."""

    events: Path
    out: Path
    window_s: float
    min_events: int


def parse_args(argv: Sequence[str] | None = None) -> AggregateCorrelationArgs:
    """The run's options from `argv` (the process's arguments when `None`)."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--events", type=Path, default=Path("events.jsonl"))
    parser.add_argument(
        "--out",
        type=Path,
        help=(
            f"vectors file to write, normally <dataset-dir>/{CORRELATION_DATASET_FILENAME} "
            "(deprecated default: inside the package, where train_correlation without "
            "options reads it)"
        ),
    )
    parser.add_argument(
        "--window",
        type=float,
        default=60.0,
        help="correlation window in seconds (default 60 = CorrelationEngine::new)",
    )
    parser.add_argument(
        "--min-events",
        type=int,
        default=1,
        help="skip PIDs with fewer than N events in the window",
    )
    args = parser.parse_args(argv)
    out = args.out
    if out is None:
        warnings.warn(
            f"aggregate_correlation without --out writes into the package ({LEGACY_OUT_PATH}); "
            f"pass --out <dataset-dir>/{CORRELATION_DATASET_FILENAME}. The default will be "
            "removed in the next release.",
            FutureWarning,
            stacklevel=2,
        )
        out = LEGACY_OUT_PATH
    return AggregateCorrelationArgs(
        events=args.events, out=out, window_s=args.window, min_events=args.min_events
    )


def run(args: AggregateCorrelationArgs) -> int:
    """Write one correlation vector per PID of the event log; returns the exit status, 1 when
    the log has no usable event."""
    events = charger_events(args.events)
    if not events:
        print("[error] no usable events in the log.", file=sys.stderr)
        return 1

    vecteurs = vecteurs_par_pid(events, int(args.window_s * 1_000_000_000), args.min_events)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for v in vecteurs:
            f.write(json.dumps(v) + "\n")

    print(
        f"{len(events)} events → {len(vecteurs)} vectors "
        f"(1 per PID, window {args.window_s:g}s, min-events {args.min_events}) → {args.out}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Aggregate an event log from the command line `argv`; returns the exit status."""
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
