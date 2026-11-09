"""Empirical calibration of the Bayesian filter's Log-Likelihood Ratios (LLR).

Reads `events.jsonl` (raw events logged by the agent) and `alerts.ndjson`
(emitted alerts), computes the per-PID BehaviorVectors within a 60s sliding
window, splits benign/malicious based on the alerted PIDs, and generates the
empirical LLRs + the corresponding Rust code for `log_likelihood_ratio()`.

Usage (from the repository root; the report goes to stdout without --out):
    python -m synthaea_ml.calibration.calibrate_llr \
        [--events events.jsonl] [--alerts alerts.ndjson] [--window 60] [--min-mal 1]
        [--out reports/calibration/llr_report_<run>.txt]
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from synthaea_ml.data.wire import normalize_for_behavior
from synthaea_ml.features.behavior import extract_behavior_features

# ── Constants ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BinSpec:
    """How one behavior feature is binned: the single source of both the discretizer
    that counts the empirical bins and the Rust match arm generated from their LLRs.

    A feature without `edges` is binary: "1" above 0.5, "0" otherwise. Otherwise 0.0
    maps to `absent` or `zero` (when set), then a value takes the label of the first
    edge it falls under (`value <= edge` when `inclusive`, `value < edge` otherwise),
    and `overflow` past the last edge.

    `absent` marks 0.0 as "never happened": in Rust every edge arm is guarded by
    `value > 0.0` and the else arm returns the absent LLR, so values past the last
    edge share it too (the `overflow` LLR is reported but not emitted). `zero` is an
    ordinary first bin, tested first in Rust.
    """

    name: str
    current_llr: str
    """The hand-calibrated LLR, shown in the report for comparison."""
    edges: tuple[tuple[float, str], ...] = ()
    absent: str | None = None
    zero: str | None = None
    overflow: str = ""
    inclusive: bool = False
    multiline: bool = False
    """Lay the generated Rust arm out as a block, one branch per line."""

    @property
    def labels(self) -> tuple[str, ...]:
        """Every bin label `discretize` can return."""
        if not self.edges:
            return ("0", "1")
        first = tuple(z for z in (self.absent, self.zero) if z is not None)
        return (*first, *(label for _, label in self.edges), self.overflow)

    def discretize(self, value: float) -> str:
        """The bin label of `value`."""
        if not self.edges:
            return "1" if value > 0.5 else "0"
        zero = self.absent if self.absent is not None else self.zero
        if zero is not None and value == 0.0:
            return zero
        for edge, label in self.edges:
            if (value <= edge) if self.inclusive else (value < edge):
                return label
        return self.overflow

    def rust_branches(self) -> tuple[list[tuple[str, str]], str]:
        """The Rust `if` conditions in order with the bin each returns, and the bin of
        the final `else`."""
        op = "<=" if self.inclusive else "<"
        guard = "value > 0.0 && " if self.absent is not None else ""
        branches = [(f"{guard}value {op} {edge:.1f}", label) for edge, label in self.edges]
        if self.absent is not None:
            return branches, self.absent
        if self.zero is not None:
            branches.insert(0, ("value == 0.0", self.zero))
        return branches, self.overflow


BIN_SPECS: tuple[BinSpec, ...] = (
    BinSpec("has_exec", current_llr="0.0"),
    BinSpec("has_connect", current_llr="0.3"),
    BinSpec("has_filewrite", current_llr="0.7"),
    BinSpec(
        "time_exec_to_connect_ms",
        current_llr="0.8/<500ms,0.3/<2s",
        edges=((500.0, "<500"), (2000.0, "<2000")),
        absent="absent",
        overflow=">=2000",
        multiline=True,
    ),
    BinSpec(
        "time_exec_to_filewrite_ms",
        current_llr="0.6",
        edges=((500.0, "<500"),),
        absent="absent",
        overflow=">=500",
    ),
    BinSpec("is_suspicious_path", current_llr="2.0"),
    BinSpec(
        "connect_count",
        current_llr="0.1*n,max1.5",
        edges=((5.0, "1-4"), (15.0, "5-14")),
        zero="0",
        overflow=">=15",
        multiline=True,
    ),
    BinSpec(
        "distinct_dports",
        current_llr="0.5",
        edges=((1.0, "0-1"), (3.0, "2-3")),
        overflow=">=4",
        inclusive=True,
    ),
    BinSpec("dest_is_external", current_llr="1.5"),
)
"""One spec per BehaviorVector feature, in vector order."""

FEATURE_NAMES = [spec.name for spec in BIN_SPECS]

# Hand-calibrated LLRs — kept as reference for comparison in the report.
LLR_CURRENT = [spec.current_llr for spec in BIN_SPECS]

# Laplace smoothing — prevents log(0) when a bin is unseen in a small sample.
LAPLACE_ALPHA = 0.5


# ── Parsing ────────────────────────────────────────────────────────────────────


def load_events(path: Path) -> list[dict]:
    """Load and normalize raw events from an events.jsonl file."""
    events = []
    try:
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(normalize_for_behavior(json.loads(line)))
                except json.JSONDecodeError as e:
                    print(f"[warn] events.jsonl line {i} invalid: {e}", file=sys.stderr)
    except FileNotFoundError:
        print(f"[error] {path} not found — run the agent first.", file=sys.stderr)
        sys.exit(1)
    return events


def load_malicious_pids(path: Path) -> set[int]:
    """Extract the alerted PIDs from an alerts.ndjson file."""
    pids: set[int] = set()
    if not path.exists():
        print(f"[warn] {path} not found — no known malicious PIDs.", file=sys.stderr)
        return pids
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                alert = json.loads(line)
                msg = alert.get("message", "")
                for token in msg.split():
                    if token.startswith("pid="):
                        with contextlib.suppress(ValueError):
                            pids.add(int(token[4:].rstrip(",")))
            except json.JSONDecodeError:
                pass
    return pids


# ── Sliding windowing ──────────────────────────────────────────────────────────


def group_by_pid_and_window(
    events: list[dict], window_ns: int
) -> dict[int, list[list[float]]]:
    """Group events by PID, slice into windows of window_ns nanoseconds, and
    return the BehaviorVectors per PID (one vector per non-empty window)."""
    by_pid: dict[int, list[dict]] = defaultdict(list)
    for e in events:
        if e.get("type") in ("exec", "connect", "fileopen"):
            by_pid[e["pid"]].append(e)

    bv_by_pid: dict[int, list[list[float]]] = {}

    for pid, pid_events in by_pid.items():
        pid_events.sort(key=lambda e: e["ts_ns"])
        if not pid_events:
            continue
        vectors = []
        window_start = pid_events[0]["ts_ns"]
        window: list[dict] = []
        for e in pid_events:
            if e["ts_ns"] - window_start > window_ns:
                if window:
                    vectors.append(extract_behavior_features(window))
                window_start = e["ts_ns"]
                window = [e]
            else:
                window.append(e)
        if window:
            vectors.append(extract_behavior_features(window))
        bv_by_pid[pid] = vectors

    return bv_by_pid


# ── Calibration ────────────────────────────────────────────────────────────────


def _discretize(feature_idx: int, value: float) -> str:
    """Discretize a feature value into its named bin (see `BIN_SPECS`); a value past the
    known features is named by its own text."""
    if 0 <= feature_idx < len(BIN_SPECS):
        return BIN_SPECS[feature_idx].discretize(value)
    return str(value)


def compute_empirical_llrs(
    bv_by_pid: dict[int, list[list[float]]],
    malicious_pids: set[int],
    alpha: float = LAPLACE_ALPHA,
) -> dict[int, dict[str, float]]:
    """Compute empirical LLRs for each feature bin via Laplace-smoothed counts.

    Returns feature_idx → {bin_name → llr}.
    """
    benign: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    malicious: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    n_benign = 0
    n_malicious = 0

    for pid, vectors in bv_by_pid.items():
        bucket = malicious if pid in malicious_pids else benign
        if pid in malicious_pids:
            n_malicious += len(vectors)
        else:
            n_benign += len(vectors)
        for bv in vectors:
            for i, val in enumerate(bv):
                bucket[i][_discretize(i, val)] += 1.0

    if n_benign == 0 or n_malicious == 0:
        print(
            f"[warn] n_benign={n_benign}, n_malicious={n_malicious}"
            " — not enough data to calibrate.",
            file=sys.stderr,
        )
        return {}

    llr: dict[int, dict[str, float]] = {}
    for i in range(len(BIN_SPECS)):
        all_bins = set(benign[i].keys()) | set(malicious[i].keys())
        total_b = sum(benign[i].values()) + alpha * len(all_bins)
        total_m = sum(malicious[i].values()) + alpha * len(all_bins)
        llr[i] = {}
        for b in all_bins:
            p_b = (benign[i].get(b, 0.0) + alpha) / total_b
            p_m = (malicious[i].get(b, 0.0) + alpha) / total_m
            llr[i][b] = math.log(p_m / p_b)

    return llr


# ── Report ─────────────────────────────────────────────────────────────────────


def generate_report(
    llr: dict[int, dict[str, float]],
    n_benign: int,
    n_malicious: int,
) -> str:
    """Render the calibration report as a human-readable string."""
    lines = []
    lines.append("=" * 70)
    lines.append("LLR CALIBRATION REPORT — Synthaea EDR")
    lines.append(f"Benign samples    : {n_benign}")
    lines.append(f"Malicious samples : {n_malicious}")
    lines.append("=" * 70)
    lines.append("")

    for i, name in enumerate(FEATURE_NAMES):
        if i not in llr:
            continue
        lines.append(f"[{i}] {name}")
        lines.append(f"    Current LLR : {LLR_CURRENT[i]}")
        lines.append("    Empirical LLRs per bin:")
        for bin_key, val in sorted(llr[i].items()):
            lines.append(f"        {bin_key:12s} → {val:+.3f}")
        lines.append("")

    lines.append("─" * 70)
    lines.append("SUGGESTED RUST CODE for log_likelihood_ratio():")
    lines.append("─" * 70)
    lines.append(generate_rust(llr))
    return "\n".join(lines)


def generate_rust(llr: dict[int, dict[str, float]]) -> str:
    """Generate the Rust match block for log_likelihood_ratio()."""
    rust = ["fn log_likelihood_ratio(idx: usize, value: f32) -> f32 {", "    match idx {"]
    for i, spec in enumerate(BIN_SPECS):
        rust.append(f"        // {i} — {spec.name}")
        if i in llr:
            rust.extend(_rust_arm(i, spec, llr[i]))
        else:
            rust.append(f"        {i} => 0.0,")
    rust.append("        _ => 0.0,")
    rust.append("    }")
    rust.append("}")
    return "\n".join(rust)


def _rust_arm(idx: int, spec: BinSpec, bins: dict[str, float]) -> list[str]:
    """The match arm of feature `idx`, returning the LLR of the bin `value` falls in (0.0
    for a bin the calibration never saw)."""
    if set(bins) <= {"0", "1"}:
        # Only the binary bins were seen, whatever the feature: test it as a flag.
        return [
            f"        {idx} => if value > 0.5 {{ {bins.get('1', 0.0):.2f} }} "
            f"else {{ {bins.get('0', 0.0):.2f} }},"
        ]
    if not spec.edges:
        return [f"        {idx} => {bins.get('1', 0.0):.2f},"]
    conditions, else_label = spec.rust_branches()
    branches = [f"if {cond} {{ {bins.get(label, 0.0):.2f} }}" for cond, label in conditions]
    branches.append(f"{{ {bins.get(else_label, 0.0):.2f} }}")
    if spec.multiline:
        return [
            f"        {idx} => {{",
            f"            {branches[0]}",
            *(f"            else {branch}" for branch in branches[1:]),
            "        }",
        ]
    return [f"        {idx} => {' else '.join(branches)},"]


# ── Main ───────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CalibrateLlrArgs:
    """The parsed command line of a run."""

    events: Path
    alerts: Path
    window_s: int
    min_malicious: int
    out: Path | None


def parse_args(argv: Sequence[str] | None = None) -> CalibrateLlrArgs:
    """The run's options from `argv` (the process's arguments when `None`)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", default="events.jsonl", help="Raw events file")
    parser.add_argument("--alerts", default="alerts.ndjson", help="JSON-Lines alerts file")
    parser.add_argument("--window", type=int, default=60, help="Sliding window in seconds")
    parser.add_argument("--min-mal", type=int, default=1, help="Min malicious PIDs required")
    parser.add_argument("--out", default=None, help="Report file (default: stdout)")
    args = parser.parse_args(argv)
    return CalibrateLlrArgs(
        events=Path(args.events),
        alerts=Path(args.alerts),
        window_s=args.window,
        min_malicious=args.min_mal,
        out=Path(args.out) if args.out else None,
    )


def run(args: CalibrateLlrArgs) -> int:
    """Compute the empirical LLRs and write the report; returns the exit status, 1 when the
    alerts name fewer malicious PIDs than required."""
    events = load_events(args.events)
    malicious_pids = load_malicious_pids(args.alerts)

    print(
        f"[info] {len(events)} events loaded, {len(malicious_pids)} malicious PIDs: {malicious_pids}",
        file=sys.stderr,
    )

    if len(malicious_pids) < args.min_malicious:
        print(
            f"[error] fewer than {args.min_malicious} malicious PID(s) in {args.alerts} — "
            "run AsyncRAT or another malware sample before calibrating.",
            file=sys.stderr,
        )
        return 1

    window_ns = args.window_s * 1_000_000_000
    bv_by_pid = group_by_pid_and_window(events, window_ns)

    n_benign = sum(len(v) for pid, v in bv_by_pid.items() if pid not in malicious_pids)
    n_malicious = sum(len(v) for pid, v in bv_by_pid.items() if pid in malicious_pids)

    llr = compute_empirical_llrs(bv_by_pid, malicious_pids)

    report = generate_report(llr, n_benign, n_malicious)

    if args.out is not None:
        args.out.write_text(report, encoding="utf-8")
        print(f"[info] report written to {args.out}", file=sys.stderr)
    else:
        print(report)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Calibrate from the command line `argv`; returns the exit status."""
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
