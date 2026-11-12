"""The agent's wire event (`crates/schema::Event` JSON) and its flat Python views.

A wire event nests the process identity under `meta` (`pid`, `timestamp_ns`, and
optionally `process_generation`) and serializes the event kind as a snake_case `type`
tag (`exec`, `connect`, `file_open`, `network_flow`, ...). The Python feature
extractors were written against a flat shape instead (`pid`/`ts_ns` at top level, a
`fileopen` tag). This module is the single place that translates one into the other.

There are two translators, on purpose. They serve different readers and do not agree
on the same input, so neither can be written in terms of the other without changing
an output (see `tests/test_wire.py`, which pins both and the points where they differ):

- `flatten_wire_event` builds the minimal flat event that `features/correlation.py`
  reads. It is the parity path: `capture_correlation_golden.jsonl` and the Rust side
  depend on it. It drops event types the correlation features do not use, drops
  events without `meta`, and keeps `daddr` verbatim (as a 1-element `daddr_v4`).
- `normalize_for_behavior` feeds `features/behavior.py` in the LLR calibration. It
  keeps every input field (including `is_ipv6` and a Host-Only `daddr_v4` octet list,
  which `flatten_wire_event` would drop or fail on), passes already-flat events
  through unchanged, defaults a missing pid or timestamp to 0, keeps unknown types,
  and splits a dotted-decimal `daddr` into the octet list the behavior extractor
  reads to tell internal from external destinations.
"""

from __future__ import annotations

from typing import Any

from synthaea_ml.data.records import FlatEvent

# Wire tags as `crates/schema::Event` serializes them (serde `rename_all = "snake_case"`
# on the `FileOpen`/`NetworkFlow` variants): distinct from `correlation.py`'s "fileopen"
# (see below). `network_flow` added by ADR-0008 (NetworkFlow absorbed into T2's
# daddr/dport sets alongside `connect`).
RAW_EVENT_TYPES = ("exec", "connect", "file_open", "network_flow")


def flatten_wire_event(raw: dict[str, Any]) -> FlatEvent | None:
    """Translates one wire event (`crates/schema::Event` JSON: identity nested under
    `meta`, `daddr` a single address string) into the flat shape `correlation.py` was
    written against (`pid`/`ts_ns` at top level, `fileopen` tag, `daddr_v4` as a
    hashable destination key — the address family is never read by `extract_features`,
    only used for counting distinct destinations, so the raw string is kept verbatim
    inside a 1-element list rather than actually split into octets)."""
    event_type = raw.get("type")
    meta = raw.get("meta")
    if event_type not in RAW_EVENT_TYPES or meta is None:
        return None
    e: FlatEvent = {
        "type": "fileopen" if event_type == "file_open" else event_type,
        "pid": meta["pid"],
        "ts_ns": meta["timestamp_ns"],
    }
    # The process incarnation (#590) and, on exec, the lineage and command line the T1
    # tier scores (#617): kept when the wire event carries them, absent otherwise, so a
    # capture without them flattens exactly as it always did.
    if meta.get("process_generation") is not None:
        e["process_generation"] = meta["process_generation"]
    if event_type == "exec":
        for key in ("argv", "cmdline", "image_path", "parent_comm", "parent_image_path"):
            if raw.get(key) is not None:
                e[key] = raw[key]
    if event_type in ("connect", "network_flow"):
        e["daddr_v4"] = [raw["daddr"]]
        e["dport"] = raw["dport"]
    elif event_type == "file_open":
        e["flags"] = raw["flags"]
    return e


def normalize_for_behavior(e: dict[str, Any]) -> dict[str, Any]:
    """Flatten the agent's nested event format to the flat format `behavior.py` reads.

    Agent format:
        {"type": "exec",      "meta": {"pid": ..., "timestamp_ns": ...}, "cmdline": ...}
        {"type": "connect",   "meta": {...}, "daddr_v4": [...], "dport": ..., "is_ipv6": ...}
        {"type": "connect",   "meta": {...}, "daddr": "1.2.3.4", "dport": ...}  (NAT captures)
        {"type": "connect",   "meta": {...}, "daddr": "2001:db8::1", "dport": ...}  (IPv6: kept
                                                  as `daddr_v6` text, with `is_ipv6` set)
        {"type": "file_open", "meta": {...}, "flags": ..., "path": ...}
    Flat format (every other input field is kept as is, `meta` included):
        {"type": "exec",     "pid": ..., "ts_ns": ..., "cmdline": ...}
        {"type": "connect",  "pid": ..., "ts_ns": ..., "daddr_v4": [...], "dport": ...}
        {"type": "fileopen", "pid": ..., "ts_ns": ..., "flags": ...}
    An event without `meta` is returned as is (the same object, not a copy).
    """
    meta = e.get("meta")
    if meta is None:
        return e
    e = dict(e)  # shallow copy — do not mutate the caller's dict
    e["pid"] = meta.get("pid", 0)
    e["ts_ns"] = meta.get("timestamp_ns", 0)
    if e.get("type") == "file_open":
        e["type"] = "fileopen"
    # NAT captures emit daddr as a dotted-decimal string; Host-Only captures use
    # daddr_v4 as a list of ints.  Normalise to daddr_v4 so behavior.py is uniform.
    # An IPv6 `daddr` (it contains a colon) goes to `daddr_v6`, kept as text for behavior.py.
    if "daddr" in e and "daddr_v4" not in e:
        if isinstance(e["daddr"], str) and ":" in e["daddr"]:
            e["is_ipv6"] = True
            e["daddr_v6"] = e["daddr"]
        else:
            try:
                e["daddr_v4"] = [int(b) for b in e["daddr"].split(".")]
            except (ValueError, AttributeError):
                e["daddr_v4"] = [0, 0, 0, 0]
    return e
