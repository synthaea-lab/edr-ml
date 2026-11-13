"""Extraction of the "multi-events per pid" feature vector, for a second ML scorer — a
complement to the cmdline vector (`cmdline.py`), never merged with it into a single model
(decision of 2026-08-27, see docs/design/design-correlation-ml-vector.md).

**Must stay in sync with the Rust mirror** (crates/synthaea-ml/src/correlation_features.rs):
exact same definitions, same output order.

Expected input format: a list of JSON-Lines events, one object per event, as produced by
`edr-cli capture-events` / `edr-cli run` in `events.jsonl`:
    {"type": "exec",     "pid": 1234, "ts_ns": 1000000000, "comm": "curl", "cmdline": "..."}
    {"type": "connect",  "pid": 1234, "ts_ns": 2000000000, "daddr_v4": [127,0,0,1],
                         "dport": 4444, "is_ipv6": false}
    {"type": "fileopen", "pid": 1234, "ts_ns": 3000000000, "path": "...", "flags": 65}
    {"type": "network_flow", "pid": 1234, "ts_ns": 4000000000, "daddr_v4": [10,0,0,1],
                         "dport": 9999}

`network_flow` (netlink-observed, ADR-0008) is absorbed into the same `unique_daddr_count`/
`unique_dport_count` sets as `connect` — see `extract_features` below. `connect_count` stays
`connect`-only.

`flags` is logged raw; the filtering on write intent (access mode `O_WRONLY`/`O_RDWR`,
or `O_CREAT`/`O_TRUNC`) happens here, exactly mirroring `schema::has_write_intent` on the Rust side
(`features/behavior.py` applies the same test).
"""

from synthaea_ml.data.records import FlatEvent
from synthaea_ml.features.posix import has_write_intent
from synthaea_ml.units import NS_PER_SECOND

FEATURE_NAMES = [
    "spawn_count",
    "connect_count",
    "filewrite_count",
    "unique_daddr_count",
    "unique_dport_count",
    "has_full_chain",
    "span_s",
    "event_count",
]


def _is_file_write(event: FlatEvent) -> bool:
    """Mirror of `schema::has_write_intent` (Rust, the ONE definition): a `fileopen`
    whose access mode is `O_WRONLY`/`O_RDWR`, or that carries `O_CREAT` or `O_TRUNC`. The access
    mode is a 2-bit *field* (`O_ACCMODE`), not independent bits — the invalid `0o3`
    combination is refused by the kernel (`EINVAL`), so it does not count as a write
    (this replaced a drifted bitmask test that disagreed with the Rust rules engine)."""
    if event.get("type") != "fileopen":
        return False
    flags = int(event.get("flags", 0))
    return has_write_intent(flags)


def same_generation(recorded: int | None, wanted: int | None) -> bool:
    """Whether an event stamped `recorded` can belong to the process incarnation `wanted`.

    Mirror of `correlator::bus::same_generation`: two known, different stamps are two
    incarnations of a recycled pid and never mix; a missing stamp on either side cannot
    disprove identity, which keeps stamp-less captures behaving exactly as before.
    """
    return recorded is None or wanted is None or recorded == wanted


def events_for_pid(
    events: list[FlatEvent], pid: int, generation: int | None = None
) -> list[FlatEvent]:
    """The events of one process incarnation: Python mirror of `EventBus::events_for_pid`.

    Records carry the stamp under `process_generation` (the `EventMeta` field name).
    """
    return [
        e
        for e in events
        if e["pid"] == pid and same_generation(e.get("process_generation"), generation)
    ]


def extract_features(
    events: list[FlatEvent], pid: int, generation: int | None = None
) -> list[float]:
    """`events`: all events of a correlation window (not only those of the requested pid —
    the pid filtering happens here, like `EventBus::events_for_pid` on the Rust side).

    `generation` is the process incarnation (#590): a recycled pid's earlier life must not
    feed this one's features. `None` keeps the pid-only behavior."""
    pid_events = events_for_pid(events, pid, generation)

    spawn_count = sum(1 for e in pid_events if e["type"] == "exec")
    connect_count = sum(1 for e in pid_events if e["type"] == "connect")
    filewrite_count = sum(1 for e in pid_events if _is_file_write(e))

    # ADR-0008: Connect and NetworkFlow (netlink) are absorbed into the same
    # daddr/dport sets — one count of "distinct destinations touched" regardless of
    # which sensor captured them. connect_count above stays Connect-only.
    daddrs = {
        tuple(e["daddr_v4"]) for e in pid_events if e["type"] in ("connect", "network_flow")
    }
    dports = {e["dport"] for e in pid_events if e["type"] in ("connect", "network_flow")}

    has_full_chain = (
        1.0 if (spawn_count >= 1 and connect_count >= 1 and filewrite_count >= 1) else 0.0
    )

    if pid_events:
        span_s = (
            max(e["ts_ns"] for e in pid_events) - min(e["ts_ns"] for e in pid_events)
        ) / NS_PER_SECOND
    else:
        span_s = 0.0

    return [
        float(spawn_count),
        float(connect_count),
        float(filewrite_count),
        float(len(daddrs)),
        float(len(dports)),
        has_full_chain,
        span_s,
        float(len(pid_events)),
    ]
