"""Per-PID behavioral feature extraction — mirror of BehaviorVector (Rust).

These 9 features complement the 9 cmdline features from features.py to form an
18D vector intended for training the behavioral ML model (phase 2).

Input format: list of events for a given PID.
Each event is a dict with at minimum the "type" key:

    exec    : {"type": "exec",     "ts_ns": int, "cmdline": str}
    connect : {"type": "connect",  "ts_ns": int, "daddr_v4": [b0,b1,b2,b3],
                                   "dport": int, "is_ipv6": bool}
    fileopen: {"type": "fileopen", "ts_ns": int, "flags": int}

Fileopen flags (POSIX):
    O_ACCMODE = 0o3, O_WRONLY = 0o1, O_RDWR = 0o2, O_CREAT = 0o100

This extraction is in 1-to-1 correspondence with the
`CorrelationEngine::behavior_vector_for_pid` method of crates/synthaea-correlator/src/lib.rs.
Any change to a feature must be mirrored in both files.
"""

from __future__ import annotations

from synthaea_ml.features.posix import has_write_intent

# ── Rust mirror constants ──────────────────────────────────────────────────────

# Suspicious directories — same values as is_suspicious_bv() and is_suspicious_win_path()
_SUSPICIOUS_DIRS = ["appdata", "\\temp\\", "downloads", "desktop", "\\public\\"]

# RFC-1918 private ranges + loopback
_PRIVATE_RANGES = [
    lambda a: a[0] == 10,
    lambda a: a[0] == 172 and 16 <= a[1] <= 31,
    lambda a: a[0] == 192 and a[1] == 168,
    lambda a: a[0] == 127,
]


# ── Helpers ────────────────────────────────────────────────────────────────────


def _is_suspicious(path: str) -> bool:
    """Returns True if the path contains a suspicious directory."""
    lower = path.lower()
    return any(s in lower for s in _SUSPICIOUS_DIRS)


def _is_private_ipv4(addr: list[int]) -> bool:
    """Returns True if the IPv4 address is private or loopback (RFC-1918)."""
    return any(fn(addr) for fn in _PRIVATE_RANGES)


# ── Main extraction ────────────────────────────────────────────────────────────


def extract_behavior_features(events: list[dict]) -> list[float]:
    """Extracts the 9 behavioral features from a PID's event sequence within
    its sliding window.

    The feature order is identical to BehaviorVector::to_vec():
        [has_exec, has_connect, has_filewrite,
         time_exec_to_connect_ms, time_exec_to_filewrite_ms,
         is_suspicious_path,
         connect_count, distinct_dports, dest_is_external]

    Returns a vector of 9 zeros if the list is empty (no events).
    """
    if not events:
        return [0.0] * 9

    exec_events = [e for e in events if e["type"] == "exec"]
    connect_events = [e for e in events if e["type"] == "connect"]
    write_events = [e for e in events if e["type"] == "fileopen" and has_write_intent(e.get("flags", 0))]

    has_exec = 1.0 if exec_events else 0.0
    has_connect = 1.0 if connect_events else 0.0
    has_filewrite = 1.0 if write_events else 0.0

    # Time deltas (ms)
    first_exec_ns = exec_events[0]["ts_ns"] if exec_events else None
    first_connect_ns = connect_events[0]["ts_ns"] if connect_events else None
    first_filewrite_ns = write_events[0]["ts_ns"] if write_events else None

    if first_exec_ns is not None and first_connect_ns is not None:
        time_exec_to_connect_ms = max(0.0, (first_connect_ns - first_exec_ns) / 1_000_000)
    else:
        time_exec_to_connect_ms = 0.0

    if first_exec_ns is not None and first_filewrite_ns is not None:
        time_exec_to_filewrite_ms = max(0.0, (first_filewrite_ns - first_exec_ns) / 1_000_000)
    else:
        time_exec_to_filewrite_ms = 0.0

    # Suspicious path — PID's first exec
    if exec_events:
        image = exec_events[0]["cmdline"].split("\0")[0]
        is_suspicious_path = 1.0 if _is_suspicious(image) else 0.0
    else:
        is_suspicious_path = 0.0

    # Network
    connect_count = float(len(connect_events))
    dports = {e["dport"] for e in connect_events}
    distinct_dports = float(len(dports))

    dest_is_external = 0.0
    for e in connect_events:
        if not e.get("is_ipv6", False):
            addr = e.get("daddr_v4", [0, 0, 0, 0])
            if not _is_private_ipv4(addr):
                dest_is_external = 1.0
                break

    return [
        has_exec,
        has_connect,
        has_filewrite,
        time_exec_to_connect_ms,
        time_exec_to_filewrite_ms,
        is_suspicious_path,
        connect_count,
        distinct_dports,
        dest_is_external,
    ]


def zero_behavior_features() -> list[float]:
    """Zero behavioral vector — used for baseline entries that only have
    cmdline data (no associated connect/fileopen events)."""
    return [0.0] * 9
