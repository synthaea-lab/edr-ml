"""Per-PID behavioral feature extraction — mirror of BehaviorVector (Rust).

These 9 features complement the 9 cmdline features from features.py to form an
18D vector intended for training the behavioral ML model (phase 2).

Input format: list of events for a given PID.
Each event is a dict with at minimum the "type" key:

    exec    : {"type": "exec",     "ts_ns": int, "cmdline": str,
                                   "image_path": str (optional, see below)}
    connect : {"type": "connect",  "ts_ns": int, "dport": int,
                                   "daddr_v4": [b0,b1,b2,b3]          (IPv4)
                                   "is_ipv6": true, "daddr_v6": str   (IPv6, text form)}
    fileopen: {"type": "fileopen", "ts_ns": int, "flags": int}

Fileopen flags (POSIX):
    O_ACCMODE = 0o3, O_WRONLY = 0o1, O_RDWR = 0o2, O_CREAT = 0o100, O_TRUNC = 0o1000

This extraction is in 1-to-1 correspondence with the
`CorrelationEngine::behavior_vector_for_pid` method of crates/synthaea-correlator/src/lib.rs.
Any change to a feature must be mirrored in both files.

Rust is the runtime reference (the Bayes filter runs on it), decided in
synthaea-lab/agent#230 and #203. Four rules follow from that:

1. An IPv6 destination is external unless it is loopback, unspecified, `fe80::/10`,
   `fc00::/7` or `ff00::/8`. An IPv4-mapped address (`::ffff:a.b.c.d`, what a dual-stack
   socket records for an IPv4 peer) is classified as its IPv4 address, like Rust's
   `IpAddr::to_canonical`: `::ffff:10.0.0.1` is internal, `::ffff:8.8.8.8` is external.
2. `0.0.0.0` and `::` are internal.
3. The suspicious-path match lowercases with ASCII rules only (Rust `to_ascii_lowercase`),
   never Unicode: U+212A KELVIN SIGN must not fold to `k`.
4. The path is the exec's `image_path` (what the kernel resolved), as in Rust. `argv[0]` of
   `cmdline` is whatever the caller passed (`exec -a`, a wrapper) and is trivially spoofed,
   and on Windows `cmdline` is one string with no stable `argv[0]`. It is only a fallback for
   an exec event without `image_path` (old wire records).
"""

from __future__ import annotations

import ipaddress
from typing import Any

from synthaea_ml.features.posix import has_write_intent

# ── Rust mirror constants ──────────────────────────────────────────────────────

# Suspicious directories — same values as is_suspicious_bv() and is_suspicious_win_path()
_SUSPICIOUS_DIRS = ["appdata", "\\temp\\", "downloads", "desktop", "\\public\\"]

# ASCII-only lowercasing table (A-Z only), the Rust `to_ascii_lowercase` rule.
_ASCII_LOWER = {c: c + 32 for c in range(ord("A"), ord("Z") + 1)}

# RFC-1918 private ranges + loopback + the unspecified address 0.0.0.0 (not `0.x.x.x`).
_PRIVATE_RANGES = [
    lambda a: list(a) == [0, 0, 0, 0],
    lambda a: a[0] == 10,
    lambda a: a[0] == 172 and 16 <= a[1] <= 31,
    lambda a: a[0] == 192 and a[1] == 168,
    lambda a: a[0] == 127,
]


# ── Helpers ────────────────────────────────────────────────────────────────────


def _is_suspicious(path: str) -> bool:
    """Returns True if the path contains a suspicious directory (ASCII-only lowercasing)."""
    lower = path.translate(_ASCII_LOWER)
    return any(s in lower for s in _SUSPICIOUS_DIRS)


def _is_private_ipv4(addr: list[int]) -> bool:
    """Returns True if the IPv4 address is private, loopback or unspecified."""
    return any(fn(addr) for fn in _PRIVATE_RANGES)


def _is_non_global_ipv6(addr: str) -> bool:
    """Mirror of `is_non_global_ipv6` (Rust): loopback, unspecified, link-local
    (`fe80::/10`), unique-local (`fc00::/7`) and multicast (`ff00::/8`).
    Raises `ValueError` on text that is not an IPv6 address."""
    value = int(ipaddress.IPv6Address(addr))
    seg0 = value >> 112
    return (
        value in (0, 1)
        or (seg0 & 0xFFC0) == 0xFE80
        or (seg0 & 0xFE00) == 0xFC00
        or (seg0 & 0xFF00) == 0xFF00
    )


def _is_external(e: dict[str, Any]) -> bool:
    """Whether a connect event's destination counts as external.

    An IPv6 event (`is_ipv6`, as in Rust where the address type decides) whose address is
    missing or unparseable is NOT external: the class cannot be judged, and an invented
    answer would be worse than none. An IPv4 event without `daddr_v4` is judged as
    `0.0.0.0`, so it is internal too, the same choice."""
    if e.get("is_ipv6", False):
        try:
            v6 = ipaddress.IPv6Address(e["daddr_v6"])
        except (KeyError, TypeError, ValueError):
            return False
        mapped = v6.ipv4_mapped
        if mapped is not None:
            return not _is_private_ipv4(list(mapped.packed))
        return not _is_non_global_ipv6(str(v6))
    return not _is_private_ipv4(e.get("daddr_v4", [0, 0, 0, 0]))


# ── Main extraction ────────────────────────────────────────────────────────────


def extract_behavior_features(events: list[dict[str, Any]]) -> list[float]:
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

    # Suspicious path — PID's first exec, judged on its `image_path` (as Rust does). Only an
    # exec event without `image_path` falls back to the first NUL-separated `cmdline`
    # argument; an `image_path` that is present but empty is the answer, not a gap.
    if exec_events:
        image = exec_events[0].get("image_path")
        if image is None:
            image = exec_events[0]["cmdline"].split("\0")[0]
        is_suspicious_path = 1.0 if _is_suspicious(image) else 0.0
    else:
        is_suspicious_path = 0.0

    # Network
    connect_count = float(len(connect_events))
    dports = {e["dport"] for e in connect_events}
    distinct_dports = float(len(dports))

    dest_is_external = 1.0 if any(_is_external(e) for e in connect_events) else 0.0

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
