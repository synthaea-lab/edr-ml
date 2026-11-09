"""Unit tests of the per-pid correlation vector, mirroring the Rust unit tests of
`correlation_features.rs` (formerly a `__main__` self-test in the module)."""

from __future__ import annotations

from synthaea_ml.features.correlation import FEATURE_NAMES, extract_features
from synthaea_ml.features.posix import O_CREAT, O_WRONLY


def named(vector: list[float]) -> dict[str, float]:
    """The vector keyed by feature name, so each assertion says what it checks."""
    return dict(zip(FEATURE_NAMES, vector, strict=True))


def connect(pid: int, ts_ns: int, daddr: list[int], dport: int) -> dict:
    return {"type": "connect", "pid": pid, "ts_ns": ts_ns, "daddr_v4": daddr, "dport": dport}


def network_flow(pid: int, ts_ns: int, daddr: list[int], dport: int) -> dict:
    return {"type": "network_flow", "pid": pid, "ts_ns": ts_ns, "daddr_v4": daddr, "dport": dport}


def test_an_absent_pid_yields_the_zero_vector() -> None:
    assert extract_features([], 1234) == [0.0] * 8
    assert extract_features([{"type": "exec", "pid": 1, "ts_ns": 0}], 1234) == [0.0] * 8


def test_exec_connect_and_write_make_a_full_chain() -> None:
    events = [
        {"type": "exec", "pid": 99, "ts_ns": 0},
        connect(99, 1_000_000_000, [127, 0, 0, 1], 4444),
        {"type": "fileopen", "pid": 99, "ts_ns": 2_000_000_000, "flags": O_WRONLY | O_CREAT},
    ]

    features = named(extract_features(events, 99))

    assert features["spawn_count"] == 1.0
    assert features["connect_count"] == 1.0
    assert features["filewrite_count"] == 1.0
    assert features["has_full_chain"] == 1.0
    assert features["span_s"] == 2.0
    assert features["event_count"] == 3.0


def test_a_spawn_alone_is_not_a_full_chain() -> None:
    features = named(extract_features([{"type": "exec", "pid": 1, "ts_ns": 0}], 1))

    assert features["has_full_chain"] == 0.0


def test_a_read_only_fileopen_is_not_a_filewrite() -> None:
    read_only = [{"type": "fileopen", "pid": 5, "ts_ns": 0, "flags": 0}]  # O_RDONLY

    assert named(extract_features(read_only, 5))["filewrite_count"] == 0.0


def test_repeated_destinations_are_counted_once_each() -> None:
    events = [
        connect(7, 0, [127, 0, 0, 1], 4444),
        connect(7, 1_000_000_000, [127, 0, 0, 1], 4444),
        connect(7, 2_000_000_000, [127, 0, 0, 1], 8080),
    ]

    features = named(extract_features(events, 7))

    assert features["connect_count"] == 3.0
    assert features["unique_daddr_count"] == 1.0
    assert features["unique_dport_count"] == 2.0


def test_network_flow_feeds_the_same_destination_sets_as_connect() -> None:
    # ADR-0008: netlink-observed flows count as destinations; connect_count stays
    # connect-only.
    events = [
        connect(7, 0, [127, 0, 0, 1], 4444),
        network_flow(7, 1_000_000_000, [10, 0, 0, 1], 4444),
        network_flow(7, 2_000_000_000, [10, 0, 0, 2], 9999),
    ]

    features = named(extract_features(events, 7))

    assert features["unique_daddr_count"] == 3.0
    assert features["unique_dport_count"] == 2.0
    assert features["event_count"] == 3.0


def test_events_of_other_pids_are_ignored() -> None:
    events = [
        {"type": "exec", "pid": 1, "ts_ns": 0},
        connect(2, 1_000_000_000, [1, 1, 1, 1], 4444),
    ]

    assert named(extract_features(events, 1))["connect_count"] == 0.0
    assert named(extract_features(events, 2))["spawn_count"] == 0.0
