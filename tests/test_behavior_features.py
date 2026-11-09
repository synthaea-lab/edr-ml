"""Unit tests of the per-pid behavior vector, the mirror of the Rust `BehaviorVector`
(formerly a `__main__` self-test in the module)."""

from __future__ import annotations

import pytest

from synthaea_ml.features.behavior import (
    O_CREAT,
    O_WRONLY,
    extract_behavior_features,
    zero_behavior_features,
)

# The order of `BehaviorVector::to_vec()`, as documented by `extract_behavior_features`.
BEHAVIOR_FEATURE_NAMES = (
    "has_exec",
    "has_connect",
    "has_filewrite",
    "time_exec_to_connect_ms",
    "time_exec_to_filewrite_ms",
    "is_suspicious_path",
    "connect_count",
    "distinct_dports",
    "dest_is_external",
)


def named(vector: list[float]) -> dict[str, float]:
    """The vector keyed by feature name, so each assertion says what it checks."""
    return dict(zip(BEHAVIOR_FEATURE_NAMES, vector, strict=True))


def connect(ts_ns: int, daddr: list[int], dport: int) -> dict:
    return {"type": "connect", "ts_ns": ts_ns, "daddr_v4": daddr, "dport": dport, "is_ipv6": False}


def test_an_exec_alone_from_a_system_path_is_not_suspicious() -> None:
    events = [
        {"type": "exec", "ts_ns": 1_000_000_000, "cmdline": "C:\\Windows\\System32\\svchost.exe"}
    ]

    features = named(extract_behavior_features(events))

    assert features["has_exec"] == 1.0
    assert features["has_connect"] == 0.0
    assert features["is_suspicious_path"] == 0.0


def test_a_dropper_sets_every_behavior_feature() -> None:
    events = [
        {"type": "exec", "ts_ns": 0, "cmdline": "C:\\Users\\victim\\AppData\\Roaming\\payload.exe"},
        connect(500_000_000, [8, 8, 8, 8], 4444),
        connect(600_000_000, [8, 8, 8, 8], 443),
        {"type": "fileopen", "ts_ns": 1_000_000_000, "flags": O_WRONLY | O_CREAT},
    ]

    features = named(extract_behavior_features(events))

    assert features["has_exec"] == 1.0
    assert features["has_connect"] == 1.0
    assert features["has_filewrite"] == 1.0
    assert features["time_exec_to_connect_ms"] == pytest.approx(500.0, abs=1.0)
    assert features["time_exec_to_filewrite_ms"] == pytest.approx(1000.0, abs=1.0)
    assert features["is_suspicious_path"] == 1.0  # AppData
    assert features["connect_count"] == 2.0
    assert features["distinct_dports"] == 2.0  # 4444 and 443
    assert features["dest_is_external"] == 1.0  # 8.8.8.8


def test_a_connect_to_a_private_address_is_not_external() -> None:
    features = named(extract_behavior_features([connect(1_000_000_000, [192, 168, 1, 1], 445)]))

    assert features["dest_is_external"] == 0.0


def test_the_zero_vector_has_nine_zeros() -> None:
    assert zero_behavior_features() == [0.0] * 9
