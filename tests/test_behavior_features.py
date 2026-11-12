"""Unit tests of the per-pid behavior vector, the mirror of the Rust `BehaviorVector`
(formerly a `__main__` self-test in the module)."""

from __future__ import annotations

import pytest

from synthaea_ml.features.behavior import extract_behavior_features, zero_behavior_features
from synthaea_ml.features.posix import O_CREAT, O_WRONLY

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


# ── Rules aligned on the Rust reference (agent#203, ml#51) ─────────────────────


def connect_v6(addr: str | None) -> dict:
    event = {"type": "connect", "ts_ns": 1_000_000_000, "dport": 443, "is_ipv6": True}
    if addr is not None:
        event["daddr_v6"] = addr
    return event


@pytest.mark.parametrize("addr", ["2606:4700:4700::1111", "2001:4860:4860::8888", "::ffff:10.0.0.1"])
def test_a_global_ipv6_destination_is_external(addr: str) -> None:
    # `::ffff:10.0.0.1` (IPv4-mapped) is judged as IPv6, like Rust's `is_non_global_ipv6`.
    assert named(extract_behavior_features([connect_v6(addr)]))["dest_is_external"] == 1.0


@pytest.mark.parametrize("addr", ["::1", "::", "fe80::1", "febf::1", "fc00::1", "fd12:3456::1", "ff02::1"])
def test_a_non_global_ipv6_destination_is_not_external(addr: str) -> None:
    assert named(extract_behavior_features([connect_v6(addr)]))["dest_is_external"] == 0.0


@pytest.mark.parametrize("addr", ["fec0::1", "2000::", "fbff::1"])
def test_ipv6_class_boundaries_follow_the_rust_masks(addr: str) -> None:
    # `fec0::/10` (deprecated site-local) is outside `fe80::/10`, `fbff::` outside `fc00::/7`.
    assert named(extract_behavior_features([connect_v6(addr)]))["dest_is_external"] == 1.0


@pytest.mark.parametrize("addr", [None, "not-an-ip"])
def test_an_ipv6_event_without_a_usable_address_is_not_external(addr: str | None) -> None:
    assert named(extract_behavior_features([connect_v6(addr)]))["dest_is_external"] == 0.0


def test_the_unspecified_ipv4_address_is_not_external() -> None:
    assert named(extract_behavior_features([connect(1, [0, 0, 0, 0], 22)]))["dest_is_external"] == 0.0


def test_only_the_exact_unspecified_ipv4_address_is_internal() -> None:
    assert named(extract_behavior_features([connect(1, [0, 1, 2, 3], 22)]))["dest_is_external"] == 1.0


def exec_with(image_path: str | None, cmdline: str) -> dict:
    event = {"type": "exec", "ts_ns": 0, "cmdline": cmdline}
    if image_path is not None:
        event["image_path"] = image_path
    return event


def test_suspicious_path_lowercasing_is_ascii_only() -> None:
    # U+212A KELVIN SIGN: `str.lower()` folds it to `k`, so `DESKTOP` would match `desktop`.
    kelvin = exec_with("C:\\Users\\x\\DES\u212aTOP\\a.exe", "a.exe")
    ascii_ = exec_with("C:\\Users\\x\\DESKTOP\\a.exe", "a.exe")
    assert named(extract_behavior_features([kelvin]))["is_suspicious_path"] == 0.0
    assert named(extract_behavior_features([ascii_]))["is_suspicious_path"] == 1.0


def test_the_path_is_the_image_path_not_argv0() -> None:
    spoofed = exec_with("/home/bob/Downloads/tool", "/usr/bin/ls\0-la\0")
    clean = exec_with("/usr/bin/ls", "/home/bob/Downloads/tool\0")
    assert named(extract_behavior_features([spoofed]))["is_suspicious_path"] == 1.0
    assert named(extract_behavior_features([clean]))["is_suspicious_path"] == 0.0


def test_argv0_is_only_the_fallback_for_an_exec_without_image_path() -> None:
    assert named(extract_behavior_features([exec_with(None, "/home/bob/Downloads/tool\0")]))[
        "is_suspicious_path"
    ] == 1.0


def test_an_empty_image_path_is_an_answer_not_a_gap() -> None:
    # Present but empty: judged as empty (not suspicious), no fallback to `cmdline`.
    event = exec_with("", "/home/bob/Downloads/tool\0")
    assert named(extract_behavior_features([event]))["is_suspicious_path"] == 0.0
