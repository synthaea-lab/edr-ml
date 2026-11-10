"""Characterization of the two wire-event translators (`synthaea_ml.data.wire`).

`normalize_for_behavior` (the LLR calibration's view) had no test. These cases pin its
current output on NAT and Host-Only captures, and pin the points where it differs from
`flatten_wire_event` (the correlation parity view) — the reason both exist.
"""

from __future__ import annotations

import pytest

from synthaea_ml.data.wire import flatten_wire_event, normalize_for_behavior
from synthaea_ml.features.behavior import extract_behavior_features

META = {"pid": 7, "timestamp_ns": 100}
EXEC = {"type": "exec", "meta": META, "cmdline": "/usr/bin/curl\0http://x"}
NAT_CONNECT = {"type": "connect", "meta": META, "daddr": "10.0.0.5", "dport": 443}
HOST_ONLY_CONNECT = {
    "type": "connect",
    "meta": META,
    "daddr_v4": [8, 8, 8, 8],
    "dport": 53,
    "is_ipv6": False,
}


def test_nat_connect_splits_daddr_into_octets_and_keeps_the_wire_fields() -> None:
    assert normalize_for_behavior(NAT_CONNECT) == {
        "type": "connect",
        "meta": META,
        "daddr": "10.0.0.5",
        "dport": 443,
        "pid": 7,
        "ts_ns": 100,
        "daddr_v4": [10, 0, 0, 5],
    }


def test_host_only_connect_keeps_its_octet_list() -> None:
    assert normalize_for_behavior(HOST_ONLY_CONNECT) == {
        **HOST_ONLY_CONNECT,
        "pid": 7,
        "ts_ns": 100,
    }


def test_an_existing_daddr_v4_wins_over_daddr() -> None:
    raw = {**HOST_ONLY_CONNECT, "daddr": "10.0.0.5"}
    assert normalize_for_behavior(raw)["daddr_v4"] == [8, 8, 8, 8]


@pytest.mark.parametrize("daddr", ["not-an-ip", None])
def test_an_unparsable_daddr_becomes_the_zero_address(daddr: object) -> None:
    raw = {"type": "connect", "meta": META, "daddr": daddr, "dport": 443}
    assert normalize_for_behavior(raw)["daddr_v4"] == [0, 0, 0, 0]


def test_an_ipv6_daddr_is_kept_as_text_for_behavior_py() -> None:
    raw = {"type": "connect", "meta": META, "daddr": "2606:4700:4700::1111", "dport": 443}
    out = normalize_for_behavior(raw)
    assert out["is_ipv6"] is True
    assert out["daddr_v6"] == "2606:4700:4700::1111"
    assert "daddr_v4" not in out


def test_file_open_is_retagged_and_keeps_flags_and_path() -> None:
    raw = {"type": "file_open", "meta": META, "flags": 0o101, "path": "/tmp/x"}
    assert normalize_for_behavior(raw) == {
        **raw,
        "type": "fileopen",
        "pid": 7,
        "ts_ns": 100,
    }


def test_missing_pid_and_timestamp_default_to_zero() -> None:
    out = normalize_for_behavior({"type": "exec", "meta": {}, "cmdline": "sh"})
    assert (out["pid"], out["ts_ns"]) == (0, 0)


def test_unknown_types_are_kept() -> None:
    raw = {"type": "process_exit", "meta": META}
    assert normalize_for_behavior(raw) == {**raw, "pid": 7, "ts_ns": 100}


def test_an_event_without_meta_is_returned_unchanged() -> None:
    flat = {"type": "exec", "pid": 1, "ts_ns": 2, "cmdline": "sh"}
    assert normalize_for_behavior(flat) is flat


def test_the_input_is_not_mutated() -> None:
    raw = {"type": "connect", "meta": dict(META), "daddr": "10.0.0.5", "dport": 443}
    before = {**raw, "meta": dict(raw["meta"])}
    normalize_for_behavior(raw)
    assert raw == before


# ── Why there are two translators ─────────────────────────────────────────────


def test_a_nat_private_destination_is_internal_only_with_octets() -> None:
    """`flatten_wire_event` keeps `daddr` verbatim, so the behavior extractor would
    read a private NAT address as external (`dest_is_external` is the last feature)."""
    behavior = extract_behavior_features([normalize_for_behavior(e) for e in (EXEC, NAT_CONNECT)])
    flattened = [flatten_wire_event(e) for e in (EXEC, NAT_CONNECT)]
    correlation_view = extract_behavior_features([e for e in flattened if e is not None])
    assert behavior[-1] == 0.0
    assert correlation_view[-1] == 1.0


def test_an_ipv6_destination_is_classified_only_when_is_ipv6_is_kept() -> None:
    """`flatten_wire_event` drops `is_ipv6`, so a link-local IPv6 destination would be read
    as an external IPv4 one; `normalize_for_behavior` classifies it (internal)."""
    v6 = {"type": "connect", "meta": META, "daddr": "fe80::1", "dport": 443, "is_ipv6": True}
    behavior = extract_behavior_features([normalize_for_behavior(e) for e in (EXEC, v6)])
    flattened = [flatten_wire_event(e) for e in (EXEC, v6)]
    correlation_view = extract_behavior_features([e for e in flattened if e is not None])
    assert behavior[-1] == 0.0
    assert correlation_view[-1] == 1.0


def test_flatten_wire_event_cannot_read_a_host_only_connect() -> None:
    with pytest.raises(KeyError, match="daddr"):
        flatten_wire_event(HOST_ONLY_CONNECT)


def test_flatten_wire_event_requires_the_pid_and_drops_what_behavior_keeps() -> None:
    with pytest.raises(KeyError, match="pid"):
        flatten_wire_event({"type": "exec", "meta": {}, "cmdline": "sh"})
    assert flatten_wire_event({"type": "process_exit", "meta": META}) is None
    assert flatten_wire_event({"type": "exec", "pid": 1, "ts_ns": 2}) is None


def test_the_old_import_path_still_resolves_to_the_same_flattener() -> None:
    from synthaea_ml.data import aggregate_correlation, wire

    assert aggregate_correlation.flatten_wire_event is wire.flatten_wire_event
    assert aggregate_correlation.RAW_EVENT_TYPES is wire.RAW_EVENT_TYPES
