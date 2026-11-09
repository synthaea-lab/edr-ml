"""Pin the LLR calibrator's discretizer and its Rust code generation.

The generated Rust is pasted by hand into the agent's `log_likelihood_ratio()`, so
its text is the contract: these snapshots must only change on purpose.
"""

from __future__ import annotations

import math

import pytest

from synthaea_ml.calibration.calibrate_llr import (
    FEATURE_NAMES,
    _discretize,
    compute_empirical_llrs,
    generate_report,
    generate_rust,
)

# One LLR per bin the discretizer can produce, for every feature.
FULL_LLR: dict[int, dict[str, float]] = {
    0: {"0": -0.5, "1": 1.234},
    1: {"0": 0.1, "1": -0.2},
    2: {"1": 0.7},
    3: {"absent": -0.3, "<500": 2.345, "<2000": 1.0, ">=2000": 0.4},
    4: {"absent": -0.1, "<500": 0.9, ">=500": 0.2},
    5: {"0": 0.0, "1": 2.0},
    6: {"0": -0.4, "1-4": 0.3, "5-14": 0.8, ">=15": 1.6},
    7: {"0-1": -0.2, "2-3": 0.5, ">=4": 1.1},
    8: {"0": -0.05, "1": 1.5},
}

FULL_RUST = """\
fn log_likelihood_ratio(idx: usize, value: f32) -> f32 {
    match idx {
        // 0 — has_exec
        0 => if value > 0.5 { 1.23 } else { -0.50 },
        // 1 — has_connect
        1 => if value > 0.5 { -0.20 } else { 0.10 },
        // 2 — has_filewrite
        2 => if value > 0.5 { 0.70 } else { 0.00 },
        // 3 — time_exec_to_connect_ms
        3 => {
            if value > 0.0 && value < 500.0 { 2.35 }
            else if value > 0.0 && value < 2000.0 { 1.00 }
            else { -0.30 }
        }
        // 4 — time_exec_to_filewrite_ms
        4 => if value > 0.0 && value < 500.0 { 0.90 } else { -0.10 },
        // 5 — is_suspicious_path
        5 => if value > 0.5 { 2.00 } else { 0.00 },
        // 6 — connect_count
        6 => {
            if value == 0.0 { -0.40 }
            else if value < 5.0 { 0.30 }
            else if value < 15.0 { 0.80 }
            else { 1.60 }
        }
        // 7 — distinct_dports
        7 => if value <= 1.0 { -0.20 } else if value <= 3.0 { 0.50 } else { 1.10 },
        // 8 — dest_is_external
        8 => if value > 0.5 { 1.50 } else { -0.05 },
        _ => 0.0,
    }
}"""

# Missing features fall back to 0.0; a feature whose only bins are "0"/"1" is emitted as
# a binary test whatever the feature (6 here); unknown bins on a binary feature (2 here)
# collapse to the "1" bin's LLR.
SPARSE_RUST = """\
fn log_likelihood_ratio(idx: usize, value: f32) -> f32 {
    match idx {
        // 0 — has_exec
        0 => 0.0,
        // 1 — has_connect
        1 => 0.0,
        // 2 — has_filewrite
        2 => 0.00,
        // 3 — time_exec_to_connect_ms
        3 => 0.0,
        // 4 — time_exec_to_filewrite_ms
        4 => 0.0,
        // 5 — is_suspicious_path
        5 => 0.0,
        // 6 — connect_count
        6 => if value > 0.5 { 0.00 } else { 0.25 },
        // 7 — distinct_dports
        7 => 0.0,
        // 8 — dest_is_external
        8 => 0.0,
        _ => 0.0,
    }
}"""


def test_generate_rust_covers_every_bin() -> None:
    assert generate_rust(FULL_LLR) == FULL_RUST


def test_generate_rust_on_sparse_llrs() -> None:
    assert generate_rust({6: {"0": 0.25}, 2: {"x": 0.3}}) == SPARSE_RUST


def test_generate_report_snapshot() -> None:
    report = generate_report({0: {"0": -0.5, "1": 1.0}, 3: {"absent": 0.1}}, 4, 2)
    head = "\n".join(
        [
            "=" * 70,
            "LLR CALIBRATION REPORT — Synthaea EDR",
            "Benign samples    : 4",
            "Malicious samples : 2",
            "=" * 70,
            "",
            "[0] has_exec",
            "    Current LLR : 0.0",
            "    Empirical LLRs per bin:",
            "        0            → -0.500",
            "        1            → +1.000",
            "",
            "[3] time_exec_to_connect_ms",
            "    Current LLR : 0.8/<500ms,0.3/<2s",
            "    Empirical LLRs per bin:",
            "        absent       → +0.100",
            "",
            "─" * 70,
            "SUGGESTED RUST CODE for log_likelihood_ratio():",
            "─" * 70,
            "",
        ]
    )
    assert report == head + generate_rust({0: {"0": -0.5, "1": 1.0}, 3: {"absent": 0.1}})


def test_report_lists_every_hand_calibrated_llr() -> None:
    llr = {i: {"0": 0.0} for i in range(9)}
    current = [
        line.removeprefix("    Current LLR : ")
        for line in generate_report(llr, 1, 1).splitlines()
        if line.startswith("    Current LLR : ")
    ]
    assert current == [
        "0.0",
        "0.3",
        "0.7",
        "0.8/<500ms,0.3/<2s",
        "0.6",
        "2.0",
        "0.1*n,max1.5",
        "0.5",
        "1.5",
    ]


@pytest.mark.parametrize("idx", [0, 1, 2, 5, 8])
@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.0, "0"), (0.5, "0"), (0.5000001, "1"), (1.0, "1"), (-1.0, "0"), (math.nan, "0")],
)
def test_discretize_binary(idx: int, value: float, expected: str) -> None:
    assert _discretize(idx, value) == expected


@pytest.mark.parametrize(
    ("idx", "value", "expected"),
    [
        # time_exec_to_connect_ms
        (3, 0.0, "absent"),
        (3, -1.0, "<500"),
        (3, 1.0, "<500"),
        (3, 499.0, "<500"),
        (3, 499.9, "<500"),
        (3, 500.0, "<2000"),
        (3, 1999.0, "<2000"),
        (3, 2000.0, ">=2000"),
        (3, 1e9, ">=2000"),
        (3, math.nan, ">=2000"),
        # time_exec_to_filewrite_ms
        (4, 0.0, "absent"),
        (4, 499.0, "<500"),
        (4, 500.0, ">=500"),
        (4, 2000.0, ">=500"),
        # connect_count
        (6, 0.0, "0"),
        (6, 1.0, "1-4"),
        (6, 4.0, "1-4"),
        (6, 4.5, "1-4"),
        (6, 5.0, "5-14"),
        (6, 14.0, "5-14"),
        (6, 15.0, ">=15"),
        # distinct_dports
        (7, 0.0, "0-1"),
        (7, 1.0, "0-1"),
        (7, 1.5, "2-3"),
        (7, 3.0, "2-3"),
        (7, 3.5, ">=4"),
        (7, 4.0, ">=4"),
        # out-of-range feature index
        (9, 2.5, "2.5"),
    ],
)
def test_discretize_continuous_boundaries(idx: int, value: float, expected: str) -> None:
    assert _discretize(idx, value) == expected


def test_feature_names() -> None:
    assert FEATURE_NAMES == [
        "has_exec",
        "has_connect",
        "has_filewrite",
        "time_exec_to_connect_ms",
        "time_exec_to_filewrite_ms",
        "is_suspicious_path",
        "connect_count",
        "distinct_dports",
        "dest_is_external",
    ]


def test_compute_empirical_llrs_uses_the_discretized_bins() -> None:
    benign = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    malicious = [1.0, 1.0, 1.0, 120.0, 900.0, 1.0, 20.0, 5.0, 1.0]
    llr = compute_empirical_llrs({1: [benign, benign], 2: [malicious]}, {2})
    assert set(llr) == set(range(9))
    assert set(llr[3]) == {"absent", "<500"}
    assert set(llr[4]) == {"absent", ">=500"}
    assert set(llr[6]) == {"0", ">=15"}
    assert set(llr[7]) == {"0-1", ">=4"}
    # Laplace alpha 0.5, two bins: p_m = 1.5/2, p_b = 0.5/3.
    assert llr[0]["1"] == pytest.approx(math.log((1.5 / 2) / (0.5 / 3)))
