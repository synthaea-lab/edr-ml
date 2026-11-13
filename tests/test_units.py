"""The nanosecond conversions keep the values the modules spelled out before (#41)."""

from __future__ import annotations

from synthaea_ml.units import NS_PER_HOUR, NS_PER_MILLISECOND, NS_PER_SECOND


def test_a_millisecond_is_a_million_nanoseconds() -> None:
    assert NS_PER_MILLISECOND == 1_000_000


def test_a_second_is_a_billion_nanoseconds() -> None:
    assert NS_PER_SECOND == 1_000_000_000


def test_an_hour_is_3600_seconds_of_nanoseconds() -> None:
    assert NS_PER_HOUR == 3_600_000_000_000
