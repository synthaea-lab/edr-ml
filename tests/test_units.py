"""ML-030: the nanosecond constants keep their values."""

from synthaea_ml.units import NS_PER_HOUR, NS_PER_SECOND


def test_ns_per_second_is_one_billion() -> None:
    assert NS_PER_SECOND == 1_000_000_000


def test_ns_per_hour_is_3600_seconds() -> None:
    assert NS_PER_HOUR == 3600 * NS_PER_SECOND
    assert NS_PER_HOUR == 3_600_000_000_000
