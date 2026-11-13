"""Nanosecond conversions: event timestamps (`ts_ns`) and windows are in nanoseconds."""

from __future__ import annotations

NS_PER_MILLISECOND = 1_000_000
NS_PER_SECOND = 1_000 * NS_PER_MILLISECOND
NS_PER_HOUR = 3_600 * NS_PER_SECOND
