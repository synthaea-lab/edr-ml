"""Nanosecond time-unit constants (ML-030).

The agent timestamps events in nanoseconds (`crates/schema::EventMeta::timestamp_ns`), so
every window and interval here converts from seconds or hours. These constants replace the
`1_000_000_000` / `3_600_000_000_000` literals that were repeated across the training,
calibration and feature modules. Values are unchanged; this is a naming change only.
"""

from __future__ import annotations

#: Nanoseconds in one second.
NS_PER_SECOND = 1_000_000_000

#: Nanoseconds in one hour (`3600 * NS_PER_SECOND`).
NS_PER_HOUR = 3600 * NS_PER_SECOND
