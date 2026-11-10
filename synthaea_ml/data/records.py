"""The record shapes the data and feature modules pass around, as `TypedDict`s.

They only annotate: a record is still a plain `dict` at run time, so building, reading and
serializing one is unchanged and nothing here is validated. What they fix is the schema,
which used to live in docstrings, and what mypy can now check at each use.
"""

from __future__ import annotations

from typing import Required, TypedDict


class FlatEvent(TypedDict, total=False):
    """One event in the flat shape the feature extractors read: `pid`/`ts_ns` at top level
    and a `fileopen` tag. `data.wire.flatten_wire_event` builds it from a wire event.

    Only `type`, `pid` and `ts_ns` are always there. `daddr_v4` is a list of octets, or a
    one-element list holding the raw address string when `flatten_wire_event` built it.
    """

    type: Required[str]
    pid: Required[int]
    ts_ns: Required[int]
    process_generation: int
    argv: list[str]
    cmdline: str
    image_path: str
    parent_comm: str
    parent_image_path: str
    path: str
    daddr_v4: list[int | str]
    dport: int
    flags: int


class CorrelationVector(TypedDict):
    """One row of the correlation dataset: the feature vector of one pid's window."""

    pid: int
    features: list[float]
    feature_names: list[str]
