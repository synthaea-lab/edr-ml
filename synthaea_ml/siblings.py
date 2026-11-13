"""Sibling checkouts of the public Synthaea repositories (synthaea-lab/agent, server, ml,
rules and scenarios).

The ML pipeline is developed on its own, but it reads files owned by siblings: the
scenario sidecars of synthaea-lab/scenarios (ADR-0009 replays bind to them by content
hash) and the agent's Rust parity fixtures. Clone the repositories side by side, so
siblings sit next to this one::

    <root>/{agent,server,ml,rules,scenarios}

``SYNTHAEA_ROOT`` points at that parent directory (default: this repo's parent).
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT_ENV = "SYNTHAEA_ROOT"
REQUIRE_ENV = "SYNTHAEA_REQUIRE_SIBLINGS"


def root() -> Path:
    """The parent directory holding the sibling checkouts."""
    override = os.environ.get(ROOT_ENV)
    if override:
        return Path(override).resolve()
    # synthaea_ml/siblings.py -> synthaea_ml -> <ml repo> -> <root>
    return Path(__file__).resolve().parents[2]


def path(repo: str, *parts: str) -> Path:
    """The path of ``parts`` inside sibling ``repo``, whether or not it exists."""
    return root().joinpath(repo, *parts)


def scenario_sidecars() -> Path:
    """The directory of the scenario sidecars (``<name>.yaml``) that replays and robustness
    cards bind to, inside the ``scenarios`` sibling."""
    return path("scenarios", "scenarios")


def required() -> bool:
    """Whether a missing sibling must fail rather than skip (set by CI)."""
    return os.environ.get(REQUIRE_ENV) == "1"
