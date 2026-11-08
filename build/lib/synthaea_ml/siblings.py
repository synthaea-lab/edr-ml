"""Sibling repositories of the Synthaea umbrella (``synthaea-lab/edr``).

The ML pipeline is developed on its own, but it reads files owned by siblings: lab
scenarios (ADR-0009 replays bind to them by content hash) and the agent's Rust parity
fixtures. Checked out through the umbrella, siblings sit next to this repo::

    <umbrella>/{agent,server,ml,rules,lab}

``SYNTHAEA_ROOT`` overrides the umbrella location.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT_ENV = "SYNTHAEA_ROOT"
REQUIRE_ENV = "SYNTHAEA_REQUIRE_SIBLINGS"


def root() -> Path:
    """The umbrella directory holding the sibling checkouts."""
    override = os.environ.get(ROOT_ENV)
    if override:
        return Path(override).resolve()
    # synthaea_ml/siblings.py -> synthaea_ml -> <ml repo> -> <umbrella>
    return Path(__file__).resolve().parents[2]


def path(repo: str, *parts: str) -> Path:
    """The path of ``parts`` inside sibling ``repo``, whether or not it exists."""
    return root().joinpath(repo, *parts)


def required() -> bool:
    """Whether a missing sibling must fail rather than skip (set by CI)."""
    return os.environ.get(REQUIRE_ENV) == "1"
