"""Test-side handling of sibling repositories (see ``synthaea_ml.siblings``).

A test that needs an absent sibling is skipped, unless ``SYNTHAEA_REQUIRE_SIBLINGS=1``
(set by CI), where collection fails instead: a parity test must never go quietly
missing where it is supposed to run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from synthaea_ml import siblings


def needs(repo: str, *parts: str) -> pytest.MarkDecorator:
    """A ``skipif`` mark for a test that reads ``parts`` from sibling ``repo``."""
    target = siblings.path(repo, *parts)
    missing = not target.exists()
    reason = (
        f"needs sibling repo `{repo}` ({target}); "
        f"check it out via the umbrella or set {siblings.ROOT_ENV}"
    )
    if missing and siblings.required():
        raise RuntimeError(f"{siblings.REQUIRE_ENV}=1 but this test {reason}")
    return pytest.mark.skipif(missing, reason=reason)


def sibling(repo: str, *parts: str) -> Path:
    """Shorthand for ``synthaea_ml.siblings.path``."""
    return siblings.path(repo, *parts)
