"""Fixtures shared by several test modules.

Only builders that need pytest state live here; plain builders are in ``helpers.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from helpers import RobustnessStub, SiteRarityRun, write_site_corpus


@pytest.fixture
def robustness_stub(monkeypatch: pytest.MonkeyPatch) -> RobustnessStub:
    """Replace `train_site_rarity`'s robustness run with canned outcomes.

    Call the fixture with ``(candidate, reference)``: the candidate's adapter (a
    ``WithheldRarity``, told apart by its type) gets ``candidate`` and any other model gets
    ``reference``. An outcome that is an exception is raised. Returns the calls made.
    """
    from synthaea_ml.training import train_site_rarity as rare

    def install(candidate: object, reference: object) -> list[dict]:
        calls: list[dict] = []

        def fake(
            model, scenario_yaml, tier="T0", mutation_seed=42, threshold=None, events_source=None
        ):
            calls.append(
                {"model": model, "threshold": threshold, "tier": tier, "seed": mutation_seed}
            )
            outcome = candidate if isinstance(model, rare.WithheldRarity) else reference
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        monkeypatch.setattr(rare, "run_robustness_evaluation", fake)
        return calls

    return install


@pytest.fixture
def site_rarity_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> SiteRarityRun:
    """Run `train_site_rarity.main()` on the synthetic site corpus; returns the output dir.

    Extra command-line arguments are appended after the fixed ones.
    """
    from synthaea_ml.training import train_site_rarity as rare

    def run(*extra: str) -> Path:
        out = tmp_path / "site-model"
        monkeypatch.setattr(
            "sys.argv",
            [
                rare.TRAINING_SCRIPT, "--site-corpus", str(write_site_corpus(tmp_path)),
                "--site-name", "acme", "--output-dir", str(out), "--min-hosts", "3", *extra,
            ],
        )
        rare.main()
        return out

    return run
