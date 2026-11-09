"""The training configuration every trainer shares: forest hyperparameters per model family,
the conformal calibration contract, the floor-check tolerances and the random seed.

One place, so a value cannot drift between the trainers that are meant to agree, and so the
evaluation tools that re-fit a model the way its trainer does read the same numbers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

RANDOM_SEED = 42
"""The seed of every random choice a training run makes (the forest, the calibration split,
the adversarial mutations), so a run is reproducible from its record."""


@dataclass(frozen=True)
class IsolationForestConfig:
    """The `IsolationForest` arguments a model family is trained with."""

    n_estimators: int
    contamination: float
    random_state: int = RANDOM_SEED

    def hyperparameters(self) -> dict[str, object]:
        """The keyword arguments of `IsolationForest`, as the model record stores them."""
        return asdict(self)


@dataclass(frozen=True)
class CalibrationConfig:
    """The conformal calibration contract (issue #46)."""

    fp_budget_per_endpoint_day: float
    """Maximum acceptable false positives per endpoint per day."""
    benign_rate_per_day: float
    """Estimated benign events (processes, for the correlation scorer) scored per endpoint per
    day, to turn the FP budget into a rate.

    A conservative estimate: production endpoints may see more or fewer events depending on
    workload. The threshold can be recalibrated when larger baselines become available (#44).
    """
    feature_bounds_margin: float
    """Relative margin of the feature bounds (expansion beyond the observed min/max)."""


@dataclass(frozen=True)
class FloorTolerance:
    """How much a candidate model may regress against its reference on the adversarial scenario
    suite before a floor check fails it."""

    escape_rate: float
    median_degradation: float


# contamination=0.05, a deliberate decision after testing 0.02 (see the git history of
# `train_linux.py`). At 0.02, the curl healthcheck false positive disappeared but the ML signal
# on a short, real malicious case (the isolated `base64 -d` exec event, captured in the lab on
# 2026-08-24: cmdline = "base64\0-d\0") collapsed to nearly zero (score -0.001, versus -0.035 at
# 0.05) - with only 71 baseline examples, the model's boundary is too fuzzy to eliminate this
# false positive without losing the signal on the true positive. Explicit choice: keep a sharp ML
# signal on the malicious case rather than a silent ML on a benign one, especially since this
# scenario is intercepted anyway by the deterministic T1059.004 rule regardless of the ML score
# (see threat-model.md - ML is never the final judge). The curl healthcheck false positive at
# 0.05 thus remains a known and accepted limitation (already documented before these features
# were added, see docs/PLAN.md). The Windows model uses the same value for the same reason: with
# a baseline example count of the same order of magnitude, 0.02 would drown the ML signal on
# short true positives with no real gain on false positives.
CMDLINE_FOREST = IsolationForestConfig(n_estimators=100, contamination=0.05)
"""T0: the command-line models (Linux, Windows, and the site models built on them)."""

# Provisional: the T0 value, to be settled on the first real capture (a process's behavior
# counts are tightly clustered, so a lower contamination may separate better, as
# `train_correlation.py` anticipates). Do not tune on the synthetic test corpus.
BEHAVIOR_FOREST = IsolationForestConfig(n_estimators=100, contamination=0.05)
"""T1: the behavior model (cmdline + correlation features)."""

BEHAVIOR_MIN_SAMPLES = 20
"""Fewer distinct incarnations than this cannot support a 70/30 split and a conformal
threshold; the behavior trainer (and the tools that re-fit it) refuse rather than emit a model
whose boundary is noise."""

CORRELATION_FOREST = IsolationForestConfig(n_estimators=100, contamination=0.02)
"""The correlation-vector scorer; provisional, see `train_correlation.py`."""

CALIBRATION = CalibrationConfig(
    fp_budget_per_endpoint_day=5.0,
    benign_rate_per_day=1000.0,
    feature_bounds_margin=0.05,
)

FLOOR_TOLERANCE = FloorTolerance(escape_rate=0.05, median_degradation=0.10)
"""Shared by the site model's global-model floor and the rarity floors, so they read alike."""
