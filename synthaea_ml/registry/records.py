"""The types of ``model_record.json``: the record and every block it binds.

Frozen dataclasses only, plus the scenario passing criterion that the writer caches in
``ScenarioReplayResult.passed`` and the release gate re-computes. Encoding lives in
``synthaea_ml.registry.codec`` and verification in ``synthaea_ml.registry.provenance``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

SCHEMA_VERSION = 3
"""Bumped only when a field changes in a way that breaks readers.

Bumped 1 → 2 in ADR-0009 for the addition of ``scenario_replays`` on
``TrainingRecord`` and the rename of the on-disk file to
``model_record.json``.

Bumped 2 → 3 for issues #45 and #46: added ``robustness_cards`` (adversarial
evaluation results binding mutations to model versions for evasion cost
measurement), ``conformal_calibration``, and ``feature_bounds`` (FP-budget
thresholds and OOD guards). Backward compatible (optional fields with None
defaults).
"""


# ── Dataset binding (unchanged since schema v1) ─────────────────────────────


@dataclass(frozen=True)
class DatasetVersion:
    """One entry in ``TrainingRecord.dataset_versions``.

    ``baseline_sha256`` is the manifest's ``sample_sha256`` (locking the sample
    file content), and ``sample_count`` is the manifest's ``sample_count``
    (a cheap cross-check that catches a truncated baseline before the slower
    hash comparison does).
    """

    name: str
    baseline_sha256: str
    sample_count: int


# ── Conformal calibration binding (issue #46, schema v3) ────────────────────


@dataclass(frozen=True)
class ConformalCalibration:
    """Conformal prediction calibration metadata (issue #46).

    Records the threshold computed from a calibration set to meet a stated FP
    budget (e.g., ≤5 false positives per endpoint per day). Produced by
    ``synthaea_ml.calibration.calibrate_threshold``; this is the single
    definition, re-exported from ``synthaea_ml.calibration``.
    """

    fp_budget_per_endpoint_day: float
    """Maximum acceptable false positives per endpoint per day (e.g., 5.0)."""

    threshold: float
    """decision_function threshold: anomaly if score < threshold."""

    calibration_set_size: int
    """Number of samples in the calibration set (30% of training data)."""

    benign_baseline_rate: float
    """Estimated benign events per endpoint per day (for FP rate conversion)."""

    calibrated_at: str
    """ISO 8601 UTC timestamp when calibration was performed."""


@dataclass(frozen=True)
class FeatureBounds:
    """Per-feature [min, max] bounds for out-of-distribution detection (issue #46).

    Computed from the training set (70%) with a margin to avoid false OOD
    rejections on legitimate edge cases. The Rust scorer validates feature
    vectors against these bounds before inference. Produced by
    ``synthaea_ml.calibration.compute_feature_bounds``; this is the single
    definition, re-exported from ``synthaea_ml.calibration``.
    """

    feature_names: list[str]
    """Feature names in order (must match model input)."""

    min_values: list[float]
    """Minimum allowed value per feature (inclusive)."""

    max_values: list[float]
    """Maximum allowed value per feature (inclusive)."""


# ── Scenario replay binding (ADR-0009, schema v2) ───────────────────────────


@dataclass(frozen=True)
class Environment:
    """OS/kernel/arch on which a replay ran. Minimum for v1;
    libc (glibc/musl) and toolchain versions are v2 candidates
    (ADR-0009 § Deferred).

    ``kernel`` is ``platform.release()`` verbatim — cross-platform via the
    Python stdlib helper. On Linux/macOS the value equals ``uname -r``
    (e.g. ``"6.6.0-generic"``, ``"23.6.0"``); on Windows it is the major
    version only (e.g. ``"10"``), NOT the full Windows build. A full-build
    Windows field would need its own dataclass member and is deferred until
    a case requires the extra precision.
    """

    os: str
    kernel: str
    arch: str


@dataclass(frozen=True)
class ExpectedDetection:
    """One expected ``(technique, rule)`` tuple in a scenario. Mirror of the
    yaml sidecar schema (#44's decision 2 / ADR-0009 § Schema).

    ``technique`` is an opaque string as ``Alert.technique`` and
    ``CorrelationAlert.rule`` produce it — it may be a single ATT&CK id
    (``"T1071"``) or a combined form (``"T1105/T1059/T1071"``). The passing
    criterion matches by exact string equality, not by ATT&CK decomposition
    (see ``compute_passed``).
    """

    technique: str
    rule: str
    min_count: int
    tolerance: int


@dataclass(frozen=True)
class ObservedDetection:
    """What the replay actually observed for one ``(technique, rule)`` tuple.

    Emitted by the replay engine that reads a run's ``alerts.ndjson`` and
    tallies ``(technique, rule)`` counts. Zero counts are represented by the
    tuple being absent from the list rather than by ``count == 0``, matching
    how the alerts pipeline reports.
    """

    technique: str
    rule: str
    count: int


@dataclass(frozen=True)
class ScenarioReplayResult:
    """One replay run against one scenario, in one environment.

    Binds to the source ``scenarios/<scenario_name>.yaml`` by content
    hash — mutating the yaml invalidates every record that references it, in
    the same way ``DatasetVersion.baseline_sha256`` binds datasets.

    ``passed`` is cached (writer computes it via ``compute_passed`` and stores
    it; ``verify_scenario_replay`` re-computes and rejects if the cached value
    disagrees, catching a mutated ``passed`` field).
    """

    scenario_name: str
    scenario_yaml_sha256: str
    run_at: str  # ISO 8601 UTC, e.g. "2026-09-17T10:15:00Z"
    environment: Environment
    expected_detections: list[ExpectedDetection]
    observed_detections: list[ObservedDetection]
    passed: bool


# ── Robustness binding (issue #45, schema v3) ───────────────────────────────


@dataclass(frozen=True)
class MutationTestResult:
    """One mutation test outcome (original vs mutated score).

    Emitted by the robustness evaluation runner for each combination of
    (mutator class, intensity, sample). Tracks whether the mutation caused
    evasion (score drop below threshold).
    """

    mutation_class: str
    intensity: str
    original_score: float
    mutated_score: float
    score_delta: float
    threshold: float
    escaped: bool
    seed: int


@dataclass(frozen=True)
class RobustnessCard:
    """Adversarial evaluation results for one scenario against one model.

    Binds to the source ``scenarios/<scenario_name>.yaml`` by content
    hash, parallel to ``ScenarioReplayResult``. Captures aggregate metrics
    (escape rate, degradation percentiles) alongside per-mutation details.

    Generated by ``synthaea_ml.evaluation.robustness.run_robustness_evaluation``.
    """

    scenario_name: str
    scenario_yaml_sha256: str
    tested_at: str  # ISO 8601 UTC, e.g. "2026-09-23T10:15:00Z"
    mutation_results: list[MutationTestResult]
    escape_rate: float
    median_score_degradation: float
    worst_case_degradation: float


# ── Record ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TrainingRecord:
    """The ``model_record.json`` sidecar of a registry model version.

    Kept named ``TrainingRecord`` (rather than ``ModelRecord``) for continuity
    with existing callers; the on-disk filename and human semantics both moved
    to a model-lifecycle framing in ADR-0009 but the Python type keeps its
    historic name to minimise churn across the codebase.
    """

    schema_version: int
    trained_at: str
    training_script: str
    dataset_versions: list[DatasetVersion]
    scenario_replays: list[ScenarioReplayResult] = field(default_factory=list)
    robustness_cards: list[RobustnessCard] = field(default_factory=list)
    hyperparameters: dict[str, object] = field(default_factory=dict)
    extra: dict[str, str] = field(default_factory=dict)
    conformal_calibration: ConformalCalibration | None = None
    feature_bounds: FeatureBounds | None = None


def with_scenario_replay(record: TrainingRecord, replay: ScenarioReplayResult) -> TrainingRecord:
    """``record`` with ``replay`` appended, in place of an earlier replay of the same scenario
    in the same environment: a rerun supersedes the run it repeats, while runs of one
    scenario on different machines each stay recorded."""
    others = [
        earlier
        for earlier in record.scenario_replays
        if (earlier.scenario_name, earlier.environment)
        != (replay.scenario_name, replay.environment)
    ]
    return replace(record, scenario_replays=[*others, replay])


# ── Passing criterion (ADR-0009 § Passing criterion) ────────────────────────


def compute_passed(
    expected: list[ExpectedDetection],
    observed: list[ObservedDetection],
) -> bool:
    """Compute a scenario replay's ``passed`` value from its expected and
    observed detections.

    Per ADR-0009: for each ``ExpectedDetection`` in the scenario, find the
    matching observation by exact ``(technique, rule)`` string equality.

    * Missing observation → fail.
    * ``observed_count < min_count`` → fail (under-detection).
    * ``observed_count > min_count + tolerance`` → fail (over-detection —
      noisy rule).
    * Otherwise → pass for that expected detection.

    ``passed`` is ``True`` iff every expected detection passes. AND across
    all expected — one failure is enough to fail the scenario.
    """
    obs_by_key: dict[tuple[str, str], int] = {(o.technique, o.rule): o.count for o in observed}
    for e in expected:
        key = (e.technique, e.rule)
        if key not in obs_by_key:
            return False
        count = obs_by_key[key]
        if count < e.min_count:
            return False
        if count > e.min_count + e.tolerance:
            return False
    return True
