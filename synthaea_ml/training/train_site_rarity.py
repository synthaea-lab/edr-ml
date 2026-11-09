"""Recalibrates a site model with the rarity feature and records its effect (issue #640).

Per-site adaptation (#49) rebuilds a T0 model from a site corpus. This entry point adds the
fleet-derived signal ADR-0020 describes: is the executed image on this fleet's common set
(seen on at least K hosts)? The model scores cmdline (9) + rarity (2) features, with the
rarity values taken **point in time** from the corpus itself (`synthaea_ml.data.common_set`),
never from today's counters.

It trains two models on the same time-ordered split, so the effect is measured, not assumed:

- the baseline on the 9 cmdline features;
- the candidate on cmdline + rarity;

each calibrated on the later calibration slice to the same FP budget, then scored on the
final, never-seen test slice. The test slice is benign site activity, so every flag there
is a false positive: `fp_rate_test_*` in the model record is the empirical FP rate of each
model at its own calibrated threshold. The split is by **time**, not random: a random split
would put an event's near-duplicates on both sides and flatter both models.

A model that fails the **floor** below is not written. The rarity features are withheld (0, 0)
until a snapshot covers enough hosts, which is the state of every freshly enrolled fleet, and a
model trained mostly on rows where rarity was known can score that state as anomalous without
the figure above showing it (it is taken at the natural mix of known and withheld rows). So the
run also measures, on the same test slice with the rarity forced to withheld:

- the false-positive rate, which may exceed the cmdline-only model's by at most
  `WITHHELD_FP_MARGIN`;
- with `--robustness-scenarios`, the escape rate and median degradation of the adversarial
  scenario suite, scoring the candidate with the rarity withheld against the cmdline-only model
  of the same corpus, each at its own calibrated threshold, with the tolerances of
  `train_site_model.py`'s floor check (`training.floor`, `config.FLOOR_TOLERANCE`).

With `--global-model-dir` (the shipped `model.onnx`, and `model_metadata.json` for its
threshold, scored through onnxruntime as the agent does) the same two checks are made against
the **global** model: the candidate with the rarity withheld must not flag the benign test slice
more than the global model does (plus `WITHHELD_FP_MARGIN`), nor be less robust on the scenario
suite, each model at its own threshold. A model that fails that one is not written either.
Without the option the record says the global comparison was NOT CHECKED. The global model is
scored on this corpus, so its false-positive rate is that of a model that never saw the site:
a figure of the site, not of the global model in general.

What this does not do: ship a model. The Rust extractor exists (`ml::features::rarity`, pinned to
`rarity.py` by a golden file) but nothing hands the agent a common-set snapshot and no scorer
loads an 11-feature model yet (ADR-0020). Detection quality with the rarity *known* is not
measured either: a scenario event has no image hash, so only the withheld state can be scored,
and only the false-positive side is measured on benign corpora.

Input: a site corpus JSONL (`{timestamp, event_type, event, agent_id}` per line, as
`/api/corpus/finalize` exports it). Usage:

    python -m synthaea_ml.training.train_site_rarity \\
        --site-corpus corpus-v1.jsonl --site-name acme \\
        --output-dir ml/registry/cmdline-rarity-iforest-linux-site-acme/0.1.0/
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from sklearn.ensemble import IsolationForest

from synthaea_ml.data.canonical import ml_cmdline_from_record
from synthaea_ml.data.common_set import (
    DEFAULT_INTERVAL_NS,
    DEFAULT_K,
    point_in_time_snapshots,
    timestamp_ns,
)
from synthaea_ml.evaluation.onnx_model import ShippedModel, load_shipped_model
from synthaea_ml.evaluation.robustness import run_robustness_evaluation
from synthaea_ml.features import cmdline, rarity
from synthaea_ml.hashing import sha256_file
from synthaea_ml.registry.codec import write_training_record
from synthaea_ml.registry.layout import METADATA_FILENAME, MODEL_FILENAME
from synthaea_ml.registry.records import ConformalCalibration, DatasetVersion
from synthaea_ml.training.config import (
    CALIBRATION,
    CMDLINE_FOREST,
    FLOOR_TOLERANCE,
    RANDOM_SEED,
)
from synthaea_ml.training.floor import CardComparison, Scorer
from synthaea_ml.training.pipeline import (
    CalibratedForest,
    export_onnx,
    fit_calibrated_forest,
    write_scorer_metadata,
)

TRAINING_SCRIPT = "synthaea_ml/training/train_site_rarity.py"
FEATURE_NAMES = [*cmdline.FEATURE_NAMES, *rarity.FEATURE_NAMES]

DEFAULT_MIN_HOSTS = 5
"""Hosts a snapshot must cover before the rarity evidence counts (ADR-0020)."""
MIN_ROWS = 60
"""Fewer exec events cannot support a three-way time split and a conformal threshold."""
TRAIN_FRACTION = 0.7
CALIBRATION_FRACTION = 0.15

WITHHELD_FP_MARGIN = 0.01
"""How far above the cmdline-only model's test false-positive rate the candidate's rate may be
with the rarity withheld (one percentage point). A starting point, to calibrate on a real corpus."""


def load_corpus(path: Path) -> list[dict[str, Any]]:
    """Every record of a site corpus; unreadable lines are skipped, not fatal."""
    if not path.exists():
        raise FileNotFoundError(f"site corpus not found: {path}")
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and "timestamp" in record:
            records.append(record)
    return records


def build_matrices(
    records: list[dict[str, Any]], k: int, min_hosts: int, interval_ns: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """Time-ordered `(X_cmdline, X_with_rarity, known_fraction)` over the exec records.

    The snapshots are computed over *all* records (a host that only ever ran non-exec events
    still counts as reporting), then the exec records are taken in time order.
    """
    snapshots = point_in_time_snapshots(records, k=k, interval_ns=interval_ns)
    rows = [
        (timestamp_ns(r), i)
        for i, r in enumerate(records)
        if r.get("event_type") == "exec" and isinstance(r.get("event"), dict)
    ]
    rows.sort()
    base: list[list[float]] = []
    extra: list[list[float]] = []
    for _, i in rows:
        event = records[i]["event"]
        base.append(cmdline.extract_features(ml_cmdline_from_record(event)))
        extra.append(rarity.extract_features(event.get("sha256"), snapshots[i], min_hosts))
    x_base = np.array(base, dtype=np.float32)
    x_rare = np.array(extra, dtype=np.float32)
    known = float(x_rare[:, 0].mean()) if len(x_rare) else 0.0
    return x_base, np.hstack([x_base, x_rare]), known


def time_split(n: int) -> tuple[slice, slice, slice]:
    """Train, calibration and test slices of `n` time-ordered rows."""
    a = int(n * TRAIN_FRACTION)
    b = int(n * (TRAIN_FRACTION + CALIBRATION_FRACTION))
    return slice(0, a), slice(a, b), slice(b, n)


def flag_rate(scores: np.ndarray, threshold: float) -> float:
    """The share of `scores` below `threshold`, so flagged; 0 for no scores."""
    return float((scores < threshold).mean()) if len(scores) else 0.0


@dataclass(frozen=True)
class MeasuredForest:
    """A calibrated forest and the false-positive rate it shows on the benign test slice."""

    forest: CalibratedForest
    fp_rate_test: float


def fit_and_measure(x: np.ndarray, names: list[str]) -> MeasuredForest:
    """Trains on the early slice, calibrates on the middle one, and measures on the last."""
    train, calibration, test = time_split(len(x))
    forest = fit_calibrated_forest(x[train], x[calibration], names, CMDLINE_FOREST)
    scores = forest.model.decision_function(x[test])
    return MeasuredForest(forest, flag_rate(scores, forest.calibration.threshold))


class WithheldRarity:
    """The 11-feature candidate seen as a 9-feature T0 model: the cmdline features go in, the
    two rarity columns are appended as withheld (0, 0), as on an agent with no usable snapshot.
    The robustness evaluation scores a command line with `decision_function` on the cmdline
    features alone, and a scenario event carries no image hash to look up."""

    def __init__(self, model: IsolationForest) -> None:
        """Wrap the 11-feature `model`."""
        self._model = model
        self.n_features_in_ = len(cmdline.FEATURE_NAMES)

    def decision_function(self, x: ArrayLike) -> np.ndarray:
        """The candidate's score of each 9-feature row of `x`, with the rarity withheld."""
        rows = np.asarray(x, dtype=np.float32)
        withheld = np.zeros((len(rows), len(rarity.FEATURE_NAMES)), dtype=np.float32)
        return self._model.decision_function(np.hstack([rows, withheld]))


def withhold_rows(x: np.ndarray, fraction: float, seed: int, rows: slice) -> np.ndarray:
    """A copy of `x` where a random `fraction` of the rows in `rows` has its two rarity columns
    set to withheld (0, 0), as if no usable snapshot had existed for them.

    In a real corpus the withheld rows are only those before the first snapshot boundary, a few
    percent, so the forest learns (0, 0) as a rare combination and then flags the withheld
    state of every new fleet (what `check_rarity_floor` measures). Withholding a share of the
    training rows teaches it that state. Only the rows in `rows` change: the slices that are
    calibrated and tested stay as the corpus gave them."""
    if not 0.0 <= fraction < 1.0:
        raise ValueError(f"withhold fraction must be in [0, 1), got {fraction}")
    out = x.copy()
    if fraction == 0.0:
        return out
    start, stop, _ = rows.indices(len(x))
    chosen = np.random.default_rng(seed).random(stop - start) < fraction
    out[start:stop][chosen, -len(rarity.FEATURE_NAMES):] = 0.0
    return out


def withheld_fp_rate(
    model: IsolationForest, calibration: ConformalCalibration, x_base_test: np.ndarray
) -> float:
    """The share of benign test rows flagged at the model's calibrated threshold with the rarity
    forced to withheld: the false-positive rate of a fleet that has no snapshot yet."""
    if not len(x_base_test):
        return 0.0
    scores = WithheldRarity(model).decision_function(x_base_test)
    return flag_rate(scores, calibration.threshold)


def compare_on_scenarios(
    candidate: IsolationForest,
    candidate_threshold: float,
    reference: Scorer,
    reference_threshold: float,
    reference_name: str,
    scenario_yamls: Sequence[Path],
) -> tuple[list[str], list[str]]:
    """The candidate, with the rarity withheld, against `reference` on each scenario, each model at
    its own threshold. Returns `(report lines, failures)`; a scenario `reference` detects nothing
    of is skipped, one the candidate detects nothing of that `reference` detects is a failure."""
    lines: list[str] = []
    failures: list[str] = []
    withheld_model = WithheldRarity(candidate)
    for scenario in scenario_yamls:
        name = Path(scenario).stem
        try:
            ref_card = run_robustness_evaluation(
                reference, scenario, tier="T0", mutation_seed=RANDOM_SEED, threshold=reference_threshold
            )
        except ValueError as error:
            # Nothing the reference detects: there is nothing for the candidate to lose here.
            lines.append(f"{name}: skipped, the {reference_name} detects none of it ({error})")
            continue
        try:
            cand_card = run_robustness_evaluation(
                withheld_model, scenario, tier="T0", mutation_seed=RANDOM_SEED, threshold=candidate_threshold
            )
        except ValueError as error:
            failures.append(
                f"{name}: the candidate with the rarity withheld detects none of what the "
                f"{reference_name} detects ({error})"
            )
            continue
        lines.append(
            f"{name}: escape rate {cand_card.escape_rate:.2%} vs {ref_card.escape_rate:.2%} "
            f"({reference_name}), median degradation {cand_card.median_score_degradation:.3f} vs "
            f"{ref_card.median_score_degradation:.3f}"
        )
        comparison = CardComparison(candidate=cand_card, reference=ref_card)
        if comparison.escape_rate_regressed:
            failures.append(
                f"{name}: escape rate {cand_card.escape_rate:.2%} is above the {reference_name}'s "
                f"{ref_card.escape_rate:.2%} + {FLOOR_TOLERANCE.escape_rate:.0%}"
            )
        if comparison.median_degradation_regressed:
            failures.append(
                f"{name}: median degradation {cand_card.median_score_degradation:.3f} is above "
                f"the {reference_name}'s {ref_card.median_score_degradation:.3f} + "
                f"{FLOOR_TOLERANCE.median_degradation:.2f}"
            )
    return lines, failures


@dataclass(frozen=True)
class FloorResult:
    """The outcome of one floor check: the candidate with the rarity withheld against a reference.

    `threshold` and `threshold_source` are the reference's, recorded when the reference is a
    shipped model whose threshold is read from its metadata (the global model)."""

    reference_name: str
    fp_reference: float
    fp_withheld: float
    scenario_lines: list[str]
    failures: list[str]
    threshold: float | None = None
    threshold_source: str | None = None

    @property
    def passed(self) -> bool:
        """Whether the candidate held the floor: no failure was found."""
        return not self.failures


@dataclass(frozen=True)
class _Reference:
    """What a floor compares the candidate with, and what failing the false-positive check means."""

    model: Scorer
    threshold: float
    name: str
    consequence: str


def _check_floor(
    candidate: IsolationForest,
    candidate_cal: ConformalCalibration,
    reference: _Reference,
    x_base_test: np.ndarray,
    scenario_yamls: Sequence[Path],
) -> FloorResult:
    """The two checks of a floor: the withheld-state false-positive rate against the reference's,
    and the scenario suite when `scenario_yamls` are given. Collects every failure rather than
    stopping at the first, so a run reports all of them."""
    failures: list[str] = []
    fp_reference = (
        flag_rate(reference.model.decision_function(x_base_test), reference.threshold)
        if len(x_base_test) else 0.0
    )
    fp_withheld = withheld_fp_rate(candidate, candidate_cal, x_base_test)
    if fp_withheld > fp_reference + WITHHELD_FP_MARGIN:
        failures.append(
            f"with the rarity withheld the candidate flags {fp_withheld:.2%} of the benign test "
            f"slice, above the {reference.name}'s {fp_reference:.2%} + {WITHHELD_FP_MARGIN:.2%}: "
            + reference.consequence
        )
    lines, scenario_failures = compare_on_scenarios(
        candidate, candidate_cal.threshold, reference.model, reference.threshold,
        reference.name, scenario_yamls,
    )
    return FloorResult(reference.name, fp_reference, fp_withheld, lines, failures + scenario_failures)


def check_rarity_floor(
    candidate: IsolationForest,
    candidate_cal: ConformalCalibration,
    baseline: IsolationForest,
    baseline_cal: ConformalCalibration,
    x_base_test: np.ndarray,
    scenario_yamls: Sequence[Path] = (),
) -> FloorResult:
    """The floor of the module docstring, against the same corpus's cmdline-only model."""
    reference = _Reference(
        baseline, baseline_cal.threshold, "cmdline-only model",
        "a fleet with no snapshot yet would see that false-positive rate",
    )
    return _check_floor(candidate, candidate_cal, reference, x_base_test, scenario_yamls)


def check_global_floor(
    candidate: IsolationForest,
    candidate_cal: ConformalCalibration,
    global_model: ShippedModel,
    x_base_test: np.ndarray,
    scenario_yamls: Sequence[Path] = (),
) -> FloorResult:
    """The same two checks as `check_rarity_floor`, against the global model: the candidate with
    the rarity withheld must not flag this corpus's benign test slice more than the global model
    does (plus `WITHHELD_FP_MARGIN`), nor be less robust on the scenario suite, each model at its
    own threshold."""
    reference = _Reference(
        global_model, global_model.threshold, "global model",
        "a fleet with no snapshot yet would see more false positives than with the global model",
    )
    floor = _check_floor(candidate, candidate_cal, reference, x_base_test, scenario_yamls)
    return replace(
        floor, threshold=global_model.threshold, threshold_source=global_model.threshold_source
    )


def _describe_floor(floor: FloorResult, with_scenarios: bool) -> str:
    """One line for the training record: what was checked, and what was not."""
    if not with_scenarios:
        return (
            f"PASSED (withheld false-positive rate only: {floor.fp_withheld:.2%} against "
            f"{floor.fp_reference:.2%} + {WITHHELD_FP_MARGIN:.2%}; no scenarios given, "
            "so the escape rate was not checked)"
        )
    return (
        f"PASSED (withheld false-positive rate {floor.fp_withheld:.2%} against "
        f"{floor.fp_reference:.2%} + {WITHHELD_FP_MARGIN:.2%}; "
        + "; ".join(floor.scenario_lines)
        + ")"
    )


GLOBAL_FLOOR_NOT_CHECKED = (
    "NOT CHECKED against the global model: the floor above compares the "
    "candidate with this corpus's cmdline-only model; pass --global-model-dir to "
    "compare with the global model before this model leaves the lab"
)


def _describe_global_floor(floor: FloorResult | None, with_scenarios: bool) -> str:
    """One line for the training record: what the global comparison covered, if it was made."""
    if floor is None:
        return GLOBAL_FLOOR_NOT_CHECKED
    text = (
        f"PASSED against the global model at its threshold {floor.threshold:g} "
        f"({floor.threshold_source}): withheld false-positive rate {floor.fp_withheld:.2%} "
        f"against {floor.fp_reference:.2%} + {WITHHELD_FP_MARGIN:.2%}"
    )
    if with_scenarios:
        return text + "; " + "; ".join(floor.scenario_lines)
    return text + "; no scenarios given, so the escape rate was not checked"


# --- the command line ----------------------------------------------------------------------


@dataclass(frozen=True)
class SiteRarityArgs:
    """The parsed command line of a run."""

    site_corpus: Path
    site_name: str
    output_dir: Path
    k: int
    min_hosts: int
    interval_hours: float
    withhold_fraction: float
    robustness_scenarios: tuple[Path, ...]
    global_model_dir: Path | None


def parse_args(argv: Sequence[str] | None = None) -> SiteRarityArgs:
    """The run's options from `argv` (the process's arguments when `None`)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-corpus", type=Path, required=True)
    parser.add_argument("--site-name", required=True, help="Tenant name, for provenance.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Hosts for the common set.")
    parser.add_argument("--min-hosts", type=int, default=DEFAULT_MIN_HOSTS)
    parser.add_argument(
        "--interval-hours",
        type=float,
        default=DEFAULT_INTERVAL_NS / 3_600_000_000_000,
        help="Cadence of published snapshots.",
    )
    parser.add_argument(
        "--withhold-fraction",
        type=float,
        default=0.0,
        help="Share of the training rows whose rarity is set to withheld, so the model learns "
        "the state of a fleet with no snapshot yet. 0 (the default) trains on the corpus as it is, "
        "which the floor check rejects when almost all rows have a known rarity.",
    )
    parser.add_argument(
        "--robustness-scenarios",
        type=Path,
        nargs="+",
        default=[],
        help="Scenario YAMLs for the floor check: the candidate with the rarity withheld must "
        "not be less robust than the cmdline-only model of the same corpus.",
    )
    parser.add_argument(
        "--global-model-dir",
        type=Path,
        default=None,
        help="Directory of the global model (`model.onnx`, and `model_metadata.json` for its "
        "threshold): the candidate with the rarity withheld must not be worse than it on the "
        "benign test slice or the scenarios. Without it the global comparison is not made.",
    )
    args = parser.parse_args(argv)
    return SiteRarityArgs(
        site_corpus=args.site_corpus,
        site_name=args.site_name,
        output_dir=args.output_dir,
        k=args.k,
        min_hosts=args.min_hosts,
        interval_hours=args.interval_hours,
        withhold_fraction=args.withhold_fraction,
        robustness_scenarios=tuple(args.robustness_scenarios),
        global_model_dir=args.global_model_dir,
    )


@dataclass(frozen=True)
class SiteMatrices:
    """The time-ordered feature matrices of a corpus's exec events."""

    record_count: int
    """Every record of the corpus, exec or not: the dataset's sample count."""
    cmdline: np.ndarray
    with_rarity: np.ndarray
    known_fraction: float

    @property
    def exec_events(self) -> int:
        """How many exec events, so rows, the matrices have."""
        return len(self.with_rarity)


def record_extras(
    args: SiteRarityArgs,
    matrices: SiteMatrices,
    *,
    baseline: MeasuredForest,
    candidate: MeasuredForest,
    floor: FloorResult,
    global_floor: FloorResult | None,
) -> dict[str, str]:
    """The `extra` of the training record: the run's settings, its measurements and what the
    floors covered. The keys and their string formats are part of the record; keep them."""
    with_scenarios = bool(args.robustness_scenarios)
    extras = {
        "site": args.site_name,
        "rarity_k": str(args.k),
        "rarity_min_hosts": str(args.min_hosts),
        "snapshot_interval_hours": str(args.interval_hours),
        "exec_events": str(matrices.exec_events),
        "rarity_known_fraction": f"{matrices.known_fraction:.4f}",
        "withhold_fraction": f"{args.withhold_fraction:.4f}",
        "fp_rate_test_cmdline_only": f"{baseline.fp_rate_test:.4f}",
        "fp_rate_test_with_rarity": f"{candidate.fp_rate_test:.4f}",
        "threshold_cmdline_only": f"{baseline.forest.calibration.threshold:.6f}",
        "fp_rate_test_with_rarity_withheld": f"{floor.fp_withheld:.4f}",
        "rarity_floor": _describe_floor(floor, with_scenarios),
        "global_model_floor": _describe_global_floor(global_floor, with_scenarios),
    }
    if global_floor is not None:
        extras["fp_rate_test_global_model"] = f"{global_floor.fp_reference:.4f}"
    return extras


def _require_floor(floor: FloorResult, summary: str, label: str) -> None:
    """Print what a floor measured, and stop the run before anything is written if it failed."""
    print(summary)
    for line in floor.scenario_lines:
        print(f"  scenario {line}")
    if not floor.passed:
        raise ValueError(
            f"the {label} failed, no model written:\n  - " + "\n  - ".join(floor.failures)
        )


def _load_matrices(args: SiteRarityArgs) -> SiteMatrices:
    """The corpus's time-ordered matrices; refuses a corpus too small to split."""
    records = load_corpus(args.site_corpus)
    interval_ns = int(args.interval_hours * 3_600_000_000_000)
    x_base, x_rare, known = build_matrices(records, args.k, args.min_hosts, interval_ns)
    if len(x_rare) < MIN_ROWS:
        raise ValueError(
            f"only {len(x_rare)} exec events in the corpus (need at least {MIN_ROWS})"
        )
    print(f"{len(x_rare)} exec events; rarity known for {known:.0%} of them")
    return SiteMatrices(len(records), x_base, x_rare, known)


def _train_both(
    x_base: np.ndarray, x_rare: np.ndarray, withhold_fraction: float
) -> tuple[MeasuredForest, MeasuredForest]:
    """The cmdline-only baseline and the rarity candidate, on the same time split."""
    baseline = fit_and_measure(x_base, list(cmdline.FEATURE_NAMES))
    train_rows, _, _ = time_split(len(x_rare))
    x_fit = withhold_rows(x_rare, withhold_fraction, CMDLINE_FOREST.random_state, train_rows)
    candidate = fit_and_measure(x_fit, FEATURE_NAMES)
    print(
        f"test-slice false-positive rate: cmdline only {baseline.fp_rate_test:.2%}, "
        f"with rarity {candidate.fp_rate_test:.2%} "
        f"(budget {CALIBRATION.fp_budget_per_endpoint_day:g}/endpoint/day)"
    )
    return baseline, candidate


def main(argv: Sequence[str] | None = None) -> None:
    """Train, check the floors and, if they hold, write the site model with its record."""
    args = parse_args(argv)
    global_model = load_shipped_model(args.global_model_dir) if args.global_model_dir else None

    matrices = _load_matrices(args)
    baseline, candidate = _train_both(
        matrices.cmdline, matrices.with_rarity, args.withhold_fraction
    )
    clf, cal = candidate.forest.model, candidate.forest.calibration
    base = baseline.forest
    _, _, test_rows = time_split(matrices.exec_events)
    x_test = matrices.cmdline[test_rows]

    floor = check_rarity_floor(
        clf, cal, base.model, base.calibration, x_test, args.robustness_scenarios
    )
    _require_floor(
        floor, f"with the rarity withheld: false-positive rate {floor.fp_withheld:.2%}",
        "rarity floor",
    )
    global_floor = None
    if global_model is not None:
        global_floor = check_global_floor(clf, cal, global_model, x_test, args.robustness_scenarios)
        _require_floor(
            global_floor,
            f"against the global model: false-positive rate {global_floor.fp_withheld:.2%} "
            f"with the rarity withheld, {global_floor.fp_reference:.2%} for the global model",
            "global model floor",
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    export_onnx(clf, matrices.with_rarity, args.output_dir / MODEL_FILENAME)
    write_scorer_metadata(args.output_dir / METADATA_FILENAME, cal, candidate.forest.bounds)
    write_training_record(
        args.output_dir,
        training_script=TRAINING_SCRIPT,
        dataset_versions=[
            DatasetVersion(
                name=f"site-corpus-{args.site_name}",
                baseline_sha256=sha256_file(args.site_corpus),
                sample_count=matrices.record_count,
            )
        ],
        hyperparameters=CMDLINE_FOREST.hyperparameters(),
        conformal_calibration=cal,
        feature_bounds=candidate.forest.bounds,
        extra=record_extras(
            args, matrices,
            baseline=baseline, candidate=candidate, floor=floor, global_floor=global_floor,
        ),
    )
    print(f"Site model with rarity -> {args.output_dir}")


if __name__ == "__main__":
    main()
