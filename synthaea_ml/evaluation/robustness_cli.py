"""Robustness CLI commands for CI gate integration (issue #45).

Commands:
- verify: Check robustness metrics against thresholds (CI gate)
- report: Generate human-readable robustness report

Usage:
    python -m synthaea_ml.evaluation.robustness_cli verify \\
        --model-dir ml/registry/cmdline-iforest-linux/0.3.0/ \\
        --max-escape-rate 0.15

    python -m synthaea_ml.evaluation.robustness_cli report \\
        --model-dir ml/registry/cmdline-iforest-linux/0.3.0/ \\
        --output robustness.md
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from synthaea_ml.registry.codec import load_training_record
from synthaea_ml.registry.records import RobustnessCard, TrainingRecord

SemanticVersion = tuple[int, int, int]


def _parse_semantic_version(name: str) -> SemanticVersion | None:
    """``(major, minor, patch)`` of a ``"0.3.0"``-style directory name, or None."""
    parts = name.split(".")
    if len(parts) != 3:
        return None
    try:
        major, minor, patch = (int(part) for part in parts)
    except ValueError:
        return None
    return major, minor, patch


def _previous_version_candidates(model_dir: Path, version: SemanticVersion) -> Iterator[Path]:
    """Directories that may hold the previous version, most preferred first.

    The previous patch, then every ``major.(minor-1).*`` entry, then every
    ``(major-1).*`` entry, each glob in the order the filesystem yields it.
    """
    major, minor, patch = version
    registry_dir = model_dir.parent
    if patch > 0:
        yield registry_dir / f"{major}.{minor}.{patch - 1}"
    if minor > 0:
        yield from registry_dir.glob(f"{major}.{minor - 1}.*")
    if major > 0:
        yield from registry_dir.glob(f"{major - 1}.*")


def previous_version(model_dir: Path) -> Path | None:
    """The registry version directory to check ``model_dir`` against for regressions.

    None when ``model_dir`` is not named ``major.minor.patch`` or no candidate exists.
    """
    version = _parse_semantic_version(model_dir.name)
    if version is None:
        return None
    for candidate in _previous_version_candidates(model_dir, version):
        if candidate.is_dir() and candidate != model_dir:
            return candidate
    return None


@dataclass(frozen=True)
class RobustnessThresholds:
    """Limits the ``verify`` gate holds a model's robustness cards to."""

    max_escape_rate: float = 0.15
    max_median_degradation: float = 0.20
    max_regression_increase: float = 0.05
    """Largest allowed escape-rate increase over the previous version (absolute)."""


class Gate(StrEnum):
    """A robustness check a scenario can fail."""

    ESCAPE_RATE = "escape_rate"
    MEDIAN_DEGRADATION = "median_degradation"
    REGRESSION = "regression"


@dataclass(frozen=True)
class GateFailure:
    """One scenario over one limit: ``observed`` exceeded ``limit``."""

    scenario_name: str
    gate: Gate
    observed: float
    limit: float


@dataclass(frozen=True)
class ScenarioRegression:
    """A scenario's escape rate against the previous version's, when it had one."""

    scenario_name: str
    previous_escape_rate: float | None
    escape_rate: float
    failure: GateFailure | None


@dataclass(frozen=True)
class RegressionCheck:
    """The comparison against the previous version, or why it could not run."""

    previous_version: str | None = None
    scenarios: tuple[ScenarioRegression, ...] = ()
    warning: str | None = None


@dataclass(frozen=True)
class ScenarioVerdict:
    """A robustness card and the thresholds it failed."""

    card: RobustnessCard
    failures: tuple[GateFailure, ...]


@dataclass(frozen=True)
class RobustnessVerification:
    """Outcome of ``verify_robustness``; ``render_verification`` turns it into text."""

    model_dir: Path
    scenarios: tuple[ScenarioVerdict, ...]
    regression: RegressionCheck | None
    """None when the regression check was not requested (or there were no cards)."""

    @property
    def failures(self) -> list[GateFailure]:
        """Every failed gate: thresholds per scenario, then regressions."""
        threshold_failures = [f for verdict in self.scenarios for f in verdict.failures]
        compared = self.regression.scenarios if self.regression is not None else ()
        return threshold_failures + _regression_failures(compared)

    @property
    def passed(self) -> bool:
        """True when there are cards and no gate failed."""
        return bool(self.scenarios) and not self.failures


def _threshold_failures(
    card: RobustnessCard, thresholds: RobustnessThresholds
) -> tuple[GateFailure, ...]:
    checks = (
        (Gate.ESCAPE_RATE, card.escape_rate, thresholds.max_escape_rate),
        (Gate.MEDIAN_DEGRADATION, card.median_score_degradation, thresholds.max_median_degradation),
    )
    return tuple(
        GateFailure(card.scenario_name, gate, observed, limit)
        for gate, observed, limit in checks
        if observed > limit
    )


def _scenario_regression(
    card: RobustnessCard,
    previous_card: RobustnessCard | None,
    thresholds: RobustnessThresholds,
) -> ScenarioRegression:
    if previous_card is None:
        return ScenarioRegression(card.scenario_name, None, card.escape_rate, None)
    increase = card.escape_rate - previous_card.escape_rate
    failure = None
    if increase > thresholds.max_regression_increase:
        failure = GateFailure(
            card.scenario_name, Gate.REGRESSION, increase, thresholds.max_regression_increase
        )
    return ScenarioRegression(
        card.scenario_name, previous_card.escape_rate, card.escape_rate, failure
    )


def _regression_failures(scenarios: Iterable[ScenarioRegression]) -> list[GateFailure]:
    return [scenario.failure for scenario in scenarios if scenario.failure is not None]


def _compare_with_previous(
    record: TrainingRecord, previous: TrainingRecord, thresholds: RobustnessThresholds
) -> tuple[ScenarioRegression, ...]:
    """Each current card against the previous version's card of the same scenario name."""
    previous_cards = {card.scenario_name: card for card in previous.robustness_cards}
    return tuple(
        _scenario_regression(card, previous_cards.get(card.scenario_name), thresholds)
        for card in record.robustness_cards
    )


def evaluate_robustness_gates(
    record: TrainingRecord,
    previous: TrainingRecord | None,
    thresholds: RobustnessThresholds,
) -> list[GateFailure]:
    """Every gate ``record`` fails; regressions are checked only when ``previous`` is given."""
    failures = [
        f for card in record.robustness_cards for f in _threshold_failures(card, thresholds)
    ]
    if previous is not None:
        failures += _regression_failures(_compare_with_previous(record, previous, thresholds))
    return failures


def _check_regression(
    model_dir: Path, record: TrainingRecord, thresholds: RobustnessThresholds
) -> RegressionCheck:
    previous_dir = previous_version(model_dir)
    if previous_dir is None:
        return RegressionCheck(warning="No previous version found for regression check")
    try:
        previous = load_training_record(previous_dir)
    except (FileNotFoundError, ValueError) as error:
        return RegressionCheck(
            warning=f"Could not load previous version for regression check: {error}"
        )
    return RegressionCheck(
        previous_version=previous_dir.name,
        scenarios=_compare_with_previous(record, previous, thresholds),
    )


def verify_robustness(
    model_dir: Path,
    thresholds: RobustnessThresholds | None = None,
    check_regression: bool = False,
) -> RobustnessVerification:
    """Check the robustness cards of ``model_dir/model_record.json`` against ``thresholds``.

    With ``check_regression``, also compare each scenario's escape rate with the previous
    registry version (``previous_version``). A missing or unreadable previous version is a
    warning, not a failure.

    Raises:
        FileNotFoundError: If ``model_dir`` has no model record.
        ValueError: If the model record cannot be decoded.
    """
    thresholds = thresholds or RobustnessThresholds()
    record = load_training_record(model_dir)
    scenarios = tuple(
        ScenarioVerdict(card, _threshold_failures(card, thresholds))
        for card in record.robustness_cards
    )
    regression = None
    if check_regression and scenarios:
        regression = _check_regression(model_dir, record, thresholds)
    return RobustnessVerification(model_dir, scenarios, regression)


def _render_failure(failure: GateFailure) -> str:
    if failure.gate is Gate.ESCAPE_RATE:
        return f"FAIL: escape rate {failure.observed:.2%} > {failure.limit:.2%}"
    if failure.gate is Gate.MEDIAN_DEGRADATION:
        return f"FAIL: median degradation {failure.observed:+.3f} > {failure.limit:.3f}"
    return (
        f"REGRESSION: escape rate increased by "
        f"{failure.observed:.2%} (max allowed: {failure.limit:.2%})"
    )


def _render_scenario(verdict: ScenarioVerdict) -> list[str]:
    card = verdict.card
    return [
        "",
        f"{card.scenario_name}:",
        f"  Escape rate: {card.escape_rate:.2%}",
        f"  Median degradation: {card.median_score_degradation:+.3f}",
        f"  Worst case degradation: {card.worst_case_degradation:+.3f}",
        *(f"  {_render_failure(failure)}" for failure in verdict.failures),
    ]


def _render_scenario_regression(scenario: ScenarioRegression) -> list[str]:
    if scenario.previous_escape_rate is None:
        return [f"  {scenario.scenario_name}: no previous baseline (new scenario)"]
    increase = scenario.escape_rate - scenario.previous_escape_rate
    lines = [
        f"  {scenario.scenario_name}: {scenario.previous_escape_rate:.2%} → "
        f"{scenario.escape_rate:.2%} (Δ {increase:+.2%})"
    ]
    if scenario.failure is not None:
        lines.append(f"    {_render_failure(scenario.failure)}")
    return lines


def _render_regression(regression: RegressionCheck) -> list[str]:
    if regression.warning is not None:
        return ["", f"WARNING: {regression.warning}"]
    lines = ["", f"Regression check against previous version: {regression.previous_version}"]
    for scenario in regression.scenarios:
        lines += _render_scenario_regression(scenario)
    return lines


def render_verification(verification: RobustnessVerification) -> str:
    """The ``verify`` command's report for ``verification``, one line per row."""
    if not verification.scenarios:
        return f"ERROR: No robustness cards found in {verification.model_dir}"
    lines = [f"Verifying robustness for {len(verification.scenarios)} scenario(s)..."]
    for verdict in verification.scenarios:
        lines += _render_scenario(verdict)
    if verification.regression is not None:
        lines += _render_regression(verification.regression)
    verdict_line = (
        "PASS: All robustness checks passed"
        if verification.passed
        else "FAIL: Robustness checks failed"
    )
    lines += ["", verdict_line]
    return "\n".join(lines)


def generate_report(model_dir: Path, output: Path | None = None) -> None:
    """Generate human-readable robustness report.

    Args:
        model_dir: Registry version directory containing model_record.json.
        output: Output path for report (default: stdout).
    """
    record = load_training_record(model_dir)

    if not record.robustness_cards:
        print("No robustness cards found.")
        return

    lines = []
    lines.append("# Adversarial Robustness Report")
    lines.append("")
    lines.append(f"Model: {model_dir}")
    lines.append(f"Schema version: {record.schema_version}")
    lines.append("")

    for card in record.robustness_cards:
        lines.append(f"## Scenario: {card.scenario_name}")
        lines.append("")
        lines.append(f"- Tested at: {card.tested_at}")
        lines.append(f"- YAML hash: `{card.scenario_yaml_sha256[:16]}...`")
        lines.append(f"- **Escape rate:** {card.escape_rate:.2%}")
        lines.append(f"- **Median score degradation:** {card.median_score_degradation:+.3f}")
        lines.append(f"- **Worst case degradation:** {card.worst_case_degradation:+.3f}")
        lines.append("")

        lines.append("### Mutation Results")
        lines.append("")
        lines.append("| Mutation | Intensity | Original | Mutated | Delta | Escaped |")
        lines.append("|----------|-----------|----------|---------|-------|---------|")

        for result in card.mutation_results[:10]:  # Show first 10
            lines.append(
                f"| {result.mutation_class} | {result.intensity} | "
                f"{result.original_score:+.3f} | {result.mutated_score:+.3f} | "
                f"{result.score_delta:+.3f} | {result.escaped} |"
            )

        if len(card.mutation_results) > 10:
            lines.append("| ... | ... | ... | ... | ... | ... |")
            lines.append(f"| ({len(card.mutation_results) - 10} more results) |")

        lines.append("")

    report = "\n".join(lines)

    if output:
        output.write_text(report, encoding="utf-8")
        print(f"Report written to {output}")
    else:
        print(report)


@dataclass(frozen=True)
class VerifyArgs:
    """The parsed command line of a ``verify`` run."""

    model_dir: Path
    thresholds: RobustnessThresholds
    check_regression: bool


@dataclass(frozen=True)
class ReportArgs:
    """The parsed command line of a ``report`` run."""

    model_dir: Path
    output: Path | None


def parse_args(argv: Sequence[str] | None = None) -> VerifyArgs | ReportArgs:
    """The command and its options from `argv` (the process's arguments when `None`)."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify robustness metrics")
    verify_parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Registry version directory containing model_record.json",
    )
    verify_parser.add_argument(
        "--max-escape-rate",
        type=float,
        default=0.15,
        help="Maximum allowed escape rate (default: 0.15)",
    )
    verify_parser.add_argument(
        "--max-median-degradation",
        type=float,
        default=0.20,
        help="Maximum allowed median degradation (default: 0.20)",
    )
    verify_parser.add_argument(
        "--check-regression",
        "--no-regression",
        dest="check_regression",
        action="store_true",
        help=(
            "Check for regression vs previous version "
            "(--no-regression is the older name of the same flag)"
        ),
    )
    verify_parser.add_argument(
        "--max-regression-increase",
        type=float,
        default=0.05,
        help="Maximum allowed escape rate increase vs previous version (default: 0.05)",
    )

    # Report command
    report_parser = subparsers.add_parser("report", help="Generate robustness report")
    report_parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Registry version directory containing model_record.json",
    )
    report_parser.add_argument(
        "--output",
        type=Path,
        help="Output path for report (default: stdout)",
    )

    args = parser.parse_args(argv)
    if args.command == "verify":
        return VerifyArgs(
            model_dir=args.model_dir,
            thresholds=RobustnessThresholds(
                max_escape_rate=args.max_escape_rate,
                max_median_degradation=args.max_median_degradation,
                max_regression_increase=args.max_regression_increase,
            ),
            check_regression=args.check_regression,
        )
    return ReportArgs(model_dir=args.model_dir, output=args.output)


def run(args: VerifyArgs | ReportArgs) -> int:
    """Run the ``verify`` or ``report`` command; returns the exit status, 1 when ``verify``
    finds a failed gate."""
    if isinstance(args, VerifyArgs):
        verification = verify_robustness(args.model_dir, args.thresholds, args.check_regression)
        print(render_verification(verification))
        return 0 if verification.passed else 1
    generate_report(args.model_dir, args.output)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command the command line `argv` names; returns the exit status."""
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
