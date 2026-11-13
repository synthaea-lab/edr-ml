"""Record a scenario replay into a model's ``model_record.json`` (issue #49).

A replay says whether the agent, running this model, raised the alerts a scenario expects.
It is recorded after training: the model is deployed on a lab machine, the scenario runs
there, and the ``alerts.ndjson`` of that run comes back here. ``run_replay`` matches it
against the scenario's sidecar, and the result joins the record's ``scenario_replays``,
where the release gate re-verifies it against the same sidecar.

The sidecar is read from the ``scenarios`` sibling checkout by default, the directory the
gate itself checks against; without that checkout, pass ``--scenarios-root``. A missing
sidecar fails the command and leaves the record untouched.

A failing replay is still recorded, and the command still exits 0: the record states what
the run showed, and the gate decides what may ship.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from synthaea_ml import siblings
from synthaea_ml.evaluation.scenario_replay import run_replay
from synthaea_ml.registry.codec import load_training_record, save_training_record
from synthaea_ml.registry.records import Environment, ScenarioReplayResult, with_scenario_replay


class UnknownScenarioError(ValueError):
    """A sidecar's ``name`` differs from the name it was looked up by, so the release gate,
    which looks a replay's sidecar up by that ``name``, could never find it."""


@dataclass(frozen=True)
class ReplayArgs:
    """The parsed command line of a ``record-replay`` run."""

    model_dir: Path
    scenario: str
    alerts: Path
    scenarios_root: Path
    environment: Environment | None
    """The machine the scenario ran on; ``None`` for this one."""


def record_replay(args: ReplayArgs) -> ScenarioReplayResult:
    """Match ``args.alerts`` against the scenario's sidecar and record the replay in the
    model record; returns the recorded replay.

    Raises:
        FileNotFoundError: If the model record, the sidecar or the alerts file is missing.
        UnknownScenarioError: If the sidecar's ``name`` is not ``args.scenario``.
    """
    sidecar = args.scenarios_root / f"{args.scenario}.yaml"
    if not sidecar.is_file():
        raise FileNotFoundError(f"scenario sidecar not found: {sidecar}")
    record = load_training_record(args.model_dir)
    replay = run_replay(sidecar, args.alerts, environment=args.environment)
    if replay.scenario_name != args.scenario:
        raise UnknownScenarioError(
            f"{sidecar} is named {replay.scenario_name!r}, not {args.scenario!r}"
        )
    save_training_record(args.model_dir, with_scenario_replay(record, replay))
    return replay


def describe_replay(replay: ScenarioReplayResult) -> str:
    """One line stating a replay's outcome and where it ran."""
    outcome = "passed" if replay.passed else "FAILED"
    environment = replay.environment
    return (
        f"{replay.scenario_name}: {outcome} on "
        f"{environment.os} {environment.kernel} {environment.arch}"
    )


def parse_args(argv: Sequence[str] | None = None) -> ReplayArgs:
    """The options from `argv` (the process's arguments when `None`)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Registry version directory containing model_record.json",
    )
    parser.add_argument(
        "--scenario", required=True, help="Scenario name, the stem of its sidecar yaml"
    )
    parser.add_argument(
        "--alerts", type=Path, required=True, help="alerts.ndjson of the run against it"
    )
    parser.add_argument(
        "--scenarios-root",
        type=Path,
        default=siblings.scenario_sidecars(),
        help="Directory of the sidecars (default: the scenarios sibling checkout)",
    )
    parser.add_argument(
        "--environment",
        nargs=3,
        metavar=("OS", "KERNEL", "ARCH"),
        help="The machine the scenario ran on, when not this one (e.g. a rig VM)",
    )
    args = parser.parse_args(argv)
    return ReplayArgs(
        model_dir=args.model_dir,
        scenario=args.scenario,
        alerts=args.alerts,
        scenarios_root=args.scenarios_root,
        environment=Environment(*args.environment) if args.environment else None,
    )


def run(args: ReplayArgs) -> int:
    """Record the replay `args` describe and print its outcome; returns the exit status."""
    print(describe_replay(record_replay(args)))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Record a replay from the command line `argv`; returns the exit status."""
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
