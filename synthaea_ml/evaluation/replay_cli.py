"""Bind scenario replays to a model record (issue #49).

`scenario_replay.run_replay` turns a scenario sidecar and the `alerts.ndjson` of a run
against it into a `ScenarioReplayResult`, but nothing wrote those results into a model
record, so the `scenario_replays` section the registry format describes was always empty
(see `docs/evaluation.md`, `docs/registry.md`).

This command is that writer. Given a model version directory and one or more
`(scenario yaml, alerts.ndjson)` pairs, it replays each pair and stores the results in
`model_record.json`'s `scenario_replays`, replacing what was there (`--append` adds to it
instead). The run that produced each `alerts.ndjson` happens elsewhere (the lab rig); this
only reads it. The provenance gate (`export/verify_onnx.py`) then re-verifies each bound
replay against its scenario yaml.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from collections.abc import Sequence
from pathlib import Path

from synthaea_ml.cli_logging import configure_cli_logging
from synthaea_ml.evaluation.scenario_replay import run_replay
from synthaea_ml.registry.codec import load_training_record, write_record
from synthaea_ml.registry.records import ScenarioReplayResult


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """The command's options from `argv` (the process's arguments when `None`)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Registry version directory containing model_record.json",
    )
    parser.add_argument(
        "--replay",
        dest="replays",
        action="append",
        nargs=2,
        metavar=("SCENARIO_YAML", "ALERTS_NDJSON"),
        required=True,
        help="A scenario sidecar and the alerts.ndjson of a run against it; repeatable",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Add to the record's existing replays instead of replacing them",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    """Replay each pair and write the results into the record; returns the exit status."""
    results: list[ScenarioReplayResult] = []
    for scenario_yaml, alerts_ndjson in args.replays:
        results.append(run_replay(Path(scenario_yaml), Path(alerts_ndjson)))

    record = load_training_record(args.model_dir)
    bound = [*record.scenario_replays, *results] if args.append else results
    write_record(args.model_dir, dataclasses.replace(record, scenario_replays=bound))

    for r in results:
        print(f"{'PASS' if r.passed else 'FAIL'} {r.scenario_name}")
    print(f"wrote {len(bound)} scenario replay(s) to {args.model_dir}")
    return 0 if all(r.passed for r in results) else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command the command line `argv` names; returns the exit status."""
    configure_cli_logging()
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
