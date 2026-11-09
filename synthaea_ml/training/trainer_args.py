"""The command line shared by the dataset trainers (`train_linux`, `train_windows`,
`train_behavior`): one or more datasets in, one registry version directory out, and an
optional robustness evaluation.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainerArgs:
    """The parsed command line of a trainer run."""

    datasets: tuple[Path, ...]
    output_dir: Path
    robustness_scenarios: tuple[Path, ...]
    robustness_events: Path | None


@dataclass(frozen=True)
class TrainerHelp:
    """The `--help` text of each trainer option, which differs between the tiers."""

    dataset: str
    output_dir: str
    robustness_scenarios: str
    robustness_events: str


CMDLINE_TRAINER_HELP = TrainerHelp(
    dataset=(
        "One or more baseline directories (each contains baseline.jsonl + "
        "manifest.json). Multiple datasets concatenate their samples and "
        "each shows up as its own DatasetVersion in the training record."
    ),
    output_dir="Registry version directory (model.onnx + training.json are written here).",
    robustness_scenarios=(
        "Optional scenario yamls for adversarial robustness evaluation (issue #45)."
    ),
    robustness_events=(
        "Optional events.jsonl or baseline.jsonl file for robustness evaluation. "
        "If provided, uses real events from this file instead of synthetic events."
    ),
)
"""The help of the T0 command-line trainers (`train_linux`, `train_windows`)."""


def parse_trainer_args(
    argv: Sequence[str] | None, *, description: str | None, help_text: TrainerHelp
) -> TrainerArgs:
    """The trainer options from `argv` (the process's arguments when `None`)."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--dataset", type=Path, required=True, nargs="+", help=help_text.dataset
    )
    parser.add_argument("--output-dir", type=Path, required=True, help=help_text.output_dir)
    parser.add_argument(
        "--robustness-scenarios",
        type=Path,
        nargs="*",
        default=[],
        help=help_text.robustness_scenarios,
    )
    parser.add_argument("--robustness-events", type=Path, help=help_text.robustness_events)
    args = parser.parse_args(argv)
    return TrainerArgs(
        datasets=tuple(args.dataset),
        output_dir=args.output_dir,
        robustness_scenarios=tuple(args.robustness_scenarios),
        robustness_events=args.robustness_events,
    )
