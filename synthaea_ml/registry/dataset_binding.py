"""Bind a record to the datasets it was trained on: a ``DatasetVersion`` built from a
baseline's verified manifest, under a name derived from that manifest.
"""

from __future__ import annotations

from pathlib import Path

from synthaea_ml.data.manifest import DEFAULT_BASELINE_FILENAME, load_manifest, verify_manifest
from synthaea_ml.registry.records import DatasetVersion


def default_dataset_name(baseline_dir: Path) -> str:
    """Derive a human-legible dataset name from the manifest next to a
    baseline.

    Convention: ``<platform>__<workload_label>__<host_id>__<YYYY-MM-DD>``.
    Double underscores separate the four coordinates so a workload label
    containing a single hyphen (``desktop-user``) does not become ambiguous.
    The date is ``capture_start[:10]`` — day-level granularity is what we bin
    by; sub-day precision belongs in the manifest, not in the name.

    Raises:
        FileNotFoundError: If the manifest is missing.
    """
    manifest = load_manifest(baseline_dir)
    capture_day = manifest.capture_start[:10]  # "2026-09-11" from ISO string
    return (
        f"{manifest.platform}__{manifest.workload_label}__{manifest.host_id}__{capture_day}"
    )


def dataset_version_from_manifest(
    baseline_dir: Path,
    *,
    name: str | None = None,
    baseline_filename: str = DEFAULT_BASELINE_FILENAME,
) -> DatasetVersion:
    """Verify the baseline against its manifest and return a ``DatasetVersion``.

    A caller who reads this back — the release gate or a re-training tool —
    can trust ``baseline_sha256`` to point at unmutated content because we
    just re-hashed the file to check.

    Args:
        baseline_dir: Directory containing ``baseline.jsonl`` and
            ``manifest.json``.
        name: Optional override. Defaults to
            ``default_dataset_name(baseline_dir)``.
        baseline_filename: Name of the samples file the manifest was written
            for. A T1 behavior capture is ``events.jsonl``, not ``baseline.jsonl``.

    Raises:
        FileNotFoundError: If baseline or manifest is missing.
        ValueError: If the baseline hash or sample count no longer matches
            the manifest (see ``verify_manifest``).
    """
    verify_manifest(baseline_dir, baseline_filename=baseline_filename)
    manifest = load_manifest(baseline_dir)
    return DatasetVersion(
        name=name if name is not None else default_dataset_name(baseline_dir),
        baseline_sha256=manifest.sample_sha256,
        sample_count=manifest.sample_count,
    )
