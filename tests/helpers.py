"""Builders shared by several test modules.

Plain functions rather than fixtures: they are called from other helpers and from
``pytest.mark.parametrize`` arguments, where a fixture cannot reach. Builders that need
pytest state (``monkeypatch``) are fixtures in ``conftest.py`` instead.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from synthaea_ml.data.manifest import DEFAULT_BASELINE_FILENAME, write_manifest

# The call signatures of the `robustness_stub` and `site_rarity_run` fixtures (conftest.py).
RobustnessStub = Callable[[object, object], list[dict]]
SiteRarityRun = Callable[..., Path]


def read_golden_jsonl(path: Path) -> list[dict]:
    """The rows of a golden JSONL fixture, blank lines skipped."""
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> Path:
    """Write ``rows`` as one JSON object per line (newline-terminated) and return ``path``."""
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def write_baseline_dataset(
    directory: Path,
    lines: list[str],
    *,
    capture_start: datetime,
    capture_end: datetime,
    platform: str = "linux",
    os_version: str = "test",
    workload_label: str = "dev",
    hostname: str = "testhost",
    baseline_filename: str = DEFAULT_BASELINE_FILENAME,
) -> Path:
    """A dataset directory: ``lines`` as its baseline file, plus a manifest that binds it."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / baseline_filename).write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_manifest(
        directory,
        platform=platform,
        os_version=os_version,
        workload_label=workload_label,
        capture_start=capture_start,
        capture_end=capture_end,
        hostname=hostname,
        baseline_filename=baseline_filename,
    )
    return directory


def write_scenario_yaml(scenarios_root: Path, name: str, contents: str) -> tuple[Path, str]:
    """Write a scenario yaml sidecar and return its path and content hash.

    The tests do not need a schema-valid yaml since the record binds by content hash,
    not by parse: any deterministic bytes will do.
    """
    scenarios_root.mkdir(parents=True, exist_ok=True)
    path = scenarios_root / f"{name}.yaml"
    path.write_bytes(contents.encode("utf-8"))
    return path, hashlib.sha256(contents.encode("utf-8")).hexdigest()


def robustness_card(escape: float, degradation: float) -> SimpleNamespace:
    """The two fields of a robustness card that the floor checks read."""
    return SimpleNamespace(escape_rate=escape, median_score_degradation=degradation)


def cmdline_rows(n: int, seed: int) -> np.ndarray:
    """``n`` seeded rows of the 9 cmdline features."""
    return np.random.default_rng(seed).normal(size=(n, 9)).astype(np.float32)


def write_site_corpus(tmp_path: Path, hosts: int = 8, days: int = 20, per_day: int = 6) -> Path:
    """A synthetic site: `hosts` agents run mostly the same tools, plus per-host oddities."""
    rows: list[dict] = []
    common = [f"{i:064x}" for i in range(6)]
    for day in range(days):
        for h in range(hosts):
            for n in range(per_day):
                shared = (day + h + n) % 4 != 0
                sha = common[(h + n) % len(common)] if shared else f"{(h * 1000 + day * 10 + n):064x}"
                stamp = f"2026-09-{1 + day:02d}T{8 + n:02d}:{h:02d}:00Z"
                argv = ["/usr/bin/tool", f"--mode{n % 3}", "x" * (1 + (day + n) % 5)]
                rows.append(
                    {
                        "timestamp": stamp,
                        "event_type": "exec",
                        "agent_id": f"agent-{h}",
                        "event": {"argv": argv, "image_path": argv[0], "sha256": sha},
                    }
                )
    return write_jsonl(tmp_path / "corpus-v1.jsonl", rows)
