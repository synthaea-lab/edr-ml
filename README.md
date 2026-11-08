# ml — The ML Space

Everything about the learned half of the detection stack lives here: datasets, feature
definitions, model training, calibration, evaluation, and export to ONNX for on-device
inference by `crates/ml`. This is a first-class part of the product, not a scripts
folder — the thesis is that a learned model raises the cost of evasion through
generalization, and that only holds if this pipeline is reproducible and measured.

## Inference tiers

| Tier | Runs | Model | Trained from |
| --- | --- | --- | --- |
| T0 — event scoring | on-device | per-event anomaly (e.g. cmdline features, Isolation Forest) | benign captures per platform |
| T1 — behavior | on-device | aggregated per-entity behavior scoring | scenario + benign captures |
| T2 — correlation | on-device | case scorer over correlated detections (with Bayesian LLR calibration) | labeled correlation traces |
| T3 — fleet | server-side | rarity / per-tenant baselines | fleet telemetry (control plane) |

T0–T2 export to ONNX and ship to agents as signed data via canary rings, never embedded
in the binary (ADR-0002); T3 lives with the control plane and never ships to endpoints.
T3 also closes the loop: it curates per-site benign corpora from fleet telemetry, from
which T0–T2 are recalibrated per deployment, under safety constraints of their own.

## Repository layout

| Path | Purpose |
| --- | --- |
| `synthaea_ml/` | The Python package — see subpackage docstrings; CLIs run as `python -m synthaea_ml.<pkg>.<module>` |
| `synthaea_ml/features/` | Feature definitions — the Rust parity seam (fixtures shared with `crates/ml`) |
| `synthaea_ml/data/` | Capture parsing, dataset building, baseline manifests |
| `synthaea_ml/training/` | Training entry points per tier/platform, shared training config |
| `synthaea_ml/calibration/` | Bayesian LLR calibration, conformal FP-budget thresholds, OOD guards |
| `synthaea_ml/evaluation/` | Metrics, FP governance, scenario-replay and robustness evaluation |
| `synthaea_ml/registry/` | Registry file layout and the machine-readable model record |
| `synthaea_ml/export/` | ONNX parity verification (`verify_onnx`) |
| `datasets/baselines/` | Committed benign baselines, one directory per capture, manifested per `synthaea_ml.data.manifest`; raw captures and labeled traces stay on disk ([datasets/README.md](datasets/README.md)) |
| `registry/` | Versioned model artifacts + model cards — what ships ([registry/README.md](registry/README.md)) |
| `reports/calibration/` | LLR calibration reports, the receipts behind calibrated constants ([reports/README.md](reports/README.md)) |
| `tests/` | Unit tests |
| `tests/fixtures/` | Rust/Python parity golden data; `generators/` holds the scripts that write it ([tests/fixtures/README.md](tests/fixtures/README.md)) |

## The three rules

1. **Feature parity is enforced, not hoped for.** Every feature has a definition here
   and in `crates/ml`; both test against the same fixture vectors. Drift is a CI
   failure, and drift in production is a silent model lobotomy. The same discipline
   covers score attribution (`tests/attribution_reference.py` vs `crates/ml`'s forest
   walk, pinned by `attribution_golden.json`) and the inference path itself
   (`crates/ml`'s `ort` scorer vs onnxruntime, pinned by `scorer_golden.json`): the
   Rust feature vector, its explanation, and the score the shipped runtime computes
   from it are each locked to the Python reference.
2. **No model ships without an evaluation record.** Each registry entry carries a model
   card: training data provenance, metrics, FP rate against benign baselines, and the
   scenario-replay results — plus a **robustness card**: measured score degradation
   under attacker-style mutations of known-bad samples. FP
   governance and robustness regression are both release gates.
3. **Reproducibility.** A registry model can be rebuilt from its recorded dataset
   versions and config. Exploratory work is never load-bearing (see Contributing).

Migrated from `old/ml` (issue #14): features, training entry points, LLR calibration,
capture/aggregation, `verify_onnx` (now a registry-wide oracle:
`python -m synthaea_ml.export.verify_onnx`), and the Rust/Python parity fixtures under
`tests/fixtures/`. The two old models live in `registry/` with reference-only cards —
they predate the evaluation gates and do not meet ship criteria.

## Tests and sibling checkouts

`ruff check .`, `mypy` and `pytest tests` run from this directory (`pip install -e ".[dev]"`
first: it pins the linters CI uses). Some tests read files owned by
public sibling repositories: the agent's Rust parity fixtures (`crates/ml/tests/fixtures`
in synthaea-lab/agent) and the scenario sidecars and capture scripts of
synthaea-lab/scenarios (`scenarios/`, `ml-capture/`; model records bind to the yaml
sidecars by content hash, ADR-0009). Check them out next to this repo as `agent/` and
`scenarios/` (alongside `server/` and `rules/` if you have them), or point `SYNTHAEA_ROOT`
at their parent. A test whose sibling is absent is skipped; `SYNTHAEA_REQUIRE_SIBLINGS=1`
(set by CI) makes it fail instead.

## Contributing

Exploratory work (notebooks, one-off scripts) stays out of this repository until it
matters: anything load-bearing graduates into `synthaea_ml/` with tests. Nothing
exploratory is imported by the package or run by CI.
