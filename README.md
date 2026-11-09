# Synthaea ML

The machine-learning pipeline of [Synthaea](https://github.com/synthaea-lab), an open-source
EDR/XDR. This repository builds the models the endpoint agent scores events with: it holds the
benign datasets, the feature definitions, training, calibration, evaluation, and export to ONNX
for on-device inference by the agent's [`ml` crate](https://github.com/synthaea-lab/agent/tree/main/crates/ml).

The premise is that a learned model raises the cost of evasion by generalizing beyond exact
signatures. That only holds if the pipeline is reproducible and every model is measured, so
both are enforced here rather than left to convention.

> **Status: alpha (0.1.0).** The pipeline, its features and its evaluation gates are in place.
> No model in the registry meets the ship criteria yet: the two committed models are
> reference-only and serve as verification inputs.

## How it fits

Models are grouped in tiers by what they score and where they run.

| Tier | Runs | Scores | Trained from |
| --- | --- | --- | --- |
| T0, event | on the endpoint | a single event, such as a command line (Isolation Forest) | benign captures, one model per platform |
| T1, behavior | on the endpoint | one process incarnation: command line and correlated activity | captures of normal activity |
| T2, correlation | on the endpoint | correlated detections, with Bayesian likelihood-ratio calibration | labeled correlation traces |
| T3, fleet | on the control plane | rarity and per-site baselines | fleet telemetry |

T0 to T2 are exported to ONNX and delivered to agents as content, separate from the agent
binary. T3 stays with the [control plane](https://github.com/synthaea-lab/server) and never
runs on endpoints; it also curates per-site benign corpora from which the endpoint tiers are
recalibrated for a deployment.

The [architecture overview](https://www.synthaea.com/en/docs/architecture) places these tiers
among the other detection layers.

## Principles

1. **Feature parity is tested, not assumed.** Every feature is defined twice, here in Python
   and in the agent in Rust, and both sides assert against the same golden vectors. The same
   applies to score attribution and to the ONNX inference path, so the feature vector, its
   explanation and its score are each pinned to the Python reference.
2. **No model ships without an evaluation record.** Each registry version carries a model
   card and a machine-readable record: training data provenance, metrics, the false-positive
   rate against benign baselines, scenario-replay results, and a robustness card that measures
   score degradation under attacker-style mutations of known-bad samples.
3. **Every model is reproducible.** A registry model can be rebuilt from the dataset versions
   and hyperparameters it records. Baselines are content-hashed, so the data cannot change
   under a model that names it.

## Getting started

Requires Python 3.11 or later and Git.

```bash
git clone https://github.com/synthaea-lab/ml
cd ml
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

On Windows, create the environment with `py -m venv .venv` and activate it with
`.venv\Scripts\activate`; the `python -m` commands below are the same on every platform.

Verify every model in the registry under ONNX Runtime (output schema, finite and
deterministic scores, label and score agreement, and dataset provenance):

```bash
python -m synthaea_ml.export.verify_onnx
```

Check that a committed baseline still matches its manifest:

```bash
python -m synthaea_ml.data.manifest verify datasets/baselines/linux-wsl2-solkapc-2026-09-09-v2
```

Train a T0 command-line model on that baseline. The trainer verifies the manifest first, then
writes `model.onnx`, `model_metadata.json` and `model_record.json` to the output directory:

```bash
python -m synthaea_ml.training.train_linux --dataset datasets/baselines/linux-wsl2-solkapc-2026-09-09-v2 --output-dir out/cmdline-iforest-linux
```

Every command-line entry point runs as `python -m synthaea_ml.<package>.<module>` and
documents its arguments with `--help`. `pip install -e .` also puts each one on the `PATH` as a
`synthaea-ml-*` command (for example `synthaea-ml-train-linux` or `synthaea-ml-verify-onnx`);
`[project.scripts]` in [`pyproject.toml`](pyproject.toml) lists them all.

## Repository layout

| Path | Contents |
| --- | --- |
| [`synthaea_ml/`](synthaea_ml) | The Python package |
| `synthaea_ml/features/` | Feature definitions, mirrored by the agent's Rust implementation |
| `synthaea_ml/data/` | Capture parsing, dataset building and baseline manifests |
| `synthaea_ml/training/` | Training entry points per tier and platform, and their shared configuration |
| `synthaea_ml/calibration/` | Likelihood-ratio calibration and conformal false-positive thresholds |
| `synthaea_ml/evaluation/` | Scenario replay, robustness evaluation and attacker-style mutations |
| `synthaea_ml/registry/` | Registry layout and the model record |
| `synthaea_ml/export/` | ONNX verification |
| [`datasets/`](datasets/README.md) | Committed benign baselines and their manifests |
| [`registry/`](registry/README.md) | Versioned models with their model cards |
| [`reports/`](reports/README.md) | Calibration reports behind calibrated constants |
| [`tests/`](tests/fixtures/README.md) | Unit tests and the Rust/Python parity fixtures |

## Contributing

Contributions are welcome. [CONTRIBUTING.md](CONTRIBUTING.md) explains how to set up the
sibling checkouts the parity tests need, run the checks, and change a feature or a fixture.
Every commit carries a Developer Certificate of Origin sign-off (`git commit -s`); there is no
CLA. The [project contributing guide](https://www.synthaea.com/en/docs/contributing) covers
the other repositories.

## Security

Do not report vulnerabilities in public issues. Use GitHub private vulnerability reporting on
this repository, as described in the
[security policy](https://github.com/synthaea-lab/.github/blob/main/SECURITY.md).

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
