# Technical documentation

How the Synthaea ML pipeline works: where training data comes from, how features are defined
and kept identical to the agent's Rust extractors, how models are trained, calibrated and
evaluated, and how a model version is recorded, verified and handed to the agent. Each page
links into the code that implements what it describes. When the code and a page disagree,
fix the page in the same pull request.

The [repository README](../README.md) is the short introduction: the model tiers, the three
principles (feature parity, an evaluation record for every model, reproducibility) and a
first run. These pages are the detail behind it.

## Start here

- [architecture.md](architecture.md): the package layers, the dependency rules between them,
  and the path from a raw agent capture to a model the agent loads.
- [cli.md](cli.md): every command-line entry point, its inputs, outputs and exit status.

## Data and features

- [datasets.md](datasets.md): the agent's event format, the three dataset shapes, manifests and
  content hashes, site corpora, and how the capture workloads produce training data.
- [features.md](features.md): every feature vector (command line, correlation, T1, lineage,
  rarity, behavior), its exact definition and the golden fixture that pins it to Rust.

## Models

- [training.md](training.md): the shared training steps, the configuration, and each trainer:
  T0 command line, T1 behavior, correlation scorer, and the two per-site trainers.
- [calibration.md](calibration.md): the conformal false-positive threshold, the
  out-of-distribution bounds, and the likelihood-ratio calibration behind the correlator.
- [evaluation.md](evaluation.md): scenario replay, adversarial robustness cards and their gate,
  and the lineage-gain measurement.
- [registry.md](registry.md): the registry layout, `model_record.json`, the provenance gate
  in `verify_onnx`, and what the agent loads.

## Working on the repository

- [reproducibility.md](reproducibility.md): seeds, content hashes, byte-pinned artifacts and line
  endings, and what still varies between runs.
- [testing.md](testing.md): the test suite, the Rust/Python golden fixtures and the sibling
  checkouts they need.
- [development.md](development.md): setup, the checks CI runs, lint and type-check ratchets,
  logging and command-line conventions.

## Related documentation

- The agent's [on-device ML](https://github.com/synthaea-lab/agent/blob/main/docs/detection/ml.md)
  page covers the consuming side: the Rust extractors, the ONNX scorers, attribution and how the
  correlator uses a score.
- The scenarios repository documents the
  [capture workloads](https://github.com/synthaea-lab/scenarios/blob/main/docs/ml-capture.md)
  that record training data and the
  [sidecar format](https://github.com/synthaea-lab/scenarios/blob/main/docs/sidecar-format.md)
  that scenario replays and robustness cards bind to.
