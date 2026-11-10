# Model registry

[`registry/`](../registry) holds the model versions this repository publishes, one directory per
model and one subdirectory per version. This page covers the files of a version, the
machine-readable record that binds a model to its data and evaluations, the gate that re-checks
that binding, and how a model reaches the agent.

## Layout

```text
registry/<model>/<version>/
    model.onnx           the model (committed despite the repository-wide *.onnx ignore)
    model_metadata.json  conformal threshold and feature bounds, read by the agent's scorer
    model_record.json    datasets, evaluations, hyperparameters, calibration
    card.md              the model card, for people
```

The three generated file names are constants in [`registry/layout.py`](../synthaea_ml/registry/layout.py).
The trainers write `model.onnx`, `model_metadata.json` and `model_record.json`
([training.md](training.md)); `card.md` is written by hand. The model names in use follow
`<family>-<algorithm>-<platform>`, for example `cmdline-iforest-linux`, and versions are
`major.minor.patch` directory names: the robustness gate finds the previous version by parsing
them ([evaluation.md](evaluation.md#the-gate)).

### The ONNX model

The model is an `IsolationForest` converted by skl2onnx (opset 18, `ai.onnx.ml` 3) with one float
input `X` of shape `[N, features]` and two outputs of shape `[N, 1]`: `label` (-1 anomalous, 1
normal) and `scores` (`decision_function`, negative is anomalous). The agent reads `scores`. It also parses the tree
ensemble out of the same bytes to attribute a score to features, so the explanation always
describes the model that produced the score.

### The model card

A card states the model's purpose and tier, its training data, metrics, the false-positive rate
against benign baselines, scenario-replay and robustness results, known limitations, and the
parity fixtures it was verified against. It is the human summary of what the record holds.

## The model record

`model_record.json` is a `TrainingRecord` ([`registry/records.py`](../synthaea_ml/registry/records.py)),
written by [`write_training_record`](../synthaea_ml/registry/codec.py):

| Field | Content |
| --- | --- |
| `schema_version` | 3 |
| `trained_at` | UTC, `YYYY-MM-DDTHH:MM:SSZ` |
| `training_script` | The trainer, as a repository path, for example `synthaea_ml/training/train_linux.py` |
| `dataset_versions` | One `{name, baseline_sha256, sample_count}` per dataset, from its verified manifest. Must not be empty. |
| `scenario_replays` | `ScenarioReplayResult`s ([evaluation.md](evaluation.md#scenario-replay)) |
| `robustness_cards` | `RobustnessCard`s ([evaluation.md](evaluation.md#robustness)) |
| `hyperparameters` | The `IsolationForest` arguments, from `training/config.py` |
| `extra` | String-to-string run metadata; the site rarity trainer records its measurements here |
| `conformal_calibration` | Omitted when absent; see [calibration.md](calibration.md#conformal-threshold) |
| `feature_bounds` | Omitted when absent; the same bounds as `model_metadata.json` |

A dataset is bound by the hash of its content, not by its path. It can move without invalidating
the record, but it cannot change.

### Encoding

The record is written as `json.dumps(..., indent=2, sort_keys=True)` plus a newline. The release
gate hashes those bytes, so the format is a contract, pinned by
[`tests/test_training_record_golden.py`](../tests/test_training_record_golden.py).

Reading ([`load_training_record`](../synthaea_ml/registry/codec.py)) checks the type of every
value before building a dataclass, and raises `RecordFormatError` (a `ValueError`) naming the
offending key. Nested blocks must carry exactly their dataclass's keys; unknown top-level keys are
ignored. Schema versions 2 and 3 are readable: version 3 only added optional blocks.

## Provenance gate

[`verify_onnx`](../synthaea_ml/export/verify_onnx.py) is the check a model passes before it
ships. For each `model.onnx` given, or every `registry/*/*/model.onnx` by default:

1. **Runtime checks under onnxruntime.** The model has a `scores` output. The platform's sanity
   command lines ([features.md](features.md#sanity-samples)) are scored twice: every score is
   finite, both runs are identical, and when a `label` output exists, `label == -1` exactly when
   `score < 0`. The platform comes from the record's first dataset name, else from the model
   directory's `-linux` or `-windows` suffix; a model whose platform cannot be determined fails
   rather than be scored with the wrong samples.
2. **Provenance**, through [`verify_training_record`](../synthaea_ml/registry/provenance.py):
   - each dataset version is looked up as the one baseline directory named `<name>` or whose
     manifest derives `<name>` ([datasets.md](datasets.md#dataset-names)); its samples file
     (`baseline.jsonl`, `events.jsonl` or `correlation_train.jsonl`) must still match its
     manifest, and the manifest must still carry the hash and sample count the record locked;
   - each scenario replay's sidecar is looked up as `<scenarios checkout>/scenarios/<name>.yaml`
     and must still hash to `scenario_yaml_sha256`, and `passed` is recomputed from the recorded
     counts, so an edited `passed` is caught.

   Robustness cards are bound to their sidecar by hash in the record, but the gate does not
   re-verify that hash today.

A version without `model_record.json` passes with a warning, unless `SYNTHAEA_STRICT_PROVENANCE`
is `1`, `true` or `yes`; then it fails. Strict mode is the release setting. The command exits 1
when any model failed, or when there is no model to check.

```bash
synthaea-ml-verify-onnx                                   # every registry model
SYNTHAEA_STRICT_PROVENANCE=1 synthaea-ml-verify-onnx registry/cmdline-iforest-linux/0.2.0/model.onnx
```

The scenarios checkout is found through `SYNTHAEA_ROOT` ([testing.md](testing.md#sibling-checkouts)).
The sidecars are byte-protected in the scenarios repository: editing an existing one requires a
retrain of every model bound to it.

### What a shippable version needs

- `model.onnx`, `model_metadata.json`, `model_record.json` from a trainer, and a `card.md`;
- every dataset version present under `datasets/baselines/` with its manifest (found by name, see
  [datasets.md](datasets.md#dataset-names));
- `verify_onnx` passing with `SYNTHAEA_STRICT_PROVENANCE=1`;
- robustness cards that pass `synthaea-ml-robustness verify`, with `--check-regression` when a
  previous version exists.

## Committed models

| Model | Version | Tier | Files | Status |
| --- | --- | --- | --- | --- |
| [`cmdline-iforest-linux`](../registry/cmdline-iforest-linux/0.1.0/card.md) | 0.1.0 | T0 | `model.onnx`, `card.md` | Reference only |
| [`cmdline-iforest-windows`](../registry/cmdline-iforest-windows/0.1.0/card.md) | 0.1.0 | T0 | `model.onnx`, `card.md` | Reference only |

Both were trained before datasets were versioned and before the evaluation gates existed. They
have no record and no metadata, so they pass `verify_onnx` only in non-strict mode. They are the
ONNX oracle inputs for `verify_onnx` and its tests, not models to ship.

## Delivery to the agent

Models are content, delivered separately from the agent binary, and read from disk at load time.
The agent expects the registry layout under its state directory (`<state_dir>/ml/registry`), or
under `ml/registry` relative to its working directory in a source checkout, and reads
`model.onnx` with its optional `model_metadata.json`:

- A missing or invalid model degrades the agent to "no ML evidence"; detection continues.
- A model whose input width differs from the extractor's is refused.
- Without metadata, the threshold defaults to 0 and no bounds are checked (T0, T2). T1 requires
  both the threshold and the bounds for its 17 features, in order.
- The correlation scorer is the only model the agent loads today, as
  `correlation-iforest-<os>/0.1.0/`.

The agent's [on-device ML](https://github.com/synthaea-lab/agent/blob/main/docs/detection/ml.md)
page documents loading, scoring and the degraded modes in full.
