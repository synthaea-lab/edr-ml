# Reproducibility

A registry model must be rebuildable from what its record names: the dataset versions, the
trainer and the hyperparameters. This page lists what makes a run repeatable, what pins it, and
what still varies.

## What fixes a run

| Input | Fixed by |
| --- | --- |
| Training data | Each dataset version's `baseline_sha256` and `sample_count`, verified against the manifest before training and again by the provenance gate ([registry.md](registry.md#provenance-gate)) |
| Sample order | Deduplication keeps first occurrences in file order ([`unique_in_order`](../synthaea_ml/data/baselines.py)); the T1 trainer keeps first-seen vector order; `train_site_rarity` sorts by time |
| Calibration split | `train_test_split` with `RANDOM_SEED` (42) for the trainers; a time split for `train_site_rarity`; seeds 0 to N-1 for `t1_lineage_gain` |
| Forest | `random_state=RANDOM_SEED` in every `IsolationForestConfig`, recorded in `hyperparameters` |
| Calibration budget | `training/config.py`, and the values recorded in `conformal_calibration` |
| Mutations | The LCG seeded with `RANDOM_SEED`, the fixed loop order, and the PRNG state recorded per result |
| Export | skl2onnx at a pinned target opset |
| Scenario expectations | The sidecar's SHA-256 in every replay and robustness card |
| Trainer | `training_script` in the record, and the commit of this repository the model was trained at |

Shared values are defined once in [`training/config.py`](../synthaea_ml/training/config.py), so
the trainers and the evaluation tools that refit a model the same way cannot drift apart.
[`tests/test_training_config.py`](../tests/test_training_config.py) pins them.

## What pins it

- [`tests/test_trainer_artifacts.py`](../tests/test_trainer_artifacts.py) requires the T0 and
  T1 trainers to write exactly the bytes a direct scikit-learn, skl2onnx and calibration
  computation produces, for `model.onnx`, `model_metadata.json` and the record. The reference is
  recomputed in the same run rather than stored, so the pin holds across library versions and
  operating systems.
- [`tests/test_training_record_golden.py`](../tests/test_training_record_golden.py) pins the
  record's on-disk bytes, which the release gate hashes.
- [`tests/test_robustness_snapshot.py`](../tests/test_robustness_snapshot.py) pins a full
  robustness card, mutation by mutation.
- The parity goldens pin every feature vector to the agent's ([testing.md](testing.md#golden-fixtures)).

## Bytes on disk

A content hash depends on line endings. [`.gitattributes`](../.gitattributes) forces LF on
dataset files and manifests, and on the golden fixtures, so a Windows checkout with
`core.autocrlf` does not change a single hash. The scenario sidecars are protected the same way
in the scenarios repository. All hashing goes through one streamed helper,
[`hashing.sha256_file`](../synthaea_ml/hashing.py), so every binding hashes the same bytes the
same way.

## What still varies

- **Timestamps.** `trained_at`, `calibrated_at`, `tested_at` and `run_at` record when a step ran.
  Two runs of the same trainer produce records that differ in those fields and nowhere else.
  Compare records with the timestamps removed.
- **Library versions.** The package requires minimum versions of numpy, scikit-learn, skl2onnx
  and onnxruntime, not exact ones. A different scikit-learn can build a different forest from the
  same seed. To rebuild a model bit for bit, use the versions it was trained with; record them in
  the model card.
- **Environment of a replay.** A scenario replay records the operating system, kernel release and
  architecture it ran on, because alert counts can depend on them.

## Rebuilding a model

1. Check out the repository at the commit the model was trained at, and install the library
   versions noted in its card.
2. Make sure every dataset in `dataset_versions` is present under `datasets/baselines/<name>/`
   and verifies: `synthaea-ml-manifest verify datasets/baselines/<name>`.
3. Run the `training_script` with those datasets and a scratch `--output-dir`, with the same
   robustness scenarios and events if the record has cards.
4. Compare `model.onnx` and `model_metadata.json` byte for byte, and the record without its
   timestamps.
