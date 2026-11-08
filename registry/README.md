# ml/registry — Model Registry

The models that ship. One directory per model, one subdirectory per version:

    registry/<model>/<version>/
      model.onnx           # the artifact (large binaries may move to LFS/release storage)
      card.md              # model card — REQUIRED, see below
      model_metadata.json  # conformal threshold + feature bounds the agent's scorer reads
      model_record.json    # machine-readable record: dataset hashes, replay results,
                           # robustness cards, hyperparameters (training_record.py)

`model.onnx`, `model_metadata.json` and `model_record.json` are written by the trainers
(`synthaea_ml.training.train_*`, file names in `synthaea_ml.registry.layout` and
`synthaea_ml.registry.training_record`). There is no separate `config.json`: the
hyperparameters a version was trained with are recorded in `model_record.json`, and the
defaults live in `synthaea_ml.training.config`. The two
committed `cmdline-iforest-{linux,windows}/0.1.0/` entries predate the record and hold
only `model.onnx` and `card.md`: `verify_onnx` passes them with a warning, and fails them
under `SYNTHAEA_STRICT_PROVENANCE=1`.

The model card records: purpose and tier, training data (dataset versions), metrics,
FP rate against benign baselines, scenario-replay results, known limitations, and the
parity-fixture hash it was verified against. No card, no ship — the control plane's
content distribution only picks up carded versions.
