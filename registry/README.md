# registry

The versioned models, one directory per model and one subdirectory per version:

```text
registry/<model>/<version>/
    model.onnx           the model
    card.md              the model card, required for every version
    model_metadata.json  conformal threshold and feature bounds read by the agent's scorer
    model_record.json    dataset hashes, scenario replays, robustness cards, hyperparameters
```

The trainers in `synthaea_ml.training` write `model.onnx`, `model_metadata.json` and
`model_record.json`; the file names are defined in `synthaea_ml.registry`. Default
hyperparameters live in `synthaea_ml.training.config`, and the values a version was trained
with are recorded in its `model_record.json`.

A model card records the model's purpose and tier, its training data, metrics, false-positive
rate against benign baselines, scenario-replay results, known limitations, and the parity
fixtures it was verified against.

`python -m synthaea_ml.export.verify_onnx` checks every version under ONNX Runtime and
re-verifies the dataset and scenario hashes its record names. With
`SYNTHAEA_STRICT_PROVENANCE=1`, a version without a record fails.

## Models

| Model | Version | Tier | Status |
| --- | --- | --- | --- |
| [`cmdline-iforest-linux`](cmdline-iforest-linux/0.1.0/card.md) | 0.1.0 | T0 | Reference only |
| [`cmdline-iforest-windows`](cmdline-iforest-windows/0.1.0/card.md) | 0.1.0 | T0 | Reference only |

Both predate the evaluation gates and hold only `model.onnx` and `card.md`. They serve as
verification inputs and do not meet the ship criteria.
