"""The file names of a registry version directory, shared by the trainers that write a model
and the tools that load one (the record's own name is `training_record.MODEL_RECORD_FILENAME`).
"""

MODEL_FILENAME = "model.onnx"
METADATA_FILENAME = "model_metadata.json"
"""The conformal threshold and feature bounds the agent's scorer reads next to the model."""
