"""The file names of a registry version directory, shared by the trainers that write a model
and the tools that load one.
"""

MODEL_FILENAME = "model.onnx"
METADATA_FILENAME = "model_metadata.json"
"""The conformal threshold and feature bounds the agent's scorer reads next to the model."""
MODEL_RECORD_FILENAME = "model_record.json"
"""The machine-readable record binding the model to its datasets and scenarios."""
