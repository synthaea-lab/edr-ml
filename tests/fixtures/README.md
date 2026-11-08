# ml/tests/fixtures

Golden data for the Rust/Python parity seam (see `../README.md`). The files here are
committed and asserted against by the pytest suite and, for the parity goldens, by
`crates/ml` in synthaea-lab/agent. The scripts that write them live in `generators/`,
one `gen_<fixture>.py` per golden. Regenerate only on a deliberate change to a feature
definition, model recipe or wire format, never to paper over a red parity test.

Run a generator from the ml repository root, e.g.
`python tests/fixtures/generators/gen_features_golden.py`. Each one resolves its
output paths from its own location, so the working directory does not matter.

## Golden data (this directory)

| File | Written by | Checked by |
| --- | --- | --- |
| `features_golden.jsonl` | `gen_features_golden.py` | `test_features_golden.py`, `crates/ml/tests/features_golden.rs` |
| `rarity_golden.jsonl` | `gen_rarity_golden.py` | `test_rarity_golden.py`, `crates/ml/tests/rarity_golden.rs` |
| `common_set_snapshot_golden.json` | `gen_common_set_snapshot_golden.py` | `crates/ml/tests/common_set_snapshot_golden.rs`, the server's snapshot builder test |
| `capture_events.jsonl`, `capture_correlation_golden.jsonl` | `gen_capture_correlation_golden.py` | `test_capture_parity.py`, `crates/ml/tests/capture_parity.rs` |
| `t1_events.jsonl`, `t1_golden.jsonl` | `gen_t1_golden.py` | `test_t1_parity.py`, `crates/ml/tests/t1_golden.rs` |
| `mutation_parity_golden.jsonl` | `gen_mutation_parity_golden.py` | `test_mutation_parity.py` |
| `lineage_golden.jsonl` | no generator in this repo | `test_lineage_golden.py`, `crates/ml/tests/lineage_golden.rs` |

## Goldens written into synthaea-lab/agent

These generators write into the agent's `crates/ml/tests/fixtures/` (resolved through
`synthaea_ml.siblings`, so check the agent out next to this repo or set `SYNTHAEA_ROOT`):

| Generator | Writes |
| --- | --- |
| `gen_attribution_golden.py` | `isoforest_small.onnx`, `attribution_golden.json` |
| `gen_scorer_golden.py` | `cmdline_scorer.onnx`, `scorer_golden.json` |
| `gen_correlation_scorer_golden.py` | `correlation_scorer.onnx`, `correlation_scorer_golden.jsonl` |
| `gen_t1_scorer_golden.py` | `t1_scorer.onnx`, `t1_scorer_golden.jsonl` (reads `t1_golden.jsonl` from here: run `gen_t1_golden.py` first if it changed) |
