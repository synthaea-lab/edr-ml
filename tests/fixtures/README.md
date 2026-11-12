# tests/fixtures

Golden data for the Rust/Python parity tests. The same files are asserted against by the
pytest suite here and by the agent's [`ml` crate](https://github.com/synthaea-lab/agent/tree/main/crates/ml),
so changing one is a coordinated change to both repositories.

Each golden is written by a script in `generators/`. Regenerate only on a deliberate change to
a feature definition, model recipe or wire format, never to make a failing parity test pass.
A generator resolves its paths from its own location, so it runs from any directory:

```bash
python tests/fixtures/generators/gen_features_golden.py
```

## Goldens in this directory

| File | Written by | Checked by |
| --- | --- | --- |
| `features_golden.jsonl` | `gen_features_golden.py` | `test_features_golden.py`, `crates/ml/tests/features_golden.rs` |
| `rarity_golden.jsonl` | `gen_rarity_golden.py` | `test_rarity_golden.py`, `crates/ml/tests/rarity_golden.rs` |
| `common_set_snapshot_golden.json` | `gen_common_set_snapshot_golden.py` | `crates/ml/tests/common_set_snapshot_golden.rs`, the server's snapshot builder test |
| `capture_events.jsonl`, `capture_correlation_golden.jsonl` | `gen_capture_correlation_golden.py` | `test_capture_parity.py`, `crates/ml/tests/capture_parity.rs` |
| `t1_events.jsonl`, `t1_golden.jsonl` | `gen_t1_golden.py` | `test_t1_parity.py`, `crates/ml/tests/t1_golden.rs` |
| `mutation_parity_golden.jsonl` | `gen_mutation_parity_golden.py` | `test_mutation_parity.py` |
| `lineage_golden.jsonl` | no generator in this repository | `test_lineage_golden.py`, `crates/ml/tests/lineage_golden.rs` |
| `behavior_golden.jsonl` | `gen_behavior_golden.py` | `test_behavior_golden.py`, the agent's `golden_tests::parity_with_behavior_py` in `crates/correlator/src/behavior.rs` (agent PRs #220, and #243 for the rows that read `daddr_v6` and `image_path`) |

`behavior_golden.jsonl` follows the Rust `BehaviorVector` where the two sides used to disagree
(agent#203, ml#51): an IPv6 destination outside the five non-global classes is external, an IPv4-mapped one is its IPv4 address (`daddr_v6` text next to
`is_ipv6: true`), `0.0.0.0` and `::` are internal, path casing is ASCII-only, and the path is the
exec's `image_path`. An exec row without `image_path` pins the documented fallback to the first
NUL-separated `cmdline` argument, so a consumer must read `image_path` when the row has one.

## Goldens written into the agent

These generators write into the agent's `crates/ml/tests/fixtures/`, found through the
sibling checkout described in [CONTRIBUTING.md](../../CONTRIBUTING.md#sibling-checkouts):

| Generator | Writes |
| --- | --- |
| `gen_attribution_golden.py` | `isoforest_small.onnx`, `attribution_golden.json` |
| `gen_scorer_golden.py` | `cmdline_scorer.onnx`, `scorer_golden.json` |
| `gen_correlation_scorer_golden.py` | `correlation_scorer.onnx`, `correlation_scorer_golden.jsonl` |
| `gen_t1_scorer_golden.py` | `t1_scorer.onnx`, `t1_scorer_golden.jsonl`; reads `t1_golden.jsonl`, so run `gen_t1_golden.py` first if it changed |
