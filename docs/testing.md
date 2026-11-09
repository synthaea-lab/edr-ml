# Testing

The suite under [`tests/`](../tests) is plain pytest. It has three jobs: pin the Python side of
every seam with the agent (features, canonical command line, inference, attribution), pin the
bytes of everything a gate hashes, and exercise each entry point end to end on synthetic data.
CI runs it on Linux, Windows and macOS ([development.md](development.md#continuous-integration)).

```bash
pytest tests                       # everything
pytest tests/test_features_golden.py -q
SYNTHAEA_REQUIRE_SIBLINGS=1 pytest tests   # as CI runs it
```

## What the suite covers

| Area | Tests |
| --- | --- |
| Feature parity | `test_features_golden`, `test_capture_parity`, `test_t1_parity`, `test_lineage_golden`, `test_rarity_golden`, `test_behavior_golden`, `test_mutation_parity`, `test_canonical`, `test_attribution_golden` |
| Feature units | `test_correlation_features`, `test_behavior_features`, `test_combined_features`, `test_rarity` |
| Data | `test_jsonl`, `test_jsonl_readers`, `test_wire`, `test_baselines`, `test_hashing`, `test_manifest`, `test_manifest_cli` |
| Training | `test_trainers`, `test_train_behavior`, `test_train_correlation`, `test_train_site_model`, `test_training_pipeline`, `test_training_config`, `test_training_floor`, `test_trainer_artifacts` |
| Site rarity floors | `test_rarity_floor`, `test_rarity_global_floor`, `test_site_rarity_messages`, `test_site_rarity_record` |
| Calibration | `test_calibrate_conformal` |
| Evaluation | `test_scenario_replay`, `test_robustness`, `test_robustness_snapshot`, `test_robustness_cli`, `test_mutations_cmdline`, `test_mutations_lineage`, `test_mutations_prng`, `test_t1_lineage_gain`, `test_odd_parent_scenarios` |
| Registry | `test_training_record`, `test_training_record_golden`, `test_record_codec`, `test_verify_onnx`, `test_verify_provenance` |
| Structure | `test_import_direction`, `test_cli_entry_points`, `test_cli_logging` |

The training and evaluation tests use synthetic captures, baselines and corpora built by
[`tests/helpers.py`](../tests/helpers.py) and the fixtures in
[`tests/conftest.py`](../tests/conftest.py). They prove the pipeline runs and what it writes, not
how well a model detects anything.

Two structural tests guard the package shape:
[`test_cli_entry_points.py`](../tests/test_cli_entry_points.py) requires every module with a
`__main__` block to be a console script and every `main` to take `argv` and return the exit
status; [`test_import_direction.py`](../tests/test_import_direction.py) keeps the training
scripts out of `verify_onnx` and the lineage-gain tool ([architecture.md](architecture.md#dependency-direction)).

## Golden fixtures

A golden is a file of inputs and the exact outputs the reference implementation produces. The
pytest suite here and the agent's `crates/ml` tests assert against the same file, so a drift on
either side fails a test. The [fixtures README](../tests/fixtures/README.md) maps each golden to its
generator and the tests that read it.

| Seam | Golden | Lives in |
| --- | --- | --- |
| Command-line features | `features_golden.jsonl` | this repository |
| Correlation features from a real capture | `capture_events.jsonl`, `capture_correlation_golden.jsonl` | this repository |
| T1 vector, per incarnation | `t1_events.jsonl`, `t1_golden.jsonl` | this repository |
| Lineage features | `lineage_golden.jsonl` | this repository |
| Rarity features | `rarity_golden.jsonl` | this repository |
| Common-set snapshot format | `common_set_snapshot_golden.json` | this repository |
| Behavior vector | `behavior_golden.jsonl` | this repository |
| Mutated feature vectors | `mutation_parity_golden.jsonl` | this repository |
| Attribution | `isoforest_small.onnx`, `attribution_golden.json` | the agent's `crates/ml/tests/fixtures/` |
| T0 inference | `cmdline_scorer.onnx`, `scorer_golden.json` | the agent |
| T2 inference | `correlation_scorer.onnx`, `correlation_scorer_golden.jsonl` | the agent |
| T1 inference | `t1_scorer.onnx`, `t1_scorer_golden.jsonl` | the agent |

The behavior and mutation-parity goldens are read only by the Python suite today; the other
goldens of this repository are asserted on both sides. The inference goldens pin the agent's `ort` scores to onnxruntime's in Python for the same
model and the same extractor. The attribution golden pins the agent's tree-path attribution to
the independent Python reference in [`tests/attribution_reference.py`](../tests/attribution_reference.py),
which parses the ONNX file with the `onnx` package. The fixture models are trained on synthetic
vectors: they pin agreement between the two sides, not detection quality.

### Regenerating a golden

Each generator in [`tests/fixtures/generators/`](../tests/fixtures/generators) resolves its paths
from its own location and runs from any directory:

```bash
python tests/fixtures/generators/gen_features_golden.py
```

Generators that write into the agent find it through `SYNTHAEA_ROOT`. `gen_t1_scorer_golden.py`
reads `t1_golden.jsonl`, so run `gen_t1_golden.py` first when that changed. `lineage_golden.jsonl`
has no generator in this repository.

Regenerate only on a deliberate change to a feature definition, a model recipe or a wire format,
and open the matching pull request on the agent at the same time. Never regenerate to make a
failing parity test pass: the failure is the drift the golden exists to catch.

## Sibling checkouts

Some tests read files owned by other public repositories: the agent's
`crates/ml/tests/fixtures/` (attribution), and the scenarios repository's `scenarios/` sidecars
and `ml-capture/odd-parents/` scripts (robustness, site floors, lineage gain, odd-parent
declarations). [`synthaea_ml/siblings.py`](../synthaea_ml/siblings.py) locates them under
`SYNTHAEA_ROOT`, which defaults to the parent of this repository:

```text
<root>/
    ml/          this repository
    agent/
    scenarios/
```

A test whose sibling is missing is skipped through [`tests/siblings.py`](../tests/siblings.py),
unless `SYNTHAEA_REQUIRE_SIBLINGS=1`, in which case collection fails. CI checks out both siblings
(sparse: the agent's ML fixtures, the scenarios' `scenarios/` and `ml-capture/`) and sets the
variable, so a parity test can never go quietly missing there.

## Writing a test

- Prefer synthetic inputs built in `tmp_path`. Do not commit real captures as test data.
- A builder used by several modules goes in `helpers.py`; one that needs pytest state goes in
  `conftest.py` as a fixture.
- A test that needs a sibling uses `needs(repo, *parts)` from `tests/siblings.py`, never a bare
  `skipif`, so CI's strict mode applies to it.
- Tests may use `assert`, `/tmp` paths and seeded non-cryptographic randomness; the lint
  exceptions for `tests/**` in `pyproject.toml` allow exactly that.
