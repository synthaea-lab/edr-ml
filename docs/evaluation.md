# Evaluation

A model is not shipped on its training loss. This page covers the three evaluations in
[`synthaea_ml/evaluation/`](../synthaea_ml/evaluation): scenario replay (did the agent raise the
alerts a scenario expects), robustness (how far an attacker-style rewrite moves a known-bad
sample's score), and the lineage-gain measurement (does a feature block earn its place).

## Scenario replay

A scenario in the [scenarios repository](https://github.com/synthaea-lab/scenarios) is a script
plus a YAML sidecar listing the alerts it must raise, each with a `technique`, a `rule`, a
`min_count` and a `tolerance` ([sidecar format](https://github.com/synthaea-lab/scenarios/blob/main/docs/sidecar-format.md)).
[`scenario_replay.run_replay`](../synthaea_ml/evaluation/scenario_replay.py) turns a sidecar and
the `alerts.ndjson` of a run against it into a `ScenarioReplayResult`:

1. [`load_expected_detections`](../synthaea_ml/evaluation/scenario_replay.py) reads the sidecar
   with `yaml.safe_load`. `name`, `expected_detections` and all four keys of every entry are
   required; a missing key raises `KeyError`.
2. Alert lines are counted by their `technique` string. Blank and unparsable lines are skipped,
   since `alerts.ndjson` is an append-only log that may end on a torn line.
3. For each expected detection with a non-zero count, an observation is recorded with that count
   and the expected `rule`. Alerts carry no rule identifier, so the rule is copied, not observed;
   two expected entries with the same technique would share one count.
4. [`compute_passed`](../synthaea_ml/registry/records.py): every expected detection must be
   observed with `min_count <= count <= min_count + tolerance`. An over-count fails, as an
   under-count does.

The result also records the sidecar's SHA-256, the time, and the environment
(`platform.system()`, `platform.release()`, `platform.machine()`). Techniques that fired but are
not expected are ignored.

`run_replay` does not run the scenario; it starts from an `alerts.ndjson` produced elsewhere. No
command-line entry point calls it yet, and no trainer writes replays into a record. When a record
does carry replays, the provenance gate re-verifies them ([registry.md](registry.md#provenance-gate)).

## Robustness

[`run_robustness_evaluation`](../synthaea_ml/evaluation/robustness.py) measures evasion cost: take
samples the model detects, rewrite them the way an attacker cheaply could, and see how many
escape. It produces one `RobustnessCard` per scenario.

### Samples under test

- With an events source (`--robustness-events` on the trainers): every exec record in the file.
  The file may be the agent's wire format (flattened first) or a baseline. At tier T1, each exec's
  `correlation_features` is computed over the whole file as one window, filtered by
  `(pid, process_generation)`.
- Without one, or if it holds no exec record: one synthetic command line per expected detection
  of the scenario, chosen by technique (a `curl ... | sh` for T1071, a `base64 -d` pipeline for
  T1059, a `wget` to `/tmp` for T1105, a generic one otherwise).

The sidecar must list at least one expected detection.

### Mutators

Every mutator works on the raw record before canonicalization, returns a copy, runs at three
intensities (light, medium, heavy), and draws from a seeded PRNG. Heavier must mean harder to
detect, never easier.

| Tier | Mutators in use |
| --- | --- |
| T0 | [`PathSubstitutionMutator`](../synthaea_ml/evaluation/mutations/cmdline.py) (`/bin/` and `/usr/bin/`, `/tmp/` and `/var/tmp/`, ...), [`ArgumentReorderMutator`](../synthaea_ml/evaluation/mutations/cmdline.py) (argv[0] stays fixed) |
| T1 | The same command-line mutators. The parent is not in the T1 vector, so a lineage mutator could only report a flattering zero. |
| T2 | None yet; the evaluation raises `NotImplementedError`. |

Three more command-line mutators exist but are disabled: base64-encoding tokens, splitting tokens
with shell quotes and padding with comments are shell-level evasions, and the sensors capture argv
after the shell has parsed it, so they produce argv values that cannot occur. They belong in a
future mutator of `sh -c` payloads. The [lineage mutators](../synthaea_ml/evaluation/mutations/lineage.py)
(fake parent name, parent path moved to system directories, lineage removed) serve the lineage
measurement and its tests.

The PRNG is [Knuth's 64-bit LCG](../synthaea_ml/evaluation/mutations/prng.py), with the same
constants as the Rust robustness tests. The draws depend on the loop order (sample, then mutator,
then intensity), and each result records the PRNG state after its mutation, so a result can be
reproduced individually. A mutation that raises `ValueError` is skipped.

### The card

For each mutation, both scores are recorded with their difference. A sample is **detected** when
its score is at or below the threshold (default 0, the forest's decision boundary; the site rarity
floor passes each model's calibrated threshold), and a mutation **escaped** when the original was
detected and the mutated score is above the threshold.

| Field | Meaning |
| --- | --- |
| `escape_rate` | Escaped results over results whose original was detected |
| `median_score_degradation` | Median of `mutated - original` over all results; positive moved toward benign |
| `worst_case_degradation` | Largest such difference |
| `mutation_results` | Every result: class, intensity, both scores, delta, threshold, escaped, PRNG state |
| `scenario_yaml_sha256`, `tested_at` | Binding to the sidecar, and when it ran |

If no sample was detected before mutation, the evaluation raises `ValueError`: an escape rate on
nothing is undefined, and the usual cause is a benign file passed as the events source.
[`tests/test_robustness_snapshot.py`](../tests/test_robustness_snapshot.py) pins a full card.
The provenance gate re-checks `scenario_yaml_sha256` against the sidecar on disk, so a card
computed against a scenario that has since changed fails it ([registry.md](registry.md#provenance-gate)).

### The gate

[`robustness_cli`](../synthaea_ml/evaluation/robustness_cli.py) reads the cards of a version's
`model_record.json`:

```bash
synthaea-ml-robustness verify --model-dir registry/cmdline-iforest-linux/0.2.0 --check-regression
synthaea-ml-robustness report --model-dir registry/cmdline-iforest-linux/0.2.0 --output robustness.md
```

`verify` fails a scenario whose escape rate exceeds `--max-escape-rate` (default 0.15) or whose
median degradation exceeds `--max-median-degradation` (default 0.20). With `--check-regression`
it also compares each scenario with the same scenario in the previous version, and fails an
escape-rate increase above `--max-regression-increase` (default 0.05). The previous version is
the previous patch directory, else a version of the previous minor, else of the previous major; a
missing or unreadable one is a warning, not a failure. A record with no card does not pass. The
exit status is 0 on pass, 1 on failure.

`report` writes Markdown: per scenario, the three metrics and the first ten mutation results.

## Lineage gain

[`t1_lineage_gain`](../synthaea_ml/evaluation/t1_lineage_gain.py) answers one question: does
adding the six [lineage features](features.md#lineage) to the T1 vector detect more at the same
false-positive budget? It is the measurement behind the T1 vector having no lineage block, and the
tool to re-run on better data.

- **Benign:** one or more behavior datasets (`events.jsonl` and manifest), reduced to distinct
  23-feature vectors (T1 then lineage). At least 20 are required.
- **Malicious:** a capture of a scenario run. Only the process tree of the runner counts: its
  incarnation first seen within 30 s before `--start-ns`, and every process first seen inside the
  window whose parent incarnation is already in the tree, excluding the runner itself. Background
  processes that ran during the window are not labeled malicious. The capture workloads write the
  three values to `run.json`.
- **Method:** for each of `--seeds` splits (default 20) of the benign vectors into 50% train, 25%
  calibration and 25% test, fit `BEHAVIOR_FOREST` with that seed on 17 and on 23 columns,
  calibrate each to the trainer's FP budget, and measure the detection rate on the malicious
  vectors, the false-positive rate on the benign test rows, and the ROC AUC.
- **Output:** a JSON report with the means and standard deviations per variant, the paired gain
  (23 minus 17) for AUC and detection, the number of seeds on which lineage is not worse, and the
  counts of distinct benign and malicious vectors.

```bash
synthaea-ml-t1-lineage-gain \
    --benign-dataset <benign-dataset-dir> \
    --malicious-capture <malicious-run>/events.jsonl \
    --root-pid 4242 --start-ns 1762700000000000000 --end-ns 1762700300000000000 \
    --output t1_lineage_gain.json
```

Report a gain with the vector counts next to it, and separately for scenarios whose parent is odd
by name, odd by path, and ordinary: a lineage gain can only show where the parent is unusual. The
report is not written into a model record.
