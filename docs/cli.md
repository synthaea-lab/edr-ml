# Command-line reference

`pip install -e .` installs one `synthaea-ml-*` command per entry point, listed in
`[project.scripts]` of [`pyproject.toml`](../pyproject.toml). Each one is also runnable as
`python -m synthaea_ml.<package>.<module>`, with the same arguments, and documents them with
`--help`. Run commands from the repository root: relative paths, and the registry and baselines
that `verify_onnx` scans, resolve from there.

| Command | Module | Purpose |
| --- | --- | --- |
| [`synthaea-ml-capture-to-baseline`](#synthaea-ml-capture-to-baseline) | `data.capture_to_baseline` | Agent capture to a deduplicated command-line baseline |
| [`synthaea-ml-aggregate-correlation`](#synthaea-ml-aggregate-correlation) | `data.aggregate_correlation` | Agent capture to one correlation vector per pid |
| [`synthaea-ml-manifest`](#synthaea-ml-manifest) | `data.manifest` | Write or verify a dataset manifest |
| [`synthaea-ml-train-linux`, `synthaea-ml-train-windows`](#synthaea-ml-train-linux-synthaea-ml-train-windows) | `training.train_linux`, `training.train_windows` | Train a T0 command-line model |
| [`synthaea-ml-train-behavior`](#synthaea-ml-train-behavior) | `training.train_behavior` | Train the T1 behavior model |
| [`synthaea-ml-train-correlation`](#synthaea-ml-train-correlation) | `training.train_correlation` | Train the correlation scorer |
| [`synthaea-ml-train-site-model`](#synthaea-ml-train-site-model) | `training.train_site_model` | Retrain a T0 model on a site corpus |
| [`synthaea-ml-train-site-rarity`](#synthaea-ml-train-site-rarity) | `training.train_site_rarity` | Site model with the rarity features, behind floor checks |
| [`synthaea-ml-calibrate-llr`](#synthaea-ml-calibrate-llr) | `calibration.calibrate_llr` | Empirical likelihood ratios for the correlator |
| [`synthaea-ml-record-replay`](#synthaea-ml-record-replay) | `evaluation.record_replay` | Record a scenario run in a model record |
| [`synthaea-ml-robustness`](#synthaea-ml-robustness) | `evaluation.robustness_cli` | Gate and report on robustness cards |
| [`synthaea-ml-t1-lineage-gain`](#synthaea-ml-t1-lineage-gain) | `evaluation.t1_lineage_gain` | Measure what lineage features add to T1 |
| [`synthaea-ml-verify-onnx`](#synthaea-ml-verify-onnx) | `export.verify_onnx` | Runtime and provenance check of registry models |

All of them exit 2 on a usage error (argparse). Progress and warnings go to stderr; a command's
own output goes to stdout ([development.md](development.md#command-line-conventions)).

## Data

### `synthaea-ml-capture-to-baseline`

```text
synthaea-ml-capture-to-baseline EVENTS_FILE [--out PATH]
```

Reads the `exec` events of an agent capture, writes one `{"argv": [...]}` line per distinct argv
to `--out` (default `baseline_benign.jsonl`), and prints the counts. Invalid lines are skipped
with a warning. Exit 1 if the capture does not exist. See [datasets.md](datasets.md#command-line-baseline).

### `synthaea-ml-aggregate-correlation`

```text
synthaea-ml-aggregate-correlation [--events PATH] [--out PATH] [--window SECONDS] [--min-events N]
```

Writes one correlation vector per pid of `--events` (default `events.jsonl`), over the pid's
window at its last event (`--window`, default 60), skipping pids with fewer than `--min-events`
(default 1). Pass `--out <dataset-dir>/correlation_train.jsonl`; without it the file goes inside
the package, with a deprecation warning. Exit 1 if the capture is missing or holds no usable
event. See [datasets.md](datasets.md#correlation-vectors).

### `synthaea-ml-manifest`

```text
synthaea-ml-manifest write BASELINE_DIR --platform {linux,windows,macos} --os-version TEXT
    --workload-label {desktop-user,dev,admin,server-idle}
    --capture-start ISO8601 --capture-end ISO8601
    [--baseline-filename NAME] [--capturer-ref TEXT] [--hostname NAME]
synthaea-ml-manifest verify BASELINE_DIR
```

`write` hashes the samples file (`baseline.jsonl` unless `--baseline-filename`) and writes
`manifest.json`. Capture times need a timezone (`Z` or an offset). `--hostname` overrides the
machine's own name for `host_id`; only its hash is stored. `verify` re-hashes `baseline.jsonl`
and recounts its lines. Exit 1 on a missing file or a mismatch. See [datasets.md](datasets.md#the-manifest).

## Training

### `synthaea-ml-train-linux`, `synthaea-ml-train-windows`

```text
synthaea-ml-train-linux --dataset DIR [DIR ...] --output-dir DIR
    [--robustness-scenarios YAML ...] [--robustness-events PATH]
```

Each `--dataset` holds `baseline.jsonl` and `manifest.json`. Writes `model.onnx`,
`model_metadata.json` and `model_record.json` to `--output-dir`, and prints the split, the
threshold, each robustness card and the sanity-sample scores. A manifest mismatch or a failed
robustness evaluation raises and exits non-zero. See [training.md](training.md#t0-command-line).

### `synthaea-ml-train-behavior`

```text
synthaea-ml-train-behavior --dataset DIR [DIR ...] --output-dir DIR
    [--robustness-scenarios YAML ...] [--robustness-events PATH]
```

Each `--dataset` holds `events.jsonl` and its manifest. `--robustness-events` is a capture of
malicious activity in the agent's format. Fails with fewer than 20 distinct behavior vectors. See
[training.md](training.md#t1-behavior).

### `synthaea-ml-train-correlation`

```text
synthaea-ml-train-correlation --dataset DIR --output-dir DIR
```

`--dataset` holds `correlation_train.jsonl` and its manifest; the two options are required
together. Without both, the deprecated in-package run. See [training.md](training.md#t2-correlation-scorer).

### `synthaea-ml-train-site-model`

```text
synthaea-ml-train-site-model --global-dataset DIR --site-corpus PATH --output-dir DIR
    --site-name NAME [--robustness-scenarios YAML] [--global-model PICKLE]
```

The global-model floor runs only with both `--robustness-scenarios` (one sidecar) and
`--global-model`; a failure exits non-zero before anything is exported. `--global-model` is
unpickled: pass only a model you trained. See [training.md](training.md#train_site_model).

### `synthaea-ml-train-site-rarity`

```text
synthaea-ml-train-site-rarity --site-corpus PATH --site-name NAME --output-dir DIR
    [--k N] [--min-hosts N] [--interval-hours H] [--withhold-fraction F]
    [--robustness-scenarios YAML ...] [--global-model-dir DIR]
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--k` | 3 | Hosts an image must have run on to be common |
| `--min-hosts` | 5 | Hosts a snapshot must cover before rarity is known |
| `--interval-hours` | 24 | Snapshot publication cadence |
| `--withhold-fraction` | 0 | Share of training rows whose rarity is set to withheld, in `[0, 1)` |
| `--robustness-scenarios` | none | Sidecars for the scenario part of each floor |
| `--global-model-dir` | none | Shipped global model to compare with; without it the record says not checked |

Writes nothing if a floor fails, and exits non-zero listing every failure. Needs at least 60
exec events. See [training.md](training.md#train_site_rarity).

## Calibration and evaluation

### `synthaea-ml-calibrate-llr`

```text
synthaea-ml-calibrate-llr [--events PATH] [--alerts PATH] [--window SECONDS] [--min-mal N] [--out PATH]
```

Defaults: `events.jsonl`, `alerts.ndjson`, a 60 s window, at least 1 malicious pid. Writes the
report to `--out`, or stdout. Exit 1 when the alerts name fewer malicious pids than `--min-mal`.
See [calibration.md](calibration.md#likelihood-ratio-calibration).

### `synthaea-ml-record-replay`

```text
synthaea-ml-record-replay --model-dir DIR --scenario NAME --alerts PATH
    [--scenarios-root DIR] [--environment OS KERNEL ARCH]
```

Matches the `alerts.ndjson` of a run of scenario `NAME` against its sidecar
(`--scenarios-root`, default the `scenarios` sibling checkout's `scenarios/`) and adds the replay
to `model_record.json`, replacing an earlier replay of the same scenario in the same environment.
`--environment` names the machine the scenario ran on (a rig VM); without it, this machine is
recorded. Prints the outcome and exits 0 whether the replay passed or failed. Exit 1 if the
record, the sidecar or the alerts are missing, or the sidecar's `name` is not `NAME`. See
[evaluation.md](evaluation.md#recording-a-replay).

### `synthaea-ml-robustness`

```text
synthaea-ml-robustness verify --model-dir DIR [--max-escape-rate R] [--max-median-degradation D]
    [--check-regression] [--max-regression-increase R]
synthaea-ml-robustness report --model-dir DIR [--output PATH]
```

`verify` exits 1 when a gate fails or the record has no robustness card. `--no-regression` is an
older spelling of `--check-regression` and has the same effect. See [evaluation.md](evaluation.md#the-gate).

### `synthaea-ml-t1-lineage-gain`

```text
synthaea-ml-t1-lineage-gain --benign-dataset DIR [DIR ...] --malicious-capture PATH
    --root-pid PID --start-ns NS --end-ns NS [--seeds N] --output PATH
```

Writes the JSON report to `--output` and prints the AUC and detection summary. See
[evaluation.md](evaluation.md#lineage-gain).

### `synthaea-ml-verify-onnx`

```text
[SYNTHAEA_STRICT_PROVENANCE=1] synthaea-ml-verify-onnx [MODEL.onnx ...]
```

Checks the given models, or every `registry/*/*/model.onnx`. Exit 1 if any fails or none is
found. See [registry.md](registry.md#provenance-gate).

## Environment variables

| Variable | Read by | Effect |
| --- | --- | --- |
| `SYNTHAEA_ROOT` | `verify_onnx`, the tests, the fixture generators | Directory holding the sibling `agent/` and `scenarios/` checkouts; default: the parent of this repository |
| `SYNTHAEA_STRICT_PROVENANCE` | `verify_onnx` | `1`, `true` or `yes`: a model without `model_record.json` fails |
| `SYNTHAEA_REQUIRE_SIBLINGS` | the tests | `1`: a test whose sibling is missing fails instead of being skipped |
