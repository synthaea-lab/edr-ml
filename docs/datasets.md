# Datasets

Every model is trained on a dataset directory: one samples file and a `manifest.json` that
records where the samples came from and locks their bytes with a SHA-256. This page covers the
input the datasets are built from (the agent's event capture), the three dataset shapes, the
manifest, the per-site corpus, and the datasets committed in this repository.

## Where the data comes from

All training data is recorded by the agent itself, so the pipeline learns from exactly the
events the endpoint will later score:

- `agent capture-events --output events.jsonl` writes every normalized event as one JSON object
  per line. It is the input of every dataset builder below.
- The scenarios repository holds the [capture
  workloads](https://github.com/synthaea-lab/scenarios/blob/main/docs/ml-capture.md): a benign
  workload that runs ordinary jobs from many kinds of parent process, and a malicious run that
  executes the attack scenarios inside one measured window and writes a `run.json` with the
  runner's pid and the window's start and end in nanoseconds. That `run.json` is what the
  [lineage-gain evaluation](evaluation.md#lineage-gain) takes as `--root-pid`, `--start-ns` and
  `--end-ns`.

Raw captures and labeled traces are kept outside this repository. Only derived baselines with
a manifest are committed (see [committed baselines](#committed-baselines)).

## The agent's event format

A capture line is the agent's `schema::Event` serialized as JSON. Process identity is nested
under `meta`, and the event kind is a snake_case `type` tag:

```json
{"type": "exec", "meta": {"pid": 4242, "ppid": 1, "timestamp_ns": 1757420000000000000, "process_generation": 3},
 "argv": ["curl", "-fsS", "http://localhost/health"], "cmdline": "...", "image_path": "/usr/bin/curl",
 "parent_comm": "bash", "parent_image_path": "/usr/bin/bash"}
{"type": "connect", "meta": {"pid": 4242, "timestamp_ns": 1757420000100000000}, "daddr": "127.0.0.1", "dport": 80}
{"type": "file_open", "meta": {"pid": 4242, "timestamp_ns": 1757420000200000000}, "path": "/tmp/out", "flags": 577}
```

The feature extractors were written against a flat shape instead (`pid` and `ts_ns` at top
level, a `fileopen` tag). [`data/wire.py`](../synthaea_ml/data/wire.py) is the single place
that translates one into the other, with two translators that deliberately disagree:

| Translator | Used by | Behavior |
| --- | --- | --- |
| `flatten_wire_event` | `aggregate_correlation`, `behavior_dataset`, robustness on a capture | Keeps only `exec`, `connect`, `file_open` and `network_flow`; drops events without `meta`; keeps `process_generation` and, on `exec`, `argv`, `cmdline`, `image_path`, `parent_comm`, `parent_image_path`; keeps `daddr` verbatim as a one-element `daddr_v4`. This is the parity path the Rust side is tested against. |
| `normalize_for_behavior` | `calibrate_llr` | Keeps every field and unknown types, passes already-flat events through, defaults a missing pid or timestamp to 0, and splits a dotted-decimal `daddr` into four octets so the behavior features can tell internal from external destinations. |

[`tests/test_wire.py`](../tests/test_wire.py) pins both and the inputs on which they differ.

### Reading JSON Lines

Every reader goes through [`iter_jsonl`](../synthaea_ml/data/jsonl.py), and every caller states
what happens to a line that is not a JSON object: `RAISE`, `SKIP` or `WARN`. Lines are split on
`\n` only, never on Unicode line separators, because the agent's serializer does not escape a
raw U+2028 inside a string. Blank lines are always ignored. A torn last line, as an interrupted
capture leaves, is an invalid line like any other.

| Reader | Policy |
| --- | --- |
| Baselines for the T0 trainers ([`data/baselines.py`](../synthaea_ml/data/baselines.py)) | `RAISE` |
| `capture_to_baseline`, `aggregate_correlation` | `WARN` |
| `behavior_dataset`, robustness events source | `SKIP` |

### The canonical command line

The command-line features split tokens on NUL. Training and inference must therefore build the
same string from an exec record, and [`data/canonical.py`](../synthaea_ml/data/canonical.py)
is the Python mirror of the agent's `ExecEvent::ml_cmdline`:

- with a non-empty `argv` (Linux): each token followed by `\0`, like `/proc/<pid>/cmdline`;
- without `argv` (Windows ETW): the flat `cmdline` verbatim, as one token, with no terminator.

A space-joined command line would collapse `token_count` to 1 and inflate `max_token_length`,
so no consumer builds its input from the sensor's display string.
[`tests/test_canonical.py`](../tests/test_canonical.py) pins the mirror.

## Dataset shapes

| Shape | Samples file | Built by | Trained by |
| --- | --- | --- | --- |
| Command-line baseline | `baseline.jsonl` | [`capture_to_baseline`](../synthaea_ml/data/capture_to_baseline.py) | `train_linux`, `train_windows`, the `--global-dataset` of `train_site_model` |
| Behavior capture | `events.jsonl` | none: the raw capture itself | `train_behavior`, `t1_lineage_gain --benign-dataset` |
| Correlation vectors | `correlation_train.jsonl` | [`aggregate_correlation`](../synthaea_ml/data/aggregate_correlation.py) | `train_correlation` |

The three file names are constants in [`data/manifest.py`](../synthaea_ml/data/manifest.py)
(`DEFAULT_BASELINE_FILENAME`, `BEHAVIOR_CAPTURE_FILENAME`, `CORRELATION_DATASET_FILENAME`).

### Command-line baseline

One exec record per line. On Linux the record is `{"argv": [...]}`; on Windows it is
`{"cmdline": "C:\\..."}`, because ETW does not populate `argv`.

`capture_to_baseline` reads the `exec` events of a capture, takes their canonical argv
([`argv_from_record`](../synthaea_ml/data/canonical.py): `argv` if present, else `cmdline`
split on NUL, else `cmdline` as one token), drops exact argv duplicates in first-seen order,
and writes `{"argv": [...]}` per line. A process that ran 10 000 times adds one sample. The
default output name is `baseline_benign.jsonl`; pass `--out <dir>/baseline.jsonl` to produce
a dataset directory.

The trainers deduplicate again after concatenating several datasets, by canonical command-line
string and in order ([`unique_in_order`](../synthaea_ml/data/baselines.py)): the order of the
samples decides the calibration split, so it is kept stable.

### Behavior capture

The T1 trainer cannot learn from a list of command lines: its correlation block needs pids,
process generations and the events around each exec. Its dataset is therefore the agent capture
itself, named `events.jsonl`, with a manifest written with `--baseline-filename events.jsonl`.

[`behavior_dataset.records_from_events`](../synthaea_ml/data/behavior_dataset.py) turns it into
one record per **process incarnation** `(pid, process_generation)`:

- an incarnation with no `exec` event has no command line and no lineage, and yields no record;
- the record is the incarnation's latest `exec`, plus `correlation_features`: the eight
  correlation features over `[last_ts - 60 s, last_ts]`, where `last_ts` is the incarnation's
  last event, filtered by `(pid, generation)` exactly as the agent's `EventBus::events_for_pid`
  filters, so a recycled pid's earlier life never feeds the next one;
- an incarnation whose window holds fewer than `min_events` events (default 1) is skipped.

### Correlation vectors

`aggregate_correlation` writes one vector per pid, not per time step: each pid's window is
rebuilt as it stood at the pid's last event (`[last_ts - window, last_ts]`, `--window` default
60 s, matching the correlator's eviction rule), and pids with fewer than `--min-events` events
are dropped. Each line is:

```json
{"pid": 4242, "features": [1.0, 2.0, 0.0, 1.0, 1.0, 0.0, 3.0, 3.0], "feature_names": ["spawn_count", "..."]}
```

No benign/malicious split is made: the correlation forest is unsupervised and trains on benign
activity alone. Without `--out`, the vectors go to a deprecated location inside the package,
with a `FutureWarning`.

## The manifest

[`write_manifest`](../synthaea_ml/data/manifest.py) writes `manifest.json` next to the samples
file, as sorted, indented JSON with a trailing newline:

| Field | Meaning |
| --- | --- |
| `schema_version` | 1. A reader rejects any other value; the version is bumped only on a breaking change. |
| `host_id` | `sha256(hostname)[:12]`. The hostname itself is never stored. Stable per host, so captures of one site cluster; not anonymity, since anyone who knows the hostname can derive it. |
| `platform` | `linux`, `windows` or `macos` (the CLI restricts it to these). |
| `os_version` | Free-form, for example `WSL2 Ubuntu 24.04.5 LTS` or `11 24H2`. |
| `workload_label` | The host's role during the capture: `desktop-user`, `dev`, `admin` or `server-idle`. |
| `capture_start`, `capture_end` | UTC, ISO 8601 with a `Z` suffix. Naive datetimes are refused. |
| `sample_count` | Non-blank lines of the samples file. |
| `sample_sha256` | SHA-256 of the samples file's bytes ([`hashing.sha256_file`](../synthaea_ml/hashing.py)). |
| `capturer_ref` | Free-form note for a human. Not hashed: put nothing secret in it. |
| `extra` | String-to-string metadata, namespaced so a later typed field cannot collide. |

[`verify_manifest`](../synthaea_ml/data/manifest.py) re-hashes the samples file and recounts its
lines; either mismatch raises `ValueError`. Every trainer calls it, through
[`dataset_version_from_manifest`](../synthaea_ml/registry/dataset_binding.py), before it
reads a sample, so a baseline edited without a new manifest never reaches a model.

### Dataset names

A model record names each dataset `<platform>__<workload_label>__<host_id>__<YYYY-MM-DD>`
([`default_dataset_name`](../synthaea_ml/registry/dataset_binding.py)), the date being the day of
`capture_start`. Double underscores separate the parts so a hyphenated label stays unambiguous.
The provenance check ([registry.md](registry.md#provenance-gate)) looks a dataset up as the one
directory under `datasets/baselines/` that claims its name: a directory named `<name>`, or one
whose manifest derives `<name>`. A baseline can therefore keep its capture-session directory
name (`linux-wsl2-solkapc-2026-09-09`). The lookup fails closed:

- two claimants (two captures of one host, label and day, or a directory named `<name>` next
  to a capture whose manifest derives it) fail as ambiguous. The exact name gets no
  precedence, so give one capture a different `workload_label`;
- a `manifest.json` under `datasets/baselines/` that cannot be read fails the lookup with its
  directory named, since it might be the dataset's.

## Site corpora

The per-site trainers read a corpus exported by the control plane, one record per line:

```json
{"timestamp": "2026-09-23T10:00:00Z", "event_type": "exec", "event": {"argv": ["..."], "sha256": "..."}, "agent_id": "..."}
```

`train_site_model` takes the canonical command line of every `exec` record.
`train_site_rarity` also replays the corpus in time order to rebuild the fleet's common set as it
stood at each event ([`data/common_set.py`](../synthaea_ml/data/common_set.py)):

- an image hash is common once at least K hosts (default 3) have executed it, each host counted
  once per hash;
- a snapshot is published every interval (default 24 h); an event sees the snapshot frozen at
  the most recent boundary before it, never the live state, and nothing at or after its own
  time;
- before the first boundary there is no snapshot (`None`);
- timestamps are converted to nanoseconds in integer arithmetic; a zone-less stamp is read as
  UTC.

Using today's counters instead would let the model learn from the future: a hash that only
became common after an event would look common to it. A site corpus has no manifest; the record
binds it by the SHA-256 of the corpus file and its record count, under the name
`site-corpus-<site-name>`.

## Committed baselines

[`datasets/README.md`](../datasets/README.md) lists the baselines in
[`datasets/baselines/`](../datasets/baselines). In short:

| Platform | Samples | Manifest | Usable by the trainers |
| --- | ---: | --- | --- |
| Linux (WSL2 Ubuntu 24.04, `dev` workload) | 423 argv | yes | yes |
| Linux, earlier capture | 47 argv | no | no |
| Windows | 313 command lines | no | no |

The two unmanifested baselines predate the manifest convention, keep the file name
`baseline_benign.jsonl` and carry no provenance, so the trainers refuse them. None of the three
is yet in a directory named after its dataset name, which the provenance gate would require of a
shipped model.

[`.gitattributes`](../.gitattributes) forces LF line endings on `datasets/**/*.jsonl` and
`manifest.json`: a CRLF checkout would change every byte hash and fail every `verify`.

## Adding a dataset

```bash
# 1. Build the samples file from a capture.
synthaea-ml-capture-to-baseline events.jsonl --out datasets/baselines/<dir>/baseline.jsonl

# 2. Record provenance and the content hash.
synthaea-ml-manifest write datasets/baselines/<dir> \
    --platform linux --os-version "Ubuntu 24.04" --workload-label dev \
    --capture-start 2026-09-14T09:00:00Z --capture-end 2026-09-14T17:00:00Z \
    --capturer-ref "lab run 12"

# 3. Check it.
synthaea-ml-manifest verify datasets/baselines/<dir>
```

For a behavior or correlation dataset, put `events.jsonl` or `correlation_train.jsonl` in the
directory instead and pass `--baseline-filename` to `write`. Note that `verify` on the command
line always checks `baseline.jsonl`; the trainers and the provenance gate find the other two
file names themselves. Use `--hostname` when the capture was recorded on another machine, so
`host_id` names the right host. Review the samples before committing them: a baseline is
published with the repository.
