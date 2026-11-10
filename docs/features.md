# Features

Each module in [`synthaea_ml/features/`](../synthaea_ml/features) defines one feature vector:
its values, their order and their names (`FEATURE_NAMES`). Each one is the reference for a Rust
twin in the agent's `crates/ml`, and both sides test against the same golden file. Changing a
definition here without changing the Rust side, regenerating the golden and retraining, silently
breaks every model trained on the old definition.

| Module | Width | Scores | Used by | Golden |
| --- | ---: | --- | --- | --- |
| [`cmdline`](../synthaea_ml/features/cmdline.py) | 9 | One command line | T0, T1, site models | `features_golden.jsonl` |
| [`correlation`](../synthaea_ml/features/correlation.py) | 8 | One pid's event window | T2, T1 | `capture_correlation_golden.jsonl` |
| [`t1`](../synthaea_ml/features/t1.py) | 17 | One process incarnation | T1 | `t1_golden.jsonl` |
| [`lineage`](../synthaea_ml/features/lineage.py) | 6 | The parent of an exec | Lineage-gain measurement only | `lineage_golden.jsonl` |
| [`rarity`](../synthaea_ml/features/rarity.py) | 2 | The executed image against the fleet's common set | `train_site_rarity` | `rarity_golden.jsonl` |
| [`behavior`](../synthaea_ml/features/behavior.py) | 9 | One pid's window, for likelihood-ratio calibration | `calibrate_llr` | `behavior_golden.jsonl` |
| [`combined`](../synthaea_ml/features/combined.py) | 14 | Correlation + lineage | Tests only | none |

Python computes in float64 and the agent in float32; the Rust golden tests compare with a
relative tolerance of `1e-4`.

## Command line

`extract_features(cmdline)` takes the [canonical command line](datasets.md#the-canonical-command-line)
(argv tokens each followed by `\0` on Linux, the flat string on Windows):

| # | Name | Definition |
| ---: | --- | --- |
| 0 | `length` | Characters, NUL separators included |
| 1 | `entropy` | Shannon entropy over characters, in bits |
| 2 | `suspicious_token_count` | How many entries of `SUSPICIOUS_TOKENS` occur as substrings (each counted once) |
| 3 | `token_count` | Non-empty NUL-separated tokens |
| 4 | `max_token_length` | Length of the longest token |
| 5 | `shell_metachar_count` | Occurrences of any of `` \|;&$`()<> `` |
| 6 | `digit_ratio` | ASCII digits `0`-`9` over length; Unicode digits do not count, as in Rust's `is_ascii_digit` |
| 7 | `is_suspicious_win_path` | 1.0 if a `SUSPICIOUS_WIN_PATHS` entry occurs, case-insensitively (`\AppData\`, `\Temp\`, `\Downloads\`, `%temp%`, ...) |
| 8 | `is_legit_win_path` | 1.0 if a `LEGIT_WIN_PATHS` entry occurs, case-insensitively (`\Windows\System32\`, `\Program Files\`, ...) |

The token lists are matched as plain substrings of the whole string, without tokenizing, to stay
identical to Rust's `str::contains`. They are a known weakness, as any static list is: an attacker
who reads them can avoid them.

## Correlation

`extract_features(events, pid, generation=None)` takes every flattened event of a correlation
window and filters it to one process itself, so the caller passes the window, not the pid's
events:

| # | Name | Definition |
| ---: | --- | --- |
| 0 | `spawn_count` | `exec` events |
| 1 | `connect_count` | `connect` events only |
| 2 | `filewrite_count` | `fileopen` events with write intent |
| 3 | `unique_daddr_count` | Distinct destinations over `connect` and `network_flow` |
| 4 | `unique_dport_count` | Distinct destination ports over `connect` and `network_flow` |
| 5 | `has_full_chain` | 1.0 if `spawn_count`, `connect_count` and `filewrite_count` are all at least 1 |
| 6 | `span_s` | Last minus first event time, in seconds |
| 7 | `event_count` | Events of the process in the window |

- **Write intent** mirrors the agent's `schema::has_write_intent`: the 2-bit access mode
  (`flags & O_ACCMODE`) is `O_WRONLY` or `O_RDWR`, or `O_CREAT` is set. The invalid access mode
  `0o3` is not a write; the kernel refuses it.
- **Incarnations.** [`same_generation`](../synthaea_ml/features/correlation.py) mirrors the
  correlator: two known, different `process_generation` stamps never mix; a missing stamp on
  either side cannot disprove identity, so captures without stamps behave as pid-only.
- **Event types.** Only the four types `flatten_wire_event` keeps contribute, so `span_s` and
  `event_count` are computed over the same subset the agent counts.

## T1

The T1 vector is the 9 command-line features of an exec record followed by the 8 correlation
features carried on the record as `correlation_features` ([`features/t1.py`](../synthaea_ml/features/t1.py)).
A record without `correlation_features` scores as an empty window (eight zeros), which keeps the
context fixed when a mutation test rewrites only the command line. A context of the wrong length
raises `ValueError`.

There is no lineage block in the shipped T1 vector. Six lineage columns were measured on lab
captures with the [lineage-gain evaluation](evaluation.md#lineage-gain): they raised detection of
attacks with an unusual parent and lowered it for the large majority whose lineage is ordinary,
because an Isolation Forest spends random splits on columns that do not separate (a paired AUC
change of about -0.08, worse on every split). The agent judges the parent with rules instead.
The lineage extractor stays, parity-tested, so the measurement can be repeated on better data.

## Lineage

`extract_features(event)` reads `parent_comm` and `parent_image_path` from an exec record:

| # | Name | Definition |
| ---: | --- | --- |
| 0 | `has_parent_lineage` | Either field is present |
| 1 | `parent_comm_is_shell` | Basename of `parent_comm` equals a `SHELL_COMMS` entry, case-insensitively |
| 2 | `parent_comm_is_webserver` | Same, against `WEBSERVER_COMMS` (`nginx`, `httpd`, `w3wp.exe`, `node`, `java`, ...) |
| 3 | `parent_comm_is_office` | Same, against `OFFICE_COMMS` |
| 4 | `parent_path_is_system` | `parent_image_path` contains a `SYSTEM_PATHS` entry, case-insensitively |
| 5 | `parent_path_is_suspicious` | Same, against `SUSPICIOUS_PATHS` (`/tmp/`, `/dev/shm/`, `\AppData\`, ...) |

Name matches compare the whole basename (`cmd.exe` matches `C:\Windows\System32\cmd.exe`, `cmd`
does not); path matches are substrings. [`features/combined.py`](../synthaea_ml/features/combined.py)
concatenates the correlation and lineage vectors (14 values), taking the lineage from the latest
exec of the same incarnation; nothing in the trainers uses it.

## Rarity

The rarity feature asks whether the executed image is on the fleet's **common set**: the image
hashes executed on at least K hosts of the tenant, published by the control plane as a snapshot.
A device can test membership in a snapshot but cannot count hosts, so the model input is derived
from the snapshot, never from a host count:

| # | Name | Definition |
| ---: | --- | --- |
| 0 | `rarity_known` | 1.0 when a snapshot exists, it covers at least `min_hosts` hosts, and the event carries a valid SHA-256 (64 hex characters, case-insensitive) |
| 1 | `image_in_common_set` | 1.0 when known and the hash is in the common set; 0.0 when known means rare or new on this fleet |

Without `rarity_known`, every binary on a freshly enrolled fleet, whose snapshot is empty, would
look rare. When the evidence is withheld both values are 0. For training, the snapshot in force at
each event is rebuilt from the corpus ([datasets.md](datasets.md#site-corpora)). The agent has a
Rust extractor pinned to this one, but no agent scorer loads an 11-feature model yet.

## Behavior

[`features/behavior.py`](../synthaea_ml/features/behavior.py) mirrors the agent correlator's
`BehaviorVector`, the per-pid vector behind its Bayesian belief. It is not a model input: it is
the vector `calibrate_llr` bins to estimate likelihood ratios ([calibration.md](calibration.md#likelihood-ratio-calibration)).

| # | Name | Definition |
| ---: | --- | --- |
| 0-2 | `has_exec`, `has_connect`, `has_filewrite` | 1.0 if at least one such event (write intent as above) |
| 3 | `time_exec_to_connect_ms` | First connect minus first exec, in ms, floored at 0; 0 if either is absent |
| 4 | `time_exec_to_filewrite_ms` | Same for the first write |
| 5 | `is_suspicious_path` | The first exec's image (its first NUL token) contains `appdata`, `\temp\`, `downloads`, `desktop` or `\public\`, case-insensitively |
| 6 | `connect_count` | `connect` events |
| 7 | `distinct_dports` | Distinct destination ports of `connect` events |
| 8 | `dest_is_external` | 1.0 if an IPv4 destination is outside 10/8, 172.16/12, 192.168/16 and 127/8 |

An empty window yields nine zeros.

## Sanity samples

[`features/sanity_samples.py`](../synthaea_ml/features/sanity_samples.py) holds a few fixed
command lines per platform (benign ones from the baseline, a short malicious `base64 -d`, and a
known false positive on Linux, a container health check with `curl`). The T0 trainers print
their scikit-learn scores after fitting, and `verify_onnx` scores them through onnxruntime. They
are a sniff test and the ONNX oracle input, not a quality gate.

## Changing a feature

1. Change the Python definition and its Rust twin together.
2. Regenerate the golden with its generator in
   [`tests/fixtures/generators/`](../tests/fixtures/generators) and open the matching pull
   request on the agent ([testing.md](testing.md#golden-fixtures)).
3. Retrain every model that uses the vector: its recorded feature names and bounds no longer
   describe what the extractor produces.

Never regenerate a golden to make a failing parity test pass; the failure is the drift.
