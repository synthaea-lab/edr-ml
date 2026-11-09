# datasets

Benign baselines the models are trained and evaluated on. Baselines are committed; raw
captures (`captures/`) and labeled traces (`labeled/`) stay on local disk and are ignored by
Git.

## Layout

Each baseline is one directory, built from a capture with
`python -m synthaea_ml.data.capture_to_baseline`:

```text
baselines/<platform>-<host>-<date>[-vN]/
    baseline.jsonl     one sample per line
    manifest.json      provenance, workload label, time range, sample count and SHA-256
```

`python -m synthaea_ml.data.manifest write` creates the manifest and
`python -m synthaea_ml.data.manifest verify <dir>` checks it. A model record binds to the
manifest's `sample_sha256`, so a baseline cannot change under a model that names it. The host
is recorded as a truncated hash of its hostname, never the hostname itself.

## Committed baselines

| Directory | Manifest | Samples |
| --- | --- | ---: |
| `linux-wsl2-solkapc-2026-09-09-v2/` | yes | 423 Linux argv |
| `linux-wsl2-solkapc-2026-09-09/` | no | 47 Linux argv |
| `windows-solkapc-2026-09-08/` | no | 313 Windows command lines |

The two unmanifested baselines predate the manifest convention. They keep the original file
name `baseline_benign.jsonl` and carry no recorded provenance, so trainers do not accept them.
