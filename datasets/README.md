# ml/datasets

Benign baselines are committed; raw captures and labeled traces stay on disk.

| Path | In git | Contents |
| --- | --- | --- |
| `baselines/` | yes | Benign baselines built from captures (`python -m synthaea_ml.data.capture_to_baseline`), one directory per capture |
| `captures/` | no (`.gitignore`) | Raw agent JSONL telemetry captures, named `<platform>-<host>-<date>/` |
| `labeled/` | no (`.gitignore`) | Labeled traces (scenario runs with ground truth) for supervised eval |

## Committed baselines

A baseline directory follows the manifest convention of `synthaea_ml.data.manifest`:

    baselines/<platform>-<host>-<date>[-vN]/
        baseline.jsonl     # one sample per line
        manifest.json      # provenance, workload label, time range, sample count + sha256

The manifest is written and checked with `python -m synthaea_ml.data.manifest {write,verify}`;
its `sample_sha256` is what a registry `model_record.json` binds a model to, so a baseline
cannot change under a model that names it.

| Directory | Manifest | Samples |
| --- | --- | --- |
| `linux-wsl2-solkapc-2026-09-09-v2/` | yes | 423 (Linux argv) |
| `linux-wsl2-solkapc-2026-09-09/` | no — predates the manifest convention | 47 (Linux argv, `baseline_benign.jsonl`) |
| `windows-solkapc-2026-09-08/` | no — predates the manifest convention | 313 (Windows cmdline, `baseline_benign.jsonl`) |

The two unmanifested directories keep the sensor's original file name and have no
recorded provenance, time range or workload label.
