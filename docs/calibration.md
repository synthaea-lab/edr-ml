# Calibration

An Isolation Forest's raw score has no false-positive rate attached to it. Calibration turns a
stated budget into a threshold the agent applies, bounds the inputs the model may be trusted on,
and, separately, estimates the likelihood ratios the agent's correlator uses for its Bayesian
belief. The code is in [`synthaea_ml/calibration/`](../synthaea_ml/calibration).

## Conformal threshold

[`calibrate_threshold`](../synthaea_ml/calibration/calibrate_conformal.py) maps a false-positive
budget to a `decision_function` threshold by split-conformal calibration:

1. The trainer fits the forest on 70% of the benign rows and holds out the other 30%
   ([training.md](training.md#the-shared-steps)).
2. The held-out rows are scored.
3. The target false-positive rate is `fp_budget / benign_rate_per_day`. With the configured 5
   false positives per endpoint per day and an estimated 1000 benign events per endpoint per
   day, that is 0.5%.
4. The threshold is the score at that percentile of the held-out scores (`numpy.percentile`, so
   the 0.5th percentile here).

A score below the threshold is anomalous. On the agent, a score at or above it produces no
evidence at all, so the budget is enforced on the endpoint rather than downstream.

The result is a `ConformalCalibration` recorded in `model_record.json` and, for the threshold
alone, in `model_metadata.json`:

| Field | Meaning |
| --- | --- |
| `fp_budget_per_endpoint_day` | The budget the threshold was set for |
| `threshold` | The score below which an input is flagged |
| `calibration_set_size` | Held-out rows the percentile was taken over |
| `benign_baseline_rate` | The benign events per endpoint per day assumed |
| `calibrated_at` | UTC time of the calibration |

Two caveats follow from the method. The guarantee holds only as far as the held-out rows
resemble an endpoint's real benign activity, and the event rate of 1000 per day is an estimate:
an endpoint that runs more distinct processes sees proportionally more false positives at the
same threshold. With about a hundred held-out rows, as the committed Linux baseline gives, the
0.5th percentile is an interpolation between the two lowest scores. Larger baselines tighten both.

## Feature bounds

An Isolation Forest scores a value beyond its training range no more anomalous than the edge of
that range, so its score there means nothing. [`compute_feature_bounds`](../synthaea_ml/calibration/calibrate_conformal.py)
records, per feature, the minimum and maximum over the **training** rows, widened on each side by
5% of the observed range (by 5% of 1.0 when the feature is constant). The bounds go into
`model_metadata.json` with the feature names in model input order (values illustrative):

```json
{
  "threshold": -0.0612,
  "feature_bounds": {
    "feature_names": ["length", "entropy", "..."],
    "min_values": [-3.15, 1.21, "..."],
    "max_values": [661.15, 5.43, "..."]
  }
}
```

The agent's scorer rejects a vector outside the bounds with an out-of-bounds error, which callers
treat as no score, not as benign.

## Likelihood-ratio calibration

The agent's correlator keeps a Bayesian belief per process. Each feature of the per-pid
[behavior vector](features.md#behavior) moves the belief by a log-likelihood ratio (LLR). These
ratios started as hand-set values; [`calibrate_llr`](../synthaea_ml/calibration/calibrate_llr.py)
estimates them from a labeled run:

1. Read the agent's `events.jsonl` (normalized with `normalize_for_behavior`) and its
   `alerts.ndjson`.
2. Label a pid malicious if any alert message contains `pid=<n>`. Every other pid is benign.
3. Group each pid's `exec`, `connect` and `fileopen` events into consecutive windows of
   `--window` seconds (default 60), and compute one behavior vector per non-empty window.
4. Discretize each feature into bins: binary features into `0`/`1`;
   `time_exec_to_connect_ms` into absent, under 500, under 2000, 2000 and more;
   `time_exec_to_filewrite_ms` into absent, under 500, 500 and more; `connect_count` into 0,
   1-4, 5-14, 15 and more; `distinct_dports` into 0-1, 2-3, 4 and more.
5. Per feature and bin, compute `log(P(bin | malicious) / P(bin | benign))` with Laplace
   smoothing (alpha 0.5), so a bin unseen on one side does not give an infinite ratio.
6. Print a report: the sample counts, the current hand-set value and the empirical ratio of
   every bin, and a generated Rust body for the correlator's `log_likelihood_ratio()`.

```bash
synthaea-ml-calibrate-llr --events events.jsonl --alerts alerts.ndjson \
    --out reports/calibration/llr_report_<run>.txt
```

The labels come from the agent's own alerts, so a ratio measures what separates the processes the
agent already flagged from the rest of that run. It is evidence for a constant, not ground truth:
review it against the run, and prefer a corpus merged from several runs to a single one.

### Reports

Each committed report in [`reports/calibration/`](../reports/calibration) is the output of one
run over one capture. [`reports/README.md`](../reports/README.md) lists them with their sample
counts and status. Only one of them backs constants in use in the agent today; the others are
exploratory inputs toward a calibration on a merged corpus.
