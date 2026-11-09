# reports

Evidence behind calibrated constants. Each file in `calibration/` is the output of one
`python -m synthaea_ml.calibration.calibrate_llr` run over one capture: the empirical per-bin
log-likelihood ratios of the nine behavior-vector features, and the generated Rust
`log_likelihood_ratio()` body. Sample counts are per-process windows, as stated in each
report's header.

| Report | Capture | Benign | Malicious | Used for |
| --- | --- | ---: | ---: | --- |
| `llr_report_njrat.txt` | NjRAT run, 2026-08-28 | 97 | 23 | The likelihood-ratio table in the agent's correlator ([`bayes.rs`](https://github.com/synthaea-lab/agent/blob/main/crates/correlator/src/bayes.rs)) |
| `llr_report_windows.txt` | Windows capture of the malware3 run | 1129 | 62 | Exploratory, not shipped |
| `llr_report_malware2_nat.txt` | malware2: AsyncRAT-style beacon to Cloudflare ranges | 67 | 6 | Exploratory, not shipped |
| `llr_report_malware3_nat.txt` | malware3: loader and PowerShell storm, 21 malicious processes | 786 | 39 | Exploratory, not shipped |
| `llr_report_malware4_nat.txt` | malware4: dropper chain, T1105, T1059, T1071 | 980 | 76 | Exploratory, not shipped |
| `llr_report_malware5_nat.txt` | malware5: LOLBin loop, T1218, T1127, T1105, 163 malicious processes | 1276 | 380 | Exploratory, not shipped |

Only the NjRAT report backs constants in use today. The five 2026-09-07 runs are separate
captures, kept as inputs toward a calibration on a merged corpus.
