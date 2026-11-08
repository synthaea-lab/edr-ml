# ml/reports

Evidence artifacts behind calibrations. Each file in `calibration/` is the output of one
`python -m synthaea_ml.calibration.calibrate_llr` run over one capture: the empirical
per-bin LLRs of the nine BehaviorVector features and the generated Rust
`log_likelihood_ratio()` body. Sample counts are per-PID windows, read from each
report's header.

| Report | Run | Benign | Malicious | Backs |
| --- | --- | ---: | ---: | --- |
| `calibration/llr_report_njrat.txt` | NjRAT capture, 2026-08-28 | 97 | 23 | The LLR table living verbatim in `crates/correlator/src/bayes.rs` (synthaea-lab/agent) — the receipt for those shipped constants |
| `calibration/llr_report_windows.txt` | Windows capture of the malware3 run (commit 045b57f) | 1129 | 62 | Exploratory run; not shipped |
| `calibration/llr_report_malware2_nat.txt` | malware2 NAT capture: AsyncRAT-style beacon to Cloudflare ranges (commit 045b57f) | 67 | 6 | Exploratory run; not shipped |
| `calibration/llr_report_malware3_nat.txt` | malware3 NAT capture: loader + PowerShell storm, 21 malicious PIDs (commit f01312b) | 786 | 39 | Exploratory run; not shipped |
| `calibration/llr_report_malware4_nat.txt` | malware4 NAT capture: dropper chain T1105/T1059/T1071 (commit d5e05cf) | 980 | 76 | Exploratory run; not shipped |
| `calibration/llr_report_malware5_nat.txt` | malware5 NAT capture: LOLBin loop T1218/T1127 + T1105, 163 malicious PIDs (commit 2d43a7e) | 1276 | 380 | Exploratory run; not shipped |

Only the NjRAT report backs constants that ship today. The five 2026-09-07 runs are
separate captures, not re-runs of it, kept as inputs toward a merged-corpus calibration.
