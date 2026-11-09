"""Command-line samples every T0 model is checked against, per platform.

Two readers: each trainer prints the scikit-learn scores of these samples after fitting (a
sniff test, not a gate), and `synthaea_ml.export.verify_onnx` uses them as its parity oracle
against the shipped ONNX models. They live here, below both, so the export layer does not have
to import a training script to get them. Keep the keys and shape stable, or update
`verify_onnx` in the same change.
"""

from __future__ import annotations

# Linux - mirror of WINDOWS_SANITY_SAMPLES.
# Benign: taken as-is from the captured baseline. Suspicious: the isolated `base64 -d`
# exec event captured in the lab on 2026-08-24 (see the contamination comment on
# `training.config.CMDLINE_FOREST`)
# and the Docker healthcheck documented as a known false positive (docs/PLAN.md) -
# included here to track it, not because it necessarily has to score as an anomaly.
LINUX_SANITY_SAMPLES: dict[str, str] = {
    "benign (whoami)": "whoami\0",
    "benign (lab script)": "bash\0/tmp/benign_activity.sh\0",
    "benign (modprobe)": "/sbin/modprobe\0-q\0--\0binfmt-0090\0",
    "suspicious (base64 decode)": "base64\0-d\0",
    "known FP (curl health)": "curl\0-f\0http://backend:8000/api/health/\0",
}

# Windows - distinguishes legitimate vs suspicious cmdlines.
# The benign cmdlines reflect the ETW format (full image path, few arguments).
# The malicious cmdlines simulate encoded payloads or suspicious paths (AppData).
WINDOWS_SANITY_SAMPLES: dict[str, str] = {
    "benign (svchost)": "C:\\Windows\\System32\\svchost.exe",
    "benign (powershell)": "C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe",
    "suspicious (AppData)": (
        "C:\\Users\\user\\AppData\\Roaming\\Microsoft\\WindowsApps\\RuntimeBroker.exe"
    ),
    "suspicious (base64 ps)": "powershell.exe -EncodedCommand ZWNobyBoZWxsbw==",
    "suspicious (Temp)": "C:\\Users\\user\\AppData\\Local\\Temp\\payload.exe",
}
