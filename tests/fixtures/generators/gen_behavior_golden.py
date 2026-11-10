"""Regenerates the per-PID behavior-vector parity fixture (fixtures/behavior_golden.jsonl).

`synthaea_ml.features.behavior.extract_behavior_features` (calibration, Python) mirrors
`BehaviorVector::from_window` (correlator, Rust). The golden file pins the cases both sides
agree on: the exec/connect/fileopen presence flags, the exec-to-connect and exec-to-filewrite
delays, the suspicious-path test, the connect counters and IPv4 external destinations. Python
is the reference, as for the other goldens.

Cases where the two sides are known to disagree are NOT in this file, since a golden would
freeze one side's answer and fail the other (see `issues/issue-ML-015.md`):
IPv6 destinations, the unspecified address 0.0.0.0, and non-ASCII path casing.

Run from any directory:

    python tests/fixtures/generators/gen_behavior_golden.py

Regenerate only on a deliberate change to a behavior feature, never to make a failing parity
test pass.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from synthaea_ml.features.behavior import extract_behavior_features

OUT = Path(__file__).resolve().parents[1] / "behavior_golden.jsonl"
MS = 1_000_000


def ex(ts_ms: int, cmdline: str) -> dict:
    return {"type": "exec", "ts_ns": ts_ms * MS, "cmdline": cmdline}


def cn(ts_ms: int, daddr: list[int], dport: int) -> dict:
    return {
        "type": "connect",
        "ts_ns": ts_ms * MS,
        "daddr_v4": daddr,
        "dport": dport,
        "is_ipv6": False,
    }


def fo(ts_ms: int, flags: int) -> dict:
    return {"type": "fileopen", "ts_ns": ts_ms * MS, "flags": flags}


# name -> events. Flags: O_RDONLY 0, O_WRONLY 0o1, O_RDWR 0o2, invalid access mode 0o3,
# O_CREAT 0o100, O_TRUNC 0o1000.
CASES: dict[str, list[dict]] = {
    "empty": [],
    "exec_only_system_path": [ex(0, "/usr/bin/ls\0-la\0")],
    "exec_only_windows_system": [ex(0, "C:\\Windows\\System32\\svchost.exe\0-k\0netsvcs\0")],
    "suspicious_appdata": [ex(0, "C:\\Users\\bob\\AppData\\Roaming\\payload.exe\0")],
    "suspicious_temp_uppercase": [ex(0, "C:\\USERS\\BOB\\APPDATA\\LOCAL\\TEMP\\X.EXE\0")],
    "suspicious_downloads": [ex(0, "/home/bob/Downloads/tool\0")],
    "suspicious_desktop": [ex(0, "C:\\Users\\bob\\Desktop\\run.exe\0")],
    "suspicious_public": [ex(0, "C:\\Users\\Public\\run.exe\0")],
    "suspicious_only_first_exec_counts": [ex(0, "/usr/bin/ls\0"), ex(10, "C:\\Temp\\a.exe\0")],
    "exec_empty_cmdline": [ex(0, "")],
    "connect_public": [cn(0, [8, 8, 8, 8], 443)],
    "connect_private_10": [cn(0, [10, 1, 2, 3], 22)],
    "connect_private_172_low": [cn(0, [172, 16, 0, 1], 22)],
    "connect_private_172_high": [cn(0, [172, 31, 255, 255], 22)],
    "connect_public_172_below": [cn(0, [172, 15, 0, 1], 22)],
    "connect_public_172_above": [cn(0, [172, 32, 0, 1], 22)],
    "connect_private_192_168": [cn(0, [192, 168, 1, 1], 445)],
    "connect_public_192_169": [cn(0, [192, 169, 1, 1], 445)],
    "connect_loopback": [cn(0, [127, 0, 0, 1], 8000)],
    "connect_private_then_public": [cn(0, [10, 0, 0, 1], 80), cn(5, [1, 1, 1, 1], 80)],
    "connect_repeated_port": [cn(0, [1, 1, 1, 1], 80), cn(1, [1, 1, 1, 2], 80), cn(2, [1, 1, 1, 3], 80)],
    "connect_distinct_ports": [cn(0, [1, 1, 1, 1], 80), cn(1, [1, 1, 1, 1], 443), cn(2, [1, 1, 1, 1], 4444)],
    "write_wronly": [fo(0, 0o1)],
    "write_rdwr": [fo(0, 0o2)],
    "write_creat_with_rdonly": [fo(0, 0o100)],
    "no_write_rdonly": [fo(0, 0)],
    "no_write_invalid_access_mode": [fo(0, 0o3)],
    "write_invalid_mode_with_creat": [fo(0, 0o3 | 0o100)],
    "write_high_bits_only_rdonly": [fo(0, 0o2000000)],
    "write_trunc_with_rdonly": [fo(0, 0o1000)],
    "write_invalid_mode_with_trunc": [fo(0, 0o3 | 0o1000)],
    "dropper": [
        ex(0, "C:\\Users\\victim\\AppData\\Roaming\\payload.exe\0"),
        cn(500, [8, 8, 8, 8], 4444),
        cn(600, [8, 8, 8, 8], 443),
        fo(1000, 0o101),
    ],
    "delay_connect_before_exec_clamped": [cn(0, [8, 8, 8, 8], 80), ex(500, "/bin/sh\0")],
    "delay_write_before_exec_clamped": [fo(0, 0o1), ex(500, "/bin/sh\0")],
    "delay_uses_first_connect": [ex(0, "/bin/sh\0"), cn(300, [8, 8, 8, 8], 80), cn(900, [8, 8, 8, 8], 81)],
    "delay_uses_first_write_only": [ex(0, "/bin/sh\0"), fo(100, 0), fo(250, 0o1), fo(400, 0o2)],
    "delay_uses_first_exec": [ex(100, "/bin/sh\0"), ex(200, "/bin/ls\0"), cn(700, [8, 8, 8, 8], 80)],
    "delay_fractional_ms": [
        {"type": "exec", "ts_ns": 0, "cmdline": "/bin/sh\0"},
        {"type": "connect", "ts_ns": 1_500_000, "daddr_v4": [8, 8, 8, 8], "dport": 80, "is_ipv6": False},
    ],
    "no_exec_connect_and_write": [cn(100, [8, 8, 8, 8], 80), fo(200, 0o1)],
}


def main() -> None:
    with OUT.open("w", encoding="utf-8", newline="\n") as f:
        for name, events in CASES.items():
            row = {"name": name, "events": events, "features": extract_behavior_features(events)}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(CASES)} cases written to {OUT}")


if __name__ == "__main__":
    main()
