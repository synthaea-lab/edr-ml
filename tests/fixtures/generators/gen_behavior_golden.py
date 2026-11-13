"""Regenerates the per-PID behavior-vector parity fixture (fixtures/behavior_golden.jsonl).

`synthaea_ml.features.behavior.extract_behavior_features` (calibration, Python) mirrors
`BehaviorVector::from_window` (correlator, Rust). The golden file pins the cases both sides
agree on: the exec/connect/fileopen presence flags, the exec-to-connect and exec-to-filewrite
delays, the suspicious-path test, the connect counters and the external destinations.

Rust is the reference for the rules the two sides used to disagree on (agent#203, ml#51):
a global IPv6 destination is external, 0.0.0.0 and :: are internal, path casing is ASCII-only,
and the path is the exec's `image_path`. An exec row without `image_path` is the documented
fallback branch (first NUL-separated `cmdline` argument), kept for old wire records. IPv6
rows carry the address as `daddr_v6` text next to `is_ipv6: true`.

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


def exi(ts_ms: int, image_path: str, cmdline: str) -> dict:
    """An exec that carries `image_path`, the primary input of the suspicious-path test."""
    return {"type": "exec", "ts_ns": ts_ms * MS, "cmdline": cmdline, "image_path": image_path}


def cn6(ts_ms: int, daddr: str, dport: int) -> dict:
    return {"type": "connect", "ts_ns": ts_ms * MS, "daddr_v6": daddr, "dport": dport, "is_ipv6": True}


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
    # agent#203 case 1: IPv6 destinations (Rust `is_non_global_ipv6`).
    "connect_ipv6_global": [cn6(0, "2606:4700:4700::1111", 443)],
    "connect_ipv6_global_2001": [cn6(0, "2001:4860:4860::8888", 443)],
    "connect_ipv6_loopback": [cn6(0, "::1", 443)],
    "connect_ipv6_unspecified": [cn6(0, "::", 443)],
    "connect_ipv6_link_local": [cn6(0, "fe80::1", 443)],
    "connect_ipv6_unique_local": [cn6(0, "fd12:3456:789a::1", 443)],
    "connect_ipv6_multicast": [cn6(0, "ff02::1", 443)],
    "connect_ipv6_site_local_is_global": [cn6(0, "fec0::1", 443)],
    # IPv4-mapped: classified as the IPv4 address (Rust `to_canonical`), not as IPv6.
    "connect_ipv6_v4_mapped_private_is_internal": [cn6(0, "::ffff:10.0.0.1", 443)],
    "connect_ipv6_v4_mapped_loopback_is_internal": [cn6(0, "::ffff:127.0.0.1", 443)],
    "connect_ipv6_v4_mapped_unspecified_is_internal": [cn6(0, "::ffff:0.0.0.0", 443)],
    "connect_ipv6_v4_mapped_public_is_external": [cn6(0, "::ffff:8.8.8.8", 443)],
    "connect_ipv6_internal_then_global": [cn6(0, "fe80::1", 80), cn6(5, "2606:4700:4700::1111", 80)],
    # agent#203 case 2: the unspecified IPv4 address is internal, `0.x.x.x` is not.
    "connect_unspecified_ipv4": [cn(0, [0, 0, 0, 0], 22)],
    "connect_zero_first_octet_not_unspecified": [cn(0, [0, 1, 2, 3], 22)],
    # agent#203 case 3: ASCII-only lowercasing. U+212A KELVIN SIGN folds to `k` under
    # Unicode `str.lower()` (DES<kelvin>TOP would match `desktop`); it must not.
    "path_ascii_only_kelvin_sign": [exi(0, "C:\\Users\\bob\\DES\u212aTOP\\run.exe", "run.exe\0")],
    "path_ascii_uppercase_still_matches": [exi(0, "C:\\Users\\bob\\DESKTOP\\run.exe", "run.exe\0")],
    # agent#203 case 3: the path is `image_path`, not argv0 of `cmdline` (`exec -a`).
    "path_image_path_wins_over_argv0_suspicious_image": [
        exi(0, "/home/bob/Downloads/tool", "/usr/bin/ls\0-la\0")
    ],
    "path_image_path_wins_over_argv0_clean_image": [
        exi(0, "/usr/bin/ls", "/home/bob/Downloads/tool\0")
    ],
    "path_empty_image_path_is_not_a_fallback": [exi(0, "", "/home/bob/Downloads/tool\0")],
    # Fallback branch: no `image_path` (rows above without one: `suspicious_downloads`, ...).
    "path_fallback_argv0_without_image_path": [ex(0, "/home/bob/Downloads/tool\0")],
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
