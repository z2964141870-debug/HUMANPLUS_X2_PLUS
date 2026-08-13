#!/usr/bin/env python3
"""Finalize the six-case left-turn configuration repair without promotion."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
import tempfile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sidecar(path: Path) -> None:
    sidecar = path.with_name(path.name + ".sha256")
    if sidecar.read_text(encoding="utf-8") != f"{sha256(path)}  {path.name}\n":
        raise RuntimeError(f"sidecar mismatch: {path}")


def atomic_text(path: Path, value: str) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: dict) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    atomic_text(path.with_name(path.name + ".sha256"), f"{sha256(path)}  {path.name}\n")


def close(actual: object, expected: float) -> bool:
    return isinstance(actual, (int, float)) and abs(float(actual) - expected) <= 1.0e-12


def parse_env(resource: dict) -> dict[str, str]:
    result = {}
    for token in resource.get("command", []):
        if "=" in token:
            name, value = token.split("=", 1)
            result[name] = value
    return result


def expected_rows(prereg: dict) -> list[dict]:
    rows = []
    index = 0
    repair_index = 0
    for speed in prereg["matrix"]["speed_order"]:
        vx = prereg["matrix"]["speeds"][speed]
        for motion in prereg["matrix"]["motion_order"]:
            config = prereg["matrix"]["motions"][motion]
            for repeat in range(1, config["repeats"] + 1):
                repaired = motion == "turn_left"
                prefix = "x2_sfbrake_repair" if repaired else "x2_sfbrake"
                case_name = f"{prefix}_{speed}_{motion}_r{repeat}"
                raw_root = prereg["outputs"]["repair_raw_root" if repaired else "attempt0_raw_root"]
                rows.append({
                    "index": index, "speed": speed, "motion": motion,
                    "repeat": repeat, "vx": vx,
                    "wz": config["fixed_wz_radps"],
                    "mirror": config["mirror_policy"],
                    "supervisor": config["supervisor"],
                    "repaired": repaired, "case": case_name,
                    "repair_index": repair_index if repaired else None,
                    "raw": str(Path(raw_root) / f"{case_name}.json"),
                    "resource": str(Path(prereg["outputs"]["report_root"]) / f"{case_name}_resource.json"),
                })
                if repaired:
                    repair_index += 1
                index += 1
    if len(rows) != 24 or sum(row["repaired"] for row in rows) != 6:
        raise RuntimeError("repair matrix must resolve to 18 frozen + 6 repaired cases")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", required=True, type=Path)
    parser.add_argument("--case", required=True, action="append", type=Path)
    parser.add_argument("--resource", required=True, action="append", type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--markdown", required=True, type=Path)
    args = parser.parse_args()
    if len(args.case) != 24 or len(args.resource) != 24:
        raise RuntimeError("exactly 24 ordered case/resource inputs are required")
    for path in (args.result, Path(str(args.result) + ".sha256"), args.markdown, Path(str(args.markdown) + ".sha256")):
        if path.exists():
            raise FileExistsError(path)
    for path in (args.prereg, *args.case, *args.resource):
        verify_sidecar(path)
    prereg = json.loads(args.prereg.read_text(encoding="utf-8"))
    if prereg.get("schema") != "x2_official_state_feedback_brake_panel_repair_prereg_v1":
        raise RuntimeError("unexpected prereg schema")
    for section in ("immutable_code", "immutable_inputs", "immutable_evidence"):
        for name, record in prereg[section].items():
            path = Path(record["path"])
            if not path.is_file() or sha256(path) != record["sha256"]:
                raise RuntimeError(f"immutable drift: {section}.{name}")

    contract = prereg["controller_contract"]
    rows = []
    for spec, raw_path, resource_path in zip(expected_rows(prereg), args.case, args.resource):
        if raw_path.resolve() != Path(spec["raw"]).resolve() or resource_path.resolve() != Path(spec["resource"]).resolve():
            raise RuntimeError(f"ordered path mismatch: {spec['case']}")
        summary = json.loads(raw_path.read_text(encoding="utf-8"))["summary"]
        resource = json.loads(resource_path.read_text(encoding="utf-8"))
        env = parse_env(resource)
        checks = {
            "case": env.get("CASE_NAME") == spec["case"],
            "controller": summary.get("stop_controller") == contract["stop_controller"],
            "gain": close(summary.get("stop_brake_gain"), contract["gain"]),
            "limit": close(summary.get("stop_brake_limit_mps"), contract["limit_mps"]),
            "template_speed": close(summary.get("stop_brake_template_speed_mps"), contract["template_speed_mps"]),
            "template_floor": close(summary.get("stop_brake_template_floor"), contract["template_floor"]),
            "vx": close(summary.get("command_vx_mps"), spec["vx"]),
            "wz": summary.get("fixed_wz_radps") is None if spec["wz"] is None else close(summary.get("fixed_wz_radps"), spec["wz"]),
            "mirror": bool(summary.get("mirror_policy")) == spec["mirror"],
            "resource_mirror": (env.get("MIRROR_POLICY") == "true") if spec["mirror"] else (env.get("MIRROR_POLICY") is None),
            "supervisor": summary.get("action_bias_mode") == spec["supervisor"],
            "one_attempt": env.get("MAX_ATTEMPTS") == "1",
            "latch": summary.get("stop_hold_latch_s") is not None,
            "resource_label": resource.get("label") == (
                f"x2_official_state_feedback_repair_case{spec['repair_index']}"
                if spec["repaired"] else f"x2_official_state_feedback_panel_case{spec['index']}"
            ),
            "resource_exit": resource.get("raw_returncode") in (0, 2) and resource.get("exit_code") == resource.get("raw_returncode"),
            "resource_autonomous": not any(resource.get(name, False) for name in ("timed_out", "term_sent", "kill_sent", "forced_cleanup")),
            "resource_elapsed": float(resource["elapsed_s"]) <= prereg["resources"]["elapsed_max_s"],
            "resource_free": int(resource["disk_after"]["free_bytes"]) >= prereg["resources"]["free_after_bytes_min"],
        }
        if not all(checks.values()):
            raise RuntimeError(f"contract mismatch: {spec['case']}: {checks}")
        rows.append({
            "case_name": spec["case"], "speed": spec["speed"], "motion": spec["motion"],
            "repeat": spec["repeat"], "repaired": spec["repaired"], "checks": checks,
            "raw_sha256": sha256(raw_path), "resource_sha256": sha256(resource_path),
            "startup_gate_pass": bool(summary["startup_gate_pass"]),
            "move_gate_pass": bool(summary["move_gate_pass"]),
            "stop_gate_pass": bool(summary["stop_gate_pass"]),
            "full_gate_pass": bool(summary["full_gate_pass"]),
            "stop_hold_latch_s": summary["stop_hold_latch_s"],
            "stop_root_xy_drift_m": summary["stop_root_xy_drift_m"],
            "stop_last_1s_mean_speed_mps": summary["stop_last_1s_mean_speed_mps"],
            "stop_root_z_min_m": summary["stop_root_z_min_m"],
            "stop_root_tilt_max_rad": summary["stop_root_tilt_max_rad"],
        })

    groups = defaultdict(lambda: {"runs": 0, "move_passes": 0, "stop_passes": 0, "full_passes": 0})
    for row in rows:
        group = groups[f"{row['speed']}:{row['motion']}"]
        group["runs"] += 1
        group["move_passes"] += int(row["move_gate_pass"])
        group["stop_passes"] += int(row["stop_gate_pass"])
        group["full_passes"] += int(row["full_gate_pass"])
    decision = "COMPLETE_REPAIRED_PANEL_DIAGNOSTIC_NO_PROMOTION"
    result = {
        "schema": "x2_official_state_feedback_brake_panel_repair_result_v1",
        "decision": decision, "prereg_sha256": sha256(args.prereg),
        "runs": 24, "frozen_attempt0_runs": 18, "repair_runs": 6,
        "startup_gate_passes": sum(row["startup_gate_pass"] for row in rows),
        "move_gate_passes": sum(row["move_gate_pass"] for row in rows),
        "stop_gate_passes": sum(row["stop_gate_pass"] for row in rows),
        "full_gate_passes": sum(row["full_gate_pass"] for row in rows),
        "groups": dict(groups), "cases": rows,
        "scientific_status": "registered_repair_diagnostic_only",
        "permissions": prereg["decision_routes"][decision],
        "optimizer_steps": 0, "backward_calls": 0, "checkpoint_writes": 0,
        "baidu_access_or_upload": False, "github_remote_access_or_push": False
    }
    atomic_json(args.result, result)
    lines = ["# X2 state-feedback brake repaired panel", "", f"Decision: `{decision}`.", "",
             f"Combined gates: startup {result['startup_gate_passes']}/24, move {result['move_gate_passes']}/24, stop {result['stop_gate_passes']}/24, full {result['full_gate_passes']}/24.", ""]
    for name, group in sorted(groups.items()):
        lines.append(f"- {name}: move {group['move_passes']}/{group['runs']}, stop {group['stop_passes']}/{group['runs']}, full {group['full_passes']}/{group['runs']}")
    lines.extend(["", "This repairs configuration evidence only and cannot promote the invalid attempt-0 panel.", ""])
    atomic_text(args.markdown, "\n".join(lines))
    atomic_text(Path(str(args.markdown) + ".sha256"), f"{sha256(args.markdown)}  {args.markdown.name}\n")
    print(json.dumps({"decision": decision, "groups": dict(groups)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
