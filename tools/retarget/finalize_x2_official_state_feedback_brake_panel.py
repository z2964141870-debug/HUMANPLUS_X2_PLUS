#!/usr/bin/env python3
"""Finalize the 24-case official state-feedback brake panel."""

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


def expected_cases(prereg: dict) -> list[dict]:
    rows = []
    index = 0
    for speed in prereg["matrix"]["speed_order"]:
        vx = prereg["matrix"]["speeds"][speed]
        for motion in prereg["matrix"]["motion_order"]:
            config = prereg["matrix"]["motions"][motion]
            for repeat in range(1, config["repeats"] + 1):
                candidate = f"x2_sfbrake_{speed}_{motion}_r{repeat}"
                rows.append({
                    "index": index,
                    "speed": speed,
                    "motion": motion,
                    "repeat": repeat,
                    "vx": vx,
                    "wz": config["fixed_wz_radps"],
                    "mirror": config["mirror_policy"],
                    "candidate": candidate,
                    "source": f"stage264_{speed}_{motion}_r{repeat}",
                    "label": f"x2_official_state_feedback_panel_case{index}",
                    "raw": str(Path(prereg["outputs"]["raw_root"]) / f"{candidate}.json"),
                    "resource": str(Path(prereg["outputs"]["report_root"]) / f"{candidate}_resource.json"),
                    "supervisor": config["supervisor"],
                })
                index += 1
    if len(rows) != 24:
        raise RuntimeError("matrix must resolve to exactly 24 cases")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--case", type=Path, action="append", required=True)
    parser.add_argument("--resource", type=Path, action="append", required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    if len(args.case) != 24 or len(args.resource) != 24:
        raise RuntimeError("exactly 24 cases/resources are required")
    for path in (
        args.result, args.result.with_name(args.result.name + ".sha256"),
        args.markdown, args.markdown.with_name(args.markdown.name + ".sha256"),
    ):
        if path.exists():
            raise FileExistsError(path)
    for path in (args.prereg, *args.case, *args.resource):
        verify_sidecar(path)
    prereg = json.loads(args.prereg.read_text(encoding="utf-8"))
    if prereg.get("schema") != "x2_official_state_feedback_brake_panel_prereg_v1":
        raise RuntimeError("unexpected preregistration schema")
    for section in ("immutable_code", "immutable_inputs", "immutable_evidence"):
        for name, record in prereg[section].items():
            path = Path(record["path"])
            if not path.is_file() or sha256(path) != record["sha256"]:
                raise RuntimeError(f"immutable drift: {section}.{name}")

    expected = expected_cases(prereg)
    source = json.loads(Path(prereg["immutable_evidence"]["source_matrix"]["path"]).read_text())
    if [row["case"] for row in source["cases"]] != [row["source"] for row in expected]:
        raise RuntimeError("source-matrix order/content mismatch")
    contract = prereg["controller_contract"]
    rows = []
    for spec, case_path, resource_path in zip(expected, args.case, args.resource):
        if case_path.resolve() != Path(spec["raw"]).resolve():
            raise RuntimeError("raw path/order mismatch")
        if resource_path.resolve() != Path(spec["resource"]).resolve():
            raise RuntimeError("resource path/order mismatch")
        summary = json.loads(case_path.read_text(encoding="utf-8"))["summary"]
        resource = json.loads(resource_path.read_text(encoding="utf-8"))
        command_env = {}
        for token in resource.get("command", []):
            if "=" in token:
                name, value = token.split("=", 1)
                command_env[name] = value
        controller_checks = {
            "controller": summary.get("stop_controller") == contract["stop_controller"],
            "gain": close(summary.get("stop_brake_gain"), contract["gain"]),
            "limit": close(summary.get("stop_brake_limit_mps"), contract["limit_mps"]),
            "template_speed": close(summary.get("stop_brake_template_speed_mps"), contract["template_speed_mps"]),
            "template_floor": close(summary.get("stop_brake_template_floor"), contract["template_floor"]),
            "resource_controller": command_env.get("STOP_CONTROLLER") == contract["stop_controller"],
            "resource_gain": close(float(command_env.get("STOP_BRAKE_GAIN", "nan")), contract["gain"]),
            "resource_limit": close(float(command_env.get("STOP_BRAKE_LIMIT", "nan")), contract["limit_mps"]),
            "resource_template_speed": close(
                float(command_env.get("STOP_BRAKE_TEMPLATE_SPEED", "nan")),
                contract["template_speed_mps"],
            ),
            "resource_template_floor": close(
                float(command_env.get("STOP_BRAKE_TEMPLATE_FLOOR", "nan")),
                contract["template_floor"],
            ),
            "resource_hold_min": close(
                float(command_env.get("EVENT_HOLD_MIN_SECONDS", "nan")),
                contract["hold_min_seconds"],
            ),
            "resource_hold_speed": close(float(command_env.get("EVENT_HOLD_SPEED", "nan")), contract["hold_speed_mps"]),
            "resource_attempts": command_env.get("MAX_ATTEMPTS") == "1",
            "no_emergency_latch": summary.get("stop_emergency_tilt_rad") is None,
        }
        motion_checks = {
            "vx": close(summary.get("command_vx_mps"), spec["vx"]),
            "wz": (
                summary.get("fixed_wz_radps") is None
                if spec["wz"] is None
                else close(summary.get("fixed_wz_radps"), spec["wz"])
            ),
            "mirror": bool(summary.get("mirror_policy")) == spec["mirror"],
            "supervisor": summary.get("action_bias_mode") == spec["supervisor"],
            "case": command_env.get("CASE_NAME") == spec["candidate"],
        }
        resource_checks = {
            "schema": resource.get("schema") == "x2_gpu_deadline_ledger_phase76_v1",
            "label": resource.get("label") == spec["label"],
            "physics_exit": resource.get("raw_returncode") in (0, 2),
            "effective_exit": resource.get("exit_code") == resource.get("raw_returncode"),
            "autonomous": not any(resource.get(name, False) for name in (
                "timed_out", "term_sent", "kill_sent", "forced_cleanup"
            )),
            "elapsed": float(resource["elapsed_s"]) <= prereg["resources"]["elapsed_max_s"],
            "free": int(resource["disk_after"]["free_bytes"]) >= prereg["resources"]["free_after_bytes_min"],
        }
        if not all(controller_checks.values()):
            raise RuntimeError(f"controller mismatch: {spec['candidate']}: {controller_checks}")
        if not all(motion_checks.values()):
            raise RuntimeError(f"motion mismatch: {spec['candidate']}: {motion_checks}")
        if not all(resource_checks.values()):
            raise RuntimeError(f"resource invalid: {spec['candidate']}: {resource_checks}")
        rows.append({
            "case_name": spec["candidate"], "source_case": spec["source"],
            "speed": spec["speed"], "motion": spec["motion"], "repeat": spec["repeat"],
            "raw_sha256": sha256(case_path), "resource_sha256": sha256(resource_path),
            "controller_checks": controller_checks, "motion_checks": motion_checks,
            "resource_checks": resource_checks,
            "latch_observed": summary.get("stop_hold_latch_s") is not None,
            "stop_hold_latch_s": summary.get("stop_hold_latch_s"),
            "startup_gate_pass": bool(summary["startup_gate_pass"]),
            "move_gate_pass": bool(summary["move_gate_pass"]),
            "stop_gate_pass": bool(summary["stop_gate_pass"]),
            "full_gate_pass": bool(summary["full_gate_pass"]),
            "move_root_pitch_mean_rad": summary["move_root_pitch_mean_rad"],
            "stop_root_pitch_mean_rad": summary["stop_root_pitch_mean_rad"],
            "stop_root_xy_drift_m": summary["stop_root_xy_drift_m"],
            "stop_last_1s_mean_speed_mps": summary["stop_last_1s_mean_speed_mps"],
            "stop_root_z_min_m": summary["stop_root_z_min_m"],
            "stop_root_tilt_max_rad": summary["stop_root_tilt_max_rad"],
        })

    passes = sum(row["full_gate_pass"] and row["latch_observed"] for row in rows)
    if passes == 24:
        decision = "PASS_OFFICIAL_STATE_FEEDBACK_BRAKE_24_OF_24_LOCAL_CANDIDATE"
    elif passes >= 23 and all(row["startup_gate_pass"] and row["move_gate_pass"] for row in rows):
        decision = "PARTIAL_OFFICIAL_STATE_FEEDBACK_BRAKE_PANEL_STOP"
    else:
        decision = "FAIL_OFFICIAL_STATE_FEEDBACK_BRAKE_PANEL_STOP"
    groups = defaultdict(lambda: {"passes": 0, "runs": 0})
    for row in rows:
        group = groups[f"{row['speed']}:{row['motion']}"]
        group["runs"] += 1
        group["passes"] += int(row["full_gate_pass"] and row["latch_observed"])
    result = {
        "schema": "x2_official_state_feedback_brake_panel_result_v1",
        "decision": decision, "prereg_sha256": sha256(args.prereg),
        "passes": passes, "runs": 24, "groups": dict(groups), "cases": rows,
        "permissions": prereg["decision_routes"][decision],
        "optimizer_steps": 0, "backward_calls": 0, "checkpoint_writes": 0,
        "baidu_access_or_upload": False, "github_remote_access_or_push": False,
    }
    atomic_json(args.result, result)
    lines = ["# X2 official state-feedback brake 24-case panel", "",
             f"Decision: `{decision}` — {passes}/24.", ""]
    for name, group in sorted(groups.items()):
        lines.append(f"- {name}: {group['passes']}/{group['runs']}")
    lines.extend(["", "Zero training was performed; this is a local simulation candidate only.", ""])
    atomic_text(args.markdown, "\n".join(lines))
    atomic_text(args.markdown.with_name(args.markdown.name + ".sha256"),
                f"{sha256(args.markdown)}  {args.markdown.name}\n")
    print(json.dumps({"decision": decision, "passes": passes, "groups": dict(groups)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
