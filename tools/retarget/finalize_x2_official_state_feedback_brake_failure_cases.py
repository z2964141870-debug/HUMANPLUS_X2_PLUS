#!/usr/bin/env python3
"""Finalize the two-case official state-feedback brake feasibility run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
    expected = f"{sha256(path)}  {path.name}\n"
    if sidecar.read_text(encoding="utf-8") != expected:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--case", type=Path, action="append", required=True)
    parser.add_argument("--resource", type=Path, action="append", required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    if len(args.case) != 2 or len(args.resource) != 2:
        raise RuntimeError("exactly two cases/resources are required")
    outputs = (
        args.result, args.result.with_name(args.result.name + ".sha256"),
        args.markdown, args.markdown.with_name(args.markdown.name + ".sha256"),
    )
    if any(path.exists() for path in outputs):
        raise FileExistsError("a final output already exists")
    for path in (args.prereg, *args.case, *args.resource):
        verify_sidecar(path)

    prereg = json.loads(args.prereg.read_text(encoding="utf-8"))
    if prereg.get("schema") != "x2_official_state_feedback_brake_prereg_v1":
        raise RuntimeError("unexpected preregistration schema")
    for section in ("immutable_code", "immutable_inputs", "immutable_evidence"):
        for name, record in prereg[section].items():
            path = Path(record["path"])
            if not path.is_file() or sha256(path) != record["sha256"]:
                raise RuntimeError(f"immutable drift: {section}.{name}")

    contract = prereg["controller_contract"]
    command_tokens = {
        f"STOP_CONTROLLER={contract['stop_controller']}",
        f"STOP_BRAKE_GAIN={contract['gain']}",
        f"STOP_BRAKE_LIMIT={contract['limit_mps']}",
        f"STOP_BRAKE_TEMPLATE_SPEED={contract['template_speed_mps']}",
        f"STOP_BRAKE_TEMPLATE_FLOOR={contract['template_floor']}",
        f"EVENT_HOLD_MIN_SECONDS={contract['hold_min_seconds']}",
        f"EVENT_HOLD_SPEED={contract['hold_speed_mps']}",
    }
    rows = []
    for index, (case_path, resource_path) in enumerate(zip(args.case, args.resource)):
        expected = prereg["cases"][index]
        if case_path.resolve() != Path(expected["raw_result"]).resolve():
            raise RuntimeError("raw case path/order mismatch")
        if resource_path.resolve() != Path(expected["resource"]).resolve():
            raise RuntimeError("resource path/order mismatch")
        payload = json.loads(case_path.read_text(encoding="utf-8"))
        summary = payload["summary"]
        resource = json.loads(resource_path.read_text(encoding="utf-8"))
        command = set(resource.get("command", []))
        controller_checks = {
            "controller": summary.get("stop_controller") == contract["stop_controller"],
            "gain": close(summary.get("stop_brake_gain"), contract["gain"]),
            "limit": close(summary.get("stop_brake_limit_mps"), contract["limit_mps"]),
            "template_speed": close(
                summary.get("stop_brake_template_speed_mps"), contract["template_speed_mps"]
            ),
            "template_floor": close(
                summary.get("stop_brake_template_floor"), contract["template_floor"]
            ),
            "resource_tokens": command_tokens.issubset(command),
            "no_emergency_latch": summary.get("stop_emergency_tilt_rad") is None,
        }
        if not all(controller_checks.values()):
            raise RuntimeError(f"controller contract mismatch: {controller_checks}")
        if abs(float(summary["command_vx_mps"]) - expected["vx_mps"]) > 1.0e-12:
            raise RuntimeError("candidate speed mismatch")
        expected_wz = expected["fixed_wz_radps"]
        actual_wz = summary["fixed_wz_radps"]
        if expected_wz is None:
            if actual_wz is not None:
                raise RuntimeError("unexpected turn command")
        elif actual_wz is None or abs(float(actual_wz) - expected_wz) > 1.0e-12:
            raise RuntimeError("turn command mismatch")
        if bool(summary["mirror_policy"]) is not expected["mirror_policy"]:
            raise RuntimeError("mirror mode mismatch")
        if f"CASE_NAME={expected['case_name']}" not in command:
            raise RuntimeError("resource command case identity mismatch")
        resource_checks = {
            "schema": resource.get("schema") == "x2_gpu_deadline_ledger_phase76_v1",
            "label": resource.get("label") == expected["label"],
            "physics_exit": resource.get("raw_returncode") in (0, 2),
            "effective_exit_matches": resource.get("exit_code") == resource.get("raw_returncode"),
            "no_timeout_or_signal": not any(resource.get(name, False) for name in (
                "timed_out", "term_sent", "kill_sent", "forced_cleanup"
            )),
            "elapsed": float(resource["elapsed_s"]) <= prereg["resources"]["elapsed_max_s"],
            "free_space": int(resource["disk_after"]["free_bytes"])
            >= prereg["resources"]["free_after_bytes_min"],
        }
        if not all(resource_checks.values()):
            raise RuntimeError(f"invalid resource: {expected['case_name']}: {resource_checks}")
        rows.append({
            "case_name": expected["case_name"],
            "raw_result_sha256": sha256(case_path),
            "resource_sha256": sha256(resource_path),
            "resource_checks": resource_checks,
            "controller_checks": controller_checks,
            "latch_observed": summary.get("stop_hold_latch_s") is not None,
            "stop_hold_latch_s": summary.get("stop_hold_latch_s"),
            "startup_gate_pass": bool(summary["startup_gate_pass"]),
            "move_gate_pass": bool(summary["move_gate_pass"]),
            "stop_gate_pass": bool(summary["stop_gate_pass"]),
            "full_gate_pass": bool(summary["full_gate_pass"]),
            "move_root_pitch_mean_rad": summary["move_root_pitch_mean_rad"],
            "stop_root_pitch_mean_rad": summary["stop_root_pitch_mean_rad"],
            "stop_last_1s_mean_speed_mps": summary["stop_last_1s_mean_speed_mps"],
            "stop_root_xy_drift_m": summary["stop_root_xy_drift_m"],
            "stop_root_z_min_m": summary["stop_root_z_min_m"],
            "stop_root_tilt_max_rad": summary["stop_root_tilt_max_rad"],
            "policy_slot_inference_counts": summary["policy_slot_inference_counts"],
        })

    if all(row["full_gate_pass"] and row["latch_observed"] for row in rows):
        decision = "PASS_OFFICIAL_STATE_FEEDBACK_BRAKE_PANEL_PREREG_ONLY"
    elif all(row["startup_gate_pass"] and row["move_gate_pass"] for row in rows):
        decision = "PARTIAL_OFFICIAL_STATE_FEEDBACK_BRAKE_SKILL_ONLY"
    else:
        decision = "FAIL_OFFICIAL_STATE_FEEDBACK_BRAKE_INTEGRATION_STOP"
    result = {
        "schema": "x2_official_state_feedback_brake_result_v1",
        "decision": decision,
        "prereg_sha256": sha256(args.prereg),
        "cases": rows,
        "permissions": prereg["decision_routes"][decision],
        "optimizer_steps": 0,
        "backward_calls": 0,
        "checkpoint_writes": 0,
        "baidu_access_or_upload": False,
        "github_remote_access_or_push": False,
    }
    atomic_json(args.result, result)
    lines = ["# X2 official state-feedback brake", "", f"Decision: `{decision}`.", ""]
    for row in rows:
        lines.append(
            f"- {row['case_name']}: full={row['full_gate_pass']}, "
            f"latch={row['stop_hold_latch_s']}; stop drift={row['stop_root_xy_drift_m']:.6f} m, "
            f"tail speed={row['stop_last_1s_mean_speed_mps']:.6f} m/s, "
            f"zmin={row['stop_root_z_min_m']:.6f} m, tilt={row['stop_root_tilt_max_rad']:.6f} rad."
        )
    lines.extend(["", "Zero training was performed. A pass only unlocks a fresh panel preregistration.", ""])
    atomic_text(args.markdown, "\n".join(lines))
    atomic_text(args.markdown.with_name(args.markdown.name + ".sha256"),
                f"{sha256(args.markdown)}  {args.markdown.name}\n")
    print(json.dumps({"decision": decision, "cases": rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
