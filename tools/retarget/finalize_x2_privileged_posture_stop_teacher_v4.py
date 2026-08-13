#!/usr/bin/env python3
"""Independently finalize the X2 posture/stop teacher feasibility screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from cwi_x2.privileged_posture_stop_teacher import (
    SEGMENTS,
    feasibility_gates,
    summarize_segment,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sidecar(path: Path) -> None:
    sidecar = path.with_name(path.name + ".sha256")
    expected = f"{sha256(path)}  {path.name}\n"
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8") != expected:
        raise RuntimeError(f"sidecar mismatch: {path}")


def atomic_text(path: Path, text: str) -> None:
    sidecar = path.with_name(path.name + ".sha256")
    if path.exists() or sidecar.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    sidecar.write_text(f"{sha256(path)}  {path.name}\n", encoding="utf-8")
    with sidecar.open("r+b") as stream:
        os.fsync(stream.fileno())


def summarize(rows: list[dict]) -> dict[str, dict[str, float]]:
    return {
        segment: summarize_segment([row for row in rows if row["segment"] == segment])
        for segment in SEGMENTS
    }


def paired_bootstrap(source_rows, candidate_rows, field, *, seed: int, draws: int):
    source = {(int(row["env_id"]), row["segment"]): row for row in source_rows}
    candidate = {(int(row["env_id"]), row["segment"]): row for row in candidate_rows}
    if set(source) != set(candidate):
        raise RuntimeError("source/candidate validation keys differ")
    keys = sorted(source)
    values = np.asarray(
        [candidate[key][field] - source[key][field] for key in keys],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise RuntimeError("non-finite paired validation values")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(draws, len(values)))
    sample = values[indices].mean(axis=1)
    return {
        "point": float(values.mean()),
        "ci95": [float(np.quantile(sample, 0.025)), float(np.quantile(sample, 0.975))],
        "environment_pairs": len(values),
        "draws": draws,
    }


def close(left, right, tolerance=1.0e-9):
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(close(left[key], right[key], tolerance) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(close(a, b, tolerance) for a, b in zip(left, right))
    if isinstance(left, (float, int)) and isinstance(right, (float, int)):
        return abs(float(left) - float(right)) <= tolerance * max(1.0, abs(float(left)), abs(float(right)))
    return left == right


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--resource", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.prereg, args.screen, args.resource):
        verify_sidecar(path)
    prereg = json.loads(args.prereg.read_text(encoding="utf-8"))
    for section in ("immutable_code", "immutable_inputs", "immutable_evidence"):
        for name, record in prereg[section].items():
            path = Path(record["path"])
            if not path.is_file() or sha256(path) != record["sha256"]:
                raise RuntimeError(f"{section} mismatch during finalization: {name}")
    screen = json.loads(args.screen.read_text(encoding="utf-8"))
    resource = json.loads(args.resource.read_text(encoding="utf-8"))
    if screen["preregistration_sha256"] != sha256(args.prereg):
        raise RuntimeError("screen/preregistration link mismatch")
    expected_outputs = prereg["outputs"]
    for name, path in (("screen", args.screen), ("resource", args.resource), ("result", args.result), ("markdown", args.markdown)):
        if Path(expected_outputs[name]).resolve() != path.resolve():
            raise RuntimeError(f"output path mismatch: {name}")
    limits = prereg["gates"]
    per_env = screen["validation"]["per_env"]
    expected_rows = prereg["runtime"]["num_envs"] * len(SEGMENTS)
    if any(len(per_env[name]) != expected_rows for name in ("source_direct", "fixed_half_direct", "feedback_gated")):
        raise RuntimeError("validation row count changed")
    summaries = {name: summarize(rows) for name, rows in per_env.items()}
    if not close(summaries, screen["validation"]["summaries"]):
        raise RuntimeError("screen summaries do not reproduce from per-env records")
    gates = feasibility_gates(summaries["source_direct"], summaries["feedback_gated"], limits)
    validation = prereg["validation"]
    cruise_bootstrap = paired_bootstrap(
        [row for row in per_env["source_direct"] if row["segment"] == "cruise"],
        [row for row in per_env["feedback_gated"] if row["segment"] == "cruise"],
        "pitch_mean_rad", seed=validation["bootstrap_seed"], draws=validation["bootstrap_draws"],
    )
    decel_bootstrap = paired_bootstrap(
        [row for row in per_env["source_direct"] if row["segment"] == "decelerate"],
        [row for row in per_env["feedback_gated"] if row["segment"] == "decelerate"],
        "pitch_mean_rad", seed=validation["bootstrap_seed"] + 1, draws=validation["bootstrap_draws"],
    )
    gates["cruise_pitch_ci"] = cruise_bootstrap["ci95"][0] >= limits["cruise_pitch_ci_lower_min"]
    gates["decel_pitch_ci"] = decel_bootstrap["ci95"][0] >= limits["decel_pitch_ci_lower_min"]
    gates["reset_fingerprint_exact"] = bool(screen["technical_checks"]["reset_fingerprint_exact"])
    if not close(gates, screen["validation"]["gates"]):
        raise RuntimeError("screen gates do not reproduce independently")
    if not close(cruise_bootstrap, screen["validation"]["cruise_pitch_paired_bootstrap"]):
        raise RuntimeError("cruise bootstrap does not reproduce")
    if not close(decel_bootstrap, screen["validation"]["decelerate_pitch_paired_bootstrap"]):
        raise RuntimeError("deceleration bootstrap does not reproduce")

    resource_limits = prereg["resource_limits"]
    resource_checks = {
        "exit_zero": resource["exit_code"] == 0 and resource["raw_returncode"] == 0,
        "autonomous": bool(resource["autonomous_exit"]) and not any(
            resource[name] for name in ("timed_out", "term_sent", "kill_sent", "forced_cleanup")
        ),
        "deadline": resource["elapsed_s"] <= resource_limits["deadline_seconds"],
        "gpu": resource.get("gpu") is not None and resource["gpu"]["memory_used_peak_mib"] <= resource_limits["maximum_gpu_memory_mib"],
        "disk_delta": resource["disk_used_delta_bytes"] <= resource_limits["maximum_disk_delta_bytes"],
        "disk_after": resource["disk_after"]["free_bytes"] >= resource_limits["minimum_free_disk_after_bytes"],
        "screen_size": args.screen.stat().st_size <= resource_limits["maximum_result_bytes"],
    }
    technical_valid = all(screen["technical_checks"].values()) and all(resource_checks.values())
    full_pass = technical_valid and all(gates.values())
    cruise_names = [
        "cruise_pitch_mean", "cruise_pitch_p05", "decel_pitch_mean", "decel_pitch_p05",
        "cruise_zero_termination", "cruise_zero_timeout", "decel_zero_termination", "decel_zero_timeout",
        "cruise_velocity", "cruise_lateral", "cruise_yaw", "cruise_support", "cruise_slip", "cruise_flight",
        "cruise_root_height", "cruise_tilt", "decel_velocity", "decel_lateral", "decel_yaw",
        "decel_support", "decel_slip", "decel_flight", "decel_root_height", "decel_tilt",
        "cruise_pitch_ci", "decel_pitch_ci", "residual_bound", "residual_slew", "no_action_clip",
        "physical_effectiveness", "physical_sign",
    ]
    cruise_pass = technical_valid and all(gates[name] for name in cruise_names)
    hold_names = [name for name in gates if name.startswith("hold_") or name.startswith("candidate_")]
    hold_pass = technical_valid and all(gates[name] for name in hold_names)
    if not technical_valid:
        decision = "FAIL_IMPLEMENTATION_STOP"
    elif full_pass:
        decision = "PASS_STATE_FEEDBACK_TEACHER_BC_PREREG_ONLY"
    elif cruise_pass and not hold_pass:
        decision = "CRUISE_FEASIBLE_BRAKE_HOLD_SKILL_PREREG_ONLY"
    else:
        decision = "FAIL_STATE_FEEDBACK_TEACHER_NEW_ACTOR_PREREG_ONLY"
    if screen["decision"] != decision:
        raise RuntimeError("runner and independent finalizer decision differ")
    result = {
        "schema": "x2_privileged_posture_stop_teacher_final_v1",
        "preregistration_sha256": sha256(args.prereg),
        "screen_sha256": sha256(args.screen),
        "resource_sha256": sha256(args.resource),
        "decision": decision,
        "technical_valid": technical_valid,
        "resource_checks": resource_checks,
        "gates": gates,
        "summaries": summaries,
        "paired_bootstrap": {"cruise": cruise_bootstrap, "decelerate": decel_bootstrap},
        "search_best_lexicographic_key": screen["search"]["best_lexicographic_key"],
        "permissions": {
            "bc_dagger_preregistration_unlocked": decision == "PASS_STATE_FEEDBACK_TEACHER_BC_PREREG_ONLY",
            "brake_hold_skill_preregistration_unlocked": decision == "CRUISE_FEASIBLE_BRAKE_HOLD_SKILL_PREREG_ONLY",
            "new_actor_preregistration_unlocked": decision == "FAIL_STATE_FEEDBACK_TEACHER_NEW_ACTOR_PREREG_ONLY",
            "training_unlocked": False,
            "deployment_unlocked": False,
            "whole_body_training_unlocked": False,
        },
    }
    atomic_text(args.result, json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    markdown = (
        "# X2 state-feedback posture/stop teacher v4\n\n"
        f"Decision: `{decision}`.\n\n"
        "This was a zero-optimizer privileged feasibility experiment.  It does not unlock "
        "training or deployment; it only selects the next separately preregistered route.\n"
    )
    atomic_text(args.markdown, markdown)
    print(json.dumps({"decision": decision, "technical_valid": technical_valid}, indent=2))


if __name__ == "__main__":
    main()
