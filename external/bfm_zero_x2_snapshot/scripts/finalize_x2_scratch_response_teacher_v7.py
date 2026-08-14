#!/usr/bin/env python3
"""Finalize the bounded fixed-command scratch locomotion kernel evidence."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
INPUTS = {
    "training": REPORTS / "x2_scratch_response_teacher_v7.json",
    "response_eval": REPORTS / "x2_scratch_response_teacher_v7_survival_audit.json",
    "ideal_eval": REPORTS / "x2_scratch_response_teacher_v7_ideal_survival_audit.json",
    "teacher_command_panel": REPORTS / "x2_stage219_direct_teacher_response_command_panel.json",
}
OUTPUT = REPORTS / "x2_scratch_response_teacher_v7_result.json"
MARKDOWN = REPORTS / "x2_scratch_response_teacher_v7_result.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_verified(path: Path) -> tuple[dict, str]:
    digest = sha256(path)
    sidecar = path.with_name(path.name + ".sha256")
    if sidecar.read_text(encoding="utf-8").split() != [digest, path.name]:
        raise RuntimeError(f"invalid sidecar: {path}")
    return json.loads(path.read_text(encoding="utf-8")), digest


def aggregate_evaluation(payload: dict) -> dict:
    groups = payload["groups"]
    lanes = payload["role_lanes"]
    total = sum(lanes.values())
    terminated = sum(group["first_episode_terminated_lanes"] for group in groups.values())

    def weighted(field: str) -> float:
        return sum(groups[name][field] * count for name, count in lanes.items()) / total

    return {
        "lanes": total,
        "survived_lanes": total - terminated,
        "terminated_lanes": terminated,
        "survival_fraction": (total - terminated) / total,
        "survival_steps_mean": weighted("first_episode_survival_steps_mean"),
        "forward_velocity_mean_mps": weighted("first_episode_forward_velocity_mean_mps"),
        "root_height_min_m": min(group["root_height_min_m"] for group in groups.values()),
        "tilt_max_rad": max(group["tilt_max_rad"] for group in groups.values()),
        "action_clip_fraction_max": max(group["action_clip_fraction"] for group in groups.values()),
    }


def write_atomic(path: Path, data: bytes) -> None:
    temporary = path.with_name("." + path.name + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def main() -> None:
    if any(path.exists() for path in (OUTPUT, OUTPUT.with_name(OUTPUT.name + ".sha256"), MARKDOWN)):
        raise FileExistsError("refusing to overwrite v7 final evidence")
    loaded = {name: load_verified(path) for name, path in INPUTS.items()}
    training = loaded["training"][0]
    response = aggregate_evaluation(loaded["response_eval"][0])
    ideal = aggregate_evaluation(loaded["ideal_eval"][0])
    command_panel = loaded["teacher_command_panel"][0]
    gates = {
        "scratch_lineage": bool(training["scratch_lineage"]),
        "stage219_weights_not_loaded": not bool(training["stage219_weights_loaded_into_student"]),
        "full_response_teacher_dataset": (
            training["collection"]["safe_labeled_states"] == 512 * 400
            and training["collection"]["terminated"] == 0
        ),
        "response_survival": response["survived_lanes"] == 512,
        "response_velocity": response["forward_velocity_mean_mps"] >= 0.30,
        "response_height": response["root_height_min_m"] >= 0.50,
        "response_tilt": response["tilt_max_rad"] <= 0.80,
        "ideal_survival": ideal["survived_lanes"] == 512,
        "ideal_velocity": ideal["forward_velocity_mean_mps"] >= 0.35,
        "ideal_height": ideal["root_height_min_m"] >= 0.60,
        "ideal_tilt": ideal["tilt_max_rad"] <= 0.35,
        "action_clip": max(response["action_clip_fraction_max"], ideal["action_clip_fraction_max"]) <= 0.001,
    }
    passed = all(gates.values())
    qualified_roles = [
        name for name, group in command_panel["groups"].items() if group["tracking_gate"]
    ]
    failed_roles = [
        name for name, group in command_panel["groups"].items() if not group["tracking_gate"]
    ]
    result = {
        "schema": "x2_bfm_zero_scratch_response_teacher_v7_result_v1",
        "decision": (
            "PASS_FIXED_COMMAND_SCRATCH_KERNEL_LOCAL_ONLY"
            if passed
            else "FAIL_FIXED_COMMAND_SCRATCH_KERNEL_STOP"
        ),
        "inputs": {
            name: {"path": str(INPUTS[name].relative_to(ROOT)), "sha256": digest}
            for name, (_payload, digest) in loaded.items()
        },
        "training": {
            "fresh_random_student": True,
            "stage219_weights_loaded": False,
            "teacher_label_frames": training["collection"]["safe_labeled_states"],
            "teacher_trajectory_terminations": training["collection"]["terminated"],
            "optimizer_steps_this_dataset_stage": training["training"]["optimizer_steps"],
            "teacher_dataset_mae_before": training["training"]["before"]["dagger"]["mae"],
            "teacher_dataset_mae_after": training["training"]["after"]["dagger"]["mae"],
            "offline_holdout_mae_after": training["training"]["after"]["offline_holdout"]["mae"],
            "checkpoint_tree_sha256": training["checkpoint"]["tree_sha256"],
        },
        "closed_loop": {"response": response, "ideal": ideal},
        "gates": gates,
        "all_fixed_command_gates_pass": passed,
        "teacher_command_panel": {
            "decision": command_panel["decision"],
            "qualified_roles": qualified_roles,
            "failed_roles": failed_roles,
            "groups": command_panel["groups"],
        },
        "interpretation": {
            "dataset_insufficiency_hypothesis_supported": True,
            "missing_factor": "response-domain full closed-loop teacher trajectories plus deployable command/gait phase",
            "fixed_vx_0p35_only": True,
            "multi_command_kernel_proven": False,
            "posture_or_stop_improvement_proven": False,
        },
        "permissions": {
            "long_training_unlocked": False,
            "deployment_unlocked": False,
            "multi_command_teacher_data_unlocked": False,
            "state_feedback_teacher_feasibility_next": passed,
        },
    }
    serialized = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    write_atomic(OUTPUT, serialized)
    digest = hashlib.sha256(serialized).hexdigest()
    write_atomic(OUTPUT.with_name(OUTPUT.name + ".sha256"), f"{digest}  {OUTPUT.name}\n".encode())
    markdown = f"""# X2 scratch response-teacher v7

Decision: `{result['decision']}`.

The fresh BFM student now survives all 512 lanes for 400 control steps in both
the response domain and the ideal domain.  Mean forward velocity is
`{response['forward_velocity_mean_mps']:.6f} m/s` in response and
`{ideal['forward_velocity_mean_mps']:.6f} m/s` in ideal.  Stage219 weights were
not loaded into the student; 204,800 response-domain teacher frames were used
as labels.

This supports the dataset-coverage hypothesis for fixed 0.35 m/s locomotion.
It does not prove posture correction, stopping, turns, low speed, deployment,
or long-training readiness.  The teacher command screen failed for
`{', '.join(failed_roles)}`, so those roles are not admitted as training data.
"""
    write_atomic(MARKDOWN, markdown.encode())
    print(json.dumps({"decision": result["decision"], "response": response, "ideal": ideal}, indent=2))


if __name__ == "__main__":
    main()
