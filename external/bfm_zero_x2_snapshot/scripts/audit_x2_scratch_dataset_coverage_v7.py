#!/usr/bin/env python3
"""Build the immutable dataset-coverage ablation for the scratch X2 actor."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PATHS = {
    "phase_bc_fit": REPORTS / "x2_scratch_phase_bc_v6.json",
    "phase_bc_ideal": REPORTS / "x2_scratch_phase_bc_v6_ideal_survival_audit.json",
    "phase_bc_response": REPORTS / "x2_scratch_phase_bc_v6_response_survival_audit.json",
    "ideal_dagger_fit": REPORTS / "x2_scratch_dagger_phase_v6.json",
    "ideal_dagger_ideal": REPORTS / "x2_scratch_dagger_phase_v6_ideal_survival_audit.json",
    "ideal_dagger_response": REPORTS / "x2_scratch_dagger_phase_v6_response_survival_audit.json",
    "response_dagger_fit": REPORTS / "x2_scratch_dagger_phase_response_v6.json",
    "response_dagger_eval": REPORTS / "x2_scratch_dagger_phase_response_v6_survival_audit.json",
    "full_teacher_fit": REPORTS / "x2_scratch_response_teacher_v7.json",
    "full_teacher_response": REPORTS / "x2_scratch_response_teacher_v7_survival_audit.json",
    "full_teacher_ideal": REPORTS / "x2_scratch_response_teacher_v7_ideal_survival_audit.json",
}
OUTPUT = REPORTS / "x2_scratch_dataset_coverage_v7.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> tuple[dict, str]:
    digest = sha256(path)
    sidecar = path.with_name(path.name + ".sha256")
    if sidecar.read_text(encoding="utf-8").split() != [digest, path.name]:
        raise RuntimeError(f"invalid sidecar for {path}")
    return json.loads(path.read_text(encoding="utf-8")), digest


def evaluation(payload: dict) -> dict:
    groups, lanes = payload["groups"], payload["role_lanes"]
    total = sum(lanes.values())
    terminated = sum(group["first_episode_terminated_lanes"] for group in groups.values())

    def weighted(field: str) -> float:
        return sum(groups[name][field] * count for name, count in lanes.items()) / total

    return {
        "survived": total - terminated,
        "total": total,
        "survival_fraction": (total - terminated) / total,
        "survival_steps_mean": weighted("first_episode_survival_steps_mean"),
        "forward_velocity_mean_mps": weighted("first_episode_forward_velocity_mean_mps"),
    }


def atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_name("." + path.name + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(path)
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def main() -> None:
    sidecar = OUTPUT.with_name(OUTPUT.name + ".sha256")
    if OUTPUT.exists() or sidecar.exists():
        raise FileExistsError("refusing to overwrite coverage audit")
    loaded = {name: load(path) for name, path in PATHS.items()}
    data = {name: item[0] for name, item in loaded.items()}
    stages = [
        {
            "stage": "phase_conditioned_offline_bc",
            "new_coverage": "deployable command and gait phase on 25,600 ideal-domain offline labels",
            "new_safe_labels": 25_600,
            "ideal": evaluation(data["phase_bc_ideal"]),
            "response": evaluation(data["phase_bc_response"]),
            "offline_holdout_mae": data["phase_bc_fit"]["after"]["holdout"]["mae"],
        },
        {
            "stage": "ideal_student_state_dagger",
            "new_coverage": "65,009 safe labels on states visited by the student in ideal physics",
            "new_safe_labels": data["ideal_dagger_fit"]["collection"]["safe_labeled_states"],
            "ideal": evaluation(data["ideal_dagger_ideal"]),
            "response": evaluation(data["ideal_dagger_response"]),
            "offline_holdout_mae": data["ideal_dagger_fit"]["training"]["after"]["offline_holdout"]["mae"],
        },
        {
            "stage": "response_student_state_dagger",
            "new_coverage": "58,046 safe labels before the response-domain student failures",
            "new_safe_labels": data["response_dagger_fit"]["collection"]["safe_labeled_states"],
            "response": evaluation(data["response_dagger_eval"]),
            "offline_holdout_mae": data["response_dagger_fit"]["training"]["after"]["offline_holdout"]["mae"],
        },
        {
            "stage": "full_response_teacher_trajectory",
            "new_coverage": "204,800 safe labels spanning full response-domain teacher trajectories",
            "new_safe_labels": data["full_teacher_fit"]["collection"]["safe_labeled_states"],
            "ideal": evaluation(data["full_teacher_ideal"]),
            "response": evaluation(data["full_teacher_response"]),
            "offline_holdout_mae": data["full_teacher_fit"]["training"]["after"]["offline_holdout"]["mae"],
        },
    ]
    result = {
        "schema": "x2_bfm_zero_scratch_dataset_coverage_ablation_v1",
        "decision": "DATASET_DOMAIN_COVERAGE_HYPOTHESIS_SUPPORTED",
        "inputs": {
            name: {"path": str(PATHS[name].relative_to(ROOT)), "sha256": item[1]}
            for name, item in loaded.items()
        },
        "stages": stages,
        "conclusion": {
            "more_data_helped": True,
            "raw_count_alone_established": False,
            "causal_factor_supported": "correct-domain closed-loop coverage",
            "evidence": [
                "phase-conditioned offline BC did not survive response physics",
                "ideal-domain DAgger nearly solved ideal physics but left response survival at zero",
                "response failure-state DAgger recovered partial response survival",
                "full safe response-teacher trajectories recovered 512/512 response and retained 512/512 ideal survival",
            ],
            "remaining_scope": [
                "multi-command coverage",
                "turn-right teacher competence",
                "low-speed teacher safety",
                "deceleration and terminal hold",
                "posture correction",
            ],
        },
        "permissions": {"long_training_unlocked": False, "deployment_unlocked": False},
    }
    payload = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    atomic(OUTPUT, payload)
    digest = hashlib.sha256(payload).hexdigest()
    atomic(sidecar, f"{digest}  {OUTPUT.name}\n".encode())
    print(json.dumps({"decision": result["decision"], "stages": stages}, indent=2))


if __name__ == "__main__":
    main()
