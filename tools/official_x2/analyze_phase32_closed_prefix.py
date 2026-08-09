#!/usr/bin/env python3
"""Analyze the one-shot Phase32 closed AimDK MuJoCo prefix capture."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from official_x2.audit_phase29_closed_wrapper_divergence import ISAAC_JOINTS, decode_row
from official_x2.decode_phase32_mujoco_trace import decode
from official_x2.replay_official_trace_direct_mujoco import yaw_tilt


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relevant_steps(rows: list[dict]) -> tuple[int, list[dict]]:
    last_reset = max(row["sequence"] for row in rows if row["call_kind"] == 1)
    return last_reset, [row for row in rows if row["call_kind"] == 3 and row["sequence"] > last_reset]


def root_alignment_score(physics: dict, telemetry: dict) -> float:
    yaw, tilt = yaw_tilt(np.asarray(physics["qpos"][3:7], dtype=np.float64))
    yaw_error = math.atan2(math.sin(yaw - telemetry["root_yaw_rad"]),
                           math.cos(yaw - telemetry["root_yaw_rad"]))
    xyz_error = np.linalg.norm(np.asarray(physics["qpos"][:3]) - np.asarray([
        telemetry["root_x_m"], telemetry["root_y_m"], telemetry["root_z_m"]]))
    return float(xyz_error + abs(yaw_error) + abs(tilt - telemetry["root_tilt_rad"]))


def compare_direct(off_path: Path, on_path: Path) -> dict:
    off = json.loads(off_path.read_text(encoding="utf-8"))
    on = json.loads(on_path.read_text(encoding="utf-8"))
    qpos = float(np.max(np.abs(np.asarray([row["qpos"] for row in off["rows"]]) -
                                      np.asarray([row["qpos"] for row in on["rows"]]))))
    qvel = float(np.max(np.abs(np.asarray([row["qvel"] for row in off["rows"]]) -
                                      np.asarray([row["qvel"] for row in on["rows"]]))))
    return {"qpos_absmax": qpos, "qvel_absmax": qvel,
            "control_wall_s": off["step_wall_s"], "hook_wall_s": on["step_wall_s"],
            "wall_ratio": on["step_wall_s"] / off["step_wall_s"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mmap", type=Path, required=True)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--toy-off", type=Path, required=True)
    parser.add_argument("--toy-on", type=Path, required=True)
    parser.add_argument("--official-off", type=Path, required=True)
    parser.add_argument("--official-on", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    decoded = decode(args.mmap, include_rows=True)
    last_reset, steps = relevant_steps(decoded["rows"])
    rollout = json.loads(args.rollout.read_text(encoding="utf-8"))
    telemetry = rollout["trace"]
    alignments = []
    for index, row in enumerate(telemetry[:14]):
        best = min(steps, key=lambda physics: root_alignment_score(physics, row))
        alignments.append({"telemetry_index": index, "stage": row["stage"],
                           "elapsed_s": row["elapsed_s"], "physics_time_s": best["time_s"],
                           "score": root_alignment_score(best, row)})

    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    qadr = {name: int(model.joint(name).qposadr[0]) for name in ISAAC_JOINTS}
    dadr = {name: int(model.joint(name).dofadr[0]) for name in ISAAC_JOINTS}
    first_stand = telemetry[10]
    expected_qpos, expected_qvel = decode_row(model, first_stand, qadr, dadr)
    stand_step = min(steps, key=lambda row: root_alignment_score(row, first_stand))
    stand_metrics = {
        "telemetry_index": 10,
        "physics_time_s": stand_step["time_s"],
        "joint_q_absmax_vs_obs": max(abs(stand_step["qpos"][qadr[name]] - expected_qpos[qadr[name]]) for name in ISAAC_JOINTS),
        "joint_dq_absmax_vs_obs": max(abs(stand_step["qvel"][dadr[name]] - expected_qvel[dadr[name]]) for name in ISAAC_JOINTS),
        "root_xyz_absmax_vs_telemetry": max(abs(stand_step["qpos"][index] - expected_qpos[index]) for index in range(3)),
        "root_velocity_absmax_vs_obs_reconstruction": max(abs(stand_step["qvel"][index] - expected_qvel[index]) for index in range(6)),
        "qacc_l2": float(np.linalg.norm(stand_step["qacc"])),
        "qacc_warmstart_l2": float(np.linalg.norm(stand_step["qacc_warmstart"])),
        "ctrl_l2": float(np.linalg.norm(stand_step["ctrl"])),
        "ncon": stand_step["ncon"], "nisland": stand_step["solver_nisland"],
        "nefc": stand_step["solver_nefc"],
    }
    first_ctrl = next(row for row in steps if max(map(abs, row["ctrl"])) > 1e-12)
    first_contact = next(row for row in steps if row["ncon"] > 0)
    direct = {"toy": compare_direct(args.toy_off, args.toy_on),
              "official_scene": compare_direct(args.official_off, args.official_on)}
    no_perturbation = all(result["qpos_absmax"] <= 1e-12 and result["qvel_absmax"] <= 1e-12 and
                          result["hook_wall_s"] <= max(2 * result["control_wall_s"], result["control_wall_s"] + .1)
                          for result in direct.values())
    report = {
        "stage": "BASE Phase32",
        "scope": "LD_PRELOAD observation and one closed prefix capture; no training/hardware/WBT",
        "assets": {"mmap": str(args.mmap), "mmap_sha256": sha256(args.mmap),
                   "rollout": str(args.rollout), "rollout_sha256": sha256(args.rollout),
                   "scene_sha256": sha256(args.scene)},
        "direct_no_perturbation": direct,
        "no_perturbation_gate_pass": no_perturbation,
        "closed_rollout_gate": {key: rollout["summary"].get(key) for key in
                                ("startup_gate_pass", "move_gate_pass", "stop_gate_pass", "full_gate_pass")},
        "closed_trace": {**decoded["header"], "call_counts": decoded["call_counts"],
                         "data_pointer_counts": decoded["data_pointer_counts"],
                         "sequence_contiguous": decoded["sequence_contiguous"],
                         "last_reset_sequence": last_reset, "relevant_step_count": len(steps),
                         "first_relevant_step_time_s": steps[0]["time_s"],
                         "last_relevant_step_time_s": steps[-1]["time_s"],
                         "first_nonzero_ctrl_time_s": first_ctrl["time_s"],
                         "first_contact_time_s": first_contact["time_s"]},
        "telemetry_alignment": {"rows": alignments,
                                "median_physics_minus_elapsed_s": float(np.median([
                                    row["physics_time_s"] - row["elapsed_s"] for row in alignments])),
                                "max_alignment_score": max(row["score"] for row in alignments)},
        "first_stand_actor_row_hidden_state": stand_metrics,
        "decision": "CLOSED_PREFIX_OBSERVABLE_WITHOUT_DETECTED_PHYSICS_PERTURBATION" if no_perturbation else "NO_PERTURBATION_GATE_FAILED",
        "causal_boundary": "The capture identifies hidden integration state and call order for this new episode. It does not prove the old Stage250 episode had bit-identical hidden state or isolate one causal field.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": report["decision"], "first_stand": stand_metrics}, indent=2))


if __name__ == "__main__":
    main()
