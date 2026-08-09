#!/usr/bin/env python3
"""Single-variable 0.2 s MuJoCo 3.3.7 probe for BASE Phase30.

The visible initial state, ten Phase28 PD targets, PD law, XML, timestep, and
control cadence are frozen.  This program must be executed with MuJoCo 3.3.7;
it does not run ONNX and therefore cannot introduce an ONNX Runtime variable.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from official_x2.audit_phase29_closed_wrapper_divergence import (
    ISAAC_JOINTS,
    compare_state,
    decode_row,
    quat_geodesic,
    sha256,
)
from official_x2.replay_official_trace_direct_mujoco import (
    JOINTS,
    pd_gains,
    yaw_tilt,
)


ROOT = Path("/home/humanplus/projects/ZHY/CWI_CrossEmbodiment_Sim")
OFFICIAL = Path("/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1")
DEFAULT_SCENE = OFFICIAL / "worktree/x2_rl_deploy/x2_rl_deploy_mujoco/configuration/robot/lx2501_3_t2d5/model_info/scene.xml"
DEFAULT_HISTORICAL = OFFICIAL / "results/official_native_strict_20260807/stage250_video_straight.json"
DEFAULT_CONTROL = OFFICIAL / "cache/phase28_stage250_extended_direct/stage250_straight_control_ticks.json"
DEFAULT_DIRECT34 = OFFICIAL / "cache/phase28_stage250_extended_direct/stage250_straight_substeps.jsonl.gz"
DEFAULT_VENDOR_MUJOCO = OFFICIAL / "worktree/x2_rl_deploy/x2_rl_deploy_mujoco/lib/libmujoco.so.3.3.7"
DEFAULT_CANDIDATE_CACHE = OFFICIAL / "cache/phase30_mujoco337_probe/stage250_prefix_0p2s.jsonl.gz"
DEFAULT_OUTPUT = ROOT / "reports/official_x2/phase30_mujoco_version_probe.json"

EXPECTED_HASHES = {
    "scene": "7fceb3e1357be29b72344db2f571d7e5655a7d1b1db4ea2571556aee28bb1b63",
    "historical": "310924d7887987c319ebb832ae08a5e47a08dfc94d069ac93c9f96d61fe30c64",
    "phase28_control": "9abc1574535c1f36ae36a094c0be77019ce6a5bdd5aaefc5aa0bc19af3b0f40a",
    "phase28_substeps": "2581e8228425e39bbd4f228be32566c6488b9f28916ff8827520b9ae324ecaab",
    "vendor_mujoco337": "9bfb4d37d1182338eaf24efdf8390e718e4f1b8597a3674fde3884dde62ff97f",
}

CONTROL_DT = 0.02
PHYSICS_DT = 0.001
TICKS = 10
SUBSTEPS = 20


def version_main_cause(reductions: list[float]) -> bool:
    """Frozen Phase30 decision: every primary error must shrink by >=50%."""
    if len(reductions) != 3 or not all(np.isfinite(reductions)):
        raise ValueError("expected three finite primary reductions")
    return all(float(reduction) >= 0.50 for reduction in reductions)


def _load_gzip(path: Path, limit: int) -> list[dict[str, Any]]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for _ in range(limit):
            line = handle.readline()
            if not line:
                break
            rows.append(json.loads(line))
    return rows


def _contact_signature(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, Any]:
    floor = int(model.geom("floor").id)
    sides: set[str] = set()
    count = 0
    for index in range(data.ncon):
        contact = data.contact[index]
        pair = (int(contact.geom1), int(contact.geom2))
        if floor not in pair:
            continue
        other = pair[1] if pair[0] == floor else pair[0]
        body = model.body(int(model.geom_bodyid[other])).name
        if body == "left_ankle_roll_link":
            sides.add("left")
            count += 1
        elif body == "right_ankle_roll_link":
            sides.add("right")
            count += 1
    return {"foot_contact_count": count, "foot_sides": sorted(sides)}


def run_prefix(scene: Path, historical: dict, controls: list[dict]) -> list[dict[str, Any]]:
    if mujoco.mj_versionString() != "3.3.7":
        raise RuntimeError(f"Phase30 candidate must use MuJoCo 3.3.7, got {mujoco.mj_versionString()}")
    model = mujoco.MjModel.from_xml_path(str(scene.resolve()))
    if not math.isclose(float(model.opt.timestep), PHYSICS_DT, abs_tol=1e-12):
        raise RuntimeError(f"timestep drift: {model.opt.timestep}")
    if (model.nq, model.nv, model.na, model.nu) != (38, 37, 0, 31):
        raise RuntimeError("compiled model dimensions drift")
    valid = [row for row in historical["trace"] if len(row.get("obs", [])) == 93]
    initial = valid[0]
    data = mujoco.MjData(model)
    qpos_adr = {name: int(model.joint(name).qposadr[0]) for name in ISAAC_JOINTS}
    dof_adr = {name: int(model.joint(name).dofadr[0]) for name in ISAAC_JOINTS}
    qpos, qvel = decode_row(model, initial, qpos_adr, dof_adr)
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mujoco.mj_forward(model, data)

    gains = pd_gains()
    all_qpos = {name: int(model.joint(name).qposadr[0]) for name in JOINTS}
    all_dof = {name: int(model.joint(name).dofadr[0]) for name in JOINTS}
    actuator = {name: int(model.actuator(f"motor_{name}").id) for name in JOINTS}
    rows: list[dict[str, Any]] = []
    for tick in range(TICKS):
        target_values = controls[tick]["pd_target_rad"]
        if controls[tick]["pd_target_joint_order"] != list(JOINTS):
            raise RuntimeError("PD target joint order drift")
        targets = dict(zip(JOINTS, map(float, target_values)))
        ctrl = np.zeros(model.nu, dtype=np.float64)
        for name in JOINTS:
            kp, kd = gains[name]
            aid = actuator[name]
            torque = kp * (targets[name] - data.qpos[all_qpos[name]]) - kd * data.qvel[all_dof[name]]
            low, high = model.actuator_ctrlrange[aid]
            ctrl[aid] = float(np.clip(torque, low, high))
        for substep in range(SUBSTEPS):
            data.ctrl[:] = ctrl
            mujoco.mj_step(model, data)
            _, tilt = yaw_tilt(np.asarray(data.qpos[3:7]))
            rows.append({
                "physics_step": tick * SUBSTEPS + substep + 1,
                "control_tick": tick,
                "substep": substep,
                "time_s": float(data.time),
                "qpos": np.asarray(data.qpos).tolist(),
                "qvel": np.asarray(data.qvel).tolist(),
                "root_tilt_rad": float(tilt),
                "ctrl": np.asarray(data.ctrl).tolist(),
                "contact": _contact_signature(model, data),
                "frozen_final_clip_action": controls[tick]["final_clip_action"],
                "frozen_pd_target_rad": target_values,
            })
    return rows


def analyze(
    scene: Path,
    historical: dict,
    controls: list[dict],
    direct34: list[dict],
    candidate337: list[dict],
) -> dict[str, Any]:
    model = mujoco.MjModel.from_xml_path(str(scene.resolve()))
    qpos_adr = {name: int(model.joint(name).qposadr[0]) for name in ISAAC_JOINTS}
    dof_adr = {name: int(model.joint(name).dofadr[0]) for name in ISAAC_JOINTS}
    valid = [row for row in historical["trace"] if len(row.get("obs", [])) == 93]
    historical_states = [decode_row(model, row, qpos_adr, dof_adr) for row in valid[:11]]
    aligned = []
    for tick in range(1, 11):
        index = tick * SUBSTEPS - 1
        hq, hv = historical_states[tick]
        rows = {"direct34": direct34[index], "candidate337": candidate337[index]}
        entry = {"elapsed_s": tick * CONTROL_DT, "historical_trace_index": tick + 10}
        for key, row in rows.items():
            qpos = np.asarray(row["qpos"], dtype=np.float64)
            qvel = np.asarray(row["qvel"], dtype=np.float64)
            entry[key] = compare_state(qpos, qvel, hq, hv, qpos_adr, dof_adr)
            entry[key]["tilt_abs_error_rad"] = abs(float(row.get("root_tilt_rad", yaw_tilt(qpos[3:7])[1])) - float(valid[tick]["root_tilt_rad"]))
        entry["paired_337_vs_34"] = compare_state(
            np.asarray(rows["candidate337"]["qpos"]), np.asarray(rows["candidate337"]["qvel"]),
            np.asarray(rows["direct34"]["qpos"]), np.asarray(rows["direct34"]["qvel"]),
            qpos_adr, dof_adr,
        )
        aligned.append(entry)

    first = aligned[0]
    at_006 = aligned[2]
    metrics = {
        "joint_dq_absmax_at_0p02": {
            "control34": first["direct34"]["joint_velocity"]["abs_max"],
            "candidate337": first["candidate337"]["joint_velocity"]["abs_max"],
        },
        "root_angvel_absmax_at_0p02": {
            "control34": first["direct34"]["root_angular_velocity"]["abs_max"],
            "candidate337": first["candidate337"]["root_angular_velocity"]["abs_max"],
        },
        "tilt_abs_error_at_0p06": {
            "control34": at_006["direct34"]["tilt_abs_error_rad"],
            "candidate337": at_006["candidate337"]["tilt_abs_error_rad"],
        },
    }
    for value in metrics.values():
        denominator = max(float(value["control34"]), 1e-12)
        value["relative_reduction"] = float((value["control34"] - value["candidate337"]) / denominator)

    reductions = [float(value["relative_reduction"]) for value in metrics.values()]
    significant = version_main_cause(reductions)
    paired_max = {
        "qpos": max(float(row["paired_337_vs_34"]["joint_position"]["abs_max"]) for row in aligned),
        "qvel": max(float(row["paired_337_vs_34"]["joint_velocity"]["abs_max"]) for row in aligned),
        "root_quaternion_rad": max(float(row["paired_337_vs_34"]["root_quaternion_geodesic_rad"]) for row in aligned),
    }
    return {
        "aligned_boundaries": aligned,
        "primary_metrics": metrics,
        "preregistered_decision_rule": (
            "version_is_main_cause only if all three historical-error metrics shrink by >=50%; "
            "otherwise version_is_not_main_cause"
        ),
        "all_three_reduced_at_least_50pct": significant,
        "paired_337_vs_34_prefix_max": paired_max,
        "decision": "VERSION_IS_MAIN_CAUSE" if significant else "VERSION_IS_NOT_MAIN_CAUSE",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--historical", type=Path, default=DEFAULT_HISTORICAL)
    parser.add_argument("--control", type=Path, default=DEFAULT_CONTROL)
    parser.add_argument("--direct34", type=Path, default=DEFAULT_DIRECT34)
    parser.add_argument("--vendor-mujoco", type=Path, default=DEFAULT_VENDOR_MUJOCO)
    parser.add_argument("--candidate-cache", type=Path, default=DEFAULT_CANDIDATE_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    actual_hashes = {
        "scene": sha256(args.scene),
        "historical": sha256(args.historical),
        "phase28_control": sha256(args.control),
        "phase28_substeps": sha256(args.direct34),
        "vendor_mujoco337": sha256(args.vendor_mujoco),
    }
    if actual_hashes != EXPECTED_HASHES:
        raise RuntimeError(f"frozen asset hash drift: {actual_hashes}")
    mapped_libraries = sorted({
        line.split()[-1]
        for line in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines()
        if "libmujoco.so" in line
    })
    if mapped_libraries != [str(args.vendor_mujoco.resolve())]:
        raise RuntimeError(
            "Phase30 must LD_PRELOAD the exact closed-wrapper vendor library; "
            f"mapped={mapped_libraries}"
        )
    historical = json.loads(args.historical.read_text(encoding="utf-8"))
    controls = json.loads(args.control.read_text(encoding="utf-8"))[:TICKS]
    direct34 = _load_gzip(args.direct34, TICKS * SUBSTEPS)
    candidate337 = run_prefix(args.scene, historical, controls)
    if len(direct34) != 200 or len(candidate337) != 200:
        raise RuntimeError("prefix row count mismatch")

    args.candidate_cache.parent.mkdir(parents=True, exist_ok=False)
    with gzip.open(args.candidate_cache, "wt", encoding="utf-8", compresslevel=6) as handle:
        for row in candidate337:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    result = analyze(args.scene, historical, controls, direct34, candidate337)
    report = {
        "stage": "BASE Phase30",
        "scope": "single-variable 0.2 s direct MuJoCo version probe; no ONNX, training, official ROS, WBT, Git, cloud, or hardware",
        "runtime": {
            "python_mujoco": mujoco.__version__,
            "mj_version_string": mujoco.mj_versionString(),
            "mapped_mujoco_libraries": mapped_libraries,
            "exact_vendor_library_preloaded": True,
        },
        "frozen_contract": {
            "hashes": actual_hashes,
            "initial_state": "Stage250 first valid stand row, identical to Phase28",
            "actions": "first 10 Phase28 final_clip_action values replayed open-loop",
            "pd_targets": "first 10 Phase28 pd_target_rad values; same 31-joint order",
            "pd_law": "same gains, torque clip, and zero-order hold as Phase28",
            "control_dt_s": CONTROL_DT,
            "physics_dt_s": PHYSICS_DT,
            "ticks": TICKS,
            "substeps_per_tick": SUBSTEPS,
            "only_intended_variable": "MuJoCo 3.4.0 -> 3.3.7",
        },
        "result": result,
        "cache": {"path": str(args.candidate_cache), "sha256": sha256(args.candidate_cache), "rows": 200},
        "conclusion": (
            "MuJoCo version mismatch is excluded as the main cause; stop Phase30."
            if result["decision"] == "VERSION_IS_NOT_MAIN_CAUSE"
            else "MuJoCo version is a strong main-cause candidate; further closed-wrapper attribution is required."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "primary_metrics": result["primary_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
