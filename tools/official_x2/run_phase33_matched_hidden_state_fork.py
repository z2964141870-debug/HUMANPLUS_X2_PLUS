#!/usr/bin/env python3
"""BASE Phase33 matched fork from the Phase32 closed AimDK snapshot."""

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


ANCHOR_TIME_S = 0.226
HORIZONS = (10, 25, 50)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def integration_state(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(model, spec), dtype=np.float64)
    mujoco.mj_getState(model, data, state, spec)
    return state


def restore_integration_state(model: mujoco.MjModel, data: mujoco.MjData, state: np.ndarray) -> None:
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    mujoco.mj_setState(model, data, state, spec)
    mujoco.mj_forward(model, data)
    # forward refreshes position/velocity-derived caches but can replace the
    # warmstart vector; restore the integration vector once more.
    mujoco.mj_setState(model, data, state, spec)


def contact_pairs_from_trace(row: dict) -> list[tuple[int, int]]:
    return sorted((min(int(c["geom1"]), int(c["geom2"])),
                   max(int(c["geom1"]), int(c["geom2"]))) for c in row["contacts"])


def contact_pairs_from_data(data: mujoco.MjData) -> list[tuple[int, int]]:
    return sorted((min(int(data.contact[i].geom1), int(data.contact[i].geom2)),
                   max(int(data.contact[i].geom1), int(data.contact[i].geom2)))
                  for i in range(data.ncon))


def quat_geodesic(left: np.ndarray, right: np.ndarray) -> float:
    left = left / np.linalg.norm(left)
    right = right / np.linalg.norm(right)
    dot = min(1.0, max(-1.0, abs(float(np.dot(left, right)))))
    return 2.0 * math.acos(dot)


def compare_to_closed(data: mujoco.MjData, closed: dict, assigned_ctrl: np.ndarray) -> dict:
    qpos = np.asarray(closed["qpos"], dtype=np.float64)
    qvel = np.asarray(closed["qvel"], dtype=np.float64)
    pairs_direct = contact_pairs_from_data(data)
    pairs_closed = contact_pairs_from_trace(closed)
    return {
        "qpos_absmax": float(np.max(np.abs(data.qpos - qpos))),
        "qvel_absmax": float(np.max(np.abs(data.qvel - qvel))),
        "root_xyz_l2": float(np.linalg.norm(data.qpos[:3] - qpos[:3])),
        "root_quat_geodesic_rad": quat_geodesic(np.asarray(data.qpos[3:7]), qpos[3:7]),
        "ctrl_absmax": float(np.max(np.abs(assigned_ctrl - np.asarray(closed["ctrl"])))),
        "ncon_direct": int(data.ncon), "ncon_closed": int(closed["ncon"]),
        "contact_pairs_equal": pairs_direct == pairs_closed,
        "contact_pairs_direct": pairs_direct,
        "contact_pairs_closed": pairs_closed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--mmap", type=Path, required=True)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if mujoco.mj_versionString() != "3.3.7":
        raise RuntimeError(f"exact vendor 3.3.7 required, got {mujoco.mj_versionString()}")
    mapped = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                     if "libmujoco.so" in line})
    expected_lib = "/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/worktree/x2_rl_deploy/x2_rl_deploy_mujoco/lib/libmujoco.so.3.3.7"
    if mapped != [expected_lib]:
        raise RuntimeError(f"exact vendor library not uniquely mapped: {mapped}")

    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    decoded = decode(args.mmap, include_rows=True)
    last_reset = max(row["sequence"] for row in decoded["rows"] if row["call_kind"] == 1)
    steps = [row for row in decoded["rows"] if row["call_kind"] == 3 and row["sequence"] > last_reset]
    anchor = min(steps, key=lambda row: abs(row["time_s"] - ANCHOR_TIME_S))
    if abs(anchor["time_s"] - ANCHOR_TIME_S) > 1e-12:
        raise RuntimeError("frozen anchor missing")
    suffix = [row for row in steps if row["time_s"] > anchor["time_s"] + 1e-12][:max(HORIZONS)]
    if len(suffix) != max(HORIZONS):
        raise RuntimeError("closed suffix too short")

    # A: construct every available mjSTATE_INTEGRATION component.  The scene has
    # na=npluginstate=nmocap=nuserdata=0; unrecorded applied forces remain their
    # reset zeros.  qacc_warmstart is explicitly restored from the closed row.
    seed = mujoco.MjData(model)
    seed.time = anchor["time_s"]
    seed.qpos[:] = anchor["qpos"]
    seed.qvel[:] = anchor["qvel"]
    if model.na:
        seed.act[:] = anchor["act"]
    seed.ctrl[:] = anchor["ctrl"]
    seed.qacc_warmstart[:] = anchor["qacc_warmstart"]
    full_state = integration_state(model, seed)
    full = mujoco.MjData(model)
    restore_integration_state(model, full, full_state)

    # B: reproduce Phase28's visible-only contract exactly.  Phase28 did not
    # have mmap qpos/qvel: it reconstructed them from one 93D observation plus
    # root telemetry, then created fresh mjData and called mj_forward.
    rollout = json.loads(args.rollout.read_text(encoding="utf-8"))
    first_stand = next(row for row in rollout["trace"]
                       if row.get("stage") == "stand" and len(row.get("obs", [])) == 93)
    qadr = {name: int(model.joint(name).qposadr[0]) for name in ISAAC_JOINTS}
    dadr = {name: int(model.joint(name).dofadr[0]) for name in ISAAC_JOINTS}
    visible_qpos, visible_qvel = decode_row(model, first_stand, qadr, dadr)
    visible = mujoco.MjData(model)
    visible.qpos[:] = visible_qpos
    visible.qvel[:] = visible_qvel
    mujoco.mj_forward(model, visible)

    initial = {
        "full_qpos_absmax": float(np.max(np.abs(full.qpos - np.asarray(anchor["qpos"])))),
        "full_qvel_absmax": float(np.max(np.abs(full.qvel - np.asarray(anchor["qvel"])))),
        "full_time_abs_error": abs(float(full.time) - float(anchor["time_s"])),
        "full_warmstart_absmax": float(np.max(np.abs(full.qacc_warmstart - np.asarray(anchor["qacc_warmstart"])))),
        "visible_qpos_absmax_vs_closed": float(np.max(np.abs(visible.qpos - np.asarray(anchor["qpos"])))),
        "visible_qvel_absmax_vs_closed": float(np.max(np.abs(visible.qvel - np.asarray(anchor["qvel"])))),
        "visible_time_s": float(visible.time),
        "visible_warmstart_l2": float(np.linalg.norm(visible.qacc_warmstart)),
        "closed_warmstart_l2": float(np.linalg.norm(anchor["qacc_warmstart"])),
    }
    trajectory = {"full": [], "visible": []}
    for closed in suffix:
        ctrl = np.asarray(closed["ctrl"], dtype=np.float64)
        for name, data in (("full", full), ("visible", visible)):
            data.ctrl[:] = ctrl
            mujoco.mj_step(model, data)
            trajectory[name].append(compare_to_closed(data, closed, ctrl))

    horizon_results = {}
    for horizon in HORIZONS:
        entry = {}
        for name in ("full", "visible"):
            prefix = trajectory[name][:horizon]
            at = prefix[-1]
            entry[name] = {
                "at_horizon": at,
                "prefix_qpos_absmax": max(row["qpos_absmax"] for row in prefix),
                "prefix_qvel_absmax": max(row["qvel_absmax"] for row in prefix),
                "prefix_root_xyz_l2_max": max(row["root_xyz_l2"] for row in prefix),
                "prefix_root_quat_rad_max": max(row["root_quat_geodesic_rad"] for row in prefix),
                "all_contact_pairs_equal": all(row["contact_pairs_equal"] for row in prefix),
            }
        horizon_results[str(horizon)] = entry

    exact_tolerance = 1e-12
    full_exact_50 = (horizon_results["50"]["full"]["prefix_qpos_absmax"] <= exact_tolerance and
                     horizon_results["50"]["full"]["prefix_qvel_absmax"] <= exact_tolerance and
                     horizon_results["50"]["full"]["all_contact_pairs_equal"])
    visible_exact_50 = (horizon_results["50"]["visible"]["prefix_qpos_absmax"] <= exact_tolerance and
                        horizon_results["50"]["visible"]["prefix_qvel_absmax"] <= exact_tolerance and
                        horizon_results["50"]["visible"]["all_contact_pairs_equal"])
    if full_exact_50 and not visible_exact_50:
        decision = "CAPTURED_HIDDEN_INTEGRATION_STATE_EXPLAINS_PREFIX_DIVERGENCE"
    elif full_exact_50 and visible_exact_50:
        decision = "VISIBLE_STATE_IS_SUFFICIENT_FOR_THIS_MATCHED_SUFFIX"
    elif not full_exact_50:
        decision = "CAPTURED_STATE_INCOMPLETE_OR_APPLICATION_TIMING_UNREPRODUCED"
    else:
        decision = "UNRESOLVED"
    result = {
        "stage": "BASE Phase33",
        "scope": "single matched direct fork; no field ablation, training, WBT, ROS, Git, cloud, or hardware",
        "assets": {"scene": str(args.scene), "scene_sha256": sha256(args.scene),
                   "mmap": str(args.mmap), "mmap_sha256": sha256(args.mmap),
                   "rollout": str(args.rollout), "rollout_sha256": sha256(args.rollout),
                   "mapped_mujoco": mapped},
        "contract": {"anchor_time_s": ANCHOR_TIME_S, "horizons_physics_ticks": list(HORIZONS),
                     "physics_dt_s": float(model.opt.timestep),
                     "future_input": "closed post-step ctrl at each next 1ms step",
                     "full_branch": "captured available mjSTATE_INTEGRATION, including time/qpos/qvel/ctrl/warmstart",
                     "visible_branch": "fresh mjData + qpos/qvel reconstructed by Phase28 decode_row from same-episode 93D/root telemetry + mj_forward"},
        "model_dimensions": {"nq": model.nq, "nv": model.nv, "na": model.na, "nu": model.nu,
                             "npluginstate": model.npluginstate, "nmocap": model.nmocap,
                             "nuserdata": model.nuserdata},
        "initial_restore": initial,
        "horizons": horizon_results,
        "full_exact_through_50": full_exact_50,
        "visible_exact_through_50": visible_exact_50,
        "decision": decision,
        "causal_boundary": "A failure of full-state replay means the observer omitted required state or the closed wrapper applies commands/callbacks at an unrecorded point. It does not justify per-field tuning.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "initial_restore": initial,
                      "horizons": {key: {name: {metric: value for metric, value in branch.items() if metric.startswith("prefix_") or metric == "all_contact_pairs_equal"}
                                         for name, branch in item.items()} for key, item in horizon_results.items()}}, indent=2))


if __name__ == "__main__":
    main()
