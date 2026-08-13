#!/usr/bin/env python3
"""Continuous contact/COM path for the certified Phase26 left half-cycle."""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import Bounds, NonlinearConstraint, minimize


HERE = Path(__file__).resolve().parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


p18 = load_module("p18", HERE / "phase18_joint_contact_generator/run_phase18_joint_contact_generator.py")
p24 = load_module("p24", HERE / "run_phase24_contact_manifold_path.py")
p25 = load_module("p25", HERE / "run_phase25_load_transfer_skeleton.py")

DT = 0.02
TRANSITION_TICKS = 25
HOLD_TICKS = 6
SEGMENTS = (
    (("left", "right"), None, "flat", "center", "left"),
    (("left",), "right", "rise", "left", "left"),
    (("left",), "right", "fall", "left", "left"),
)


class ContinuousLoadFrame(p24.PathFrame):
    def __init__(self, *args, com_target: np.ndarray, **kwargs):
        self.com_target = np.asarray(com_target)
        super().__init__(*args, **kwargs)

    def eq(self, x, jac=False):
        result = self.eval(x)
        values, rows = [], []
        for side in self.active:
            index = self.fixed[side]
            values.append(result["signed"][side][index] - 0.00025)
            rows.append(result["sj"][side][index])
        values.extend(result["com"][:2] - self.com_target)
        rows.extend(result["cj"][:2])
        for side in self.active:
            values.extend(result["foot"][side][:2] - self.anchors[side])
            rows.extend(result["fj"][side][:2])
        return np.asarray(rows) if jac else np.asarray(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--keyframes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.scene))
    boundary = json.loads(args.boundary.read_text())["boundary"]
    keyframe_result = json.loads(args.keyframes.read_text())
    nodes = keyframe_result["steps"][:4]
    if len(nodes) != 4 or not all(node["success"] for node in nodes):
        raise RuntimeError("Phase26 left-half keyframes are not strictly feasible")
    targets = {name: np.asarray(value) for name, value in keyframe_result["targets"].items()}
    floor, helper = p18.physics.foot_geom_contract(model)
    feet = {
        side: sorted(
            geom for geom in helper[side]
            if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_SPHERE
            and model.geom_contype[geom] != 0
        )
        for side in p18.SIDES
    }
    names = []
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name and model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            names.append(name)
    qadr = np.asarray([
        model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
        for name in names
    ])
    lower = np.asarray([names.index(name) for name in p18.LOWER15])
    joint_ids = np.asarray([
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in p18.LOWER15
    ])
    dofs = model.jnt_dofadr[joint_ids]
    limits = model.jnt_range[joint_ids]
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    base_q = np.asarray(boundary["qpos"], dtype=float)[qadr]
    anchors = {side: np.asarray(nodes[0]["foot_centroid_xy"][side]) for side in p18.SIDES}

    def state_from_node(node: dict) -> dict:
        q = base_q.copy()
        q[lower] = np.asarray(node["lower_q"])
        return {"root": np.asarray(node["root"]), "quat": np.asarray(node["quat_xyzw"]), "q": q, "lower": lower}

    frames = [state_from_node(nodes[0])] * (HOLD_TICKS + 1)
    diagnostics = []
    failed = None
    started = time.perf_counter()
    for segment, (active, swing_side, profile, target_a, target_b) in enumerate(SEGMENTS):
        start = state_from_node(nodes[segment])
        end = state_from_node(nodes[segment + 1])
        probe = p25.LoadFrame(
            model, feet, floor, pelvis, dofs, qadr, start, active, (), anchors,
            com_target=targets[target_a],
        )
        exact_start = probe.eval(np.zeros(21))
        fixed = {side: int(np.argmin(exact_start["signed"][side])) for side in active}
        for tick in range(1, TRANSITION_TICKS):
            u = tick / TRANSITION_TICKS
            progress = p24.smooth5(u)
            interpolated = p24.interpolate(nodes[segment], nodes[segment + 1], u)
            interpolated["q"] = base_q.copy()
            interpolated["lower"] = lower
            interpolated["q"][lower] = (
                (1.0 - progress) * np.asarray(nodes[segment]["lower_q"])
                + progress * np.asarray(nodes[segment + 1]["lower_q"])
            )
            com_target = (1.0 - progress) * targets[target_a] + progress * targets[target_b]
            if profile == "rise":
                clearance = 0.00025 + (0.012 - 0.00025) * progress
            elif profile == "fall":
                clearance = 0.012 - (0.012 - 0.00025) * progress
            else:
                clearance = 0.0
            swing = (swing_side,) if swing_side else ()
            evaluator = ContinuousLoadFrame(
                model, feet, floor, pelvis, dofs, qadr, interpolated, active, swing, anchors,
                swing_clearance=clearance, fixed=fixed, com_target=com_target,
            )
            low = np.r_[[-0.08, -0.08, -0.05], [-0.2] * 3, [-0.3] * 15]
            high = -low
            low[6:] = np.maximum(low[6:], limits[:, 0] - interpolated["q"][lower])
            high[6:] = np.minimum(high[6:], limits[:, 1] - interpolated["q"][lower])
            solution = minimize(
                lambda x: float(x @ x), np.zeros(21), jac=lambda x: 2.0 * x,
                method="SLSQP", bounds=Bounds(low, high),
                constraints=[
                    NonlinearConstraint(lambda x: evaluator.eq(x), 0.0, 0.0, jac=lambda x: evaluator.eq(x, True)),
                    NonlinearConstraint(lambda x: evaluator.ineq(x), 0.0, np.inf, jac=lambda x: evaluator.ineq(x, True)),
                ],
                options={"maxiter": 80, "ftol": 1e-9, "disp": False},
            )
            exact = evaluator.eval(solution.x)
            eq_max = float(np.max(np.abs(evaluator.eq(solution.x))))
            ineq_min = float(np.min(evaluator.ineq(solution.x)))
            feasible = eq_max <= 1e-6 and ineq_min >= -1e-8
            diagnostics.append({
                "segment": segment, "tick": tick, "u": u,
                "active": list(active), "swing": list(swing),
                "clearance_m": clearance, "com_target_xy": com_target.tolist(),
                "objective_converged": bool(solution.success), "status": int(solution.status),
                "eq_max_abs": eq_max, "ineq_min": ineq_min,
                "correction_l2": float(np.linalg.norm(solution.x)), "feasible": feasible,
            })
            if not feasible:
                failed = diagnostics[-1]
                break
            frames.append({"root": exact["root"], "quat": exact["quat"], "q": exact["q"], "lower": lower})
        if failed is not None:
            break
        frames.append(end)
        for _ in range(HOLD_TICKS):
            frames.append(end)

    if failed is None:
        roots = np.asarray([frame["root"] for frame in frames])
        q = np.asarray([frame["q"][lower] for frame in frames])
        steps = np.max(np.abs(np.diff(q, axis=0)), axis=1)
        acceleration = np.linalg.norm(np.diff(roots[:, :2], n=2, axis=0) / DT**2, axis=1)
        continuity = {
            "joint_step_p95_rad": float(np.percentile(steps, 95)),
            "joint_step_max_rad": float(np.max(steps)),
            "root_horizontal_acceleration_max_mps2": float(np.max(acceleration)),
        }
        gates = {
            "joint_step_p95": continuity["joint_step_p95_rad"] <= 0.1,
            "joint_step_max": continuity["joint_step_max_rad"] <= 0.15,
            "root_horizontal_acceleration": continuity["root_horizontal_acceleration_max_mps2"] <= 4.0,
        }
    else:
        continuity = None
        gates = {"all_intermediate_frames_hard_feasible": False}
    complete = failed is None and all(gates.values())
    result = {
        "stage": "Phase27 left-half continuous load/contact path",
        "execution": {"attempted_intermediate_frames": len(diagnostics), "completed_frames": len(frames), "wall_time_s": time.perf_counter() - started, "mj_step_calls": 0, "gpu": False},
        "diagnostics": diagnostics, "failed_frame": failed,
        "continuity": continuity, "gates": gates,
        "frames": [{"root": f["root"].tolist(), "quat_xyzw": f["quat"].tolist(), "lower_q": f["q"][lower].tolist()} for f in frames],
        "decision": {
            "left_half_continuous_path_complete": complete,
            "dynamics_truth": False, "physics_unlocked": False, "training_unlocked": False,
            "next": "centroidal force preflight on this left half" if complete else "stop without alternate configuration",
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"execution": result["execution"], "failed_frame": failed, "continuity": continuity, "gates": gates, "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()
