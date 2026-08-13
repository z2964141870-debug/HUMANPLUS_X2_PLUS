#!/usr/bin/env python3
"""Build a 50 Hz contact-manifold path between Phase22 keyframes; no physics."""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import Bounds, NonlinearConstraint, minimize
from scipy.spatial.transform import Rotation, Slerp


HERE = Path(__file__).resolve().parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


p18 = load_module("p18", HERE / "phase18_joint_contact_generator/run_phase18_joint_contact_generator.py")
p22 = load_module("p22", HERE / "run_phase22_sequential_contact_skeleton.py")

DT = 0.02
TRANSITION_TICKS = 25
HOLD_TICKS = 6
SEGMENTS = (
    (("left", "right"), None, "flat"),
    (("left",), "right", "rise"),
    (("left",), "right", "fall"),
    (("right",), "left", "rise"),
    (("right",), "left", "fall"),
    (("left", "right"), None, "flat"),
)


def smooth5(u: float) -> float:
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def interpolate(a: dict, b: dict, u: float) -> dict:
    s = smooth5(u)
    rotation = Slerp(
        [0.0, 1.0], Rotation.from_quat(np.asarray([a["quat_xyzw"], b["quat_xyzw"]]))
    )([s]).as_quat()[0]
    return {
        "root": (1.0 - s) * np.asarray(a["root"]) + s * np.asarray(b["root"]),
        "quat": rotation,
        "q": None,
        "lower": None,
    }


class PathFrame(p22.Frame):
    def __init__(self, *args, swing_clearance: float, fixed: dict, **kwargs):
        self.swing_clearance = swing_clearance
        super().__init__(*args, **kwargs)
        self.fixed = dict(fixed)
        self.cache = None

    def ineq(self, x, jac=False):
        result = self.eval(x)
        values, rows = [], []
        for side in p18.SIDES:
            threshold = self.swing_clearance if side in self.swing else -0.00001
            values.extend(result["signed"][side] - threshold)
            rows.extend(result["sj"][side])
        return np.asarray(rows) if jac else np.asarray(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--skeleton", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.scene))
    boundary = json.loads(args.boundary.read_text())["boundary"]
    nodes = json.loads(args.skeleton.read_text())["steps"]
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

    def node_state(node: dict) -> dict:
        q = base_q.copy()
        q[lower] = np.asarray(node["lower_q"])
        return {
            "root": np.asarray(node["root"]),
            "quat": np.asarray(node["quat_xyzw"]),
            "q": q,
            "lower": lower,
        }

    def active_indices(state: dict, active: tuple) -> dict:
        probe = p22.Frame(model, feet, floor, pelvis, dofs, qadr, state, active, (), anchors)
        exact = probe.eval(np.zeros(21))
        return {side: int(np.argmin(exact["signed"][side])) for side in active}

    frames = []
    diagnostics = []
    first = node_state(nodes[0])
    for _ in range(HOLD_TICKS + 1):
        frames.append(first)
    started = time.perf_counter()
    failed = None

    for segment_index, (active, swing_side, profile) in enumerate(SEGMENTS):
        start = node_state(nodes[segment_index])
        end = node_state(nodes[segment_index + 1])
        fixed = active_indices(start, active)
        for tick in range(1, TRANSITION_TICKS):
            u = tick / TRANSITION_TICKS
            base = interpolate(nodes[segment_index], nodes[segment_index + 1], u)
            base["q"] = base_q.copy()
            base["lower"] = lower
            base["q"][lower] = (
                (1.0 - smooth5(u)) * np.asarray(nodes[segment_index]["lower_q"])
                + smooth5(u) * np.asarray(nodes[segment_index + 1]["lower_q"])
            )
            if profile == "rise":
                clearance = 0.00025 + (0.012 - 0.00025) * smooth5(u)
            elif profile == "fall":
                clearance = 0.012 - (0.012 - 0.00025) * smooth5(u)
            else:
                clearance = 0.0
            swing = (swing_side,) if swing_side else ()
            evaluator = PathFrame(
                model, feet, floor, pelvis, dofs, qadr, base, active, swing, anchors,
                swing_clearance=clearance, fixed=fixed,
            )
            low = np.r_[[-0.08, -0.08, -0.05], [-0.2] * 3, [-0.3] * 15]
            high = -low
            low[6:] = np.maximum(low[6:], limits[:, 0] - base["q"][lower])
            high[6:] = np.minimum(high[6:], limits[:, 1] - base["q"][lower])
            solution = minimize(
                lambda x: float(x @ x), np.zeros(21), jac=lambda x: 2.0 * x,
                method="SLSQP", bounds=Bounds(low, high),
                constraints=[
                    NonlinearConstraint(
                        lambda x: evaluator.eq(x), 0.0, 0.0,
                        jac=lambda x: evaluator.eq(x, True),
                    ),
                    NonlinearConstraint(
                        lambda x: evaluator.ineq(x), 0.0, np.inf,
                        jac=lambda x: evaluator.ineq(x, True),
                    ),
                ],
                options={"maxiter": 80, "ftol": 1e-9, "disp": False},
            )
            exact = evaluator.eval(solution.x)
            eq_max = float(np.max(np.abs(evaluator.eq(solution.x))))
            ineq_min = float(np.min(evaluator.ineq(solution.x)))
            feasible = eq_max <= 1e-6 and ineq_min >= -1e-8
            diagnostics.append({
                "segment": segment_index,
                "tick": tick,
                "u": u,
                "active": list(active),
                "swing": list(swing),
                "clearance_m": clearance,
                "objective_converged": bool(solution.success),
                "status": int(solution.status),
                "eq_max_abs": eq_max,
                "ineq_min": ineq_min,
                "correction_l2": float(np.linalg.norm(solution.x)),
                "feasible": feasible,
            })
            if not feasible:
                failed = diagnostics[-1]
                break
            frames.append({
                "root": exact["root"], "quat": exact["quat"],
                "q": exact["q"], "lower": lower,
            })
        if failed is not None:
            break
        frames.append(end)
        for _ in range(HOLD_TICKS):
            frames.append(end)

    if failed is None:
        root = np.asarray([frame["root"] for frame in frames])
        q = np.asarray([frame["q"][lower] for frame in frames])
        joint_step = np.max(np.abs(np.diff(q, axis=0)), axis=1)
        root_acc = np.linalg.norm(np.diff(root[:, :2], n=2, axis=0) / DT**2, axis=1)
        continuity = {
            "joint_step_p95_rad": float(np.percentile(joint_step, 95)),
            "joint_step_max_rad": float(np.max(joint_step)),
            "root_horizontal_acceleration_max_mps2": float(np.max(root_acc)),
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
    serializable_frames = [{
        "root": frame["root"].tolist(),
        "quat_xyzw": frame["quat"].tolist(),
        "lower_q": frame["q"][lower].tolist(),
    } for frame in frames]
    result = {
        "stage": "Phase24 continuous contact-manifold path",
        "execution": {
            "attempted_intermediate_frames": len(diagnostics),
            "completed_frames": len(frames),
            "wall_time_s": time.perf_counter() - started,
            "mj_step_calls": 0,
            "gpu": False,
        },
        "diagnostics": diagnostics,
        "failed_frame": failed,
        "continuity": continuity,
        "gates": gates,
        "frames": serializable_frames,
        "decision": {
            "contact_manifold_path_complete": complete,
            "physics_unlocked": False,
            "training_unlocked": False,
            "next": "review before adding forward semantics" if complete else "stop without alternate configuration",
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "execution": result["execution"], "failed_frame": failed,
        "continuity": continuity, "gates": gates, "decision": result["decision"],
    }, indent=2))


if __name__ == "__main__":
    main()
