#!/usr/bin/env python3
"""Sequential quasi-static COM/load-transfer keyframes; no dynamics or physics."""
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
p22 = load_module("p22", HERE / "run_phase22_sequential_contact_skeleton.py")

STEPS = (
    ("DS_CENTER", ("left", "right"), (), "center"),
    ("DS_LOAD_LEFT", ("left", "right"), (), "left"),
    ("L_SUPPORT_R_SWING", ("left",), ("right",), "left"),
    ("DS_R_TOUCHDOWN_LEFT_LOADED", ("left", "right"), (), "left"),
    ("DS_LOAD_RIGHT", ("left", "right"), (), "right"),
    ("R_SUPPORT_L_SWING", ("right",), ("left",), "right"),
    ("DS_L_TOUCHDOWN_RIGHT_LOADED", ("left", "right"), (), "right"),
    ("DS_SETTLE_CENTER", ("left", "right"), (), "center"),
)


class LoadFrame(p22.Frame):
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.scene))
    boundary = json.loads(args.boundary.read_text())["boundary"]
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
    qpos = np.asarray(boundary["qpos"], dtype=float)
    state = {
        "root": qpos[:3].copy(), "quat": qpos[3:7][[1, 2, 3, 0]].copy(),
        "q": qpos[qadr].copy(), "lower": lower,
    }
    probe = p22.Frame(model, feet, floor, pelvis, dofs, qadr, state, ("left", "right"), (), {})
    initial = probe.eval(np.zeros(21))
    anchors = {side: initial["foot"][side][:2].copy() for side in p18.SIDES}
    targets = {
        "left": anchors["left"],
        "right": anchors["right"],
        # Preserve the coherent closed-AimDK boundary exactly.  Its COM is
        # safely inside the DS polygon but is not the feet-centroid midpoint.
        "center": initial["com"][:2].copy(),
    }

    rows = []
    started = time.perf_counter()
    for index, (label, active, swing, target_name) in enumerate(STEPS):
        evaluator = LoadFrame(
            model, feet, floor, pelvis, dofs, qadr, state, active, swing, anchors,
            com_target=targets[target_name],
        )
        if index == 0:
            x = np.zeros(21)
            objective_converged = True
            status = 0
            message = "native boundary accepted"
            # Phase41 owns the native soft-contact qualification.
            eq_max = float(np.max(np.abs(evaluator.eq(x)[2:])))
            ineq_min = float(min(np.min(evaluator.eval(x)["signed"][side]) + 0.0005 for side in p18.SIDES))
            feasible = eq_max <= 1e-6 and ineq_min >= 0.0
        else:
            low = np.r_[[-0.25, -0.25, -0.12], [-0.35] * 3, [-0.65] * 15]
            high = -low
            low[6:] = np.maximum(low[6:], limits[:, 0] - state["q"][lower])
            high[6:] = np.minimum(high[6:], limits[:, 1] - state["q"][lower])
            solution = minimize(
                lambda value: float(value @ value), np.zeros(21), jac=lambda value: 2.0 * value,
                method="SLSQP", bounds=Bounds(low, high),
                constraints=[
                    NonlinearConstraint(
                        lambda value: evaluator.eq(value), 0.0, 0.0,
                        jac=lambda value: evaluator.eq(value, True),
                    ),
                    NonlinearConstraint(
                        lambda value: evaluator.ineq(value), 0.0, np.inf,
                        jac=lambda value: evaluator.ineq(value, True),
                    ),
                ],
                options={"maxiter": 100, "ftol": 1e-9, "disp": False},
            )
            x = solution.x
            objective_converged = bool(solution.success)
            status = int(solution.status)
            message = str(solution.message)
            eq_max = float(np.max(np.abs(evaluator.eq(x))))
            ineq_min = float(np.min(evaluator.ineq(x)))
            feasible = eq_max <= 1e-6 and ineq_min >= -1e-8
        exact = evaluator.eval(x)
        rows.append({
            "index": index, "label": label, "active": list(active), "swing": list(swing),
            "com_target": target_name, "success": feasible,
            "objective_converged": objective_converged, "status": status, "message": message,
            "eq_max_abs": eq_max, "ineq_min": ineq_min,
            "root": exact["root"].tolist(), "quat_xyzw": exact["quat"].tolist(),
            "lower_q": exact["q"][lower].tolist(),
            "com_xy": exact["com"][:2].tolist(),
            "foot_centroid_xy": {side: exact["foot"][side][:2].tolist() for side in p18.SIDES},
        })
        if not feasible:
            break
        state = {
            "root": exact["root"], "quat": exact["quat"],
            "q": exact["q"], "lower": lower,
        }

    complete = len(rows) == len(STEPS) and all(row["success"] for row in rows)
    result = {
        "stage": "Phase25 quasi-static load-transfer skeleton",
        "execution": {"completed_steps": len(rows), "wall_time_s": time.perf_counter() - started, "mj_step_calls": 0, "gpu": False},
        "targets": {name: value.tolist() for name, value in targets.items()},
        "steps": rows,
        "decision": {
            "load_transfer_skeleton_complete": complete,
            "dynamics_truth": False,
            "physics_unlocked": False,
            "training_unlocked": False,
            "next": "time-parameterize COM transfer before force feasibility" if complete else "stop without alternate configuration",
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"execution": result["execution"], "steps": rows, "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()
