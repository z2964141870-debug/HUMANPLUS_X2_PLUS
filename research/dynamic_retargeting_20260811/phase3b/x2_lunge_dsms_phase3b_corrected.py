#!/usr/bin/env python3
"""Corrected Phase3b DSMS contract and one short-prefix solve.

The official XML remains a motor/torque model for independent replay.  Inside
the optimizer only, each motor is represented as the exactly equivalent
official external position-PD servo: target q at 50 Hz, PD evaluated at every
1 kHz MuJoCo step, and force clipped by the original motor ctrlrange.
"""
from __future__ import annotations

import argparse, dataclasses, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path("/home/humanplus/projects/ZHY/dsms_workspace")
UPSTREAM = ROOT / "shooting-for-contact"
REPO = Path("/home/humanplus/projects/ZHY/CWI_CrossEmbodiment_Sim")
SCENE = Path("/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/worktree/x2_rl_deploy/x2_rl_deploy_mujoco/configuration/robot/lx2501_3_t2d5/model_info/scene.xml")
CONTROL = Path("/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/worktree/x2_rl_deploy/x2_rl_deploy_controller/config/motion_control.yaml")
LUNGE = Path("/home/humanplus/projects/ZHY/x2_wbt_official_v1/motion_lib_x2_official_v1/silver_contact/phase30_time_dilation/x2_phase30_time_dilation_1p46.pkl")
OUT = ROOT / "phase3b"
OUT.mkdir(parents=True, exist_ok=True)

sys.path[:0] = [str(UPSTREAM), str(UPSTREAM / "examples" / "g1_gait"),
                str(REPO / "tools"), str(REPO)]
os.environ.setdefault("TRAJOPT_ROOT_DIR", str(UPSTREAM))

import joblib, mujoco
import src.multi_shooting as ms_module
from src.dynamics import Dynamics, DynamicsConfig
from src.multi_shooting import MultiShootingConfig
from src.spline import SplineConfig
from examples.g1_gait.g1_gait import G1GaitTO
from utils.file_utils import save_trajectory
import retarget.run_x2_forefoot_official_physics_screen as physics
import retarget.run_x2_native_gold_trackability_phase12 as phase12
from scipy.spatial.transform import Rotation, Slerp


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def adapt_entry_50hz(entry):
    source_fps = float(entry["fps"]); target_fps = 50.0
    q0 = np.asarray(entry["dof"], dtype=np.float64)
    p0 = np.asarray(entry["root_trans_offset"], dtype=np.float64)
    r0 = np.asarray(entry["root_rot"], dtype=np.float64)
    duration = (len(q0)-1)/source_fps
    t0 = np.arange(len(q0), dtype=np.float64)/source_fps
    n = int(round(duration*target_fps))+1
    t = np.linspace(0.0, duration, n)
    q = np.column_stack([np.interp(t,t0,q0[:,i]) for i in range(q0.shape[1])])
    p = np.column_stack([np.interp(t,t0,p0[:,i]) for i in range(3)])
    r = Slerp(t0, Rotation.from_quat(r0))(t).as_quat()
    dq = np.gradient(q, 1.0/target_fps, axis=0, edge_order=1)
    dp = np.gradient(p, 1.0/target_fps, axis=0, edge_order=1)
    dr = np.zeros((n,3), dtype=np.float64)
    inc = (Rotation.from_quat(r[:-1]).inv()*Rotation.from_quat(r[1:])).as_rotvec()*target_fps
    dr[0],dr[-1]=inc[0],inc[-1]
    if n>2: dr[1:-1]=0.5*(inc[:-1]+inc[1:])
    adapted = {"dof":q, "root_trans_offset":p, "root_rot":r, "fps":target_fps,
               "joint_names_mujoco":list(entry["joint_names_mujoco"])}
    return adapted, {"q":q,"dq":dq,"root_pos":p,"root_quat_xyzw":r,
                     "root_lin_vel":dp,"root_ang_vel":dr,"fps":target_fps}


def load_reference(prefix_frames: int = 18):
    entry = joblib.load(LUNGE)["PHUMA-LUNGE-R-001"]
    adapted, ref = adapt_entry_50hz(entry)
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    control = physics.build_control_contract(model, CONTROL)
    n = min(prefix_frames, len(ref["q"]))
    X = np.zeros((n, model.nq + model.nv), dtype=np.float64)
    X[:, :3] = ref["root_pos"][:n]
    X[:, 3:7] = ref["root_quat_xyzw"][:n][:, [3, 0, 1, 2]]
    X[:, model.nq:model.nq+3] = ref["root_lin_vel"][:n]
    X[:, model.nq+3:model.nq+6] = ref["root_ang_vel"][:n]
    for i, name in enumerate(control.joint_names):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        X[:, model.jnt_qposadr[jid]] = ref["q"][:n, i]
        X[:, model.nq + model.jnt_dofadr[jid]] = ref["dq"][:n, i]
    return entry, adapted, ref, X, control


def phase12_reference(entry, ref):
    """Add the FK/contact fields required by the frozen Phase12 replay."""
    model = mujoco.MjModel.from_xml_path(str(SCENE)); data = mujoco.MjData(model)
    names = list(entry["joint_names_mujoco"])
    qadr = [int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in names]
    body_ids = np.arange(1, model.nbody, dtype=np.int64)
    body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(i)) for i in body_ids]
    body_pos = np.zeros((len(ref["q"]), len(body_ids), 3)); body_quat = np.zeros((len(ref["q"]), len(body_ids), 4))
    contact = {"left": np.zeros(len(ref["q"]), dtype=bool), "right": np.zeros(len(ref["q"]), dtype=bool)}
    floor, feet = physics.foot_geom_contract(model); geom_side = {g: s for s, gs in feet.items() for g in gs}
    for i in range(len(ref["q"])):
        mujoco.mj_resetData(model, data)
        data.qpos[:3] = ref["root_pos"][i]; data.qpos[3:7] = ref["root_quat_xyzw"][i][[3, 0, 1, 2]]
        data.qpos[qadr] = ref["q"][i]; data.qvel[:3] = ref["root_lin_vel"][i]; data.qvel[3:6] = ref["root_ang_vel"][i]
        mujoco.mj_forward(model, data); body_pos[i] = data.xpos[body_ids]; body_quat[i] = data.xquat[body_ids][:, [1, 2, 3, 0]]
        current = phase12.contact_now(model, data, floor, geom_side)
        for side in contact: contact[side][i] = bool(current[side])
    return {**ref, "joint_names": names, "body_names": body_names,
            "body_pos": body_pos, "body_quat_xyzw": body_quat, "contact": contact}


class OfficialPDDynamics(Dynamics):
    """Official motor model rewritten in-memory as an exactly equivalent PD servo."""
    def __init__(self, config: DynamicsConfig):
        # Preserve the authored motor model and official Euler integrator.
        config = dataclasses.replace(config, actuator_mode="position", integrator=None)
        super().__init__(config)
        m = self.model
        contract = physics.build_control_contract(m, CONTROL)
        torque_range = m.actuator_ctrlrange.copy()
        joint_target_range = np.array([m.jnt_range[m.actuator_trnid[a, 0]] for a in range(m.nu)], dtype=np.float64)

        m.actuator_gaintype[:] = mujoco.mjtGain.mjGAIN_FIXED
        m.actuator_biastype[:] = mujoco.mjtBias.mjBIAS_AFFINE
        m.actuator_dyntype[:] = mujoco.mjtDyn.mjDYN_NONE
        m.actuator_gainprm[:] = 0.0
        m.actuator_biasprm[:] = 0.0
        m.actuator_gainprm[:, 0] = contract.kp
        m.actuator_biasprm[:, 1] = -contract.kp
        m.actuator_biasprm[:, 2] = -contract.kd
        m.actuator_gear[:] = 0.0
        m.actuator_gear[:, 0] = 1.0
        m.actuator_ctrlrange[:] = joint_target_range
        m.actuator_ctrllimited[:] = 1
        m.actuator_forcerange[:] = torque_range
        m.actuator_forcelimited[:] = 1
        # Policy/reference excludes head2: target is fixed nominal before bounds are built.
        for a, name in enumerate(contract.actuator_joint_names):
            if name in ("head_yaw_joint", "head_pitch_joint"):
                m.actuator_ctrlrange[a] = [0.0, 0.0]
        self.servo_kp = contract.kp.copy()
        self.servo_kd = contract.kd.copy()
        self._load_actuator_and_limits()
        self.data = mujoco.MjData(m)
        self.official_contract = contract
        self.original_torque_range = torque_range


# MultiShootingBase resolves this module-global constructor at runtime.
ms_module.Dynamics = OfficialPDDynamics


@dataclasses.dataclass
class TrackConfig:
    w_base_pos: float = 10.0
    w_base_ori: float = 10.0
    w_base_linvel: float = 0.1
    w_base_angvel: float = 0.1
    w_joint_pos: float = 0.1
    w_joint_vel: float = 0.01
    term_scale: float = 10.0
    Rtau: float = 1e-6
    Rrate: float = 1e-2
    exact_vel_grad: bool = False
    ee_anchor: str = "torso_link"
    def ee_track_expanded(self):
        return (
            (1, "pelvis", 1.0, 1.0, 1.0, 0.01, 0.01, False),
            (1, "left_ankle_roll_link", 1.0, 1.0, 1.0, 0.01, 0.01, False),
            (1, "right_ankle_roll_link", 1.0, 1.0, 1.0, 0.01, 0.01, False),
            (1, "left_wrist_yaw_link", 1.0, 1.0, 1.0, 0.01, 0.01, True),
            (1, "right_wrist_yaw_link", 1.0, 1.0, 1.0, 0.01, 0.01, True),
        )


def initialize_data(model, data, x):
    data.qpos[:] = x[:model.nq]
    data.qvel[:] = x[model.nq:model.nq + model.nv]
    mujoco.mj_forward(model, data)


def external_pd_servo_equivalence(X, contract, steps=100):
    raw = mujoco.MjModel.from_xml_path(str(SCENE)); raw.opt.timestep = 0.001
    servo_dyn = OfficialPDDynamics(DynamicsConfig(model_path=str(SCENE), sim_dt=0.001, integrator=None, actuator_mode="position"))
    servo = servo_dyn.model
    dr, ds = mujoco.MjData(raw), mujoco.MjData(servo)
    initialize_data(raw, dr, X[0]); initialize_data(servo, ds, X[0])
    qerr = verr = 0.0
    for k in range(steps):
        phase = min((k * 0.001) / 0.02, len(X) - 1.0)
        i = min(int(np.floor(phase)), len(X) - 1); j = min(i + 1, len(X) - 1); a = phase - i
        target_state = (1.0-a)*X[i] + a*X[j]
        target = target_state[servo_dyn.act_qadr]
        q = dr.qpos[servo_dyn.act_qadr]; qd = dr.qvel[servo_dyn.act_vadr]
        dr.ctrl[:] = np.clip(contract.kp*(target-q)-contract.kd*qd,
                             raw.actuator_ctrlrange[:, 0], raw.actuator_ctrlrange[:, 1])
        ds.ctrl[:] = target
        mujoco.mj_step(raw, dr); mujoco.mj_step(servo, ds)
        qerr = max(qerr, float(np.max(np.abs(dr.qpos-ds.qpos))))
        verr = max(verr, float(np.max(np.abs(dr.qvel-ds.qvel))))
    return {"steps": steps, "qpos_max_abs": qerr, "qvel_max_abs": verr,
            "pass": qerr <= 1e-10 and verr <= 1e-9}


def baseline_free_root(entry, ref):
    row = phase12.simulate(phase12_reference(entry, ref), phase12.MODE_FREE)
    return {"simulated_duration_s": row["simulated_duration_s"], "fall_time_s": row["fall_time_s"],
            "duration_fraction": row["duration_fraction"],
            "pass": abs(row["simulated_duration_s"] - 0.575) <= 0.02}


def build_problem(X, max_iter=300):
    sim_dt, node_dt = 0.001, 0.02
    n_fine = (len(X)-1)*20; K = 20; N = int(np.ceil(n_fine/K))
    dyn_cfg = DynamicsConfig(model_path=str(SCENE), sim_dt=sim_dt, integrator=None,
                             actuator_mode="position", n_threads=8, fd_eps=1e-6,
                             fd_centered=True, friction_cone=None)
    spline = SplineConfig(M=max(2, int(round(N*K*sim_dt*50.0))+1), spline_type="linear")
    ms_cfg = MultiShootingConfig(N=N, node_dt=node_dt, spline=spline,
        ipopt_options={"linear_solver":"mumps", "print_level":5, "max_iter":max_iter,
                       "tol":1e-2, "acceptable_tol":1e-2,
                       "acceptable_constr_viol_tol":1e-2, "acceptable_iter":5,
                       "hessian_approximation":"limited-memory", "mu_strategy":"adaptive"},
        keep_best_sol=True, keep_best_sol_rho=1e2)
    return G1GaitTO(dyn_cfg, ms_cfg, TrackConfig(), X, 0.02)


def preflight():
    entry, adapted, ref, X, contract = load_reference()
    problem = build_problem(X, max_iter=1)
    X0, U0 = problem.warm_start(problem.N_sim)
    z = problem.pack(X0[::problem.K][:problem.N+1], problem.spline.fit(U0))
    checks = {
        "scene_sha256": sha(SCENE), "control_sha256": sha(CONTROL), "motion_sha256": sha(LUNGE),
        "mujoco_version": mujoco.__version__,
        "integrator_official_euler": int(problem.dyn.model.opt.integrator) == int(mujoco.mjtIntegrator.mjINT_EULER),
        "reference_velocity_max_abs": float(np.max(np.abs(X[:, problem.dyn.nq:]))),
        "reference_velocity_nonzero": bool(np.max(np.abs(X[:, problem.dyn.nq:])) > 1e-6),
        "pd_exact": bool(np.array_equal(problem.dyn.kp, contract.kp) and np.array_equal(problem.dyn.kd, contract.kd)),
        "constraints_registered": int(problem.n_constraints),
        "constraints_returned": int(len(problem.constraints(z))),
        "constraints_length_exact": len(problem.constraints(z)) == problem.n_constraints,
        "head_bounds": [[float(problem._lb[problem.n_states+a]), float(problem._ub[problem.n_states+a])]
                        for a,n in enumerate(contract.actuator_joint_names) if n.startswith("head_")],
    }
    checks["head_bounds_zero"] = all(x == [0.0, 0.0] for x in checks["head_bounds"])
    checks["servo_equivalence"] = external_pd_servo_equivalence(X, contract)
    checks["baseline"] = baseline_free_root(entry, ref)
    checks["pass"] = bool(checks["integrator_official_euler"] and checks["reference_velocity_nonzero"]
                          and checks["pd_exact"] and checks["constraints_length_exact"]
                          and checks["head_bounds_zero"] and checks["servo_equivalence"]["pass"]
                          and checks["baseline"]["pass"])
    (OUT/"phase3b_preflight.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))
    return checks


def solve():
    entry, adapted, ref, Xref, contract = load_reference()
    problem = build_problem(Xref, max_iter=300)
    X0, U0 = problem.warm_start(problem.N_sim)
    t0 = time.perf_counter(); X, U, info = problem.solve(X0, U0); wall = time.perf_counter()-t0
    Xfine = problem.stitched_trajectory(X, U)
    ends = problem.dyn.rollout_batch(X[:problem.N], U.reshape(problem.N, problem.K, problem.nu))[:, problem.K]
    defects = np.array([problem.dyn.state_diff(X[i+1], ends[i]) for i in range(problem.N)])
    out = OUT/"x2_lunge_phase3b_prefix.npz"
    save_trajectory(str(out), time=np.arange(len(Xfine))*0.001, state=Xfine, input=U,
                    model=str(SCENE), reference=problem.X_ref, spline_type="linear",
                    defects=defects, node_dt=np.asarray(0.02))
    status = info.get("status_msg", ""); status = status.decode() if isinstance(status,(bytes,bytearray)) else str(status)
    result = {"status": int(info.get("status",-999)), "status_msg":status,
              "status_ok": info.get("status") in (0,1), "wall_s":wall,
              "obj": float(info.get("obj_val",np.nan)), "max_defect":float(np.max(np.abs(defects))),
              "N":problem.N, "K":problem.K, "controls":list(U.shape), "states":list(Xfine.shape),
              "artifact":str(out)}
    (OUT/"phase3b_solve_result.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2)); return result


if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("mode",choices=("preflight","solve")); a=ap.parse_args()
    preflight() if a.mode=="preflight" else solve()
