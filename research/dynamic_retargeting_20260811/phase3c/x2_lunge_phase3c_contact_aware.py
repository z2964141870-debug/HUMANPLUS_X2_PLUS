#!/usr/bin/env python3
"""One preregistered contact-aware DSMS solve on the corrected Phase3b contract."""
from __future__ import annotations

import dataclasses, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path("/home/humanplus/projects/ZHY/dsms_workspace")
sys.path.insert(0, str(ROOT / "phase3b"))
import x2_lunge_dsms_phase3b_corrected as b

import mujoco
from src.dynamics import DynamicsConfig
from src.multi_shooting import MultiShootingConfig
from src.spline import SplineConfig
from examples.g1_gait.g1_gait import G1GaitTO
from utils.file_utils import save_trajectory

OUT = ROOT / "phase3c"; OUT.mkdir(parents=True, exist_ok=True)
TARGET_CLEARANCE = 0.00025


@dataclasses.dataclass
class ContactTrackConfig(b.TrackConfig):
    def ee_track_expanded(self):
        return (
            (1, "pelvis", 1.0, 1.0, 1.0, 0.01, 0.01, False),
            (1, "left_ankle_roll_link", 100.0, 300.0, 1.0, 20.0, 0.01, False),
            (1, "right_ankle_roll_link", 100.0, 300.0, 1.0, 20.0, 0.01, False),
            (1, "left_wrist_yaw_link", 1.0, 1.0, 1.0, 0.01, 0.01, True),
            (1, "right_wrist_yaw_link", 1.0, 1.0, 1.0, 0.01, 0.01, True),
        )


def build_problem(X, max_iter):
    sim_dt, node_dt, K = .001, .02, 20
    N = int(np.ceil(((len(X)-1)*20)/K))
    dyn = DynamicsConfig(model_path=str(b.SCENE), sim_dt=sim_dt, integrator=None,
                         actuator_mode="position", n_threads=8, fd_eps=1e-6,
                         fd_centered=True, friction_cone=None)
    spline = SplineConfig(M=max(2, int(round(N*K*sim_dt*50.0))+1), spline_type="linear")
    ms = MultiShootingConfig(N=N, node_dt=node_dt, spline=spline,
        ipopt_options={"linear_solver":"mumps", "print_level":5, "max_iter":max_iter,
          "tol":1e-2, "acceptable_tol":1e-2, "acceptable_constr_viol_tol":1e-2,
          "acceptable_iter":5, "hessian_approximation":"limited-memory", "mu_strategy":"adaptive"},
        keep_best_sol=True, keep_best_sol_rho=1e2)
    problem = G1GaitTO(dyn, ms, ContactTrackConfig(), X, .02)
    patch_contact_targets(problem)
    return problem


def patch_contact_targets(problem):
    """Set both ankle position/twist targets to exact official-sole DS anchors."""
    model, data = problem.dyn.model, mujoco.MjData(problem.dyn.model)
    _, feet = b.physics.foot_geom_contract(model)
    ankle_names = ("left_ankle_roll_link", "right_ankle_roll_link")
    ankle_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in ankle_names]
    n = len(problem.X_ref); body_pos = np.zeros((n, 2, 3)); sole_center = np.zeros((n, 2, 2)); min_clear = np.zeros((n, 2))
    for k, x in enumerate(problem.X_ref):
        data.qpos[:] = x[:model.nq]; data.qvel[:] = x[model.nq:model.nq+model.nv]; mujoco.mj_forward(model, data)
        for side_i, side in enumerate(("left", "right")):
            gs = list(feet[side]); body_pos[k, side_i] = data.xpos[ankle_ids[side_i]]
            sole_center[k, side_i] = np.mean(data.geom_xpos[gs, :2], axis=0)
            min_clear[k, side_i] = min(float(data.geom_xpos[g, 2] - model.geom_size[g, 0]) for g in gs)
    # EE position row order is pelvis, left ankle, right ankle, wrists.
    for side_i, row in enumerate((1, 2)):
        delta_xy = sole_center[0, side_i] - sole_center[:, side_i]
        problem.ee.pos_ref[:, row, :2] = body_pos[:, side_i, :2] + delta_xy
        problem.ee.pos_ref[:, row, 2] = body_pos[:, side_i, 2] + (TARGET_CLEARANCE - min_clear[:, side_i])
        # EE velocity row order is identical because every configured body has nonzero w_v.
        problem.ee.v_ref[:, row, :] = 0.0
    problem.contact_target_diagnostic = {
        "target_clearance_m": TARGET_CLEARANCE,
        "left_xy_anchor": sole_center[0, 0].tolist(), "right_xy_anchor": sole_center[0, 1].tolist(),
        "source_min_clearance_range_m": [float(min_clear.min()), float(min_clear.max())],
        "target_translation_max_m": float(max(np.max(np.linalg.norm(sole_center[:, 0]-sole_center[0, 0], axis=1)),
                                                  np.max(np.linalg.norm(sole_center[:, 1]-sole_center[0, 1], axis=1)),
                                                  np.max(np.abs(TARGET_CLEARANCE-min_clear))))
    }


def preflight():
    _, _, _, X, _ = b.load_reference(); p = build_problem(X, 1)
    X0, U0 = p.warm_start(p.N_sim); z = p.pack(X0[::p.K][:p.N+1], p.spline.fit(U0))
    out = {"parent_preflight": json.loads((ROOT/"phase3b/phase3b_preflight.json").read_text()),
           "contact_target": p.contact_target_diagnostic,
           "constraints_registered": int(p.n_constraints), "constraints_returned": int(len(p.constraints(z))),
           "pass": bool(len(p.constraints(z)) == p.n_constraints)}
    (OUT/"phase3c_preflight.json").write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps(out,indent=2)); return out


def solve():
    _, _, _, Xref, _ = b.load_reference(); p = build_problem(Xref, 300)
    X0, U0 = p.warm_start(p.N_sim); t0=time.perf_counter(); X,U,info=p.solve(X0,U0); wall=time.perf_counter()-t0
    Xfine=p.stitched_trajectory(X,U); ends=p.dyn.rollout_batch(X[:p.N],U.reshape(p.N,p.K,p.nu))[:,p.K]
    defects=np.array([p.dyn.state_diff(X[i+1],ends[i]) for i in range(p.N)])
    path=OUT/"x2_lunge_phase3c_contact_prefix.npz"
    save_trajectory(str(path),time=np.arange(len(Xfine))*.001,state=Xfine,input=U,model=str(b.SCENE),reference=p.X_ref,spline_type="linear",defects=defects,node_dt=np.asarray(.02))
    msg=info.get("status_msg",""); msg=msg.decode() if isinstance(msg,(bytes,bytearray)) else str(msg)
    out={"status":int(info.get("status",-999)),"status_msg":msg,"status_ok":info.get("status") in (0,1),
         "wall_s":wall,"obj":float(info.get("obj_val",np.nan)),"max_defect":float(np.max(np.abs(defects))),
         "defect_p95":float(np.percentile(np.abs(defects),95)),"artifact":str(path),"contact_target":p.contact_target_diagnostic}
    (OUT/"phase3c_solve_result.json").write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps(out,indent=2)); return out


if __name__ == "__main__":
    mode=sys.argv[1] if len(sys.argv)>1 else "preflight"
    preflight() if mode=="preflight" else solve()
