#!/usr/bin/env python3
"""Read-only capability audit for the next joint geometry/contact generator."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform

import mujoco
import numpy as np
import scipy
from scipy import optimize, sparse
from scipy.sparse import linalg as sparse_linalg


ROOT = Path("/home/humanplus/projects/ZHY")
REPO = ROOT / "CWI_CrossEmbodiment_Sim"
DSMS = ROOT / "dsms_workspace/shooting-for-contact"
SCENE = ROOT / "x2_official_rl_deploy_v1/worktree/x2_rl_deploy/x2_rl_deploy_mujoco/configuration/robot/lx2501_3_t2d5/model_info/scene.xml"
MOTION = ROOT / "x2_wbt_official_v1/motion_lib_x2_official_v1/silver_contact/phase30_time_dilation/x2_phase30_time_dilation_1p46.pkl"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    matrix = sparse.eye(4, format="csr")
    lsqr = sparse_linalg.lsqr(matrix, np.ones(4), iter_lim=4)
    slsqp = optimize.minimize(lambda x: float(np.dot(x, x)), np.ones(2), method="SLSQP")
    qpsolvers_spec = importlib.util.find_spec("qpsolvers")
    qpsolvers = None
    if qpsolvers_spec is not None:
        import qpsolvers as qpsolvers_module
        qpsolvers = sorted(qpsolvers_module.available_solvers)
    result = {
        "stage": "dynamic retargeting Phase17 next-generator capability audit",
        "execution": {"mj_forward_calls": 1, "mj_step_calls": 0, "optimizer_problem_size": "4x4 LSQR + 2D SLSQP smoke only", "gpu": False},
        "runtime": {"python": platform.python_version(), "mujoco": mujoco.__version__, "scipy": scipy.__version__},
        "assets": {
            "official_scene_exists": SCENE.is_file(), "official_scene_sha256": sha256(SCENE),
            "phase30_motion_exists": MOTION.is_file(), "phase30_motion_sha256": sha256(MOTION),
            "shooting_for_contact_tree_exists": DSMS.is_dir(),
            "phase15_result_exists": (REPO / "research/dynamic_retargeting_20260811/phase15_morphology_normalization/phase15_result.json").is_file(),
            "phase13_result_exists": (REPO / "research/dynamic_retargeting_20260811/phase13_support_margin_audit/phase13_result.json").is_file(),
        },
        "apis": {
            "mj_jacGeom": hasattr(mujoco, "mj_jacGeom"), "mj_jacBody": hasattr(mujoco, "mj_jacBody"),
            "mj_jacSubtreeCom": hasattr(mujoco, "mj_jacSubtreeCom"), "mj_geomDistance": hasattr(mujoco, "mj_geomDistance"),
            "scipy_sparse_lsqr": bool(lsqr[1] in (1, 2)), "scipy_slsqp": bool(slsqp.success),
            "scipy_milp": hasattr(optimize, "milp"), "qpsolvers_available": qpsolvers,
            "cyipopt_importable": importlib.util.find_spec("cyipopt") is not None,
        },
        "decision": {
            "offline_joint_geometry_generator_implementable_now": True,
            "raw_mujoco_replay_implementable_now": True,
            "faithful_ddr_or_omnitrack_upstream_available": False,
            "long_training_unlocked": False,
            "selected_next_method": "X2-specific staged contact/foot-placement/root/lower-q generator using sparse Jacobians, Phase15 init, discrete schedule enumeration, then one raw official-PD replay",
        },
    }
    required = [
        result["assets"]["official_scene_exists"], result["assets"]["phase30_motion_exists"],
        result["apis"]["mj_jacGeom"], result["apis"]["mj_jacSubtreeCom"],
        result["apis"]["scipy_sparse_lsqr"], result["apis"]["scipy_slsqp"],
    ]
    if not all(required):
        result["decision"]["offline_joint_geometry_generator_implementable_now"] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
