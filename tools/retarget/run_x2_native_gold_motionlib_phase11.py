#!/usr/bin/env python3
"""Ingest and round-trip Phase10 X2 native Gold through SONIC MotionLib.

Run this script in the existing ``x2-sonic-isaaclab`` environment.  It uses
the native MotionLib loader first.  The only adapter restores state-rich
actual velocities/contact that native MotionLib otherwise recomputes; source
Gold pose/root/fps/clips remain immutable.  No policy, PPO, LoRA, physics
replay, checkpoint, BASE, Git, cloud, or real robot is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import torch


REPO = Path(__file__).resolve().parents[2]
SONIC_ROOT = Path(
    "/home/humanplus/x2_teleop_final/x2_sonic/sonic_x2_sandbox"
)
SRC_ROOT = REPO / "src"
for value in (SONIC_ROOT, SRC_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from easydict import EasyDict
from gear_sonic.utils.motion_lib.motion_lib_base import FixHeightMode
from gear_sonic.utils.motion_lib.motion_lib_robot import MotionLibRobot

from x2_native_gold_motionlib_adapter import (
    apply_recorded_state_adapter,
    select_wbt29,
    wbt29_indices,
)


DEFAULT_GOLD = Path(
    "/home/humanplus/projects/ZHY/x2_wbt_official_v1/motion_lib_x2_official_v1/"
    "gold_dynamic_native_seed_v1"
)
DEFAULT_HELD = DEFAULT_GOLD / "held_out/official_native_dance_held_out.pkl"
DEFAULT_MODEL_CONTRACT = REPO / "reports/retarget/x2_official_joint_body_map.json"
DEFAULT_JSON = REPO / "reports/retarget/x2_native_gold_motionlib_phase11.json"
DEFAULT_MD = REPO / "reports/retarget/x2_native_gold_motionlib_phase11.md"
OFFICIAL_MODEL_DIR = Path(
    "/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/worktree/x2_rl_deploy/"
    "x2_rl_deploy_mujoco/configuration/robot/lx2501_3_t2d5/model_info"
)
OFFICIAL_X2_XML = OFFICIAL_MODEL_DIR / "x2.xml"

Q_MAX_ERROR_RAD = 5.0e-6
ROOT_POS_MAX_ERROR_M = 1.0e-7
ROOT_QUAT_MAX_ERROR_RAD = 2.0e-3
FK_POS_MAX_ERROR_M = 2.0e-5
FK_ORI_MAX_ERROR_RAD = 2.0e-4
ADAPTED_STATE_MAX_ERROR = 1.0e-7


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def native_config(motion_file: Path) -> EasyDict:
    return EasyDict({
        "motion_file": str(motion_file),
        "smpl_motion_file": "zeros",
        "asset": EasyDict({
            "assetRoot": str(OFFICIAL_MODEL_DIR),
            "assetFileName": "x2.xml",
            "urdfFileName": "",
        }),
        "extend_config": [],
        "target_fps": 50,
        "step_dt": 1.0 / 50.0,
        "multi_thread": False,
        "fix_height": FixHeightMode.no_fix,
        "use_parallel_fk": False,
        "randomize_heading": False,
        "freeze_frame_aug": False,
        "randomize_wrist_poses": False,
        "cat_upper_body_poses": False,
        # This native inference is audited against, then the adapter restores
        # Phase10 model_contact.  It is not treated as hardware contact truth.
        "foot_contact_label_mode": "motion_local_height",
        "zero_root_xy": False,
    })


def concatenate(entries: dict[str, dict[str, Any]], keys: list[str], field: str) -> np.ndarray:
    return np.concatenate([np.asarray(entries[key][field]) for key in keys], axis=0)


def quat_geodesic_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1.0e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1.0e-12)
    dots = np.abs(np.sum(a * b, axis=-1))
    return 2.0 * np.arccos(np.clip(dots, -1.0, 1.0))


def native_roundtrip(motion_lib: MotionLibRobot, entries: dict[str, Any]) -> dict[str, Any]:
    keys = list(motion_lib.curr_motion_keys)
    q = concatenate(entries, keys, "dof")
    dq = concatenate(entries, keys, "dof_vel")
    root = concatenate(entries, keys, "root_trans_offset")
    quat = concatenate(entries, keys, "root_rot")
    root_lin = concatenate(entries, keys, "root_lin_vel_w_mps")
    root_ang = concatenate(entries, keys, "root_ang_vel")
    source_contact = {
        side: np.concatenate([np.asarray(entries[key]["model_contact"][side]) for key in keys])
        for side in ("left", "right")
    }
    loaded_q = motion_lib.dof_pos.detach().cpu().numpy()
    loaded_dq = motion_lib.dof_vel.detach().cpu().numpy()
    loaded_root = motion_lib.body_pos_w[:, 0].detach().cpu().numpy()
    loaded_quat = motion_lib.body_quat_w[:, 0].detach().cpu().numpy()
    native_contact = {
        "left": motion_lib.feet_l.detach().cpu().numpy().reshape(-1).astype(bool),
        "right": motion_lib.feet_r.detach().cpu().numpy().reshape(-1).astype(bool),
    }
    return {
        "keys_exact": keys == list(entries.keys()),
        "clip_frame_counts": [int(value) for value in motion_lib._motion_num_frames.cpu().tolist()],
        "clip_fps": [float(value) for value in motion_lib._motion_fps.cpu().tolist()],
        "length_starts": [int(value) for value in motion_lib.length_starts.cpu().tolist()],
        "q_error_abs_p95_max_rad": [float(np.percentile(np.abs(loaded_q - q), 95)), float(np.max(np.abs(loaded_q - q)))],
        "native_recomputed_dq_error_abs_p95_max_radps": [float(np.percentile(np.abs(loaded_dq - dq), 95)), float(np.max(np.abs(loaded_dq - dq)))],
        "root_pos_error_abs_max_m": float(np.max(np.abs(loaded_root - root))),
        "root_quat_geodesic_p95_max_rad": [float(np.percentile(quat_geodesic_xyzw(loaded_quat, quat), 95)), float(np.max(quat_geodesic_xyzw(loaded_quat, quat)))],
        "native_recomputed_root_lin_error_abs_p95_max_mps": [float(np.percentile(np.abs(motion_lib.body_lin_vel_w[:, 0].cpu().numpy() - root_lin), 95)), float(np.max(np.abs(motion_lib.body_lin_vel_w[:, 0].cpu().numpy() - root_lin)))],
        "native_recomputed_root_ang_error_abs_p95_max_radps": [float(np.percentile(np.abs(motion_lib.body_ang_vel_w[:, 0].cpu().numpy() - root_ang), 95)), float(np.max(np.abs(motion_lib.body_ang_vel_w[:, 0].cpu().numpy() - root_ang)))],
        "native_inferred_contact_agreement": {
            side: float(np.mean(native_contact[side] == source_contact[side])) for side in ("left", "right")
        },
    }


def adapter_roundtrip(motion_lib: MotionLibRobot, entries: dict[str, Any]) -> dict[str, Any]:
    keys = list(motion_lib.curr_motion_keys)
    dq = concatenate(entries, keys, "dof_vel")
    root_lin = concatenate(entries, keys, "root_lin_vel_w_mps")
    root_ang = concatenate(entries, keys, "root_ang_vel")
    source_contact = {
        side: np.concatenate([np.asarray(entries[key]["model_contact"][side]) for key in keys])
        for side in ("left", "right")
    }
    return {
        "dof_vel_error_abs_max_radps": float(np.max(np.abs(motion_lib.dof_vel.cpu().numpy() - dq))),
        "root_lin_vel_error_abs_max_mps": float(np.max(np.abs(motion_lib.body_lin_vel_w[:, 0].cpu().numpy() - root_lin))),
        "root_ang_vel_error_abs_max_radps": float(np.max(np.abs(motion_lib.body_ang_vel_w[:, 0].cpu().numpy() - root_ang))),
        "contact_exact": {
            "left": bool(np.array_equal(motion_lib.feet_l.cpu().numpy().reshape(-1).astype(bool), source_contact["left"])),
            "right": bool(np.array_equal(motion_lib.feet_r.cpu().numpy().reshape(-1).astype(bool), source_contact["right"])),
        },
    }


def official_fk_roundtrip(
    motion_lib: MotionLibRobot,
    first_entry: dict[str, Any],
    official_names: list[str],
) -> dict[str, Any]:
    """Compare held-out clip0 loader FK to official x2.xml at identical q/root."""
    model = mujoco.MjModel.from_xml_path(str(OFFICIAL_X2_XML))
    data = mujoco.MjData(model)
    qpos_addresses = [
        int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
        for name in official_names
    ]
    loader_names = list(motion_lib.mesh_parsers.body_names)
    model_body_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in loader_names
    ]
    if any(value < 0 for value in model_body_ids):
        raise ValueError("loader body missing in official x2.xml")
    count = len(first_entry["dof"])
    loaded_pos = motion_lib.body_pos_w[:count].detach().cpu().numpy()
    loaded_quat = motion_lib.body_quat_w[:count].detach().cpu().numpy()
    pos_errors = []
    ori_errors = []
    for frame in range(count):
        data.qpos[:3] = first_entry["root_trans_offset"][frame]
        xyzw = first_entry["root_rot"][frame]
        data.qpos[3:7] = [xyzw[3], xyzw[0], xyzw[1], xyzw[2]]
        data.qpos[qpos_addresses] = first_entry["dof"][frame]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        expected_pos = data.xpos[model_body_ids]
        expected_rot = Rotation.from_matrix(data.xmat[model_body_ids].reshape(-1, 3, 3)).as_quat()
        pos_errors.append(np.abs(loaded_pos[frame] - expected_pos))
        ori_errors.append(quat_geodesic_xyzw(loaded_quat[frame], expected_rot))
    pos_errors = np.asarray(pos_errors)
    ori_errors = np.asarray(ori_errors)
    return {
        "frames": count,
        "body_count": len(loader_names),
        "loader_body_order": loader_names,
        "position_error_abs_p95_max_m": [float(np.percentile(pos_errors, 95)), float(np.max(pos_errors))],
        "orientation_geodesic_p95_max_rad": [float(np.percentile(ori_errors, 95)), float(np.max(ori_errors))],
    }


def render(report: dict[str, Any]) -> str:
    native = report["native_loader"]
    adapter = report["state_adapter"]["roundtrip"]
    fk = report["official_fk_roundtrip"]
    lines = [
        "# X2 Native Gold → SONIC MotionLib Phase11",
        "",
        f"- 裁决：**{report['decision']['status']}**。",
        "- 仅做 Phase10 held-out MotionLib ingestion/FK round-trip；没有 PPO、LoRA、checkpoint、BASE、官方物理重放或真机。",
        "- source trace 自身60秒稳定不等于 reference replay 稳定；本阶段没有把二者混写。",
        "",
        "## 假设",
        "",
        "Phase10 Gold 的 actual pose/root 能被 SONIC 原生 MotionLib 无损读入；原生 loader 对速度/contact 的派生行为可通过独立最小 adapter 显式恢复，而不修改源 Gold。",
        "",
        "## 干预与对照",
        "",
        "- 对照：原生 `MotionLibRobot`，`no_fix`、50Hz、关闭随机 heading/wrist/freeze augmentation。",
        "- 干预：只在内存中恢复 `dof_vel`、root lin/ang velocity 和 Phase10 model contact；pose/root/FK/fps/clip 不允许变化。",
        "- zero-update gate：固定 held-out clip0，比较源 Gold、MotionLib FK 与官方 `x2.xml` FK；无动作修正。",
        "",
        "## 结果",
        "",
        f"- clips/fps/starts：`{native['clip_frame_counts']}` / `{native['clip_fps']}` / `{native['length_starts']}`。",
        f"- 原生 q p95/max：`{native['q_error_abs_p95_max_rad'][0]:.2e}/{native['q_error_abs_p95_max_rad'][1]:.2e}rad`；root pos max `{native['root_pos_error_abs_max_m']:.2e}m`；root quat max `{native['root_quat_geodesic_p95_max_rad'][1]:.2e}rad`。",
        f"- 原生 loader 会显式重算 dq：与 actual dq p95/max `{native['native_recomputed_dq_error_abs_p95_max_radps'][0]:.3f}/{native['native_recomputed_dq_error_abs_p95_max_radps'][1]:.3f}rad/s`，因此启用独立 state adapter。",
        f"- adapter 后 dq/root-lin/root-ang max error：`{adapter['dof_vel_error_abs_max_radps']:.2e}/{adapter['root_lin_vel_error_abs_max_mps']:.2e}/{adapter['root_ang_vel_error_abs_max_radps']:.2e}`；contact exact `{adapter['contact_exact']}`。",
        f"- official FK position p95/max `{fk['position_error_abs_p95_max_m'][0]:.2e}/{fk['position_error_abs_p95_max_m'][1]:.2e}m`；orientation p95/max `{fk['orientation_geodesic_p95_max_rad'][0]:.2e}/{fk['orientation_geodesic_p95_max_rad'][1]:.2e}rad`。",
        f"- WBT29 q/dq max error `{report['wbt29_contract']['q_error_abs_max_rad']:.2e}/{report['wbt29_contract']['dq_error_abs_max_radps']:.2e}`；head q/dq max `{report['wbt29_contract']['head_q_abs_max_rad']:.2e}/{report['wbt29_contract']['head_dq_abs_max_radps']:.2e}`。",
        "",
        "## 结论",
        "",
        f"- 结果：{report['decision']['result']}",
        f"- 结论：{report['decision']['conclusion']}",
        f"- 下一步：{report['decision']['next_step']}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--held-out", type=Path, default=DEFAULT_HELD)
    parser.add_argument("--model-contract", type=Path, default=DEFAULT_MODEL_CONTRACT)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    entries = joblib.load(args.held_out)
    model_contract = json.loads(args.model_contract.read_text(encoding="utf-8"))
    official_names = model_contract["control_boundaries"]["official_mjcf_actuated_31"]
    wbt_names = model_contract["control_boundaries"]["wbt_target_29"]
    head_names = model_contract["control_boundaries"]["head_locked_2"]

    motion_lib = MotionLibRobot(native_config(args.held_out), len(entries), "cpu")
    motion_lib.load_motions_for_evaluation()
    native = native_roundtrip(motion_lib, entries)
    before_q = motion_lib.dof_pos.detach().clone()
    before_root = motion_lib.body_pos_w.detach().clone()
    before_quat = motion_lib.body_quat_w.detach().clone()
    adapter_meta = apply_recorded_state_adapter(motion_lib, args.held_out)
    adapted = adapter_roundtrip(motion_lib, entries)
    adapter_pose_exact = {
        "q": bool(torch.equal(before_q, motion_lib.dof_pos)),
        "root": bool(torch.equal(before_root, motion_lib.body_pos_w)),
        "quat": bool(torch.equal(before_quat, motion_lib.body_quat_w)),
    }
    first_key = list(entries)[0]
    fk = official_fk_roundtrip(motion_lib, entries[first_key], official_names)

    keys = list(motion_lib.curr_motion_keys)
    source_q = concatenate(entries, keys, "dof")
    source_dq = concatenate(entries, keys, "dof_vel")
    indices = wbt29_indices(official_names, wbt_names)
    loaded_wbt_q = select_wbt29(motion_lib.dof_pos, official_names, wbt_names).cpu().numpy()
    loaded_wbt_dq = select_wbt29(motion_lib.dof_vel, official_names, wbt_names).cpu().numpy()
    head_indices = [official_names.index(name) for name in head_names]
    wbt = {
        "indices_official31": indices.tolist(),
        "q_error_abs_max_rad": float(np.max(np.abs(loaded_wbt_q - source_q[:, indices]))),
        "dq_error_abs_max_radps": float(np.max(np.abs(loaded_wbt_dq - source_dq[:, indices]))),
        "head_q_abs_max_rad": float(np.max(np.abs(motion_lib.dof_pos[:, head_indices].cpu().numpy()))),
        "head_dq_abs_max_radps": float(np.max(np.abs(motion_lib.dof_vel[:, head_indices].cpu().numpy()))),
    }

    checks = {
        "native_keys_exact": native["keys_exact"],
        "clip_boundaries_exact_3x400": native["clip_frame_counts"] == [400, 400, 400]
        and native["length_starts"] == [0, 400, 800],
        "fps_exact_50": native["clip_fps"] == [50.0, 50.0, 50.0],
        "native_q_roundtrip": native["q_error_abs_p95_max_rad"][1] <= Q_MAX_ERROR_RAD,
        "native_root_position_roundtrip": native["root_pos_error_abs_max_m"] <= ROOT_POS_MAX_ERROR_M,
        "native_root_orientation_roundtrip": native["root_quat_geodesic_p95_max_rad"][1] <= ROOT_QUAT_MAX_ERROR_RAD,
        "adapter_does_not_change_pose_fk": all(adapter_pose_exact.values()),
        "adapter_restores_actual_state": max(
            adapted["dof_vel_error_abs_max_radps"],
            adapted["root_lin_vel_error_abs_max_mps"],
            adapted["root_ang_vel_error_abs_max_radps"],
        ) <= ADAPTED_STATE_MAX_ERROR,
        "adapter_restores_model_contact": all(adapted["contact_exact"].values()),
        "official_fk_position_roundtrip": fk["position_error_abs_p95_max_m"][1] <= FK_POS_MAX_ERROR_M,
        "official_fk_orientation_roundtrip": fk["orientation_geodesic_p95_max_rad"][1] <= FK_ORI_MAX_ERROR_RAD,
        "wbt29_exact": wbt["q_error_abs_max_rad"] <= Q_MAX_ERROR_RAD
        and wbt["dq_error_abs_max_radps"] <= ADAPTED_STATE_MAX_ERROR,
        "head_lock_exact": wbt["head_q_abs_max_rad"] <= Q_MAX_ERROR_RAD
        and wbt["head_dq_abs_max_radps"] <= ADAPTED_STATE_MAX_ERROR,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    passed = all(checks.values())
    report = {
        "schema_version": "x2_native_gold_motionlib_phase11_v1",
        "provenance": {
            "held_out_gold": {"path": str(args.held_out), "sha256": sha256(args.held_out)},
            "model_contract": {"path": str(args.model_contract), "sha256": sha256(args.model_contract)},
            "official_x2_xml": {"path": str(OFFICIAL_X2_XML), "sha256": sha256(OFFICIAL_X2_XML)},
            "sonic_motionlib": str(SONIC_ROOT / "gear_sonic/utils/motion_lib/motion_lib_base.py"),
        },
        "truth_boundary": {
            "phase10_source_is_official_simulation_not_hardware": True,
            "contact_is_model_geometry_not_hardware_grf_cop_wrench": True,
            "source_trace_stability_is_not_replay_stability": True,
            "official_physics_replay_executed": False,
            "ppo_lora_training_checkpoint_base_git_baidu_real_robot": False,
        },
        "hypothesis": "Phase10 native actual pose/root is natively MotionLib-compatible; a minimal in-memory adapter is sufficient only for auxiliary actual velocity/contact fields.",
        "intervention": "Native loader first; then restore dq/root velocities/model contact in memory without altering q/root/FK/fps/clips.",
        "control": "Unmodified held-out Gold through MotionLibRobot with no_fix and all training augmentations disabled.",
        "native_loader": native,
        "state_adapter": {
            "required_because": "native MotionLib recomputes dq/root velocity/contact and ignores Phase10 recorded auxiliary fields",
            "metadata": adapter_meta,
            "pose_fk_unchanged": adapter_pose_exact,
            "roundtrip": adapted,
        },
        "official_fk_roundtrip": fk,
        "wbt29_contract": wbt,
        "zero_update_gate": {
            "scope": first_key,
            "thresholds": {
                "q_max_error_rad": Q_MAX_ERROR_RAD,
                "root_pos_max_error_m": ROOT_POS_MAX_ERROR_M,
                "root_quat_max_error_rad": ROOT_QUAT_MAX_ERROR_RAD,
                "fk_pos_max_error_m": FK_POS_MAX_ERROR_M,
                "fk_orientation_max_error_rad": FK_ORI_MAX_ERROR_RAD,
                "adapted_state_max_error": ADAPTED_STATE_MAX_ERROR,
            },
            "checks": checks,
            "pass": passed,
        },
        "decision": {
            "status": "PHASE11_MOTIONLIB_INGESTION_PASSED" if passed else "PHASE11_MOTIONLIB_INGESTION_REJECTED",
            "failed_checks": [name for name, value in checks.items() if not value],
            "result": (
                "Phase10 held-out Gold已通过SONIC MotionLib原生pose/root/FK与最小state adapter round-trip；未发现clip、fps、WBT29或head-lock静默改写。"
                if passed
                else "MotionLib ingestion/FK契约未通过预注册zero-update门；按规则停止，不进入physics或训练。"
            ),
            "conclusion": (
                "Gold可作为SONIC/X2 pipeline ingestion sanity set；actual velocity/contact必须通过独立adapter显式恢复，不能假设原生loader会保留。"
                if passed
                else "当前Gold不能安全接入SONIC/X2 MotionLib；失败只评价schema/FK契约。"
            ),
            "next_step": (
                "若继续，只允许在held-out clip0做一次冻结prescribed/free reference-trackability smoke；仍不得训练。"
                if passed
                else "停止，不扩adapter或训练；先修复明确失败的loader/FK字段。"
            ),
        },
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(render(report), encoding="utf-8")
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
