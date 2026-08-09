#!/usr/bin/env python3
"""CPU zero-step probe for Phase23 WBT29 and Gold split hooks.

No simulator is stepped and no policy, optimizer, or Isaac application is
created.  SONIC MotionLib is used only as a CPU kinematic loader.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import torch


REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
SONIC = Path("/home/humanplus/x2_teleop_final/x2_sonic/sonic_x2_sandbox")
for path in (SRC, SONIC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from easydict import EasyDict
from gear_sonic.utils.motion_lib.motion_lib_base import FixHeightMode
from gear_sonic.utils.motion_lib.motion_lib_robot import MotionLibRobot

from x2_faithful_any2any_phase23 import (
    GoldSplitSpec,
    ImmutableGoldMotionLibHook,
    POLICY_TERM_ORDER,
    WBT29PolicyContract,
)


GOLD = Path(
    "/home/humanplus/projects/ZHY/x2_wbt_official_v1/motion_lib_x2_official_v1/"
    "gold_dynamic_native_seed_v1"
)
TRAIN = GOLD / "train/official_native_dance_train.pkl"
HELD = GOLD / "held_out/official_native_dance_held_out.pkl"
MODEL_CONTRACT = REPO / "reports/retarget/x2_official_joint_body_map.json"
JOINT_UTILS = SONIC / "gear_sonic/envs/env_utils/joint_utils.py"
OFFICIAL_MODEL_DIR = Path(
    "/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/worktree/x2_rl_deploy/"
    "x2_rl_deploy_mujoco/configuration/robot/lx2501_3_t2d5/model_info"
)

FROZEN_HASHES = {
    "train": "644dc7534b63a7831bfe4156941b01508003f2834d2ccdac7b227369bad8ef8b",
    "held_out": "45ffda2f8ddc64cbeb4ccc714d37a0c98ca328e8e6473edb12670dcdfd19a3dc",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def literal_assignment(path: Path, name: str) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                return list(ast.literal_eval(node.value))
    raise KeyError(name)


def native_config(path: Path) -> EasyDict:
    return EasyDict(
        {
            "motion_file": str(path),
            "smpl_motion_file": "zeros",
            "asset": EasyDict(
                {
                    "assetRoot": str(OFFICIAL_MODEL_DIR),
                    "assetFileName": "x2.xml",
                    "urdfFileName": "",
                }
            ),
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
            "foot_contact_label_mode": "motion_local_height",
            "zero_root_xy": False,
        }
    )


def cat(entries: dict[str, dict[str, Any]], field: str) -> np.ndarray:
    return np.concatenate([np.asarray(entry[field]) for entry in entries.values()], axis=0)


def probe_split(spec: GoldSplitSpec, expected_clips: int) -> dict[str, Any]:
    entries = joblib.load(spec.source_pkl)
    motion = MotionLibRobot(native_config(spec.source_pkl), len(entries), "cpu")
    motion.load_motions_for_evaluation()
    before = {
        "dof_pos": motion.dof_pos.detach().clone(),
        "body_pos_w": motion.body_pos_w.detach().clone(),
        "body_quat_w": motion.body_quat_w.detach().clone(),
    }
    hook = ImmutableGoldMotionLibHook(spec)
    hook_meta = hook.attach(motion)
    expected_dq = torch.as_tensor(cat(entries, "dof_vel"), dtype=motion.dof_vel.dtype)
    expected_root_lin = torch.as_tensor(
        cat(entries, "root_lin_vel_w_mps"), dtype=motion.body_lin_vel_w.dtype
    )
    expected_root_ang = torch.as_tensor(
        cat(entries, "root_ang_vel"), dtype=motion.body_ang_vel_w.dtype
    )
    expected_contact = {
        side: np.concatenate(
            [np.asarray(entry["model_contact"][side]) for entry in entries.values()]
        )
        for side in ("left", "right")
    }
    checks = {
        "clip_count": len(motion.curr_motion_keys) == expected_clips,
        "sampler_keys_exact": list(motion.curr_motion_keys) == hook_meta["sampler_keys"],
        "pose_q_unchanged": bool(torch.equal(before["dof_pos"], motion.dof_pos)),
        "root_position_fk_unchanged": bool(
            torch.equal(before["body_pos_w"], motion.body_pos_w)
        ),
        "root_orientation_fk_unchanged": bool(
            torch.equal(before["body_quat_w"], motion.body_quat_w)
        ),
        "actual_dq_restored": bool(torch.equal(expected_dq, motion.dof_vel.cpu())),
        "actual_root_lin_velocity_restored": bool(
            torch.equal(expected_root_lin, motion.body_lin_vel_w[:, 0].cpu())
        ),
        "actual_root_ang_velocity_restored": bool(
            torch.equal(expected_root_ang, motion.body_ang_vel_w[:, 0].cpu())
        ),
        "model_contact_left_restored": bool(
            np.array_equal(
                expected_contact["left"], motion.feet_l.cpu().numpy().reshape(-1)
            )
        ),
        "model_contact_right_restored": bool(
            np.array_equal(
                expected_contact["right"], motion.feet_r.cpu().numpy().reshape(-1)
            )
        ),
    }
    return {
        "hook": hook_meta,
        "motionlib": {
            "clip_frames": [int(value) for value in motion._motion_num_frames.tolist()],
            "fps": [float(value) for value in motion._motion_fps.tolist()],
            "length_starts": [int(value) for value in motion.length_starts.tolist()],
        },
        "checks": checks,
        "pass": all(checks.values()),
    }


def render(report: dict[str, Any]) -> str:
    lines = [
        "# X2 WBT29 + Native Gold Hook — Phase23",
        "",
        f"- 裁决：**{report['decision']['status']}**。",
        "- 本阶段只有纯单测与 CPU MotionLib zero-step probe；未启动 Isaac physics、PPO、LoRA、optimizer 或真机。",
        "- `model_contact` 是官方仿真碰撞几何标签，不是实机 GRF/COP/wrench。",
        "",
        "## 假设",
        "",
        "B1/B2 可以通过两个小型、可组合的边界实现：策略始终只看 G1 source-semantic 29DOF，模拟器31DOF中的头部保持显式 nominal；Gold train/held-out 由哈希和key分别锁定，状态adapter只恢复actual dq/root velocity/model contact。",
        "",
        "## 干预",
        "",
        "- 新增 name-driven WBT29 observation/history/action gather/scatter。",
        "- 新增 immutable Gold split hook；train可供后续optimizer sampler，held-out硬标记不可进入optimizer。",
        "",
        "## 对照",
        "",
        "- 31→source29→target29/31 round-trip 与原始按名切片逐元素比较。",
        "- MotionLib attach 前后逐元素比较 q、root pose 和全身 FK；actual dq/root velocity/contact 与源PKL比较。",
        "",
        "## 结果",
        "",
        f"- B1 checks：`{report['wbt29_zero_step_probe']['checks']}`。",
        f"- train keys：`{report['gold_splits']['train']['hook']['sampler_keys']}`。",
        f"- held-out keys：`{report['gold_splits']['held_out']['hook']['sampler_keys']}`。",
        f"- train/held CPU probes：`{report['gold_splits']['train']['pass']}` / `{report['gold_splits']['held_out']['pass']}`。",
        f"- policy history/action dims：`{report['contract']['flat_policy_dim']}` / `{report['contract']['policy_action_dim']}`；head exclusion：`{report['contract']['head_excluded2']}`。",
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
    parser.add_argument(
        "--json",
        type=Path,
        default=REPO / "reports/retarget/x2_faithful_wbt29_gold_phase23.json",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=REPO / "reports/retarget/x2_faithful_wbt29_gold_phase23.md",
    )
    args = parser.parse_args()

    model = json.loads(MODEL_CONTRACT.read_text(encoding="utf-8"))
    boundaries = model["control_boundaries"]
    contract = WBT29PolicyContract.build(
        boundaries["official_mjcf_actuated_31"],
        boundaries["wbt_target_29"],
        literal_assignment(JOINT_UTILS, "G1_ISAACLab_ORDER"),
        boundaries["head_locked_2"],
    )
    train_spec = GoldSplitSpec(
        "train", TRAIN, FROZEN_HASHES["train"], "official_native_dance_train_", True
    )
    held_spec = GoldSplitSpec(
        "held_out",
        HELD,
        FROZEN_HASHES["held_out"],
        "official_native_dance_held_out_",
        False,
    )

    train_entries = joblib.load(TRAIN)
    first = next(iter(train_entries.values()))
    q31 = torch.as_tensor(first["dof"][:10]).unsqueeze(0)
    dq31 = torch.as_tensor(first["dof_vel"][:10]).unsqueeze(0)
    source_q = contract.official_to_source(q31)
    target_q = contract.source_to_target(source_q)
    nominal31 = q31.clone()
    scattered = contract.source_action_to_official(source_q, nominal31)
    terms = {
        "gravity_dir": torch.zeros(1, 10, 3),
        "base_ang_vel": torch.as_tensor(first["root_ang_vel"][:10]).unsqueeze(0),
        "joint_pos": q31,
        "joint_vel": dq31,
        "actions": torch.zeros(1, 10, 29),
    }
    adapted = contract.adapt_policy_history(terms)
    flat = contract.flatten_policy_history(adapted)
    target_indices = torch.as_tensor(contract.official_to_target_indices)
    head_indices = torch.as_tensor(contract.head_indices)
    expected_target = q31.index_select(-1, target_indices)
    checks = {
        "source29_dim": source_q.shape[-1] == 29,
        "target_roundtrip_exact": bool(torch.equal(target_q, expected_target)),
        "official_scatter_nonhead_exact": bool(
            torch.equal(scattered.index_select(-1, target_indices), expected_target)
        ),
        "official_scatter_head_nominal_exact": bool(
            torch.equal(
                scattered.index_select(-1, head_indices),
                nominal31.index_select(-1, head_indices),
            )
        ),
        "history_terms_exact": tuple(adapted) == POLICY_TERM_ORDER,
        "history_joint_pos_source_exact": bool(torch.equal(adapted["joint_pos"], source_q)),
        "history_joint_vel_source_exact": bool(
            torch.equal(adapted["joint_vel"], contract.official_to_source(dq31))
        ),
        "flat_policy_dim_930": flat.shape == (1, 930),
    }
    split_probes = {
        "train": probe_split(train_spec, 4),
        "held_out": probe_split(held_spec, 3),
    }
    train_keys = set(split_probes["train"]["hook"]["sampler_keys"])
    held_keys = set(split_probes["held_out"]["hook"]["sampler_keys"])
    split_checks = {
        "paths_are_distinct": TRAIN.resolve() != HELD.resolve(),
        "keys_are_disjoint": not bool(train_keys & held_keys),
        "train_optimizer_eligible": split_probes["train"]["hook"]["optimizer_eligible"] is True,
        "held_optimizer_forbidden": split_probes["held_out"]["hook"]["optimizer_eligible"] is False,
        "hashes_frozen": sha256(TRAIN) == FROZEN_HASHES["train"]
        and sha256(HELD) == FROZEN_HASHES["held_out"],
    }
    passed = all(checks.values()) and all(split_checks.values()) and all(
        item["pass"] for item in split_probes.values()
    )
    manifest = contract.manifest()
    manifest["contract_hash"] = canonical_hash(manifest)
    report = {
        "schema_version": "x2_faithful_wbt29_gold_phase23_v1",
        "truth_boundary": {
            "cpu_motionlib_zero_step_only": True,
            "isaac_physics_training_optimizer_network_mutation": False,
            "live_train_eval_entrypoint_wired": False,
            "head_absent_from_policy": True,
            "contact_is_model_estimate_not_hardware_grf_cop_wrench": True,
        },
        "provenance": {
            "implementation": str(REPO / "src/x2_faithful_any2any_phase23.py"),
            "model_contract": {"path": str(MODEL_CONTRACT), "sha256": sha256(MODEL_CONTRACT)},
            "gold_train": {"path": str(TRAIN), "sha256": sha256(TRAIN)},
            "gold_held_out": {"path": str(HELD), "sha256": sha256(HELD)},
        },
        "contract": manifest,
        "wbt29_zero_step_probe": {"checks": checks, "pass": all(checks.values())},
        "gold_splits": split_probes,
        "split_isolation": {"checks": split_checks, "pass": all(split_checks.values())},
        "decision": {
            "status": (
                "B1_B2_IMPLEMENTATION_READY_CPU_PROBED" if passed else "B1_B2_REJECTED"
            ),
            "result": (
                "WBT29 gather/scatter/history and immutable train/held Gold hooks pass exact CPU zero-step contracts."
                if passed
                else "At least one WBT29 or Gold split contract failed."
            ),
            "conclusion": (
                "B1/B2 are implementation-ready as isolated composable hooks; they are not yet wired by a faithful live train/eval config, and live Isaac/PPO integration remains deliberately unexecuted."
                if passed
                else "Do not integrate into a live environment; fix only the failed contract."
            ),
            "next_step": (
                "Freeze this manifest, then address Phase22 B3-B5 before the preregistered live zero-update gate."
                if passed
                else "Stop without physics or optimizer expansion."
            ),
        },
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.markdown.write_text(render(report), encoding="utf-8")
    print(json.dumps({"status": report["decision"]["status"], "checks": sum(checks.values()) + sum(split_checks.values())}))


if __name__ == "__main__":
    main()
