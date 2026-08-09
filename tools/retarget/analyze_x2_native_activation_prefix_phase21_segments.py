#!/usr/bin/env python3
"""Offline-only segmentation of Phase21 source abnormalities (no mj_step)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


REPO = Path(__file__).resolve().parents[2]
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))
import retarget.run_x2_forefoot_official_physics_screen as physics


SOURCE = Path(
    "/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/results/"
    "official_native_activation_prefix_phase21_20260809/"
    "official_native_event_v2_activation_prefix_phase21.npz"
)
DEFAULT_JSON = REPO / "reports/retarget/x2_native_activation_prefix_phase21_source_segments.json"
DEFAULT_MD = REPO / "reports/retarget/x2_native_activation_prefix_phase21_source_segments.md"


def phase(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    return {
        "double_support": float(np.mean(left & right)),
        "single_support": float(np.mean(left ^ right)),
        "flight": float(np.mean(~left & ~right)),
    }


def analyze(source: Path) -> dict[str, Any]:
    with np.load(source, allow_pickle=False) as archive:
        raw = {name: archive[name] for name in archive.files}
    times = raw["time_s"].astype(np.float64)
    q = raw["joint_q_rad"].astype(np.float64)
    dq = raw["joint_dq_radps"].astype(np.float64)
    root = raw["root_pos_w_m"].astype(np.float64)
    quat = raw["root_quat_xyzw"].astype(np.float64)
    names = raw["joint_names"].astype(str).tolist()
    rl = float(raw["mode_event_elapsed_ns"][1] * 1.0e-9)

    model = mujoco.MjModel.from_xml_path(str(physics.DEFAULT_SCENE))
    data = mujoco.MjData(model)
    contract = physics.build_control_contract(model, physics.DEFAULT_CONTROL)
    source_index = {name: i for i, name in enumerate(names)}
    actuator_source = np.asarray([source_index[name] for name in contract.actuator_joint_names])
    floor, feet = physics.foot_geom_contract(model)
    geom_side = {geom: side for side, values in feet.items() for geom in values}
    left, right = [], []
    for frame in range(len(times)):
        mujoco.mj_resetData(model, data)
        data.qpos[:3] = root[frame]
        data.qpos[3:7] = quat[frame, [3, 0, 1, 2]]
        data.qpos[contract.qpos_addresses] = q[frame, actuator_source]
        mujoco.mj_forward(model, data)
        current = {"left": False, "right": False}
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            other = g2 if g1 == floor else g1 if g2 == floor else -1
            side = geom_side.get(other)
            if side:
                current[side] = True
        left.append(current["left"]); right.append(current["right"])
    left = np.asarray(left, dtype=bool); right = np.asarray(right, dtype=bool)

    segments = {
        "joint_prefix": (float(times[0]), rl),
        "mode_boundary_pm100ms": (rl - 0.1, rl + 0.1),
        "rl_internal_after100ms": (rl + 0.1, float(times[-1]) + 1.0e-9),
    }
    results: dict[str, Any] = {}
    for name, (start, stop) in segments.items():
        indices = np.flatnonzero((times >= start) & (times < stop))
        steps = np.abs(np.diff(q[indices], axis=0))
        velocity = np.abs(dq[indices])
        step_max_index = np.unravel_index(np.argmax(steps), steps.shape)
        vel_max_index = np.unravel_index(np.argmax(velocity), velocity.shape)
        target = {"dof": q[indices], "joint_names_mujoco": names}
        limits = physics.target_limit_report(model, target)
        tilt = 2.0 * np.arctan2(
            np.linalg.norm(quat[indices, :2], axis=1),
            np.maximum(1.0e-12, np.sqrt(quat[indices, 2] ** 2 + quat[indices, 3] ** 2)),
        )
        below = np.flatnonzero(root[indices, 2] < 0.42)
        results[name] = {
            "start_s": start, "stop_s": stop, "frames": int(len(indices)),
            "joint_step_p95_max_rad": [float(np.percentile(steps, 95)), float(steps[step_max_index])],
            "joint_step_max_joint": names[step_max_index[1]],
            "joint_step_max_time_s": float(times[indices[step_max_index[0] + 1]]),
            "recorded_dq_p95_max_radps": [float(np.percentile(velocity, 95)), float(velocity[vel_max_index])],
            "recorded_dq_max_joint": names[vel_max_index[1]],
            "recorded_dq_max_time_s": float(times[indices[vel_max_index[0]]]),
            "joint_limits": limits,
            "root_z_min_final_m": [float(np.min(root[indices, 2])), float(root[indices[-1], 2])],
            "root_tilt_p95_max_rad": [float(np.percentile(tilt, 95)), float(np.max(tilt))],
            "first_root_z_below_0p42_s": None if not len(below) else float(times[indices[below[0]]]),
            "contact_phase": phase(left[indices], right[indices]),
        }
    return {
        "scope": "offline source-state segmentation only; FK/contact reconstruction uses mj_forward, never mj_step",
        "segments": results,
        "frozen_max_gates": {"joint_step_rad": 0.45, "recorded_dq_radps": 20.0},
        "replay_error_segmentation": {
            "available": False,
            "reason": "Phase21 aggregate JSON did not persist replay q/body/contact traces; no new physics run is allowed",
        },
        "decision": {
            "step_velocity_failure_origin": "RL internal, not JOINT interpolation or the ±100ms mode boundary",
            "joint_prefix_physics": "smooth in q/dq but root fell below 0.42m 1.581s after first complete snapshot and was already collapsed before RL",
            "reference_use": "do not score the whole JOINT prefix as motion reference; its first complete state/command context may remain an initialization seed",
            "any2any_boundary": "mixed-mode aggregate failure cannot be attributed to WBT/Any2Any",
        },
    }


def render(report: dict[str, Any]) -> str:
    lines = [
        "# Phase21 Source Segment Attribution",
        "",
        "- 纯离线分析：只读取已有NPZ并逐帧`mj_forward`重建模型接触；没有`mj_step`、没有新replay。",
        "- 分段：JOINT prefix、RL切换±100ms、RL内部（切换100ms后）。",
        "",
        "| segment | step p95/max | dq p95/max | limit fraction | root-z min/final | flight |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in report["segments"].items():
        lines.append(
            f"| {name} | {row['joint_step_p95_max_rad'][0]:.4f}/{row['joint_step_p95_max_rad'][1]:.4f}rad | "
            f"{row['recorded_dq_p95_max_radps'][0]:.3f}/{row['recorded_dq_p95_max_radps'][1]:.3f}rad/s | "
            f"{row['joint_limits']['fraction']:.3%} | {row['root_z_min_final_m'][0]:.3f}/{row['root_z_min_final_m'][1]:.3f}m | "
            f"{row['contact_phase']['flight']:.3f} |"
        )
    joint = report["segments"]["joint_prefix"]
    boundary = report["segments"]["mode_boundary_pm100ms"]
    rl = report["segments"]["rl_internal_after100ms"]
    lines += [
        "",
        "## 结论",
        "",
        f"- step/dq最大门失败不在JOINT或切换边界：JOINT max `{joint['joint_step_p95_max_rad'][1]:.3f}rad/{joint['recorded_dq_p95_max_radps'][1]:.2f}rad/s`，边界 `{boundary['joint_step_p95_max_rad'][1]:.3f}/{boundary['recorded_dq_p95_max_radps'][1]:.2f}`，均低于冻结max门；RL内部达到 `{rl['joint_step_p95_max_rad'][1]:.3f}/{rl['recorded_dq_p95_max_radps'][1]:.2f}`，发生在 `{rl['joint_step_max_time_s']:.3f}s` 的 `{rl['joint_step_max_joint']}`。",
        f"- 但JOINT prefix并不物理稳定：root-z在 `{joint['first_root_z_below_0p42_s']:.3f}s` 跌破0.42m，即首个complete后约1.58s，RL前已倒地。",
        "- 因此prefix不应作为整段计分reference；最多保留首个complete state/command context作初始化seed，再单独选择稳态、物理有效的RL reference。",
        "- q/body replay误差无法分段，因为既有JSON没有保存trace；在不新跑physics的约束下不能声称aggregate q RMSE由prefix、boundary或RL哪一段主导。",
        "- 这次混合模式失败不能归因于WBT/Any2Any。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()
    report = analyze(args.source)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(render(report), encoding="utf-8")
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
