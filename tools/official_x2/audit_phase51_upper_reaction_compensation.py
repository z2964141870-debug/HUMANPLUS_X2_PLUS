#!/usr/bin/env python3
"""Phase51 offline centroidal audit for training-free upper reaction compensation.

This tool never runs physics or changes the controller.  It computes the
centroidal momentum induced by the Phase50 *target* arm trajectory with the
official X2 model, then asks whether a conservation-derived high-level yaw or
lateral feedforward explains the observed B1-minus-A motion disturbance.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


REPO = Path(__file__).resolve().parents[2]
SCENE = Path("/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/worktree/x2_rl_deploy/x2_rl_deploy_mujoco/configuration/robot/lx2501_3_t2d5/model_info/scene.xml")
A_ROLLOUT = Path("/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/results/phase34_full_stage250_trace/phase34_stage250_closed_full_once_d230.json")
B1_ROLLOUT = Path("/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/results/phase50_hybrid_upper_closed/phase50_stage250_hybrid_upper_B_once_d229.json")
UPPER_ARTIFACT = REPO / "artifacts/official_x2/x2_hybrid_phase44_upper_motion.npz"
PHASE50_REPORT = REPO / "reports/retarget/x2_hybrid_upper_closed_phase50.json"
OUTPUT_JSON = REPO / "reports/retarget/x2_upper_reaction_compensation_phase51.json"
OUTPUT_MD = REPO / "reports/retarget/x2_upper_reaction_compensation_phase51.md"

ARM14 = (
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_yaw_joint", "left_wrist_pitch_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint",
)
DT = 0.02
MAX_LAG_STEPS = 15
MIN_DIRECTION_CORRELATION = 0.30
MIN_EXPLAINED_R2 = 0.10
MAX_YAW_COMMAND_RADPS = 0.10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def move_rows(path: Path) -> list[dict[str, Any]]:
    rows = [row for row in json.loads(path.read_text())["trace"] if row["stage"] == "move"]
    if len(rows) != 200:
        raise RuntimeError(f"{path}: expected 200 move rows, got {len(rows)}")
    return rows


def best_lag(signal: np.ndarray, response: np.ndarray) -> dict[str, float | int]:
    candidates = []
    for lag in range(MAX_LAG_STEPS + 1):
        x = signal if lag == 0 else signal[:-lag]
        y = response if lag == 0 else response[lag:]
        corr = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 1e-12 and np.std(y) > 1e-12 else 0.0
        candidates.append({"lag_steps": lag, "lag_s": lag * DT, "correlation": corr, "r2": corr * corr})
    return max(candidates, key=lambda row: abs(float(row["correlation"])))


def centroidal_reaction(model: mujoco.MjModel, q: np.ndarray) -> dict[str, np.ndarray]:
    data = mujoco.MjData(model)
    pelvis = int(model.body("pelvis").id)
    total_mass = float(model.body_subtreemass[pelvis])
    qd = np.gradient(q, DT, axis=0)
    linear_momentum, angular_momentum, izz = [], [], []
    for q_frame, qd_frame in zip(q, qd):
        data.qpos[:] = model.qpos0
        data.qvel[:] = 0.0
        for name, position, velocity in zip(ARM14, q_frame, qd_frame):
            data.qpos[int(model.joint(name).qposadr[0])] = float(position)
            data.qvel[int(model.joint(name).dofadr[0])] = float(velocity)
        mujoco.mj_forward(model, data)
        mujoco.mj_subtreeVel(model, data)
        linear_momentum.append(total_mass * data.subtree_linvel[pelvis].copy())
        angular_momentum.append(data.subtree_angmom[pelvis].copy())

        # Configuration-dependent whole-robot centroidal Izz: momentum caused
        # by unit free-root yaw velocity with all joints locked.
        data.qvel[:] = 0.0
        data.qvel[5] = 1.0
        mujoco.mj_forward(model, data)
        mujoco.mj_subtreeVel(model, data)
        izz.append(float(data.subtree_angmom[pelvis, 2]))
    linear_momentum = np.asarray(linear_momentum)
    angular_momentum = np.asarray(angular_momentum)
    izz = np.asarray(izz)
    return {
        "P": linear_momentum,
        "H": angular_momentum,
        "Izz": izz,
        "yaw_reaction_radps": -angular_momentum[:, 2] / izz,
        "lateral_reaction_mps": -linear_momentum[:, 1] / total_mass,
        "total_mass_kg": np.asarray(total_mass),
    }


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)), "p05": float(np.quantile(values, .05)),
        "median": float(np.median(values)), "p95": float(np.quantile(values, .95)),
        "max": float(np.max(values)), "rms": float(np.sqrt(np.mean(values ** 2))),
        "integral": float(np.trapz(values, dx=DT)),
    }


def render(report: dict[str, Any]) -> str:
    y = report["candidate_interfaces"]["yaw_command"]
    l = report["candidate_interfaces"]["lateral_command"]
    return f"""# WBT Phase51：training-free 上肢反作用补偿离线裁决

## 假设

Phase50 的上肢 target q/q̇ 会产生可由 official X2 MJCF 计算的 centroidal momentum；若它与 B1−A 的航向或横向扰动在符号、量纲和时序上对应，则可通过一个有界高层 command 前馈补偿，而不直接修改腿12 action。

## 干预

本阶段只有只读模型计算，没有 adapter 改动、physics、训练或 gain 扫描。上肢动量是**官方仿真模型估计**，不是实机力/动量真值。

计算合同：

```text
upper target q/qdot + official MJCF
→ whole-body centroidal P_upper, H_upper with root fixed
→ yaw reaction = -H_upper,z / Izz(q)
→ lateral reaction = -P_upper,y / M
```

## 离线结果

- yaw reaction：范围 [{y['signal']['min']:.5f}, {y['signal']['max']:.5f}] rad/s，RMS={y['signal']['rms']:.5f}，4s 积分={y['signal']['integral']:.5f} rad。
- lateral reaction：范围 [{l['signal']['min']:.6f}, {l['signal']['max']:.6f}] m/s，RMS={l['signal']['rms']:.6f}，4s 积分={l['signal']['integral']:.6f} m。
- 实际 B1−A yaw 位移差：{report['observed_disturbance']['yaw_displacement_B1_minus_A_rad']:.5f} rad；横向位移差：{report['observed_disturbance']['lateral_displacement_B1_minus_A_m']:.5f} m。
- yaw reaction→yaw-rate disturbance 最佳 lag={y['regression']['lag_s']:.2f}s，corr={y['regression']['correlation']:.3f}，R²={y['regression']['r2']:.3f}。
- lateral reaction→lateral-velocity disturbance 最佳 lag={l['regression']['lag_s']:.2f}s，corr={l['regression']['correlation']:.3f}，R²={l['regression']['r2']:.3f}。

## 唯一预注册 B2（未执行）

- 最小接口：`command_wz`，不是 waist/leg joint residual。
- `wz_ff(t)=clip(-H_upper,z/Izz, ±0.10 rad/s)`；actor command 为 `clip(wz_heading+wz_ff, ±0.10)`。
- 零上肢 target velocity 时 `wz_ff=0`，严格回退 Stage250。
- 不使用 Phase50 回归拟合 gain，不做 time shift，不扫 cap；B2 只允许一条 closed episode。

## 裁决

**{report['decision']['status']}**

{report['decision']['conclusion']}

## 下一步

{report['decision']['next_step']}
"""


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    a, b = move_rows(A_ROLLOUT), move_rows(B1_ROLLOUT)
    upper_names = tuple(np.load(UPPER_ARTIFACT)["joint_names"].tolist())
    if upper_names != ARM14:
        raise RuntimeError("Phase44 artifact upper order changed")
    q = np.asarray([row["upper_target_rad"] for row in b], dtype=np.float64)
    reaction = centroidal_reaction(model, q)
    yaw_signal = reaction["yaw_reaction_radps"]
    lateral_signal = reaction["lateral_reaction_mps"]
    yaw_response = np.asarray([row["root_yaw_rate_radps"] for row in b]) - np.asarray([row["root_yaw_rate_radps"] for row in a])
    lateral_response = np.asarray([row["root_vy_b_mps"] for row in b]) - np.asarray([row["root_vy_b_mps"] for row in a])
    yaw_regression = best_lag(yaw_signal, yaw_response)
    lateral_regression = best_lag(lateral_signal, lateral_response)
    yaw_displacement = ((b[-1]["root_yaw_rad"] - b[0]["root_yaw_rad"]) - (a[-1]["root_yaw_rad"] - a[0]["root_yaw_rad"]))
    lateral_displacement = ((b[-1]["root_y_m"] - b[0]["root_y_m"]) - (a[-1]["root_y_m"] - a[0]["root_y_m"]))
    yaw_response_integral = float(np.trapz(yaw_response, dx=DT))
    lateral_response_integral = float(np.trapz(lateral_response, dx=DT))

    yaw_dimension_pass = float(np.max(np.abs(yaw_signal))) <= MAX_YAW_COMMAND_RADPS + 1e-12
    yaw_direction_pass = yaw_regression["correlation"] >= MIN_DIRECTION_CORRELATION
    yaw_explanation_pass = yaw_regression["r2"] >= MIN_EXPLAINED_R2
    yaw_integral = float(np.trapz(yaw_signal, dx=DT))
    lateral_integral = float(np.trapz(lateral_signal, dx=DT))
    yaw_net_pass = np.sign(yaw_integral) == np.sign(yaw_response_integral) and abs(yaw_integral) >= 0.1 * abs(yaw_response_integral)
    lateral_direction_pass = lateral_regression["correlation"] >= MIN_DIRECTION_CORRELATION
    lateral_explanation_pass = lateral_regression["r2"] >= MIN_EXPLAINED_R2
    lateral_net_pass = np.sign(lateral_integral) == np.sign(lateral_response_integral) and abs(lateral_integral) >= 0.1 * abs(lateral_response_integral)

    yaw_proven = yaw_dimension_pass and yaw_direction_pass and yaw_explanation_pass and yaw_net_pass
    lateral_proven = lateral_direction_pass and lateral_explanation_pass and lateral_net_pass
    execute = bool(yaw_proven or lateral_proven)
    report = {
        "schema_version": "x2_upper_reaction_compensation_phase51_v1",
        "scope": "offline model calculation and preregistration only; no physics/training/controller modification",
        "provenance": {
            "scene": str(SCENE), "scene_sha256": sha256(SCENE),
            "A_phase34_rollout": str(A_ROLLOUT), "A_sha256": sha256(A_ROLLOUT),
            "B1_phase50_rollout": str(B1_ROLLOUT), "B1_sha256": sha256(B1_ROLLOUT),
            "upper_artifact": str(UPPER_ARTIFACT), "upper_artifact_sha256": sha256(UPPER_ARTIFACT),
            "phase50_report_sha256": sha256(PHASE50_REPORT), "mujoco_version": mujoco.__version__,
        },
        "model_contract": {
            "total_mass_kg": float(reaction["total_mass_kg"]),
            "Izz_kgm2": quantiles(reaction["Izz"]),
            "target_frames": len(q), "fps": 50.0,
            "centroidal_values_are_model_estimates_not_hardware_truth": True,
        },
        "candidate_interfaces": {
            "yaw_command": {
                "formula": "clip(-H_upper_z/Izz(q), +/-0.10rad/s)", "signal": quantiles(yaw_signal),
                "regression": yaw_regression,
                "gates": {"dimension_and_bound": yaw_dimension_pass, "direction_corr_ge_0p30": yaw_direction_pass, "r2_ge_0p10": yaw_explanation_pass, "net_sign_and_10pct_magnitude": bool(yaw_net_pass)},
                "proven_offline": yaw_proven,
            },
            "lateral_command": {
                "formula": "-P_upper_y/M", "signal": quantiles(lateral_signal),
                "regression": lateral_regression,
                "gates": {"direction_corr_ge_0p30": lateral_direction_pass, "r2_ge_0p10": lateral_explanation_pass, "net_sign_and_10pct_magnitude": bool(lateral_net_pass)},
                "proven_offline": lateral_proven,
            },
        },
        "observed_disturbance": {
            "yaw_rate_B1_minus_A": quantiles(yaw_response),
            "lateral_body_velocity_B1_minus_A": quantiles(lateral_response),
            "yaw_displacement_B1_minus_A_rad": float(yaw_displacement),
            "lateral_displacement_B1_minus_A_m": float(lateral_displacement),
            "yaw_rate_response_integral_rad": yaw_response_integral,
            "lateral_body_velocity_response_integral_m": lateral_response_integral,
        },
        "preregistered_B2_if_unlocked": {
            "control": "Phase50 B1 exact + conservation yaw feedforward into command_wz only",
            "formula": "command_wz=clip(existing_heading_wz + clip(-H_upper_z/Izz,+/-0.10),+/-0.10)",
            "gain": 1.0, "time_shift_s": 0.0, "parameter_scan": False,
            "zero_upper_exact_fallback": True, "direct_leg12_write": False,
            "episodes": 1, "retries": 0,
            "gates": "Phase50 B1 functional/upper/root/heading/lateral/stop/contact/slip gates; upper must not regress and heading/lateral must improve",
        },
        "decision": {
            "physics_execution_unlocked": execute,
            "status": "B2_PHYSICS_PREREG_UNLOCKED" if execute else "STOP_OFFLINE_SIGNAL_DOES_NOT_EXPLAIN_PHASE50_DISTURBANCE",
            "conclusion": (
                "至少一个守恒式前馈在符号、量纲、时序和净效应上通过离线证据，可按唯一B2合同运行。" if execute else
                "守恒式信号量纲有界，但不能解释Phase50的系统性漂移：yaw净积分与实际漂移异号且仅约1%，最佳相关也低；lateral信号同样未过方向/R²/净效应门。此时运行B2相当于盲试，按任务停止。"
            ),
            "next_step": (
                "仅运行一次预注册B2，不得调gain/lag。" if execute else
                "若继续，应先获得独立上肢动作或对称/反对称上肢panel来辨识扰动映射；不能用同一Phase50 episode拟合gain后再在同episode宣称验证。"
            ),
        },
    }
    OUTPUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    OUTPUT_MD.write_text(render(report), encoding="utf-8")
    print(json.dumps(report["decision"], ensure_ascii=False))


if __name__ == "__main__":
    main()
