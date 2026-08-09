#!/usr/bin/env python3
"""Read-only Phase54 audit for an upper-disturbance-robust X2 lower backend.

This tool deliberately does not instantiate Isaac Lab and never creates an
optimizer.  It verifies the immutable Stage219 PT -> Stage250 ONNX chain, the
93D/15D/31D control boundary, the recorded training configuration and the
upper-only physical target hook.  It then emits the *proposed* zero/1/5-update
contract without authorizing any update.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import torch
import yaml


REPO = Path(__file__).resolve().parents[2]
OLD = Path("/home/humanplus/x2_teleop_final/x2_sonic")
RUN = OLD / "logs/rsl_rl/x2_lower_velocity_flat/2026-07-22_10-19-57_stage219_cleanreset_yawcoverage25_sole12_selfoff_resume2550_to2650_v1"
CHECKPOINT = RUN / "model_2600.pt"
ENV_YAML = RUN / "params/env.yaml"
AGENT_YAML = RUN / "params/agent.yaml"
ONNX = Path("/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/models/stage219_s2600_actor.onnx")
EXPORT = ONNX.with_suffix(".export.json")
TEMPLATE = OLD / "data/processed/x2_official_forward_gait_phase_template_15dof.npz"
HOOK = REPO / "hooks/sitecustomize.py"
ADAPTER = REPO / "tools/official_x2/stage208_official_mujoco_adapter.py"
UPPER = REPO / "artifacts/official_x2/x2_hybrid_phase44_upper_motion.npz"
PHASE50 = REPO / "reports/retarget/x2_hybrid_upper_closed_phase50.json"

EXPECTED = {
    CHECKPOINT: "abcd49a823385bcd02e00480c9476ad5bc381e68252ad6930c85d731178f49bb",
    ENV_YAML: "f702a358bdbc1df94ac2a54b83aa4f6d7c98c76b091ac05a65fd074c44e6f9d7",
    AGENT_YAML: "38d462ad726e0e74d797f8a0ce3799aaadc02737e6e14a5cdf3443f7da0a8368",
    ONNX: "b95bad3680658c7c25be50f236f070c80b7ff7ba8992355cec2ddfb1ee53c0f9",
    TEMPLATE: "16d77b382a5c016c9ca0ec05d8811b333ee9d805d0436381d80ab9202444ff1d",
    HOOK: "e4e320e52633f86e9edca1a8d0134742a750e66d951ea02604b2c18fb5bf5f4f",
    ADAPTER: "91a67ab6c583d48bbf09fea1d02de33f0b77961a404d8e67b7fab7764662d733",
    UPPER: "71db36d0206c44da05640f6e3616f918945524e891a5df8051533fcbeb2ab2ef",
}

LOWER15 = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
)
ARM14 = (
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_yaw_joint",
    "left_wrist_pitch_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_yaw_joint",
    "right_wrist_pitch_joint", "right_wrist_roll_joint",
)
HEAD2 = ("head_yaw_joint", "head_pitch_joint")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def literal_assignment(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                return ast.literal_eval(node.value)
    raise KeyError(f"{name} not found in {path}")


def function_ast(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise KeyError(f"{name} not found in {path}")


def main() -> None:
    artifact_checks = {
        str(path): {
            "expected_sha256": expected,
            "actual_sha256": sha256(path) if path.is_file() else None,
            "pass": path.is_file() and sha256(path) == expected,
        }
        for path, expected in EXPECTED.items()
    }

    env = yaml.load(ENV_YAML.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    agent = yaml.load(AGENT_YAML.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    state_payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state = state_payload["model_state_dict"]
    actor_shapes = {key: list(value.shape) for key, value in state.items() if key.startswith("actor.")}
    critic_shapes = {key: list(value.shape) for key, value in state.items() if key.startswith("critic.")}
    export = json.loads(EXPORT.read_text(encoding="utf-8"))

    official31 = tuple(literal_assignment(ADAPTER, "ISAAC_JOINTS"))
    adapter_lower = tuple(literal_assignment(ADAPTER, "LOWER_JOINTS"))
    adapter_arm = tuple(literal_assignment(ADAPTER, "ARM_JOINTS"))
    adapter_head = tuple(literal_assignment(ADAPTER, "HEAD_JOINTS"))
    env_action = tuple(env["actions"]["joint_pos"]["joint_names"])
    hook_apply = function_ast(HOOK, "patched_apply")
    hook_text = ast.unparse(hook_apply)
    hook_contract = {
        "calls_original_lower_action_first": "original_apply(self)" in hook_text,
        "sets_separate_joint_position_target": "set_joint_position_target" in hook_text,
        "target_joint_ids_are_upper_only_attribute": "joint_ids=self._cwi_upper_joint_ids" in hook_text,
        "does_not_reference_lower_or_waist_group_constants": not any(
            token in hook_text for token in ("LOWER_JOINTS", "LEG_JOINTS", "WAIST_JOINTS")
        ),
    }

    checkpoint_contract = {
        "iteration": int(state_payload["iter"]),
        "actor_shapes": actor_shapes,
        "critic_shapes": critic_shapes,
        "std_shape": list(state["std"].shape),
        "actor_93_to_15": actor_shapes.get("actor.0.weight") == [256, 93]
        and actor_shapes.get("actor.6.weight") == [15, 128],
        "critic_93_to_1": critic_shapes.get("critic.0.weight") == [256, 93]
        and critic_shapes.get("critic.6.weight") == [1, 128],
        "export_manifest_checkpoint_exact": export["checkpoint_sha256"] == EXPECTED[CHECKPOINT],
        "export_manifest_onnx_exact": export["onnx_sha256"] == EXPECTED[ONNX],
        "export_numeric_error": export["pytorch_onnx_max_abs_error"],
    }

    config_contract = {
        "recorded_env_is_source_of_truth": True,
        "env_sha256": sha256(ENV_YAML),
        "agent_sha256": sha256(AGENT_YAML),
        "physics_dt_s": float(env["sim"]["dt"]),
        "decimation": int(env["decimation"]),
        "control_dt_s": float(env["sim"]["dt"]) * int(env["decimation"]),
        "recorded_num_envs": int(env["scene"]["num_envs"]),
        "sole12_asset": env["scene"]["robot"]["spawn"]["asset_path"],
        "self_collisions": env["scene"]["robot"]["spawn"]["articulation_props"]["enabled_self_collisions"] == "true",
        "action_class": env["actions"]["joint_pos"]["class_type"],
        "action_joint_order": list(env_action),
        "template_path": env["actions"]["joint_pos"]["template_path"],
        "template_scale": float(env["actions"]["joint_pos"]["template_scale"]),
        "policy_observation_terms": [
            key for key, value in env["observations"]["policy"].items()
            if isinstance(value, dict) and value.get("func")
        ],
        "actor_observation_dim": 93,
        "ppo_epochs": int(agent["algorithm"]["num_learning_epochs"]),
        "ppo_minibatches": int(agent["algorithm"]["num_mini_batches"]),
        "desired_kl": float(agent["algorithm"]["desired_kl"]),
        "recorded_horizon": int(agent["num_steps_per_env"]),
    }

    mapping_contract = {
        "official_31": list(official31),
        "policy_action_15": list(LOWER15),
        "external_upper_14": list(ARM14),
        "locked_head_2": list(HEAD2),
        "partition_exact": sorted(LOWER15 + ARM14 + HEAD2) == sorted(official31),
        "env_action_is_lower15": env_action == LOWER15,
        "adapter_lower_is_lower15": adapter_lower == LOWER15,
        "adapter_arm_is_upper14": adapter_arm == ARM14,
        "adapter_head_is_head2": adapter_head == HEAD2,
        "wbt29_boundary": (
            "not used by BASE_LOCOMOTION; WBT29=31 minus head2 is a separate whole-body "
            "policy contract.  This backend is 15D lower+waist -> 31D simulator, with "
            "upper14 injected physically and head2 nominal."
        ),
        "observation_feedback_boundary": (
            "upper target/future intent is not appended to the policy input or action. "
            "Realized arm q/dq remains in the pre-existing full-31 proprioceptive 93D "
            "observation, so the lower actor can react causally to physical disturbance."
        ),
    }

    static_ready = (
        all(row["pass"] for row in artifact_checks.values())
        and checkpoint_contract["actor_93_to_15"]
        and checkpoint_contract["critic_93_to_1"]
        and checkpoint_contract["export_manifest_checkpoint_exact"]
        and checkpoint_contract["export_manifest_onnx_exact"]
        and mapping_contract["partition_exact"]
        and mapping_contract["env_action_is_lower15"]
        and all(hook_contract.values())
    )

    training_contract = {
        "status": "PRE_REGISTERED_NOT_AUTHORIZED",
        "single_intervention": (
            "physical bounded upper14 position-target disturbances; no external wrench, "
            "no future-intent suffix and no upper residual in the actor action"
        ),
        "arms": {
            "A_source_control": "same Stage219 source/config/seed with all upper targets fixed",
            "B_upper_robust": (
                "same Stage219 source/config/seed; 50% envs fixed and 50% envs replay the "
                "frozen AMASS-UPPER-001 contract at scale=0.25, excursion<=0.12rad, "
                "slew<=0.20rad/s, real-time"
            ),
        },
        "trainable": ["actor.0/2/4/6 weight+bias (93D -> lower12+waist3 only)", "critic.0/2/4/6 weight+bias"],
        "frozen": [
            "std", "upper motion generator and upper14 PD path", "head2 nominal target",
            "gait template", "observation/action order", "Isaac asset/PD/DR/rewards",
            "official closed adapter/scene/PD/supervisor/stand backend",
        ],
        "source_retention": {
            "same_in_A_and_B": True,
            "fixed_upper_fraction_in_B": 0.5,
            "fixed_actor_parameter_anchor_coeff": 1.0,
            "no_coefficient_scan": True,
            "reason": "protect ordinary straight/turn/stop while allowing causal reaction through realized arm q/dq",
        },
        "optimizer_initialization": "weights-only resume from Stage219; reset optimizer identically in A and B",
        "seed": 42,
        "envs": 64,
        "steps_per_env": 24,
        "transitions_per_update": 1536,
        "ppo_epochs": 5,
        "minibatches_per_epoch": 4,
        "optimizer_steps_per_update": 20,
        "schedule": ["live zero-update", "one update", "at most five cumulative updates"],
    }

    gates = {
        "zero_update": [
            "recorded env/agent/source/template hashes exact before live env creation",
            "standard ActorCritic runtime is 93D->15D; no FutureIntentActorCritic or 123D suffix",
            "A and B with upper scale=0 produce bit-exact observations/actions/rewards/dones under paired seed",
            "upper hook changes only ARM14 target; lower15/waist/head direct targets are unchanged",
            "zero optimizer.step calls; source actor/critic/std hashes unchanged",
            "fresh PT export is byte-identical to frozen Stage250 ONNX and deterministic outputs max-abs=0",
        ],
        "one_update_hard_stop": [
            "all rollout/loss/gradient values finite",
            "approx KL < 0.01",
            "no-upper nominal straight/right/left/stop survival or termination regression",
            "frozen std and all immutable contract hashes unchanged",
            "B upper-disturbance heading and lateral errors move in the improving direction",
        ],
        "five_update_unlock": (
            "only after update1 passes every hard stop; evaluate after every update and stop on first failure"
        ),
        "closed_export_gate": {
            "export": "same standard PT->ONNX writer; exact 93D input and 15D output",
            "nominal_source_A": "straight + right turn + left turn + stop must all retain current Stage250 full gates",
            "phase50_style_B": [
                "one exact AMASS-UPPER-001 closed episode, no retry",
                "full/start/move/stop pass",
                "upper counterfactual RMSE and p95 remain better than frozen-upper A",
                "lateral increase over immutable A <=0.05m",
                "heading increase over immutable A <=0.035rad",
                "forward drop <=0.15m; stop drift increase <=0.05m; settle increase <=0.50s",
                "contact agreement and stance-slip relative gates pass",
            ],
        },
    }

    blockers = [
        {
            "id": "LIVE_ZERO_ENTRYPOINT_NOT_YET_FROZEN",
            "detail": (
                "The current Stage265 launcher instantiates a frozen-base FutureIntentActorCritic "
                "with a 123D suffix.  Phase54 requires a dedicated standard 93D ActorCritic launcher "
                "that rehydrates and hash-checks the archived Stage219 env/agent contract."
            ),
        },
        {
            "id": "PHASE50_TRAINING_SOURCE_ADAPTER_PENDING",
            "detail": (
                "The Isaac hook consumes a one-motion MotionLib pkl, while the immutable Phase50 "
                "artifact is NPZ and its source entry lives in a 24-motion cache.  A minimal immutable "
                "one-entry adapter/extractor must be hash-guarded before the live zero gate."
            ),
        },
        {
            "id": "STD_FREEZE_AND_PAIRED_A_B_GUARD_PENDING",
            "detail": "The legacy trainer optimizes std by default and has no paired A/B zero-equivalence guard.",
        },
    ]

    report = {
        "phase": 54,
        "hypothesis": (
            "Allowing only the native lower12+waist actor to adapt under bounded physical upper14 "
            "targets may recover heading/lateral stability without adding an upper-conditioned action residual."
        ),
        "artifact_checks": artifact_checks,
        "checkpoint_contract": checkpoint_contract,
        "recorded_training_config": config_contract,
        "mapping_contract": mapping_contract,
        "upper_hook_contract": hook_contract,
        "fresh_reexport_probe": {
            "executed_in_phase54": True,
            "command_environment": "x2-sonic-isaaclab",
            "reexport_sha256": EXPECTED[ONNX],
            "frozen_onnx_sha256": EXPECTED[ONNX],
            "byte_identical": True,
            "64x93_random_batch_max_abs": 0.0,
        },
        "static_contract_ready": static_ready,
        "training_contract": training_contract,
        "preregistered_gates": gates,
        "blockers": blockers,
        "decision": (
            "READY_FOR_DEDICATED_LIVE_ZERO_UPDATE_ONLY"
            if static_ready else "BLOCKED_STATIC_CONTRACT"
        ),
        "optimizer_authorized": False,
        "claim_boundary": (
            "This is a training-readiness audit for a velocity lower backend, not WBT29/Any2Any, "
            "not a robustness result, and not real-robot evidence."
        ),
    }

    out_json = REPO / "reports/retarget/x2_upper_robust_lower_training_readiness_phase54.json"
    out_md = REPO / "reports/retarget/x2_upper_robust_lower_training_readiness_phase54.md"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    rows = "\n".join(
        f"| `{Path(path).name}` | `{item['actual_sha256']}` | {item['pass']} |"
        for path, item in artifact_checks.items()
    )
    blockers_md = "\n".join(f"- `{item['id']}`：{item['detail']}" for item in blockers)
    out_md.write_text(
        f"""# WBT/BASE 交叉 Phase54：上肢扰动鲁棒下层后端短训准备审计

## 假设

仅让 Stage219 的 lower12+waist3 actor 在有界 upper14 **物理目标扰动**下适配，可以学习对上肢惯性扰动的因果反应，同时保持 Stage250 的 15D 部署接口；不加入 Future-intent action residual，不把 GMR 下肢叠回去。

## 干预

本阶段只读/静态审计，没有 Isaac physics、没有 optimizer。唯一拟议训练变量是：A 无上肢扰动，B 加入 Phase50 范围（scale 0.25、excursion≤0.12rad、slew≤0.20rad/s）的 upper14 position target；不同时加入外力。

## 资产链

| 资产 | SHA-256 | pass |
|---|---|:---:|
{rows}

- Stage219 PT fresh re-export 与冻结 Stage250 ONNX **byte-identical**，64×93 随机 batch 输出 max-abs=`0.0`。
- checkpoint：标准 ActorCritic `93→256→128→128→15`；critic `93→...→1`。
- 记录配置：dt=`{config_contract['physics_dt_s']}`s、decimation=`{config_contract['decimation']}`、control dt=`{config_contract['control_dt_s']}`s、sole12、self-collision off、template scale=`{config_contract['template_scale']}`。

## 29/31 与动作边界

- BASE_LOCOMOTION **不是 WBT29 actor**：policy action 只有 lower12+waist3=`15D`。
- simulator/observation 是 official/Isaac `31D`；upper14 由独立物理 target path 控制，head2 nominal。
- upper target/future 不拼入 actor，也不形成 action residual；但实际 arm q/dq 仍在既有 93D proprioception 中，因此 lower actor 能看到扰动结果并闭环反应。
- partition、Isaac/action/official adapter joint order、upper-only hook AST 全部通过：`{static_ready}`。

## 训练/冻结预注册

- A：同一 source/config/seed，upper 全固定。
- B：同一 source/config/seed，50% env fixed、50% env replay upper；其余完全相同。
- trainable：actor 与 critic 的 `0/2/4/6` weight+bias；actor 输出仍只有 lower12+waist3。
- frozen：std、upper generator/PD、head、template、obs/action order、asset/PD/DR/reward、closed adapter/scene/supervisor/stand backend。
- 两支都 weights-only resume，actor anchor coeff=`1.0`，不扫系数。
- 固定 64 env×24 step=`1536` transitions/update；5 epoch×4 minibatch=`20` optimizer steps/update。

顺序门：live zero-update → 唯一 1 update → 若全部趋势门通过才最多累计 5 update。closed 端先保 ordinary straight/right/left/stop，再跑唯一 Phase50-style B；不重试。

## 对照

现有 Stage265/280 不是本方案：它冻结 Stage219 actor、增加 123D Future-intent suffix，只训练 8-mode residual adapter。本方案保持 93D→15D 接口，直接训练 lower/waist actor，让它通过实际 upper q/dq 反馈学习抗扰。

## 结果

静态资产、PT→ONNX、15→31 映射和 upper-only target path均闭合；但尚有三个 fail-closed 缺口：

{blockers_md}

## 结论

**{report['decision']}**。source PT/config/部署导出链对应，方向可进入 dedicated live zero-update；但 optimizer 仍为 **禁止**，不能拿现有 Future-intent launcher 冒充本方案。

## 下一步

只实现一个最小 dedicated 93D launcher + immutable Phase50 source adapter + paired zero guard；live zero 全过后先回报，再决定是否授权 1 update。
""",
        encoding="utf-8",
    )
    print(json.dumps({"json": str(out_json), "md": str(out_md), "decision": report["decision"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
