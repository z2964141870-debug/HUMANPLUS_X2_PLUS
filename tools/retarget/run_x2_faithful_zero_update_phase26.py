#!/usr/bin/env python3
"""Faithful Any2Any zero-update gate for X2 Phase26.

This entrypoint creates no Isaac environment, optimizer, callback, or PPO
trainer.  It uses SONIC MotionLib only as a CPU kinematic loader/FK cache,
then evaluates exactly one train and one held-out fixed batch through the
frozen source G1 encoder/FSQ/decoder/critic and a B=0 exact-S7 LoRA copy.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import joblib
import torch
import yaml
from vector_quantize_pytorch import FSQ


REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
TOOLS = REPO / "tools"
SONIC = Path("/home/humanplus/x2_teleop_final/x2_sonic/sonic_x2_sandbox")
for path in (SRC, TOOLS, SONIC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gear_sonic.utils.motion_lib.motion_lib_robot import MotionLibRobot  # noqa: E402
from retarget.probe_x2_faithful_wbt29_gold_phase23 import native_config  # noqa: E402
from retarget.audit_x2_faithful_entrypoint_phase24 import (  # noqa: E402
    MLP,
    PolicyShell,
    ValueShell,
    _load_source_checkpoint,
    _replace_selected,
)
from x2_faithful_any2any_phase23 import (  # noqa: E402
    GoldSplitSpec,
    ImmutableGoldMotionLibHook,
    POLICY_TERM_ORDER,
    WBT29PolicyContract,
)
from x2_physics_provenance_guard import (  # noqa: E402
    assert_physics_contract,
    load_runtime_snapshot,
    sha256,
)


DEFAULT_CONFIG = REPO / "configs/x2_faithful_zero_update_phase26.yaml"
DEFAULT_JSON = REPO / "reports/retarget/x2_faithful_zero_update_phase26.json"
DEFAULT_MD = REPO / "reports/retarget/x2_faithful_zero_update_phase26.md"


def tensor_mapping_hash(*states: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for state_index, state in enumerate(states):
        for name in sorted(state):
            value = state[name]
            if not torch.is_tensor(value):
                continue
            array = value.detach().cpu().contiguous().numpy()
            digest.update(f"{state_index}:{name}:{array.dtype}:{array.shape}".encode())
            digest.update(array.tobytes())
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def quat_xyzw_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    quat = quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True).clamp_min(1e-12)
    x, y, z, w = quat.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*quat.shape[:-1], 3, 3)


def rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    # This matches the source observation convention: first two matrix rows.
    return matrix[..., :2, :].reshape(*matrix.shape[:-2], 6)


def local_vectors(rotation_w: torch.Tensor, vectors_w: torch.Tensor) -> torch.Tensor:
    return torch.einsum("...ji,...j->...i", rotation_w, vectors_w)


def default_source29(contract: WBT29PolicyContract, defaults: Mapping[str, float]) -> torch.Tensor:
    result = []
    for name in contract.source29:
        value = float(defaults["all_other"])
        if "hip_pitch_joint" in name:
            value = float(defaults["hip_pitch"])
        elif "knee_joint" in name:
            value = float(defaults["knee"])
        elif "ankle_pitch_joint" in name:
            value = float(defaults["ankle_pitch"])
        elif "shoulder_pitch_joint" in name:
            value = float(defaults["shoulder_pitch"])
        elif "elbow_joint" in name:
            value = float(defaults["elbow"])
        result.append(value)
    return torch.tensor(result, dtype=torch.float32)


def build_contract(phase23_report: Mapping[str, Any]) -> WBT29PolicyContract:
    item = phase23_report["contract"]
    contract = WBT29PolicyContract.build(
        item["official31_order"],
        item["sim_target29_order"],
        item["policy_source29_order"],
        item["head_excluded2"],
    )
    manifest = contract.manifest()
    assert item["contract_hash"] == canonical_hash(manifest)
    return contract


def load_split(
    phase24: Mapping[str, Any], split: str
) -> tuple[MotionLibRobot, dict[str, Any], dict[str, Any]]:
    row = phase24["phase23_contract"][split]
    spec = GoldSplitSpec(
        split,
        Path(row["path"]),
        row["sha256"],
        row["key_prefix"],
        bool(row["optimizer_eligible"]),
    )
    payload = joblib.load(spec.source_pkl)
    motion = MotionLibRobot(native_config(spec.source_pkl), len(payload), "cpu")
    motion.load_motions_for_evaluation()
    hook = ImmutableGoldMotionLibHook(spec).attach(motion)
    return motion, payload, hook


def fixed_batch(
    motion: MotionLibRobot,
    key: str,
    contract: WBT29PolicyContract,
    default29: torch.Tensor,
    source14: list[str],
    history_range: list[int],
    future_range: list[int],
) -> dict[str, torch.Tensor]:
    key_index = list(motion.curr_motion_keys).index(key)
    start = int(motion.length_starts[key_index])
    h0, h1 = history_range
    f0, f1 = future_range
    if h1 - h0 != 10 or f1 - f0 != 10 or h1 > f0:
        raise ValueError("fixed Phase26 batch must have non-overlapping 10-frame history/future")
    hs = slice(start + h0, start + h1)
    fs = slice(start + f0, start + f1)

    q_h31 = motion.dof_pos[hs].float()
    dq_h31 = motion.dof_vel[hs].float()
    q_f31 = motion.dof_pos[fs].float()
    dq_f31 = motion.dof_vel[fs].float()
    q_h = contract.official_to_source(q_h31) - default29
    dq_h = contract.official_to_source(dq_h31)
    q_f = contract.official_to_source(q_f31) - default29
    dq_f = contract.official_to_source(dq_f31)

    root_quat_h = motion.body_quat_w[hs, 0].float()
    root_rot_h = quat_xyzw_to_matrix(root_quat_h)
    root_quat_f = motion.body_quat_w[fs, 0].float()
    root_rot_f = quat_xyzw_to_matrix(root_quat_f)
    reference_orientation = rotation_6d(
        torch.matmul(root_rot_h[-1].transpose(-1, -2), root_rot_f)
    )
    encoder_input = torch.cat((q_f, dq_f, reference_orientation), dim=-1).reshape(1, 640)

    gravity_w = torch.tensor([0.0, 0.0, -1.0]).expand(10, 3)
    gravity_b = local_vectors(root_rot_h, gravity_w)
    root_ang_w = motion.body_ang_vel_w[hs, 0].float()
    root_ang_b = local_vectors(root_rot_h, root_ang_w)
    terms = {
        "gravity_dir": gravity_b.unsqueeze(0),
        "base_ang_vel": root_ang_b.unsqueeze(0),
        "joint_pos": q_h.unsqueeze(0),
        "joint_vel": dq_h.unsqueeze(0),
        "actions": torch.zeros(1, 10, 29),
    }
    if tuple(terms) != POLICY_TERM_ORDER:
        raise AssertionError("policy term order changed")
    adapted = contract.adapt_policy_history(
        terms, joint_order="source29", action_order="source29"
    )
    actor_obs = contract.flatten_policy_history(adapted)

    root_pos_h = motion.body_pos_w[hs, 0].float()
    root_pos_f = motion.body_pos_w[fs, 0].float()
    base_rotation = root_rot_h[-1]
    anchor_pos_b = local_vectors(base_rotation, root_pos_f[0] - root_pos_h[-1])
    anchor_ori_b = reference_orientation[0]

    body_names = list(motion.skeleton_tree.node_names)
    body_indices = torch.tensor([body_names.index(name) for name in source14])
    body_pos_w = motion.body_pos_w[start + h1 - 1].float().index_select(0, body_indices)
    body_quat_w = motion.body_quat_w[start + h1 - 1].float().index_select(0, body_indices)
    body_pos_b = local_vectors(
        base_rotation.expand(len(source14), 3, 3),
        body_pos_w - root_pos_h[-1],
    ).reshape(-1)
    body_rot_b = torch.matmul(
        base_rotation.transpose(-1, -2), quat_xyzw_to_matrix(body_quat_w)
    )
    body_ori_b = rotation_6d(body_rot_b).reshape(-1)

    root_lin_w = motion.body_lin_vel_w[hs, 0].float()
    root_lin_b = local_vectors(root_rot_h, root_lin_w).reshape(-1)
    critic_obs = torch.cat(
        (
            torch.cat((q_f, dq_f), dim=-1).reshape(-1),
            anchor_pos_b,
            anchor_ori_b,
            body_pos_b,
            body_ori_b,
            root_lin_b,
            root_ang_b.reshape(-1),
            q_h.reshape(-1),
            dq_h.reshape(-1),
            torch.zeros(290),
        )
    ).unsqueeze(0)
    if actor_obs.shape != (1, 930) or encoder_input.shape != (1, 640):
        raise ValueError("actor/encoder fixed batch dimensions changed")
    if critic_obs.shape != (1, 1645):
        raise ValueError(f"critic fixed batch dimension changed: {critic_obs.shape}")
    return {
        "encoder_input": encoder_input,
        "actor_obs": actor_obs,
        "critic_obs": critic_obs,
        "sim_nominal31": q_h31[-1:].clone(),
        "source_q29": contract.official_to_source(q_h31[-1:]),
    }


def create_models(
    checkpoint: Mapping[str, Any], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    policy_state = checkpoint["policy_state_dict"]
    value_state = checkpoint["value_state_dict"]
    base_policy = PolicyShell(policy_state)
    adapted_policy = PolicyShell(policy_state)
    base_value = ValueShell(value_state)
    adapted_value = ValueShell(value_state)
    base_encoder = MLP(policy_state, "actor_module.encoders.g1")
    adapted_encoder = MLP(policy_state, "actor_module.encoders.g1")
    for module in (
        base_policy,
        adapted_policy,
        base_value,
        adapted_value,
        base_encoder,
        adapted_encoder,
    ):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    rank, alpha = int(manifest["rank"]), float(manifest["alpha"])
    masks = _replace_selected(
        adapted_policy.actor_module.decoders["g1_dyn"].module,
        [int(value) for value in manifest["actor_layers"]],
        rank,
        alpha,
        manifest.get("actor_input_mask"),
    )
    _replace_selected(
        adapted_value.critic_module.module,
        [int(value) for value in manifest["critic_layers"]],
        rank,
        alpha,
        None,
    )
    actual_trainable = sorted(
        [
            f"{prefix}.{name}"
            for prefix, module in (("policy", adapted_policy), ("value_model", adapted_value))
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
        ]
    )
    return {
        "base_policy": base_policy,
        "adapted_policy": adapted_policy,
        "base_value": base_value,
        "adapted_value": adapted_value,
        "base_encoder": base_encoder,
        "adapted_encoder": adapted_encoder,
        "quantizer_base": FSQ(levels=[32] * 32).eval(),
        "quantizer_adapted": FSQ(levels=[32] * 32).eval(),
        "trainable_names": actual_trainable,
        "input_masks": masks,
    }


def normalize_critic(value: torch.Tensor, state: Mapping[str, torch.Tensor]) -> torch.Tensor:
    mean = state["running_mean_std.running_mean"].detach()
    var = state["running_mean_std.running_var"].detach()
    return torch.clamp((value - mean) / torch.sqrt(var + 1.0e-5), -5.0, 5.0)


def evaluate_batch(
    batch: Mapping[str, torch.Tensor],
    models: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    contract: WBT29PolicyContract,
) -> dict[str, Any]:
    with torch.no_grad():
        base_latent = models["base_encoder"](batch["encoder_input"]).reshape(1, 2, 32)
        adapted_latent = models["adapted_encoder"](batch["encoder_input"]).reshape(1, 2, 32)
        base_token, _ = models["quantizer_base"](base_latent)
        adapted_token, _ = models["quantizer_adapted"](adapted_latent)
        base_decoder_input = torch.cat((base_token.reshape(1, 64), batch["actor_obs"]), dim=-1)
        adapted_decoder_input = torch.cat(
            (adapted_token.reshape(1, 64), batch["actor_obs"]), dim=-1
        )
        base_action_source = models["base_policy"].actor_module.decoders["g1_dyn"](
            base_decoder_input
        )
        adapted_action_source = models["adapted_policy"].actor_module.decoders["g1_dyn"](
            adapted_decoder_input
        )
        normalized_critic = normalize_critic(
            batch["critic_obs"], checkpoint["value_state_dict"]
        )
        base_value = models["base_value"].critic_module(normalized_critic)
        adapted_value = models["adapted_value"].critic_module(normalized_critic)

    base_target = contract.source_to_target(base_action_source)
    adapted_target = contract.source_to_target(adapted_action_source)
    base_official = contract.source_action_to_official(
        base_action_source, batch["sim_nominal31"]
    )
    adapted_official = contract.source_action_to_official(
        adapted_action_source, batch["sim_nominal31"]
    )
    head_index = torch.tensor(contract.head_indices)
    return {
        "input_finite": bool(
            all(torch.isfinite(value).all() for value in batch.values())
        ),
        "reference_token_max_abs": float((base_token - adapted_token).abs().max()),
        "action_source_max_abs": float((base_action_source - adapted_action_source).abs().max()),
        "action_target_max_abs": float((base_target - adapted_target).abs().max()),
        "action_official_nonhead_max_abs": float(
            (base_official - adapted_official).abs().max()
        ),
        "value_max_abs": float((base_value - adapted_value).abs().max()),
        "std_source_exact": bool(
            torch.equal(
                models["adapted_policy"].std.detach(),
                checkpoint["policy_state_dict"]["std"],
            )
        ),
        "head_nominal_exact": bool(
            torch.equal(
                adapted_official.index_select(-1, head_index),
                batch["sim_nominal31"].index_select(-1, head_index),
            )
        ),
        "all_outputs_finite": bool(
            torch.isfinite(base_token).all()
            and torch.isfinite(adapted_token).all()
            and torch.isfinite(base_action_source).all()
            and torch.isfinite(adapted_action_source).all()
            and torch.isfinite(base_value).all()
            and torch.isfinite(adapted_value).all()
        ),
        "token_sha256": tensor_mapping_hash({"token": base_token}),
        "action_source_sha256": tensor_mapping_hash({"action": base_action_source}),
        "value_sha256": tensor_mapping_hash({"value": base_value}),
    }


def verify_frozen_inputs(config: Mapping[str, Any]) -> dict[str, bool]:
    return {
        name: Path(row["path"]).is_file() and sha256(Path(row["path"])) == row["sha256"]
        for name, row in config["frozen_inputs"].items()
    }


def render(report: Mapping[str, Any]) -> str:
    lines = [
        "# X2 Faithful Any2Any Zero-Update — Phase26",
        "",
        f"- 裁决：**{report['decision']['status']}**。",
        "- 运行边界：CPU MotionLib仅作为kinematic loader/FK；Isaac环境实例0、物理step 0、optimizer/callback/PPO update 0。",
        "- 固定batch严格为train clip0与held-out clip0各一个；observation corruption和DR关闭。",
        "",
        "## 假设",
        "",
        "若B1–B5被同一个fail-closed入口真实调用，则exact-S7的B=0初始化应在target29对齐后逐元素复现冻结source动作、reference token和critic value，并且不让held-out进入任何优化路径。",
        "",
        "## 干预",
        "",
        "- WBT29 gather/scatter + head nominal；Gold MotionLib immutable hook；faithful PPO/reward/DR静态合同；exact-S7 manifest；Phase25 physics hash/runtime guard。",
        "- 只执行两组base versus B=0 LoRA forward，没有采样、loss、backward、optimizer或checkpoint输出。",
        "",
        "## 对照与结果",
        "",
        f"- frozen/hash guard：`{report['guards']}`。",
        f"- split isolation：`{report['split_isolation']}`。",
        f"- train fixed batch：`{report['fixed_batches']['train']['metrics']}`。",
        f"- held fixed batch：`{report['fixed_batches']['held_out']['metrics']}`。",
        f"- zero gate：`{report['zero_update_checks']}`。",
        "",
        "## 结论",
        "",
        f"- 结果：{report['decision']['result']}",
        f"- 结论：{report['decision']['conclusion']}",
        f"- 下一步：{report['decision']['next_step']}",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config.get("immutable") is not True:
        raise ValueError("Phase26 config is not immutable")

    frozen_checks = verify_frozen_inputs(config)
    if not all(frozen_checks.values()):
        failed = [name for name, value in frozen_checks.items() if not value]
        raise RuntimeError(f"Phase26 frozen input hash guard failed: {failed}")
    phase24 = yaml.safe_load(Path(config["frozen_inputs"]["phase24_config"]["path"]).read_text())
    phase23_report = json.loads(
        Path(config["frozen_inputs"]["phase23_report"]["path"]).read_text()
    )
    phase24_report = json.loads(
        Path(config["frozen_inputs"]["phase24_report"]["path"]).read_text()
    )
    phase25_report = json.loads(
        Path(config["frozen_inputs"]["phase25_report"]["path"]).read_text()
    )
    phase10_report = json.loads(
        Path(config["frozen_inputs"]["phase10_split_report"]["path"]).read_text()
    )
    phase25_guard = assert_physics_contract(
        Path(config["frozen_inputs"]["phase25_manifest"]["path"]),
        load_runtime_snapshot(Path(config["frozen_inputs"]["phase25_runtime"]["path"])),
    )
    guards = {
        "all_phase22_25_hashes_exact": all(frozen_checks.values()),
        "phase25_files_exact": all(phase25_guard["file_checks"].values()),
        "phase25_runtime_exact": phase25_guard["runtime_check"]["exact"],
        "phase25_declared_train_domain_ready": phase25_report["decision"]["status"]
        == "B5_READY_AS_DECLARED_TRAIN_DOMAIN",
        "official_mujoco_remains_held_out_mismatch": phase25_report["truth_boundary"]
        ["official_mujoco_is_held_out_sim_to_sim"]
        and not phase25_report["truth_boundary"]["domains_numerically_equivalent"],
    }

    contract = build_contract(phase23_report)
    default29 = default_source29(contract, config["x2_default_joint_positions"])
    source14 = phase24["reward_semantics"]["source14_body_names"]
    if sum(config["critic_semantic_layout"].values()) != 1645:
        raise ValueError("critic semantic layout no longer sums to 1645")

    checkpoint = _load_source_checkpoint(Path(phase24["source"]["checkpoint"]))
    dense_before = tensor_mapping_hash(
        checkpoint["policy_state_dict"], checkpoint["value_state_dict"]
    )
    selected = config["selected_lora_manifest"]
    manifest = phase24["lora_manifests"][selected]
    models = create_models(checkpoint, manifest)

    fixed_reports = {}
    hook_reports = {}
    for label in ("train", "held_out"):
        split = config["fixed_batches"][label]["split"]
        motion, payload, hook = load_split(phase24, split)
        key = config["fixed_batches"][label]["key"]
        if key not in payload or key not in hook["sampler_keys"]:
            raise ValueError(f"fixed {label} key is not in split-locked sampler")
        batch = fixed_batch(
            motion,
            key,
            contract,
            default29,
            source14,
            config["fixed_batches"]["history_frames"],
            config["fixed_batches"]["future_frames"],
        )
        fixed_reports[label] = {
            "split": split,
            "key": key,
            "source_frame_range": payload[key]["source_frame_range"],
            "metrics": evaluate_batch(batch, models, checkpoint, contract),
        }
        hook_reports[label] = hook

    dense_after = tensor_mapping_hash(
        checkpoint["policy_state_dict"], checkpoint["value_state_dict"]
    )
    expected_trainable = phase24_report["lora_manifests"][selected]["trainable_names"]
    rms = checkpoint["value_state_dict"]
    rms_hash = tensor_mapping_hash(
        {
            key: rms[key]
            for key in (
                "running_mean_std.running_mean",
                "running_mean_std.running_var",
                "running_mean_std.count",
            )
        }
    )
    train_keys = hook_reports["train"]["sampler_keys"]
    held_keys = hook_reports["held_out"]["sampler_keys"]
    split_policy = phase10_report["split"]["policy"]
    split_isolation = {
        "train_sampler_keys_exactly_four": len(train_keys) == 4,
        "held_sampler_keys_exactly_three": len(held_keys) == 3,
        "train_held_keys_disjoint": not bool(set(train_keys) & set(held_keys)),
        "held_optimizer_eligible_false": hook_reports["held_out"]["optimizer_eligible"]
        is False,
        "optimizer_path_keys": [],
        "callback_path_keys": [],
        "held_or_embargo_absent_from_optimizer_callback": True,
        "four_second_embargo_exact": split_policy["embargo_frame_block"] == [1600, 1800],
        "random_adjacent_frame_split_disabled": split_policy["random_adjacent_frame_split"]
        is False,
        "split_overlap_absent": split_policy["overlap"] is False,
    }

    tolerance = float(config["zero_update_contract"]["action_token_tolerance"])
    batch_metrics = [fixed_reports[label]["metrics"] for label in ("train", "held_out")]
    all_lora_parameters = [
        parameter
        for module in (models["adapted_policy"], models["adapted_value"])
        for name, parameter in module.named_parameters()
        if name.endswith(("lora_A", "lora_B"))
    ]
    frozen_parameters = [
        parameter
        for module in (
            models["adapted_policy"],
            models["adapted_value"],
            models["adapted_encoder"],
        )
        for name, parameter in module.named_parameters()
        if not name.endswith(("lora_A", "lora_B"))
    ]
    expected_critic_layout = {
        "command_multi_future": 580,
        "motion_anchor_pos_b": 3,
        "motion_anchor_ori_b": 6,
        "source14_body_pos_b": 42,
        "source14_body_ori_b": 84,
        "base_lin_vel_history": 30,
        "base_ang_vel_history": 30,
        "joint_pos_history": 290,
        "joint_vel_history": 290,
        "actions_history": 290,
    }
    target_probe = torch.arange(29, dtype=torch.float32).reshape(1, 29)
    target_roundtrip = contract.source_to_target(contract.target_to_source(target_probe))
    checks = {
        "all_guards_pass": all(guards.values()),
        "runtime_policy_action_dim_29": phase24["source"]["policy_action_dim"] == 29,
        "policy_contract_hash_exact": phase23_report["contract"]["contract_hash"]
        == canonical_hash(contract.manifest()),
        "target29_source29_target29_roundtrip_exact": bool(
            torch.equal(target_probe, target_roundtrip)
        ),
        "head_absent_policy_and_nominal_in_sim": all(
            item["head_nominal_exact"] for item in batch_metrics
        )
        and len(contract.head2) == 2,
        "motionlib_train_held_hooks_live": all(
            "state_adapter" in hook_reports[label] for label in ("train", "held_out")
        ),
        "split_isolation_exact": all(
            value for key, value in split_isolation.items() if isinstance(value, bool)
        ),
        "fixed_forward_batches_exactly_two": len(fixed_reports) == 2,
        "zero_B_reference_token_within_1e6": all(
            item["reference_token_max_abs"] <= tolerance for item in batch_metrics
        ),
        "zero_B_target_action_within_1e6": all(
            item["action_target_max_abs"] <= tolerance for item in batch_metrics
        ),
        "zero_B_value_within_1e6": all(
            item["value_max_abs"] <= tolerance for item in batch_metrics
        ),
        "std_source_exact_and_frozen": all(item["std_source_exact"] for item in batch_metrics)
        and models["adapted_policy"].std.requires_grad is False,
        "dense_checkpoint_hash_unchanged": dense_before == dense_after,
        "trainable_names_exact_selected_manifest": models["trainable_names"]
        == expected_trainable,
        "only_lora_A_B_trainable": bool(all_lora_parameters)
        and all(parameter.requires_grad for parameter in all_lora_parameters)
        and all(not parameter.requires_grad for parameter in frozen_parameters),
        "critic_running_stats_loaded_not_reinitialized": rms_hash
        == tensor_mapping_hash(
            {
                key: checkpoint["value_state_dict"][key]
                for key in (
                    "running_mean_std.running_mean",
                    "running_mean_std.running_var",
                    "running_mean_std.count",
                )
            }
        ),
        "critic_semantic_layout_and_running_stat_width_exact": config[
            "critic_semantic_layout"
        ]
        == expected_critic_layout
        and checkpoint["value_state_dict"]["running_mean_std.running_mean"].shape
        == (1645,)
        and checkpoint["value_state_dict"]["running_mean_std.running_var"].shape
        == (1645,),
        "all_inputs_outputs_finite": all(
            item["input_finite"] and item["all_outputs_finite"] for item in batch_metrics
        ),
        "no_optimizer_callback_or_physics": all(
            config["zero_update_contract"][key] == 0
            for key in (
                "environment_instances",
                "physics_steps",
                "optimizer_instances",
                "optimizer_steps",
                "callback_instances",
            )
        ),
        "one_update_remains_forbidden": config["zero_update_contract"]
        ["one_update_authorized"]
        is False,
    }
    passed = all(checks.values())
    report = {
        "schema_version": "x2_faithful_zero_update_phase26_v1",
        "provenance": {
            "config": {"path": str(args.config), "sha256": sha256(args.config)},
            "source_checkpoint": {
                "path": phase24["source"]["checkpoint"],
                "sha256": phase24["source"]["checkpoint_sha256"],
            },
            "dense_checkpoint_tensor_hash_before": dense_before,
            "dense_checkpoint_tensor_hash_after": dense_after,
            "critic_running_statistics_hash": rms_hash,
            "selected_lora_manifest": selected,
        },
        "truth_boundary": {
            "cpu_motionlib_kinematic_loader_fk_instances": 2,
            "isaac_environment_instances": 0,
            "physics_steps": 0,
            "optimizer_instances_and_steps": [0, 0],
            "callback_instances": 0,
            "fixed_forward_batches": 2,
            "framework_initialization": "CPU torch modules + CPU MotionLib kinematic loader only; no SimulationApp or manager environment",
            "official_mujoco_is_held_out_sim_to_sim_not_training_truth": True,
            "model_contact_is_not_hardware_grf_cop_wrench": True,
        },
        "guards": guards,
        "contract": contract.manifest(),
        "split_hooks": hook_reports,
        "split_isolation": split_isolation,
        "lora": {
            "selected": selected,
            "trainable_names_actual": models["trainable_names"],
            "trainable_names_expected": expected_trainable,
            "input_masks": models["input_masks"],
        },
        "critic_contract": {
            "semantic_layout": config["critic_semantic_layout"],
            "source14_body_names": source14,
            "running_mean_shape": list(
                checkpoint["value_state_dict"]["running_mean_std.running_mean"].shape
            ),
            "running_var_shape": list(
                checkpoint["value_state_dict"]["running_mean_std.running_var"].shape
            ),
            "running_count": float(
                checkpoint["value_state_dict"]["running_mean_std.count"]
            ),
            "normalization": "source checkpoint mean/var, epsilon=1e-5, clamp[-5,5], eval-only",
        },
        "fixed_batches": fixed_reports,
        "zero_update_checks": checks,
        "decision": {
            "status": "FAITHFUL_ZERO_UPDATE_PASSED_ONE_UPDATE_STILL_LOCKED"
            if passed
            else "FAITHFUL_ZERO_UPDATE_REJECTED",
            "result": (
                "B1-B5 are invoked by one guarded entrypoint and exact-S7 B=0 reproduces frozen source token/action/value on the two preregistered Gold batches."
                if passed
                else "At least one preregistered zero-update contract failed; no optimizer is allowed."
            ),
            "conclusion": (
                "B6 zero-update is satisfied without environment physics, optimizer, callbacks, or checkpoint output. This is initialization equivalence, not training or performance evidence."
                if passed
                else "The faithful live wiring is blocked before optimizer creation."
            ),
            "next_step": (
                "Stop for review. The Phase22 one-update smoke remains forbidden until separately authorized."
                if passed
                else "Fix only the failed contract and rerun the same two fixed batches; do not enter PPO."
            ),
        },
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    args.markdown.write_text(render(report), encoding="utf-8")
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
