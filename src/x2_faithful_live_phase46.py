"""Opt-in live WBT29 boundary and zero-update gate for Phase46.

This module is deliberately loaded only by the Phase46 launcher.  It keeps the
SONIC policy boundary in the original G1 29-joint semantic order while the X2
articulation remains 31 DoF.  The two head joints are never policy inputs or
outputs; because the action term controls only the listed 29 joints, Isaac's
head actuators retain their declared nominal-zero targets.

The zero trainer creates no optimizer and calls no environment step.  It is a
runtime contract probe, not PPO and not an evaluation of physical stability.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import joblib
import torch


SOURCE29 = (
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
)
HEAD2 = ("head_yaw_joint", "head_pitch_joint")
EXACT_S7_ACTOR = tuple(
    f"actor_module.decoders.g1_dyn.module.{index}" for index in (0, 2, 4, 6, 8, 10, 12)
)
EXACT_S7_CRITIC = tuple(
    f"critic_module.module.{index}" for index in (2, 4, 6, 8, 10)
)
# Frozen copy of the source-canonical triples used by the live SONIC X2
# observation module: (X2 default, G1 default, relative-range scale).  Keeping
# this copy in the isolated entrypoint makes CPU tests independent of Isaac.
SOURCE_CANONICAL = {
    "left_hip_pitch_joint": (-0.248, -0.312, 1.0286121673),
    "left_hip_roll_joint": (0.0, 0.0, 1.1113339701),
    "left_hip_yaw_joint": (0.0, 0.0, 1.0784513101),
    "left_knee_joint": (0.5303, 0.669, 1.2325289744),
    "left_ankle_pitch_joint": (-0.2823, -0.363, 1.1116799363),
    "left_ankle_roll_joint": (0.0, 0.0, 0.9992366412),
    "right_hip_pitch_joint": (-0.248, -0.312, 1.0286121673),
    "right_hip_roll_joint": (0.0, 0.0, 1.1113339701),
    "right_hip_yaw_joint": (0.0, 0.0, 1.0784513101),
    "right_knee_joint": (0.5303, 0.669, 1.2325289744),
    "right_ankle_pitch_joint": (-0.2823, -0.363, 1.1116799363),
    "right_ankle_roll_joint": (0.0, 0.0, 0.9973333333),
    "waist_yaw_joint": (0.0, 0.0, 0.9008947006),
    "waist_pitch_joint": (0.0, 0.0, 1.6560509554),
    "waist_roll_joint": (0.0, 0.0, 1.0655737705),
    "left_shoulder_pitch_joint": (0.4, 0.2, 1.1249218750),
    "left_shoulder_roll_joint": (0.0, 0.2, 1.2572691552),
    "left_shoulder_yaw_joint": (0.0, 0.0, 1.0242566510),
    "left_elbow_joint": (-1.2, 0.6, 1.3336729496),
    "left_wrist_yaw_joint": (0.0, 0.0, 0.6316236307),
    "left_wrist_pitch_joint": (0.0, 0.0, 2.8932437276),
    "left_wrist_roll_joint": (0.0, 0.0, 1.7187102397),
    "right_shoulder_pitch_joint": (0.4, 0.2, 1.1249218750),
    "right_shoulder_roll_joint": (0.0, -0.2, 1.2572691552),
    "right_shoulder_yaw_joint": (0.0, 0.0, 1.0242566510),
    "right_elbow_joint": (-1.2, 0.6, 1.3336729496),
    "right_wrist_yaw_joint": (0.0, 0.0, 0.6316236307),
    "right_wrist_pitch_joint": (0.0, 0.0, 2.8932437276),
    "right_wrist_roll_joint": (0.0, 0.0, 1.7187102397),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_canonical(
    joint_pos: torch.Tensor, joint_vel: torch.Tensor, joint_names: list[str]
) -> tuple[torch.Tensor, torch.Tensor]:
    unknown = sorted(set(joint_names) - set(SOURCE_CANONICAL))
    if unknown:
        raise ValueError(f"missing source-canonical joint mapping: {unknown}")
    params = joint_pos.new_tensor([SOURCE_CANONICAL[name] for name in joint_names])
    x2_default, g1_default, relative_scale = params.unbind(dim=-1)
    return (
        g1_default + (joint_pos - x2_default) * relative_scale,
        joint_vel * relative_scale,
    )


def command_multi_future_source29(
    env: Any, command_name: str, non_flatten: bool = False
) -> torch.Tensor:
    """Return the frozen G1 encoder's 29-DoF future-reference contract.

    Layout intentionally matches SONIC's existing command term: all flattened
    position frames followed by all flattened velocity frames, then (for the
    tokenizer group) reshaped to ``[N, future, 58]``.
    """

    command = env.command_manager.get_term(command_name)
    frame_count = int(command.num_future_frames)
    articulation_names = list(command.robot.joint_names)
    joint_count = len(articulation_names)
    if joint_count != 31 or set(articulation_names) != set(SOURCE29) | set(HEAD2):
        raise RuntimeError("live X2 articulation is not the frozen WBT29+head2 partition")
    pos31 = command.joint_pos_multi_future.view(command.num_envs, frame_count, joint_count)
    vel31 = command.joint_vel_multi_future.view(command.num_envs, frame_count, joint_count)
    indices = torch.as_tensor(
        [articulation_names.index(name) for name in SOURCE29],
        device=pos31.device,
        dtype=torch.long,
    )
    pos29 = pos31.index_select(-1, indices)
    vel29 = vel31.index_select(-1, indices)
    pos29, vel29 = _source_canonical(pos29, vel29, list(SOURCE29))
    values = torch.cat(
        (pos29.reshape(command.num_envs, -1), vel29.reshape(command.num_envs, -1)), dim=1
    )
    if non_flatten:
        return values.reshape(command.num_envs, frame_count, 58)
    return values


def validate_split(
    motion_lib: Any,
    *,
    source_path: Path,
    expected_sha256: str,
    expected_keys: list[str],
    expected_split: str,
    recorded_state_adapter: bool,
) -> dict[str, Any]:
    """Hash/key/split-lock a live MotionLib and optionally restore native state."""

    if _sha256(source_path) != expected_sha256:
        raise RuntimeError("Phase46 source hash differs from the frozen contract")
    payload = joblib.load(source_path)
    if list(payload) != expected_keys:
        raise RuntimeError("Phase46 source sampler keys differ from the frozen contract")
    if any(row.get("split") != expected_split for row in payload.values()):
        raise RuntimeError("Phase46 source split label differs from the frozen contract")
    live_keys = list(motion_lib.curr_motion_keys)
    if live_keys != expected_keys:
        raise RuntimeError(f"live MotionLib keys differ: {live_keys} != {expected_keys}")
    adapter = None
    if recorded_state_adapter:
        from x2_native_gold_motionlib_adapter import apply_recorded_state_adapter

        adapter = apply_recorded_state_adapter(motion_lib, source_path)
    return {
        "path": str(source_path),
        "sha256": expected_sha256,
        "split": expected_split,
        "sampler_keys": expected_keys,
        "recorded_state_adapter": adapter,
    }


def _model_state_hash(module: torch.nn.Module, *, exclude_lora: bool = False) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        if exclude_lora and ("lora_A" in name or "lora_B" in name):
            continue
        array = value.detach().cpu().contiguous().numpy()
        digest.update(f"{name}:{array.dtype}:{array.shape}".encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


class Phase46LiveZeroTrainer:
    """Trainer-shaped fail-closed probe with zero optimizer and zero env steps."""

    def __init__(
        self,
        *,
        env: Any,
        model: torch.nn.Module,
        value_model: torch.nn.Module,
        checkpoint: str | None,
        config: Any,
        log_dir: str | Path,
        **_: Any,
    ) -> None:
        if checkpoint is None:
            raise RuntimeError("Phase46 requires the frozen source checkpoint")
        self.env = env
        self.policy = model
        self.value_model = value_model
        self.config = config
        self.log_dir = Path(log_dir)
        self.checkpoint = Path(checkpoint)
        self.optimizer = None

    @staticmethod
    def _load_exact(module: torch.nn.Module, state: Mapping[str, torch.Tensor], label: str) -> None:
        missing, unexpected = module.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"{label} checkpoint is not exact at WBT29 live boundary: "
                f"missing={missing}, unexpected={unexpected}"
            )

    @staticmethod
    def _sequence_obs(obs: Mapping[str, Any]) -> dict[str, Any]:
        return {
            name: value.unsqueeze(1) if isinstance(value, torch.Tensor) and value.ndim == 2 else value
            for name, value in obs.items()
        }

    def train(self) -> None:
        from gear_sonic.trl.trainer.ppo_trainer import _apply_any2any_lora

        split = os.environ.get("PHASE46_SPLIT", "")
        if split not in ("train", "held_out"):
            raise RuntimeError("PHASE46_SPLIT must be train or held_out")
        expected_keys = json.loads(os.environ["PHASE46_EXPECTED_KEYS_JSON"])
        split_report = validate_split(
            self.env._motion_lib,
            source_path=Path(os.environ["PHASE46_SOURCE_PATH"]),
            expected_sha256=os.environ["PHASE46_SOURCE_SHA256"],
            expected_keys=expected_keys,
            expected_split=split,
            recorded_state_adapter=os.environ.get("PHASE46_RECORDED_STATE_ADAPTER") == "true",
        )

        robot = self.env.env.scene["robot"]
        articulation_names = list(robot.joint_names)
        action_term = self.env.env.action_manager.get_term("joint_pos")
        action_names = list(action_term._joint_names)
        if action_names != list(SOURCE29):
            raise RuntimeError(f"live action order is not source29: {action_names}")
        if set(articulation_names) != set(SOURCE29) | set(HEAD2) or len(articulation_names) != 31:
            raise RuntimeError("live articulation does not partition into WBT29 plus head2")
        head_ids = [articulation_names.index(name) for name in HEAD2]
        head_default = robot.data.default_joint_pos[0, head_ids].detach().cpu()
        if not torch.equal(head_default, torch.zeros_like(head_default)):
            raise RuntimeError(f"head simulator nominal is not exact zero: {head_default.tolist()}")

        checkpoint = torch.load(self.checkpoint, map_location=self.env.device, weights_only=False)
        policy_state = checkpoint.get("actor_model_state_dict", checkpoint.get("policy_state_dict"))
        value_state = checkpoint.get("value_state_dict")
        if policy_state is None or value_state is None:
            raise RuntimeError("source checkpoint lacks policy/value state")
        self._load_exact(self.policy, policy_state, "policy")
        self._load_exact(self.value_model, value_state, "value")
        base_policy = copy.deepcopy(self.policy).eval()
        base_value = copy.deepcopy(self.value_model).eval()
        self.policy.eval()
        self.value_model.eval()

        lora_report = _apply_any2any_lora(
            self.policy,
            self.value_model,
            self.config.any2any_lora,
            env=self.env,
            report_dir=None,
        )
        # LoRA modules are constructed on CPU by the shared injector. The
        # normal PPO trainer subsequently moves the assembled modules; this
        # zero-only trainer must make that boundary explicit before forward.
        self.policy.to(self.env.device)
        self.value_model.to(self.env.device)
        actor_layers = tuple(lora_report["actor_lora_layers"])
        critic_layers = tuple(lora_report["critic_lora_layers"])
        if actor_layers != EXACT_S7_ACTOR:
            raise RuntimeError(f"live actor LoRA scope differs from exact S7: {actor_layers}")
        if critic_layers != EXACT_S7_CRITIC:
            raise RuntimeError(f"live critic LoRA scope differs from exact S7: {critic_layers}")
        trainable_names = sorted(
            f"policy.{name}" for name, p in self.policy.named_parameters() if p.requires_grad
        ) + sorted(
            f"value_model.{name}" for name, p in self.value_model.named_parameters() if p.requires_grad
        )
        if any("std" in name for name in trainable_names):
            raise RuntimeError("source action std is trainable in Phase46")

        # Reset is required to materialize a real observation batch.  No action
        # is applied and env.step is never called.
        obs = self._sequence_obs(self.env.reset(flatten_dict_obs=True))
        observed = {name: list(value.shape) for name, value in obs.items() if torch.is_tensor(value)}
        tokenizer_terms = self.env.env.observation_manager.compute_group(
            "tokenizer", update_history=False
        )
        if not isinstance(tokenizer_terms, Mapping):
            raise RuntimeError("live tokenizer group is unexpectedly concatenated")
        future_reference = tokenizer_terms.get("command_multi_future_nonflat")
        future_shape = list(future_reference.shape) if torch.is_tensor(future_reference) else None
        if observed.get("actor_obs", [])[-1:] != [930]:
            raise RuntimeError(f"live actor observation is not 930: {observed.get('actor_obs')}")
        if observed.get("critic_obs", [])[-1:] != [1645]:
            raise RuntimeError(f"live critic observation is not 1645: {observed.get('critic_obs')}")
        if future_shape is None or future_shape[-2:] != [10, 58]:
            raise RuntimeError(
                "live tokenizer future reference is not 10x58: "
                f"{future_shape}"
            )

        batches = []
        with torch.no_grad():
            for index in range(2):
                base_action = base_policy(obs)
                adapted_action = self.policy(obs)
                base_value_out = base_value.evaluate(obs)
                adapted_value_out = self.value_model.evaluate(obs)
                if base_action.shape[-1] != 29:
                    raise RuntimeError(f"live actor output is not 29: {tuple(base_action.shape)}")
                batches.append(
                    {
                        "index": index,
                        "action_shape": list(base_action.shape),
                        "value_shape": list(base_value_out.shape),
                        "action_B0_max_abs": float((base_action - adapted_action).abs().max()),
                        "value_B0_max_abs": float((base_value_out - adapted_value_out).abs().max()),
                        "finite": bool(
                            torch.isfinite(base_action).all()
                            and torch.isfinite(adapted_action).all()
                            and torch.isfinite(base_value_out).all()
                            and torch.isfinite(adapted_value_out).all()
                        ),
                    }
                )

        tolerance = 1.0e-6
        passed = all(
            row["finite"]
            and row["action_B0_max_abs"] <= tolerance
            and row["value_B0_max_abs"] <= tolerance
            for row in batches
        )
        report = {
            "schema_version": "x2_faithful_live_zero_phase46_v1",
            "split": split,
            "split_contract": split_report,
            "runtime": {
                "articulation_joint_names": articulation_names,
                "action_joint_names": action_names,
                "head_excluded": list(HEAD2),
                "head_default": head_default.tolist(),
                "observation_shapes": observed,
                "actor_lora_layers": list(actor_layers),
                "critic_lora_layers": list(critic_layers),
                "trainable_names": trainable_names,
            },
            "B0_batches": batches,
            "hashes": {
                "source_checkpoint": _sha256(self.checkpoint),
                "base_policy": _model_state_hash(base_policy),
                "adapted_policy_without_lora": _model_state_hash(self.policy, exclude_lora=True),
                "base_value": _model_state_hash(base_value),
                "adapted_value_without_lora": _model_state_hash(self.value_model, exclude_lora=True),
            },
            "truth_boundary": {
                "isaac_environment_instances": 1,
                "environment_resets_for_observation": 1,
                "environment_control_steps": 0,
                "optimizer_instances": 0,
                "optimizer_steps": 0,
                "checkpoints_created": 0,
                "physical_performance_claim": False,
            },
            "decision": "PASS" if passed else "FAIL",
        }
        self.log_dir.mkdir(parents=True, exist_ok=True)
        output = self.log_dir / f"phase46_live_zero_{split}.json"
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(output), "decision": report["decision"]}, ensure_ascii=False))
        if not passed:
            raise RuntimeError("Phase46 B=0 runtime equivalence failed")
