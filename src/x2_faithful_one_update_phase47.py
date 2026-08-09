"""Fail-closed trainer for the single authorized Phase47 exact-S7 PPO update.

The class deliberately reuses the shared SONIC PPO implementation.  It adds
only immutable workload/data/scope guards plus post-update provenance.  It is
not a general training launcher and refuses more than one outer update.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import torch

from gear_sonic.trl.trainer.ppo_trainer_aux_loss import TRLAuxLossPPOTrainer
from x2_faithful_live_phase46 import (
    EXACT_S7_ACTOR,
    EXACT_S7_CRITIC,
    HEAD2,
    SOURCE29,
    _sha256,
    validate_split,
)


TRAIN_PATH = Path(
    "/home/humanplus/projects/ZHY/CWI_CrossEmbodiment_Sim/"
    "artifacts/retarget/x2_phase45_kinematic_bronze_train.pkl"
)
TRAIN_SHA256 = "06b759e7926bd841537cf515d7cc46793d47503a9fab5da95c649a2f0b6ec4a6"
TRAIN_KEYS = ["AMASS-STAND-001", "AMASS-UPPER-001", "PHUMA-LUNGE-R-001"]
SOURCE_CHECKPOINT_SHA256 = (
    "e6bdab3f64a39336b3d41877d4f497d05f58af275f288ec0e6746c283ded8909"
)
EXPECTED = {
    "seed": 0,
    "envs": 64,
    "rollout_steps_per_env": 24,
    "transitions": 1536,
    "ppo_epochs": 5,
    "mini_batches": 4,
    "gradient_accumulation_steps": 1,
    "optimizer_minibatch_steps": 20,
    "outer_updates": 1,
}


def _tensor_digest(named_tensors: list[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(named_tensors):
        value = tensor.detach().cpu().contiguous()
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _lora_scope(model: torch.nn.Module, prefix: str) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                name.rsplit(".lora_", 1)[0]
            for name, _ in model.named_parameters()
            if ".lora_" in name
            }
        )
    )


def _finite_scalar_metrics(row: dict[str, Any]) -> tuple[bool, list[str]]:
    required_prefixes = ("loss/", "grad/", "policy/")
    checked: list[str] = []
    for key, value in row.items():
        if not key.startswith(required_prefixes):
            continue
        if not isinstance(value, (int, float)):
            continue
        checked.append(key)
        if not math.isfinite(float(value)):
            return False, checked
    required = {
        "loss/policy_avg",
        "loss/value_avg",
        "loss/weighted_ppo_loss_avg",
        "grad/actor_preclip",
        "grad/critic_preclip",
        "grad/global_preclip",
        "policy/approxkl_avg",
    }
    return required.issubset(checked), checked


class Phase47OneUpdateTrainer(TRLAuxLossPPOTrainer):
    """The only authorized 1-update exact-S7 Bronze trainer."""

    def __init__(self, *args: Any, checkpoint: str | None = None, **kwargs: Any) -> None:
        if checkpoint is None or _sha256(Path(checkpoint)) != SOURCE_CHECKPOINT_SHA256:
            raise RuntimeError("Phase47 source checkpoint hash mismatch")
        super().__init__(*args, checkpoint=checkpoint, **kwargs)
        self._phase47_checkpoint = Path(checkpoint)
        self._phase47_preflight = self._validate_live_contract()
        self._phase47_trainable_before = {
            name: value.detach().cpu().clone()
            for prefix, module in (
                ("policy", self.policy_model), ("value_model", self.value_model)
            )
            for name, value in module.named_parameters()
            if value.requires_grad
            for name in (f"{prefix}.{name}",)
        }
        frozen = [
            (f"{prefix}.{name}", value)
            for prefix, module in (
                ("policy", self.policy_model), ("value_model", self.value_model)
            )
            for name, value in module.named_parameters()
            if not value.requires_grad
        ]
        self._phase47_frozen_hash_before = _tensor_digest(frozen)

    def _validate_live_contract(self) -> dict[str, Any]:
        split = validate_split(
            self.env._motion_lib,
            source_path=TRAIN_PATH,
            expected_sha256=TRAIN_SHA256,
            expected_keys=TRAIN_KEYS,
            expected_split="train",
            recorded_state_adapter=False,
        )
        if any("held_out" in key.lower() for key in self.env._motion_lib.curr_motion_keys):
            raise RuntimeError("held-out key entered the Phase47 optimizer environment")
        robot = self.env.env.scene["robot"]
        articulation = list(robot.joint_names)
        action_names = list(self.env.env.action_manager.get_term("joint_pos")._joint_names)
        if action_names != list(SOURCE29):
            raise RuntimeError("Phase47 action order is not frozen WBT29")
        if len(articulation) != 31 or set(articulation) != set(SOURCE29) | set(HEAD2):
            raise RuntimeError("Phase47 articulation is not WBT29 plus head2")
        head_ids = [articulation.index(name) for name in HEAD2]
        head = robot.data.default_joint_pos[0, head_ids].detach().cpu()
        if not torch.equal(head, torch.zeros_like(head)):
            raise RuntimeError("Phase47 head nominal is not exact zero")

        actual = {
            "seed": int(self.args.seed),
            "envs": int(self.env.num_envs),
            "rollout_steps_per_env": int(self.num_steps_per_env),
            "transitions": int(self.env.num_envs * self.num_steps_per_env),
            "ppo_epochs": int(self.args.num_ppo_epochs),
            "mini_batches": int(self.args.num_mini_batches),
            "gradient_accumulation_steps": int(self.args.gradient_accumulation_steps),
            "optimizer_minibatch_steps": int(self.optimizer_steps_per_iteration),
            "outer_updates": int(self.args.num_total_batches),
        }
        if actual != EXPECTED:
            raise RuntimeError(f"Phase47 workload drift: {actual} != {EXPECTED}")

        actor_scope = _lora_scope(self.policy_model, "policy")
        critic_scope = _lora_scope(self.value_model, "value_model")
        if actor_scope != tuple(sorted(EXACT_S7_ACTOR)):
            raise RuntimeError(f"Phase47 actor scope drift: {actor_scope}")
        if critic_scope != tuple(sorted(EXACT_S7_CRITIC)):
            raise RuntimeError(f"Phase47 critic scope drift: {critic_scope}")
        trainable = sorted(
            [f"policy.{name}" for name, p in self.policy_model.named_parameters() if p.requires_grad]
            + [f"value_model.{name}" for name, p in self.value_model.named_parameters() if p.requires_grad]
        )
        if not trainable or any(".lora_" not in name for name in trainable):
            raise RuntimeError("Phase47 trainable set contains non-LoRA tensors")
        if any("std" in name for name in trainable):
            raise RuntimeError("Phase47 source std is trainable")
        return {
            "split": split,
            "workload": actual,
            "articulation_joint_names": articulation,
            "action_joint_names": action_names,
            "head_nominal": head.tolist(),
            "actor_lora_scope": list(actor_scope),
            "critic_lora_scope": list(critic_scope),
            "trainable_names": trainable,
        }

    def train(self) -> None:
        super().train()
        if int(self.state.global_step) != 1:
            raise RuntimeError(f"Phase47 executed {self.state.global_step} outer updates, expected 1")
        diagnostics_path = Path(self.training_diagnostics_path)
        rows = [json.loads(line) for line in diagnostics_path.read_text().splitlines() if line]
        if len(rows) != 1:
            raise RuntimeError(f"Phase47 diagnostics rows={len(rows)}, expected 1")
        row = rows[0]
        if int(row.get("optimizer/effective_steps", -1)) != 20:
            raise RuntimeError("Phase47 effective optimizer step count is not 20")
        finite, checked = _finite_scalar_metrics(row)
        if not finite:
            raise RuntimeError("Phase47 loss/gradient/KL diagnostics are nonfinite or incomplete")

        frozen = [
            (f"{prefix}.{name}", value)
            for prefix, module in (
                ("policy", self.policy_model), ("value_model", self.value_model)
            )
            for name, value in module.named_parameters()
            if not value.requires_grad
        ]
        frozen_after = _tensor_digest(frozen)
        if frozen_after != self._phase47_frozen_hash_before:
            raise RuntimeError("Phase47 changed a frozen dense/reference/FSQ/std parameter")
        deltas = {}
        for prefix, module in (("policy", self.policy_model), ("value_model", self.value_model)):
            for name, value in module.named_parameters():
                full_name = f"{prefix}.{name}"
                if full_name in self._phase47_trainable_before:
                    deltas[full_name] = float(
                        (value.detach().cpu() - self._phase47_trainable_before[full_name]).abs().max()
                    )
        if not deltas or max(deltas.values()) <= 0.0:
            raise RuntimeError("Phase47 optimizer made no change to any exact-S7 tensor")

        report = {
            "schema_version": "x2_faithful_one_update_phase47_runtime_v1",
            "decision": "PASS_NUMERICAL_ONE_UPDATE_ONLY",
            "preflight": self._phase47_preflight,
            "source_checkpoint": {
                "path": str(self._phase47_checkpoint),
                "sha256": SOURCE_CHECKPOINT_SHA256,
            },
            "training_diagnostics": row,
            "finite_metric_keys": checked,
            "trainable_delta_max_abs": deltas,
            "frozen_parameter_hash": {
                "before": self._phase47_frozen_hash_before,
                "after": frozen_after,
                "unchanged": True,
            },
            "truth_boundary": {
                "outer_updates": 1,
                "optimizer_minibatch_steps": 20,
                "transitions": 1536,
                "held_out_optimizer_samples": 0,
                "five_update_authorized": False,
            },
        }
        self.log_dir.mkdir(parents=True, exist_ok=True)
        output = self.log_dir / "phase47_one_update_runtime.json"
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps({"output": str(output), "decision": report["decision"]}))


class Phase47BoundedRegressionTrainer(TRLAuxLossPPOTrainer):
    """Deterministic bounded rollout used only before/after the one update.

    The shared trainer necessarily constructs an optimizer during init, but
    this override never calls backward/step and never exposes rollout samples
    to an optimizer.  This distinction is written into every result.
    """

    def __init__(self, *args: Any, checkpoint: str | None = None, **kwargs: Any) -> None:
        if checkpoint is None:
            raise RuntimeError("Phase47 regression requires an explicit checkpoint")
        expected = os.environ["PHASE47_EVAL_CHECKPOINT_SHA256"]
        if _sha256(Path(checkpoint)) != expected:
            raise RuntimeError("Phase47 regression checkpoint hash mismatch")
        super().__init__(*args, checkpoint=checkpoint, **kwargs)
        self._phase47_eval_checkpoint = Path(checkpoint)
        self._phase47_eval_checkpoint_sha = expected

    def train(self) -> None:
        split = os.environ["PHASE47_EVAL_SPLIT"]
        source_path = Path(os.environ["PHASE47_EVAL_SOURCE_PATH"])
        expected_keys = json.loads(os.environ["PHASE47_EVAL_KEYS_JSON"])
        contract = validate_split(
            self.env._motion_lib,
            source_path=source_path,
            expected_sha256=os.environ["PHASE47_EVAL_SOURCE_SHA256"],
            expected_keys=expected_keys,
            expected_split=split,
            recorded_state_adapter=os.environ.get("PHASE47_EVAL_RECORDED_ADAPTER") == "true",
        )
        horizon = int(os.environ["PHASE47_EVAL_HORIZON"])
        label = os.environ["PHASE47_EVAL_LABEL"]
        if horizon <= 0 or horizon > 400:
            raise RuntimeError("Phase47 regression horizon is outside preregistered bounds")

        self._eval_mode()
        self.env.set_is_evaluating()
        self.model.policy.eval_mode()
        obs = self.env.reset_all()
        obs = {key: value.to(self.accelerator.device) for key, value in obs.items()}
        command = self.env.env.command_manager.get_term("motion")
        initial_motion_ids = command.motion_ids.detach().cpu().tolist()
        observed_ids = sorted(set(int(value) for value in initial_motion_ids))
        if observed_ids != list(range(len(expected_keys))):
            raise RuntimeError(
                f"Phase47 regression did not cover every frozen motion: {observed_ids}"
            )
        per_env_key = [expected_keys[int(index)] for index in initial_motion_ids]
        alive = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.accelerator.device)
        first_done = torch.full(
            (self.env.num_envs,), horizon, dtype=torch.long, device=self.accelerator.device
        )
        metric_sum: dict[str, torch.Tensor] = {}
        metric_count: dict[str, torch.Tensor] = {}
        initial_action = None
        initial_value = None
        total_reward = torch.zeros(self.env.num_envs, device=self.accelerator.device)

        with torch.no_grad():
            self.model.policy.init_rollout()
            sequence_obs = {
                key: value.unsqueeze(1) if value.ndim == 2 else value for key, value in obs.items()
            }
            initial_value = self.model.value_model.evaluate(sequence_obs).detach().cpu()
            for step in range(horizon):
                self.model.policy.rollout(obs_dict=obs)
                action = self.model.policy.action_mean.detach()
                if initial_action is None:
                    initial_action = action.cpu()
                obs, rewards, dones, _infos = self.env.step({"actions": action})
                obs = {key: value.to(self.accelerator.device) for key, value in obs.items()}
                rewards = rewards.to(self.accelerator.device).reshape(-1)
                dones = dones.to(self.accelerator.device).reshape(-1).bool()
                total_reward[alive] += rewards[alive]
                for metric_name in (
                    "error_anchor_pos", "error_anchor_rot", "error_body_pos",
                    "error_body_rot", "error_joint_pos", "error_joint_vel",
                ):
                    value = command.metrics[metric_name].detach().reshape(-1)
                    metric_sum.setdefault(metric_name, torch.zeros_like(total_reward))[alive] += value[alive]
                    metric_count.setdefault(metric_name, torch.zeros_like(total_reward))[alive] += 1
                newly_done = alive & dones
                first_done[newly_done] = step + 1
                alive &= ~dones

        rows = []
        for env_index, motion_key in enumerate(per_env_key):
            steps = int(first_done[env_index].item())
            rows.append({
                "env_index": env_index,
                "motion_key": motion_key,
                "survival_s": steps * 0.02,
                "terminated": steps < horizon,
                "reward_sum": float(total_reward[env_index].item()),
                "metrics": {
                    name: float((metric_sum[name][env_index] / metric_count[name][env_index].clamp_min(1)).item())
                    for name in metric_sum
                },
            })
        aggregate = {}
        for key in expected_keys:
            group = [row for row in rows if row["motion_key"] == key]
            aggregate[key] = {
                "episodes": len(group),
                "survival_s_mean": sum(row["survival_s"] for row in group) / len(group),
                "termination_rate": sum(row["terminated"] for row in group) / len(group),
                "tracking": {
                    name: sum(row["metrics"][name] for row in group) / len(group)
                    for name in group[0]["metrics"]
                },
            }
        report = {
            "schema_version": "x2_faithful_phase47_bounded_regression_v1",
            "label": label,
            "split_contract": contract,
            "checkpoint": {
                "path": str(self._phase47_eval_checkpoint),
                "sha256": self._phase47_eval_checkpoint_sha,
            },
            "horizon_steps": horizon,
            "control_dt_s": 0.02,
            "motion_ids": initial_motion_ids,
            "per_motion": aggregate,
            "fixed_initial_outputs": {
                "action": initial_action.tolist(),
                "value": initial_value.tolist(),
            },
            "truth_boundary": {
                "optimizer_instances_constructed_by_shared_trainer": 1,
                "optimizer_backward_calls": 0,
                "optimizer_steps": 0,
                "rollout_samples_entered_optimizer": 0,
                "physical_domain": "IsaacLab declared B5 train domain",
                "native_gold_role": "source-ability/pipeline regression only",
            },
        }
        self.log_dir.mkdir(parents=True, exist_ok=True)
        output = self.log_dir / f"phase47_regression_{label}.json"
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps({"output": str(output), "motions": aggregate}))
