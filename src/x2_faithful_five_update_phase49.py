"""One gated segment of the authorized Phase49 five-update pilot.

Each process is configured with the original five-update scheduler horizon but
is stopped after exactly one PPO outer update.  Resume segments restore the
optimizer, scheduler, trainer and environment checkpoint state.  External
physical regression gates decide whether the next segment may start.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import torch
from transformers import TrainerCallback

from gear_sonic.trl.trainer.ppo_trainer_aux_loss import TRLAuxLossPPOTrainer
from x2_faithful_live_phase46 import EXACT_S7_ACTOR, EXACT_S7_CRITIC, HEAD2, SOURCE29, _sha256, validate_split
from x2_faithful_one_update_phase47 import (
    TRAIN_KEYS,
    TRAIN_PATH,
    TRAIN_SHA256,
    _finite_scalar_metrics,
    _lora_scope,
    _tensor_digest,
)


class _StopAfterOneOuterUpdate(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        control.should_training_stop = True
        return control


class Phase49GatedSegmentTrainer(TRLAuxLossPPOTrainer):
    """Execute exactly one segment of a checkpoint-resumable five-update plan."""

    def __init__(self, *args: Any, checkpoint: str | None = None, **kwargs: Any) -> None:
        segment = int(os.environ["PHASE49_SEGMENT"])
        if segment not in range(1, 6):
            raise RuntimeError(f"Phase49 segment outside 1..5: {segment}")
        expected_sha = os.environ["PHASE49_EXPECTED_CHECKPOINT_SHA256"]
        if checkpoint is None or _sha256(Path(checkpoint)) != expected_sha:
            raise RuntimeError("Phase49 input checkpoint hash mismatch")
        super().__init__(*args, checkpoint=checkpoint, **kwargs)
        self._phase49_segment = segment
        self._phase49_checkpoint = Path(checkpoint)
        self._phase49_checkpoint_sha = expected_sha
        expected_start = segment - 1
        if int(self.state.global_step) != expected_start:
            raise RuntimeError(
                f"Phase49 resume step drift: {self.state.global_step} != {expected_start}"
            )
        self._phase49_preflight = self._validate_contract()
        self._phase49_trainable_before = {
            f"{prefix}.{name}": value.detach().cpu().clone()
            for prefix, module in (("policy", self.policy_model), ("value_model", self.value_model))
            for name, value in module.named_parameters()
            if value.requires_grad
        }
        frozen = [
            (f"{prefix}.{name}", value)
            for prefix, module in (("policy", self.policy_model), ("value_model", self.value_model))
            for name, value in module.named_parameters()
            if not value.requires_grad
        ]
        self._phase49_frozen_before = _tensor_digest(frozen)
        self.callback_handler.add_callback(_StopAfterOneOuterUpdate())

    def _validate_contract(self) -> dict[str, Any]:
        split = validate_split(
            self.env._motion_lib,
            source_path=TRAIN_PATH,
            expected_sha256=TRAIN_SHA256,
            expected_keys=TRAIN_KEYS,
            expected_split="train",
            recorded_state_adapter=False,
        )
        if any("held_out" in key.lower() for key in self.env._motion_lib.curr_motion_keys):
            raise RuntimeError("held-out key entered Phase49 optimizer")
        robot = self.env.env.scene["robot"]
        articulation = list(robot.joint_names)
        action_names = list(self.env.env.action_manager.get_term("joint_pos")._joint_names)
        if action_names != list(SOURCE29):
            raise RuntimeError("Phase49 action order drift")
        if len(articulation) != 31 or set(articulation) != set(SOURCE29) | set(HEAD2):
            raise RuntimeError("Phase49 articulation contract drift")
        actual = {
            "seed": int(self.args.seed),
            "envs": int(self.env.num_envs),
            "rollout_steps_per_env": int(self.num_steps_per_env),
            "transitions_per_update": int(self.env.num_envs * self.num_steps_per_env),
            "ppo_epochs": int(self.args.num_ppo_epochs),
            "mini_batches": int(self.args.num_mini_batches),
            "optimizer_steps_per_update": int(self.optimizer_steps_per_iteration),
            "configured_outer_updates": int(self.args.num_total_batches),
        }
        expected = {
            "seed": 0,
            "envs": 64,
            "rollout_steps_per_env": 24,
            "transitions_per_update": 1536,
            "ppo_epochs": 5,
            "mini_batches": 4,
            "optimizer_steps_per_update": 20,
            "configured_outer_updates": 5,
        }
        if actual != expected:
            raise RuntimeError(f"Phase49 workload drift: {actual} != {expected}")
        actor_scope = _lora_scope(self.policy_model, "policy")
        critic_scope = _lora_scope(self.value_model, "value_model")
        if actor_scope != tuple(sorted(EXACT_S7_ACTOR)):
            raise RuntimeError("Phase49 actor exact-S7 scope drift")
        if critic_scope != tuple(sorted(EXACT_S7_CRITIC)):
            raise RuntimeError("Phase49 critic exact-S7 scope drift")
        trainable = sorted(
            [f"policy.{name}" for name, p in self.policy_model.named_parameters() if p.requires_grad]
            + [f"value_model.{name}" for name, p in self.value_model.named_parameters() if p.requires_grad]
        )
        if not trainable or any(".lora_" not in name or "std" in name for name in trainable):
            raise RuntimeError("Phase49 trainable scope is not LoRA-only")
        return {
            "split": split,
            "workload": actual,
            "actor_lora_scope": list(actor_scope),
            "critic_lora_scope": list(critic_scope),
            "trainable_names": trainable,
        }

    def train(self) -> None:
        start = self._phase49_segment - 1
        super().train()
        if int(self.state.global_step) != self._phase49_segment:
            raise RuntimeError(
                f"Phase49 segment executed wrong count: {start}->{self.state.global_step}"
            )
        diagnostics = Path(self.training_diagnostics_path)
        rows = [json.loads(line) for line in diagnostics.read_text().splitlines() if line]
        if len(rows) != 1:
            raise RuntimeError(f"Phase49 segment diagnostics rows={len(rows)}, expected 1")
        row = rows[0]
        finite, checked = _finite_scalar_metrics(row)
        if not finite:
            raise RuntimeError("Phase49 nonfinite loss/gradient/KL")
        if int(row.get("optimizer/effective_steps", -1)) != 20:
            raise RuntimeError("Phase49 optimizer step count drift")
        kl = float(row["policy/approxkl_avg"])
        if not math.isfinite(kl) or kl >= 0.02:
            raise RuntimeError(f"Phase49 KL hard stop: {kl}")
        frozen = [
            (f"{prefix}.{name}", value)
            for prefix, module in (("policy", self.policy_model), ("value_model", self.value_model))
            for name, value in module.named_parameters()
            if not value.requires_grad
        ]
        frozen_after = _tensor_digest(frozen)
        if frozen_after != self._phase49_frozen_before:
            raise RuntimeError("Phase49 frozen parameter hash changed")
        deltas = {}
        for prefix, module in (("policy", self.policy_model), ("value_model", self.value_model)):
            for name, value in module.named_parameters():
                full = f"{prefix}.{name}"
                if full in self._phase49_trainable_before:
                    deltas[full] = float(
                        (value.detach().cpu() - self._phase49_trainable_before[full]).abs().max()
                    )
        if not deltas or max(deltas.values()) <= 0:
            raise RuntimeError("Phase49 segment changed no exact-S7 tensor")
        output = {
            "schema_version": "x2_faithful_phase49_segment_v1",
            "decision": "PASS_NUMERICAL_SEGMENT_PENDING_PHYSICAL_GATE",
            "segment": self._phase49_segment,
            "start_global_step": start,
            "end_global_step": int(self.state.global_step),
            "input_checkpoint": {
                "path": str(self._phase49_checkpoint),
                "sha256": self._phase49_checkpoint_sha,
            },
            "preflight": self._phase49_preflight,
            "training_diagnostics": row,
            "finite_metric_keys": checked,
            "trainable_delta_max_abs": deltas,
            "frozen_parameter_hash": {
                "before": self._phase49_frozen_before,
                "after": frozen_after,
                "unchanged": True,
            },
            "truth_boundary": {
                "outer_updates_this_process": 1,
                "cumulative_updates": self._phase49_segment,
                "optimizer_steps_this_process": 20,
                "cumulative_optimizer_steps": self._phase49_segment * 20,
                "held_optimizer_samples": 0,
                "next_segment_authorized": False,
            },
        }
        path = self.log_dir / "phase49_segment_runtime.json"
        path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps({"output": str(path), "decision": output["decision"]}))
