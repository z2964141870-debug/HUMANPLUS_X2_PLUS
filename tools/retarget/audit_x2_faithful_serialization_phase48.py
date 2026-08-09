#!/usr/bin/env python3
"""Phase48 exact-S7 checkpoint serialization roundtrip, with no physics/update.

The final Phase47 checkpoint is loaded as the in-memory baseline.  Fixed
Bronze-lunge and native-Gold batches are evaluated on CPU, then the exact
trainer checkpoint writer serializes the unmodified policy/value state to a
temporary file.  The temporary checkpoint is reloaded and evaluated on the
same tensors.  No optimizer is constructed and no environment/physics step is
executed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from typing import Any, Mapping

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

from gear_sonic.trl.callbacks.model_save_callback import ModelSaveCallback  # noqa: E402
from gear_sonic.trl.utils.any2any_lora_checkpoint import (  # noqa: E402
    merge_any2any_lora_state,
)
from retarget.audit_x2_faithful_entrypoint_phase24 import (  # noqa: E402
    MLP,
    PolicyShell,
    ValueShell,
)
from retarget.run_x2_faithful_bronze_zero_phase45 import (  # noqa: E402
    load_bronze,
)
from retarget.run_x2_faithful_zero_update_phase26 import (  # noqa: E402
    build_contract,
    default_source29,
    fixed_batch,
    load_split,
    normalize_critic,
)


FINAL = REPO / "logs/x2_faithful_exact_s7_one_update_phase47/last.pt"
FINAL_SHA256 = "8048298519495117d5691828d0d8507ea3952f45b5d3ea7014f449e9760db982"
PHASE45_CONFIG = REPO / "configs/x2_faithful_bronze_exact_s7_phase45.yaml"
PHASE47_RUNTIME = (
    REPO / "logs/x2_faithful_exact_s7_one_update_phase47/phase47_one_update_runtime.json"
)
DEFAULT_JSON = REPO / "reports/retarget/x2_faithful_serialization_roundtrip_phase48.json"
DEFAULT_MD = REPO / "reports/retarget/x2_faithful_serialization_roundtrip_phase48.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_hash(state: Mapping[str, torch.Tensor], names: list[str]) -> str:
    digest = hashlib.sha256()
    for name in sorted(names):
        value = state[name].detach().cpu().contiguous()
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def split_names(checkpoint: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    trainable = []
    frozen = []
    for state_name in ("policy_state_dict", "value_state_dict"):
        prefix = "policy" if state_name == "policy_state_dict" else "value_model"
        state = checkpoint[state_name]
        for name in state:
            full = f"{prefix}.{name}"
            (trainable if name.endswith((".lora_A", ".lora_B")) else frozen).append(full)
    return sorted(trainable), sorted(frozen)


def combined_hash(checkpoint: Mapping[str, Any], names: list[str]) -> str:
    values: dict[str, torch.Tensor] = {}
    for full in names:
        prefix, name = full.split(".", 1)
        state_key = "policy_state_dict" if prefix == "policy" else "value_state_dict"
        values[full] = checkpoint[state_key][name]
    return tensor_hash(values, names)


class _StateCarrier:
    def __init__(self, state: Mapping[str, torch.Tensor]):
        self._state = state

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: value.detach().clone() for name, value in self._state.items()}


class _ModelCarrier:
    def __init__(self, checkpoint: Mapping[str, Any]):
        self.policy = _StateCarrier(checkpoint["policy_state_dict"])
        self.value_model = _StateCarrier(checkpoint["value_state_dict"])


def materialized(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    policy, policy_report = merge_any2any_lora_state(
        checkpoint["policy_state_dict"], fallback_alpha=16.0
    )
    value, value_report = merge_any2any_lora_state(
        checkpoint["value_state_dict"], fallback_alpha=16.0
    )
    if policy_report["merged_layer_count"] != 7:
        raise RuntimeError("Phase48 policy exact-S7 layer count drift")
    if value_report["merged_layer_count"] != 5:
        raise RuntimeError("Phase48 critic exact-S7 layer count drift")
    return {
        "policy_state_dict": policy,
        "value_state_dict": value,
        "merge_report": {"policy": policy_report, "value": value_report},
    }


def evaluate(batch: Mapping[str, torch.Tensor], checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    policy_state = checkpoint["policy_state_dict"]
    value_state = checkpoint["value_state_dict"]
    policy = PolicyShell(policy_state).eval()
    value = ValueShell(value_state).eval()
    encoder = MLP(policy_state, "actor_module.encoders.g1").eval()
    quantizer = FSQ(levels=[32] * 32).eval()
    with torch.no_grad():
        latent = encoder(batch["encoder_input"]).reshape(1, 2, 32)
        token, _ = quantizer(latent)
        decoder_input = torch.cat((token.reshape(1, 64), batch["actor_obs"]), dim=-1)
        action = policy.actor_module.decoders["g1_dyn"](decoder_input)
        critic = normalize_critic(batch["critic_obs"], value_state)
        value_out = value.critic_module(critic)
    if not all(torch.isfinite(item).all() for item in (token, action, value_out)):
        raise RuntimeError("Phase48 fixed forward produced non-finite output")
    return {
        "token": token.detach().cpu(),
        "action": action.detach().cpu(),
        "value": value_out.detach().cpu(),
    }


def compare(before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    result = {}
    for name in ("token", "action", "value"):
        delta = (before[name] - after[name]).abs()
        result[name] = {
            "max_abs": float(delta.max()),
            "exact": bool(torch.equal(before[name], after[name])),
        }
    return result


def render(report: Mapping[str, Any]) -> str:
    r = report["roundtrip"]
    return "\n".join(
        [
            "# WBT Phase48：exact-S7 序列化闭环",
            "",
            f"- 裁决：**{report['decision']}**。",
            "- 边界：CPU MotionLib/FK固定前向；environment、physics step、optimizer、backward均为0。",
            "- 使用Phase47 final作为in-memory baseline，并调用训练使用的`ModelSaveCallback.save_checkpoint`写临时roundtrip；临时文件已删除。",
            "",
            "## 结果",
            "",
            f"- lunge：`{r['lunge']}`",
            f"- native Gold：`{r['gold']}`",
            f"- trainable/frozen hash：`{report['hashes']}`",
            f"- exact-S7 names：`{report['exact_s7']}`",
            "",
            "## 结论",
            "",
            "保存前内存态与使用同一writer另存后reload的token/action/value逐元素完全相等，最大绝对误差0，严格通过1e-6门。Phase47遗留的序列化证据缺口已闭合。",
            "",
            "## 5-update fresh-from-source预注册（未执行）",
            "",
            f"`{report['five_update_preregistered']}`",
            "",
            "本阶段没有启动训练；5-update仍需主任务单独授权。",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()
    if sha256(FINAL) != FINAL_SHA256:
        raise RuntimeError("Phase48 final checkpoint hash drift")
    config = yaml.safe_load(PHASE45_CONFIG.read_text())
    runtime = json.loads(PHASE47_RUNTIME.read_text())
    checkpoint = torch.load(FINAL, map_location="cpu", weights_only=False)
    trainable, frozen = split_names(checkpoint)
    expected_names = sorted(runtime["preflight"]["trainable_names"])
    if trainable != expected_names:
        raise RuntimeError("Phase48 exact-S7 trainable names differ from Phase47 runtime")

    phase24 = yaml.safe_load(
        Path(config["frozen_inputs"]["phase24_config"]["path"]).read_text()
    )
    phase23 = json.loads(
        Path(config["frozen_inputs"]["phase23_report"]["path"]).read_text()
    )
    phase26 = yaml.safe_load(
        Path(config["frozen_inputs"]["phase26_config"]["path"]).read_text()
    )
    contract = build_contract(phase23)
    default29 = default_source29(contract, phase26["x2_default_joint_positions"])
    source14 = phase24["reward_semantics"]["source14_body_names"]
    bronze_motion, _, _ = load_bronze(config)
    gold_motion, _, _ = load_split(phase24, "held_out")
    h = config["fixed_zero_batches"]["history_frames"]
    f = config["fixed_zero_batches"]["future_frames"]
    batches = {
        "lunge": fixed_batch(
            bronze_motion, "PHUMA-LUNGE-R-001", contract, default29, source14, h, f
        ),
        "gold": fixed_batch(
            gold_motion,
            "official_native_dance_held_out_000",
            contract,
            default29,
            source14,
            h,
            f,
        ),
    }
    before_materialized = materialized(checkpoint)
    before = {name: evaluate(batch, before_materialized) for name, batch in batches.items()}
    trainable_hash_before = combined_hash(checkpoint, trainable)
    frozen_hash_before = combined_hash(checkpoint, frozen)

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix="x2_phase48_", suffix=".pt", delete=False) as stream:
            temporary_path = Path(stream.name)
        # The live writer removes log_history from the serialized trainer
        # state.  Re-saving a checkpoint therefore requires reconstructing
        # that non-model bookkeeping field; policy/value tensors are untouched.
        writer_state = copy.deepcopy(checkpoint["state"])
        writer_state.__dict__.setdefault("log_history", [])
        ModelSaveCallback.save_checkpoint(
            _ModelCarrier(checkpoint),
            optimizer=None,
            lr_scheduler=None,
            state=writer_state,
            env_state_dict=checkpoint["env_state_dict"],
            args=checkpoint["args"],
            save_path=str(temporary_path),
        )
        reloaded = torch.load(temporary_path, map_location="cpu", weights_only=False)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    after_materialized = materialized(reloaded)
    after = {name: evaluate(batch, after_materialized) for name, batch in batches.items()}
    roundtrip = {name: compare(before[name], after[name]) for name in batches}
    trainable_hash_after = combined_hash(reloaded, trainable)
    frozen_hash_after = combined_hash(reloaded, frozen)
    all_forward = all(
        row[metric]["max_abs"] <= 1e-6 and row[metric]["exact"]
        for row in roundtrip.values()
        for metric in ("token", "action", "value")
    )
    hashes = {
        "trainable_before": trainable_hash_before,
        "trainable_after": trainable_hash_after,
        "trainable_exact": trainable_hash_before == trainable_hash_after,
        "frozen_before": frozen_hash_before,
        "frozen_after": frozen_hash_after,
        "frozen_exact": frozen_hash_before == frozen_hash_after,
    }
    passed = all_forward and hashes["trainable_exact"] and hashes["frozen_exact"]
    report = {
        "schema_version": "x2_faithful_serialization_roundtrip_phase48_v1",
        "decision": "PASS_SERIALIZATION_GATE_FIVE_UPDATE_PREREGISTERED_NOT_RUN" if passed else "REJECT",
        "provenance": {
            "phase47_final": str(FINAL),
            "phase47_final_sha256": FINAL_SHA256,
            "writer": "gear_sonic.trl.callbacks.model_save_callback.ModelSaveCallback.save_checkpoint",
            "temporary_checkpoint_deleted": True,
        },
        "roundtrip": roundtrip,
        "hashes": hashes,
        "exact_s7": {
            "trainable_names": trainable,
            "expected_names": expected_names,
            "names_exact": trainable == expected_names,
            "actor_lora_layers": 7,
            "critic_lora_layers": 5,
        },
        "truth_boundary": {
            "motionlib_cpu_loader_instances": 2,
            "fixed_forward_batches": 2,
            "environment_instances": 0,
            "physics_steps": 0,
            "optimizer_instances": 0,
            "optimizer_steps": 0,
            "backward_calls": 0,
        },
        "five_update_preregistered": {
            "authorized_in_phase48": False,
            "initialization": "fresh_from_original_SONIC_source_checkpoint_not_Phase47_final",
            "source_checkpoint_sha256": runtime["source_checkpoint"]["sha256"],
            "seed": 0,
            "outer_updates_max": 5,
            "envs": 64,
            "rollout_steps_per_env": 24,
            "transitions_per_update": 1536,
            "transitions_total_max": 7680,
            "ppo_epochs": 5,
            "minibatches_per_epoch": 4,
            "optimizer_steps_per_update": 20,
            "optimizer_steps_total_max": 100,
            "train_keys_exact": ["AMASS-STAND-001", "AMASS-UPPER-001", "PHUMA-LUNGE-R-001"],
            "held_optimizer_samples": 0,
            "lora_scope": "Phase47 exact-S7 unchanged",
            "checkpoint_policy": "save source and final only; no intermediate checkpoint",
            "per_update_hard_stops": [
                "nonfinite observation/action/reward/advantage/loss/gradient",
                "approx_KL >= 0.02",
                "dynamic lunge survival drop >0.10s from frozen source baseline",
                "dynamic lunge termination-rate increase >0.05",
                "dynamic lunge has no preregistered tracking error decrease or any increases >10%",
                "frozen parameter hash changes",
                "held/native-Gold sample enters optimizer"
            ],
            "final_native_gold_gate": {
                "survival_drop_max_fraction": 0.05,
                "termination_rate_increase_max": 0.05,
                "tracking_error_increase_max_fraction": 0.1,
                "action_value_drift": "report with the same fixed deterministic batches; no post-hoc threshold"
            },
            "final_reload_gate_max_abs": 1e-6,
            "claim_boundary": "kinematic-Bronze optimizer sanity only; not dynamic-Silver/Gold or deployment evidence"
        },
    }
    if not passed:
        raise RuntimeError(json.dumps(report, indent=2))
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    args.markdown.write_text(render(report), encoding="utf-8")
    print(json.dumps({"decision": report["decision"], "json": str(args.json)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
