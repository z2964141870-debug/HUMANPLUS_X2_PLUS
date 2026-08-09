#!/usr/bin/env python3
"""Static/CPU entrypoint for faithful X2 Any2Any Phase24.

The train/eval modes select immutable Gold hooks but do not create an
optimizer or simulator.  ``live-zero`` is fail-closed until B5 is frozen.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch
from torch import nn
import yaml


REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from x2_faithful_any2any_phase23 import GoldSplitSpec, ImmutableGoldMotionLibHook


DEFAULT_CONFIG = REPO / "configs/x2_faithful_any2any_phase24.yaml"
DEFAULT_JSON = REPO / "reports/retarget/x2_faithful_any2any_phase24.json"
DEFAULT_MD = REPO / "reports/retarget/x2_faithful_any2any_phase24.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_source_checkpoint(path: Path) -> dict[str, Any]:
    # The released checkpoint pickles only trainer bookkeeping through this
    # symbol.  It is not used by this audit; defining the lookup lets CPU torch
    # load the trusted local release without importing Isaac/pxr.
    import trl.trainer.utils as trainer_utils

    class OnlineTrainerState:
        pass

    OnlineTrainerState.__module__ = "trl.trainer.utils"
    trainer_utils.OnlineTrainerState = OnlineTrainerState
    return torch.load(path, map_location="cpu", weights_only=False)


class MLP(nn.Module):
    def __init__(self, state: dict[str, torch.Tensor], prefix: str):
        super().__init__()
        modules: list[nn.Module] = []
        linear_indices = []
        for index in range(0, 64, 2):
            weight_key = f"{prefix}.module.{index}.weight"
            if weight_key not in state:
                break
            weight = state[weight_key]
            bias = state[f"{prefix}.module.{index}.bias"]
            linear = nn.Linear(weight.shape[1], weight.shape[0])
            with torch.no_grad():
                linear.weight.copy_(weight)
                linear.bias.copy_(bias)
            modules.append(linear)
            linear_indices.append(index)
            next_weight = f"{prefix}.module.{index + 2}.weight"
            if next_weight in state:
                modules.append(nn.SiLU())
        self.module = nn.Sequential(*modules)
        self.linear_indices = tuple(linear_indices)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.module(value)


class PolicyShell(nn.Module):
    def __init__(self, state: dict[str, torch.Tensor]):
        super().__init__()
        self.std = nn.Parameter(state["std"].detach().clone(), requires_grad=False)
        self.actor_module = nn.Module()
        self.actor_module.decoders = nn.ModuleDict(
            {"g1_dyn": MLP(state, "actor_module.decoders.g1_dyn")}
        )


class ValueShell(nn.Module):
    def __init__(self, state: dict[str, torch.Tensor]):
        super().__init__()
        self.critic_module = MLP(state, "critic_module")


class AuditLoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        input_mask: torch.Tensor | None = None,
    ):
        super().__init__()
        self.base_layer = base
        for parameter in self.base_layer.parameters():
            parameter.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        self.scaling = float(alpha) / rank
        if input_mask is None:
            input_mask = torch.ones(base.in_features)
        self.register_buffer("input_mask", input_mask, persistent=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = torch.nn.functional.linear(value * self.input_mask, self.lora_A)
        residual = torch.nn.functional.linear(residual, self.lora_B)
        return self.base_layer(value) + residual * self.scaling


def _replace_selected(
    sequential: nn.Sequential,
    selected: list[int],
    rank: int,
    alpha: float,
    input_mask_cfg: dict[str, Any] | None,
) -> dict[str, Any]:
    masks = {}
    for index in selected:
        layer = sequential[index]
        if not isinstance(layer, nn.Linear):
            raise ValueError(f"selected module.{index} is not Linear")
        mask = None
        if input_mask_cfg is not None and int(input_mask_cfg["layer"]) == index:
            start, stop = [int(value) for value in input_mask_cfg["selected_columns"]]
            mask = torch.zeros(layer.in_features)
            mask[start:stop] = 1.0
            masks[f"module.{index}"] = {
                "selected_columns": [start, stop],
                "selected_count": int(mask.sum()),
                "input_width": layer.in_features,
            }
        sequential[index] = AuditLoRALinear(layer, rank, alpha, input_mask=mask)
    return masks


def _parameter_like_count(state: dict[str, torch.Tensor]) -> int:
    return sum(
        int(value.numel())
        for key, value in state.items()
        if torch.is_tensor(value)
        and (key == "std" or key.endswith(".weight") or key.endswith(".bias"))
    )


def instantiate_manifest(
    name: str,
    manifest: dict[str, Any],
    policy_state: dict[str, torch.Tensor],
    value_state: dict[str, torch.Tensor],
) -> dict[str, Any]:
    policy = PolicyShell(policy_state)
    value = ValueShell(value_state)
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    for parameter in value.parameters():
        parameter.requires_grad_(False)
    base_policy = copy.deepcopy(policy)
    base_value = copy.deepcopy(value)
    rank = int(manifest["rank"])
    alpha = float(manifest["alpha"])
    actor = policy.actor_module.decoders["g1_dyn"].module
    critic = value.critic_module.module
    masks = _replace_selected(
        actor,
        [int(item) for item in manifest["actor_layers"]],
        rank,
        alpha,
        manifest.get("actor_input_mask"),
    )
    _replace_selected(
        critic,
        [int(item) for item in manifest["critic_layers"]],
        rank,
        alpha,
        None,
    )
    generator = torch.Generator().manual_seed(2409)
    actor_input = torch.randn(2, 994, generator=generator)
    critic_input = torch.randn(2, 1645, generator=generator)
    with torch.no_grad():
        base_action = base_policy.actor_module.decoders["g1_dyn"](actor_input)
        action = policy.actor_module.decoders["g1_dyn"](actor_input)
        base_value_out = base_value.critic_module(critic_input)
        value_out = value.critic_module(critic_input)
    trainable = []
    frozen = []
    for prefix, module in (("policy", policy), ("value_model", value)):
        for parameter_name, parameter in module.named_parameters():
            row = {
                "name": f"{prefix}.{parameter_name}",
                "shape": list(parameter.shape),
                "params": int(parameter.numel()),
            }
            (trainable if parameter.requires_grad else frozen).append(row)
    trainable_count = sum(row["params"] for row in trainable)
    adapted_dense_base = _parameter_like_count(
        {
            **{
                key: value
                for key, value in policy_state.items()
                if key == "std" or key.startswith("actor_module.decoders.g1_dyn.")
            },
            **{f"value.{key}": value for key, value in value_state.items()},
        }
    )
    full_checkpoint_parameter_like = _parameter_like_count(policy_state) + _parameter_like_count(
        value_state
    )
    expected_lora_names = sorted(
        row["name"] for row in trainable if row["name"].endswith(("lora_A", "lora_B"))
    )
    checks = {
        "action_shape_29": list(action.shape) == [2, 29],
        "value_shape_1": list(value_out.shape) == [2, 1],
        "zero_B_action_exact": bool(torch.equal(base_action, action)),
        "zero_B_value_exact": bool(torch.equal(base_value_out, value_out)),
        "std_loaded_exact": bool(torch.equal(policy.std.detach(), policy_state["std"])),
        "std_frozen": policy.std.requires_grad is False,
        "only_lora_A_B_trainable": len(trainable) > 0
        and len(trainable) == len(expected_lora_names),
        "all_outputs_finite": bool(torch.isfinite(action).all() and torch.isfinite(value_out).all()),
    }
    return {
        "name": name,
        "paper_scope": manifest["paper_scope"],
        "rank": rank,
        "alpha": alpha,
        "actor_layers": [f"actor_module.decoders.g1_dyn.module.{i}" for i in manifest["actor_layers"]],
        "critic_layers": [f"critic_module.module.{i}" for i in manifest["critic_layers"]],
        "input_masks": masks,
        "trainable": trainable,
        "trainable_names": expected_lora_names,
        "trainable_tensor_count": len(trainable),
        "trainable_param_count": trainable_count,
        "adapted_dense_base_param_count": adapted_dense_base,
        "full_checkpoint_parameter_like_count": full_checkpoint_parameter_like,
        "trainable_ratio_over_adapted_dense_base": trainable_count / adapted_dense_base,
        "trainable_ratio_over_full_checkpoint_parameter_like": trainable_count
        / full_checkpoint_parameter_like,
        "frozen_contract": manifest["frozen"],
        "checks": checks,
        "pass": all(checks.values()),
    }


def select_split_hook(config: dict[str, Any], split: str) -> dict[str, Any]:
    item = config["phase23_contract"][split]
    spec = GoldSplitSpec(
        split,
        Path(item["path"]),
        item["sha256"],
        item["key_prefix"],
        bool(item["optimizer_eligible"]),
    )
    return ImmutableGoldMotionLibHook(spec).validate_source()


def _source_checks(config: dict[str, Any]) -> dict[str, bool]:
    source_cfg = yaml.safe_load(Path(config["source"]["config"]).read_text())
    algo = source_cfg["algo"]["config"]
    ppo = config["ppo"]
    ppo_keys = tuple(ppo)
    ppo_exact = all(algo[key] == ppo[key] for key in ppo_keys)
    motion = source_cfg["manager_env"]["commands"]["motion"]
    reward = config["reward_semantics"]
    source_rewards = source_cfg["manager_env"]["rewards"]
    reward_terms_exact = True
    for name, expected in reward["terms"].items():
        actual = source_rewards[name]
        reward_terms_exact &= str(actual["func"]).endswith(":" + expected["func"])
        reward_terms_exact &= actual["weight"] == expected["weight"]
        if "std" in expected:
            reward_terms_exact &= actual["params"]["std"] == expected["std"]
        if "threshold" in expected:
            reward_terms_exact &= actual["params"]["threshold"] == expected["threshold"]
    events = source_cfg["manager_env"]["events"]
    dr = config["domain_randomization"]
    dr_exact = (
        events["physics_material"]["params"]["static_friction_range"]
        == dr["physics_material"]["static_friction_range"]
        and events["physics_material"]["params"]["dynamic_friction_range"]
        == dr["physics_material"]["dynamic_friction_range"]
        and events["physics_material"]["params"]["restitution_range"]
        == dr["physics_material"]["restitution_range"]
        and events["add_joint_default_pos"]["params"]["pos_distribution_params"]
        == dr["joint_default_position"]["distribution"]
        and events["base_com"]["params"]["com_range"]
        == {key: dr["torso_com"][key] for key in ("x", "y", "z")}
        and events["push_robot"]["interval_range_s"]
        == dr["push_velocity"]["interval_range_s"]
        and events["randomize_rigid_body_mass"]["params"]["mass_distribution_params"]
        == dr["selected_body_mass_scale"]["distribution"]
    )
    return {
        "source_checkpoint_hash": sha256(Path(config["source"]["checkpoint"]))
        == config["source"]["checkpoint_sha256"],
        "source_config_hash": sha256(Path(config["source"]["config"]))
        == config["source"]["config_sha256"],
        "ppo_exact": ppo_exact,
        "source3_exact": motion["reward_point_body"] == reward["source3_body_names"]
        and motion["reward_point_body_offset"] == reward["source3_body_offsets"],
        "source14_exact": motion["body_names"] == reward["source14_body_names"],
        "reward_terms_exact": bool(reward_terms_exact),
        "source_equivalent_dr_exact": bool(dr_exact),
        "std_load_true_frozen": config["source"]["std"]
        == {"source_key": "std", "load": True, "trainable": False},
    }


def render(report: dict[str, Any]) -> str:
    a = report["lora_manifests"]["sonic_decoder_critic_primary"]
    b = report["lora_manifests"]["figure7_exact_s7"]
    lines = [
        "# X2 Faithful Any2Any Entrypoint — Phase24",
        "",
        f"- 裁决：**{report['decision']['status']}**。",
        "- 仅静态审计、Gold hook选择和CPU decoder/critic forward；没有Isaac physics、optimizer、PPO update或网络保存。",
        "- B5尚未冻结，因此live zero-update由launcher以exit 42失败关闭。",
        "",
        "## 假设",
        "",
        "B3/B4可以先作为不可变训练合同完成，而不把未冻结的目标物理资产默认为正确。SONIC论文范围与Figure7 S7必须保留为两条独立对照。",
        "",
        "## 干预与对照",
        "",
        "- 干预：锁定source PPO、source3/source14 reward语义、source-equivalent DR、冻结std和Gold split hook。",
        "- 对照A：SONIC actor dynamics decoder + critic。",
        "- 对照B：Figure7 exact S7，actor input只开放proprio列，critic input/output冻结。",
        "",
        "## 结果",
        "",
        f"- source/config checks：`{report['static_checks']}`。",
        f"- train keys：`{report['entrypoints']['train_static']['sampler_keys']}`；held keys：`{report['entrypoints']['eval_static']['sampler_keys']}`。",
        f"- A trainable：{a['trainable_param_count']}，占adapted dense `{a['trainable_ratio_over_adapted_dense_base']:.6f}`；CPU checks `{a['checks']}`。",
        f"- B trainable：{b['trainable_param_count']}，占adapted dense `{b['trainable_ratio_over_adapted_dense_base']:.6f}`；proprio mask `{b['input_masks']}`；CPU checks `{b['checks']}`。",
        "- 两条manifest的完整trainable names/shapes/masks在同名JSON中；std、reference encoders、kinematic decoder、FSQ和其他dense均声明冻结。",
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
    parser.add_argument(
        "--mode",
        choices=("audit", "train-static", "eval-static", "live-zero"),
        default="audit",
    )
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    gate = config["physics_gate"]
    if args.mode == "live-zero" and not gate["b5_target_physics_contract_frozen"]:
        print(
            json.dumps(
                {
                    "status": "REFUSED_B5_NOT_FROZEN",
                    "exit_code": gate["fail_closed_exit_code"],
                }
            )
        )
        return int(gate["fail_closed_exit_code"])
    if args.mode in ("train-static", "eval-static"):
        split = "train" if args.mode == "train-static" else "held_out"
        hook = select_split_hook(config, split)
        print(json.dumps({"status": "STATIC_HOOK_READY", "mode": args.mode, **hook}))
        return 0

    static_checks = _source_checks(config)
    train = select_split_hook(config, "train")
    held = select_split_hook(config, "held_out")
    split_checks = {
        "train_optimizer_eligible": train["optimizer_eligible"] is True,
        "held_optimizer_forbidden": held["optimizer_eligible"] is False,
        "sampler_keys_disjoint": not bool(set(train["sampler_keys"]) & set(held["sampler_keys"])),
    }
    checkpoint = _load_source_checkpoint(Path(config["source"]["checkpoint"]))
    manifests = {
        name: instantiate_manifest(
            name,
            item,
            checkpoint["policy_state_dict"],
            checkpoint["value_state_dict"],
        )
        for name, item in config["lora_manifests"].items()
    }
    passed = (
        all(static_checks.values())
        and all(split_checks.values())
        and all(item["pass"] for item in manifests.values())
        and gate["live_zero_update_allowed"] is False
    )
    report = {
        "schema_version": "x2_faithful_any2any_phase24_v1",
        "provenance": {
            "config": {"path": str(args.config), "sha256": sha256(args.config)},
            "launcher": str(REPO / "scripts/run_x2_faithful_any2any_phase24.sh"),
            "phase23_implementation": config["phase23_contract"]["implementation"],
        },
        "truth_boundary": {
            "static_and_cpu_forward_only": True,
            "physics_optimizer_ppo_update_checkpoint_write": False,
            "live_zero_update_allowed": False,
            "live_train_eval_wiring_status": "static hook explicit; physics path fail-closed pending B5",
        },
        "static_checks": static_checks,
        "entrypoints": {
            "train_static": train,
            "eval_static": held,
            "split_checks": split_checks,
        },
        "lora_manifests": manifests,
        "physics_gate": gate,
        "decision": {
            "status": "B3_B4_IMPLEMENTATION_READY_B5_FAIL_CLOSED" if passed else "B3_B4_REJECTED",
            "result": (
                "Faithful config, split-explicit entrypoints and two independent LoRA manifests pass static/CPU forward checks."
                if passed
                else "At least one faithful config, split or LoRA manifest check failed."
            ),
            "conclusion": (
                "B3/B4 are implementation-ready; this is not a live zero-update pass because B5 remains deliberately unresolved."
                if passed
                else "Do not proceed to B5 or live zero-update until the failed static check is fixed."
            ),
            "next_step": (
                "Freeze B5 target physics provenance, then wire the same immutable config into the preregistered live zero-update gate."
                if passed
                else "Stop without Isaac or optimizer expansion."
            ),
        },
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    args.markdown.write_text(render(report))
    print(json.dumps({"status": report["decision"]["status"], "manifests": len(manifests)}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
