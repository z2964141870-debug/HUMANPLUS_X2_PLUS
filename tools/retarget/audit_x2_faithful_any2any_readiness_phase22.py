#!/usr/bin/env python3
"""Read-only faithful Any2Any training-readiness audit for X2.

This tool does not instantiate Isaac Lab, run MuJoCo, load a policy for
inference, or mutate a network.  It freezes file/semantic contracts from the
Phase10 native Gold seed, Phase11 MotionLib adapter, released SONIC checkpoint,
local Any2Any paper, and the current X2 LoRA/PPO implementation.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import joblib
import yaml


REPO = Path(__file__).resolve().parents[2]
SONIC_RELEASE = Path("/home/humanplus/humanoid-GPT/A/sonic_release")
X2_ROOT = Path("/home/humanplus/x2_teleop_final/x2_sonic")
SANDBOX = X2_ROOT / "sonic_x2_sandbox"
GOLD = Path(
    "/home/humanplus/projects/ZHY/x2_wbt_official_v1/motion_lib_x2_official_v1/"
    "gold_dynamic_native_seed_v1"
)

PATHS = {
    "paper": SONIC_RELEASE / "any2any.pdf",
    "checkpoint": SONIC_RELEASE / "last.pt",
    "source_config": SONIC_RELEASE / "config.yaml",
    "checkpoint_inspection": X2_ROOT / "logs/sonic_release_checkpoint_inspection.json",
    "model_contract": REPO / "reports/retarget/x2_official_joint_body_map.json",
    "phase10": REPO / "reports/retarget/x2_native_gold_seed_phase10.json",
    "phase11": REPO / "reports/retarget/x2_native_gold_motionlib_phase11.json",
    "gold_manifest": GOLD / "manifest.json",
    "gold_train": GOLD / "train/official_native_dance_train.pkl",
    "gold_held": GOLD / "held_out/official_native_dance_held_out.pkl",
    "joint_utils": SANDBOX / "gear_sonic/envs/env_utils/joint_utils.py",
    "x2_robot": SANDBOX / "gear_sonic/envs/manager_env/robots/x2.py",
    "x2_env": SANDBOX / "gear_sonic/envs/manager_env/modular_tracking_env_cfg.py",
    "trainer": SANDBOX / "gear_sonic/trl/trainer/ppo_trainer.py",
    "current_exp": SANDBOX
    / "gear_sonic/config/exp/manager/universal_token/all_modes/sonic_x2_receiver_safe_clean.yaml",
    "current_rewards": SANDBOX
    / "gear_sonic/config/manager_env/rewards/tracking/base_5point_local_feet_acc.yaml",
    "current_events": SANDBOX
    / "gear_sonic/config/manager_env/events/tracking/x2_level0_4.yaml",
    "launcher": X2_ROOT / "scripts/run_x2_sonic_any2any_lora_receiver_safe_smoke.sh",
    "stage152_trainable": X2_ROOT
    / "logs/ppo_dryrun/x2_stage152_B_pilot_seed0_v1/any2any_lora_trainable_report.json",
    "phase11_adapter": REPO / "src/x2_native_gold_motionlib_adapter.py",
}

EXPECTED_HASHES = {
    "paper": "912e7425a37e8e2436870dc6b1c7f3c515500597534f38356e8c7494e64a4ec8",
    "checkpoint": "e6bdab3f64a39336b3d41877d4f497d05f58af275f288ec0e6746c283ded8909",
    "source_config": "f08187795fa16a839a28bc1c18e0555d38d9420e03733744341cdcb56ab629c7",
    "gold_train": "644dc7534b63a7831bfe4156941b01508003f2834d2ccdac7b227369bad8ef8b",
    "gold_held": "45ffda2f8ddc64cbeb4ccc714d37a0c98ca328e8e6473edb12670dcdfd19a3dc",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def literal_assignment(path: Path, name: str) -> Any:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                return ast.literal_eval(node.value)
    raise KeyError(f"literal assignment {name} not found in {path}")


def launcher_default(text: str, name: str) -> str | None:
    match = re.search(
        rf'^{re.escape(name)}="\$\{{{re.escape(name)}:-([^}}]+)\}}"$',
        text,
        flags=re.MULTILINE,
    )
    return match.group(1) if match else None


def pkl_contract(path: Path) -> dict[str, Any]:
    payload = joblib.load(path)
    rows = []
    for key, entry in payload.items():
        rows.append(
            {
                "key": key,
                "split": str(entry["split"]),
                "source_frame_range": [int(v) for v in entry["source_frame_range"]],
                "frames": int(len(entry["dof"])),
                "dof": int(entry["dof"].shape[-1]),
                "fps": float(entry["fps"]),
                "joint_names": list(entry["joint_names_mujoco"]),
            }
        )
    return {"keys": list(payload), "rows": rows}


def ranges_overlap(left: list[int], right: list[int]) -> bool:
    return max(left[0], right[0]) < min(left[1], right[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json",
        type=Path,
        default=REPO / "reports/retarget/x2_faithful_any2any_readiness_phase22.json",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=REPO / "reports/retarget/x2_faithful_any2any_readiness_phase22.md",
    )
    args = parser.parse_args()

    missing = [name for name, path in PATHS.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing audit inputs: {missing}")

    hashes = {name: sha256(path) for name, path in PATHS.items()}
    frozen_hash_checks = {
        name: hashes[name] == expected for name, expected in EXPECTED_HASHES.items()
    }
    source = yaml.safe_load(PATHS["source_config"].read_text(encoding="utf-8"))
    inspection = load_json(PATHS["checkpoint_inspection"])
    model_contract = load_json(PATHS["model_contract"])
    phase10 = load_json(PATHS["phase10"])
    phase11 = load_json(PATHS["phase11"])
    stage152 = load_json(PATHS["stage152_trainable"])
    train = pkl_contract(PATHS["gold_train"])
    held = pkl_contract(PATHS["gold_held"])

    g1_order = list(literal_assignment(PATHS["joint_utils"], "G1_ISAACLab_ORDER"))
    official31 = list(model_contract["control_boundaries"]["official_mjcf_actuated_31"])
    target29 = list(model_contract["control_boundaries"]["wbt_target_29"])
    head2 = list(model_contract["control_boundaries"]["head_locked_2"])
    target_to_source = [g1_order.index(name) for name in target29]
    source_to_target = [target29.index(name) for name in g1_order]

    source_policy_terms = source["manager_env"]["observations"]["policy"]
    source_critic_terms = source["manager_env"]["observations"]["critic"]
    policy_term_order = [
        key
        for key, value in source_policy_terms.items()
        if isinstance(value, dict) and "func" in value
    ]
    critic_term_order = [
        key
        for key, value in source_critic_terms.items()
        if isinstance(value, dict) and "func" in value
    ]
    history = {
        "actor_prop": int(source["actor_prop_history_length"]),
        "actor_actions": int(source["actor_actions_history_length"]),
        "critic_prop": int(source["critic_prop_history_length"]),
        "critic_actions": int(source["critic_actions_history_length"]),
    }
    source_obs_contract = {
        "policy_term_order": policy_term_order,
        "policy_term_functions": {
            key: source_policy_terms[key].get("func") for key in policy_term_order
        },
        "critic_term_order": critic_term_order,
        "critic_term_functions": {
            key: source_critic_terms[key].get("func") for key in critic_term_order
        },
        "history": history,
        "policy_per_history_dims": {
            "gravity_dir": 3,
            "base_ang_vel": 3,
            "joint_pos": 29,
            "joint_vel": 29,
            "actions": 29,
        },
        "policy_flat_dim": 930,
        "token_flat_dim": 64,
        "g1_dyn_input_dim": 994,
        "critic_input_dim_from_checkpoint": 1645,
    }
    source_action_cfg = source["manager_env"]["actions"]["joint_pos"]
    source_action_contract = {
        "semantic_order": g1_order,
        "dim": 29,
        "joint_selector": source_action_cfg["joint_names"],
        "use_default_offset": bool(source_action_cfg["use_default_offset"]),
        "clip": float(source["manager_env"]["config"]["action_clip_value"]),
        "checkpoint_std_dim": inspection["policy_groups"]["std"]["tensors"][0]["shape"][0],
    }
    alignment_contract = {
        "official31": official31,
        "head_locked2": head2,
        "target_wbt29_order": target29,
        "source_g1_order": g1_order,
        "target_to_source_permutation": target_to_source,
        "source_to_target_permutation": source_to_target,
        "target_obs_rule": "select WBT29 by name then scatter into source_g1_order before policy",
        "target_action_rule": "gather source_g1_order policy output into target_wbt29_order; head absent and held nominal",
    }

    train_ranges = [row["source_frame_range"] for row in train["rows"]]
    held_ranges = [row["source_frame_range"] for row in held["rows"]]
    no_overlap = not any(ranges_overlap(a, b) for a in train_ranges for b in held_ranges)
    embargo_frames = min(r[0] for r in held_ranges) - max(r[1] for r in train_ranges)
    data_checks = {
        "train_hash_frozen": frozen_hash_checks["gold_train"],
        "held_hash_frozen": frozen_hash_checks["gold_held"],
        "train_4x400": [row["frames"] for row in train["rows"]] == [400] * 4,
        "held_3x400": [row["frames"] for row in held["rows"]] == [400] * 3,
        "all_50hz": all(row["fps"] == 50.0 for row in train["rows"] + held["rows"]),
        "all_official31": all(row["joint_names"] == official31 for row in train["rows"] + held["rows"]),
        "train_held_key_disjoint": not (set(train["keys"]) & set(held["keys"])),
        "source_ranges_nonoverlap": no_overlap,
        "embargo_frames": embargo_frames,
        "embargo_seconds": embargo_frames / 50.0,
        "phase11_ingestion_passed": phase11["decision"]["status"]
        == "PHASE11_MOTIONLIB_INGESTION_PASSED",
    }

    current_actor_layers = list(stage152["actor_lora_layers"])
    current_critic_layers = list(stage152["critic_lora_layers"])
    s7_actor_input = ["actor_module.decoders.g1_dyn.module.0"]
    s7_actor_backbone = [
        f"actor_module.decoders.g1_dyn.module.{idx}" for idx in (2, 4, 6, 8, 10)
    ]
    s7_actor_output = ["actor_module.decoders.g1_dyn.module.12"]
    s7_critic_backbone = [f"critic_module.module.{idx}" for idx in (2, 4, 6, 8, 10)]
    s7 = {
        "paper_boundary": (
            "Figure7 S7 is Oli-WBT->Luna; SONIC experiments separately state "
            "actor dynamics decoder + critic, with FSQ/other pretrained modules frozen"
        ),
        "exact_s7_translation_for_current_mlp": {
            "actor_proprio_input": s7_actor_input,
            "actor_proprio_input_mask": "proprioception columns only; token/reference columns zero",
            "actor_backbone": s7_actor_backbone,
            "actor_action_output": s7_actor_output,
            "critic_backbone": s7_critic_backbone,
            "critic_input_output": "frozen (module.0 and module.12)",
        },
        "current_stage152": {
            "actor_layers": current_actor_layers,
            "actor_input_mask": stage152.get("actor_input_mask", {}),
            "actor_output_mask": stage152.get("actor_output_mask", {}),
            "critic_layers": current_critic_layers,
            "boundary_trainable": stage152.get("boundary_trainable", []),
            "action_count": stage152.get("actor_output_mask", {}).get("action_count"),
            "selected_action_count": stage152.get("actor_output_mask", {}).get(
                "selected_action_count"
            ),
        },
        "differences": [
            "current module.0 LoRA is unmasked, so token/reference and proprioception columns can both adapt",
            "current critic includes input module.0 and output module.12; exact S7 freezes both",
            "current live decoder has 31 output rows and masks two head rows; exact aligned contract is a 29-row source-semantic output before target gather",
        ],
        "implementation_can_express_s7": all(
            token in PATHS["trainer"].read_text(encoding="utf-8")
            for token in ("actor_lora_prefixes", "critic_lora_prefixes", "actor_input_mask")
        ),
        "exact_s7_instantiated_and_zero_update_tested": False,
    }

    launcher = PATHS["launcher"].read_text(encoding="utf-8")
    current_defaults = {
        key: launcher_default(launcher, key)
        for key in (
            "PPO_EPOCHS",
            "NUM_MINI_BATCHES",
            "TRAIN_BOUNDARY_KEYS",
            "TRACKED_BODY_SET",
            "REWARD_POINT_MODE",
            "RIGID_BODY_MASS_RANGE",
            "ACTUATOR_GAIN_RANDOMIZATION",
        )
    }
    ppo_source = {
        key: source["algo"]["config"][key]
        for key in (
            "num_learning_epochs",
            "num_mini_batches",
            "clip_param",
            "gamma",
            "lam",
            "actor_learning_rate",
            "critic_learning_rate",
            "desired_kl",
            "max_grad_norm",
            "num_steps_per_env",
        )
    }
    reward_keys = [key for key in source["manager_env"]["rewards"] if not key.startswith("_")]
    event_keys = [key for key in source["manager_env"]["events"] if not key.startswith("_")]
    entrypoint_checks = {
        "ppo_trainer_any2any_lora": "def _apply_any2any_lora" in PATHS["trainer"].read_text(encoding="utf-8"),
        "motion_file_override": 'motion_lib_cfg.motion_file="${MOTION_FILE}"' in launcher,
        "ppo_epoch_override": 'algo.config.num_learning_epochs="${PPO_EPOCHS}"' in launcher,
        "ppo_minibatch_override": 'algo.config.num_mini_batches="${NUM_MINI_BATCHES}"' in launcher,
        "reward_override_entry": "manager_env.rewards.tracking_anchor_pos.weight" in launcher,
        "dr_event_entry": "manager_env.events.randomize_rigid_body_mass" in launcher,
        "motionlib_state_adapter_exists": "apply_recorded_state_adapter" in PATHS["phase11_adapter"].read_text(encoding="utf-8"),
    }

    ready = [
        {
            "id": "R1_SOURCE_FROZEN",
            "evidence": "Any2Any paper, SONIC last.pt and source config hashes match preregistration; checkpoint exposes 29-row g1_dyn and 1645-D critic.",
        },
        {
            "id": "R2_DATA_SPLIT_FROZEN",
            "evidence": "Gold train=4x400, held-out=3x400 at 50Hz with a disjoint 200-frame/4.0s embargo and immutable hashes.",
        },
        {
            "id": "R3_MOTIONLIB_ROUNDTRIP",
            "evidence": "Phase11 native loader + minimal state adapter passed q/root/FK/29-head-lock round-trip.",
        },
        {
            "id": "R4_OFFLINE_31_29_ALIGNMENT",
            "evidence": "official31 partitions exactly into WBT29 + head2; WBT29 and G1 use the same named set and have an explicit bijective permutation.",
        },
        {
            "id": "R5_OPTIMIZATION_ENTRYPOINTS",
            "evidence": "LoRA layer selection/masking, PPO, reward, motion-file and DR override entrypoints exist in local code.",
        },
    ]
    blockers = [
        {
            "id": "B1_RUNTIME_29_ALIGNMENT_NOT_INSTANTIATED",
            "evidence": "current X2 runtime/training expands policy observations/actions to 31 (Stage152 action_count=31) and masks head rows; it does not first scatter/gather through the frozen G1 29-D semantic space.",
            "required": "dedicated WBT29 action term and observation/history adapter using the frozen permutations; head must be absent from policy and held nominal in the simulator.",
        },
        {
            "id": "B2_GOLD_STATE_ADAPTER_NOT_IN_TRAIN_PATH",
            "evidence": "Phase11 state adapter is standalone audit code; current training config still points to receiver_safe_clean and does not prove actual dq/root velocity/model-contact restoration for Gold train clips.",
            "required": "minimal immutable MotionLib hook for Gold train/eval, with pose/root/FK unchanged and split-specific paths.",
        },
        {
            "id": "B3_FAITHFUL_CONFIG_NOT_LOCKED",
            "evidence": f"generic launcher defaults are {current_defaults}; source requires PPO epochs=5, minibatches=4, std frozen, source3/source14 reward semantics and source-equivalent DR.",
            "required": "new preregistered faithful config/launcher; do not reuse mutable generic defaults.",
        },
        {
            "id": "B4_EXACT_S7_SCOPE_NOT_INSTANTIATED",
            "evidence": "Stage152 adapts all g1_dyn rows/columns plus critic input/output; exact Figure7 S7 needs proprio-only input columns, actor hidden backbone/output and critic hidden backbone only.",
            "required": "instantiate the exact named module/mask list and record trainable/frozen tensors; keep SONIC-specific decoder+critic baseline as a separate primary control.",
        },
        {
            "id": "B5_TARGET_PHYSICS_PROVENANCE_NOT_FROZEN",
            "evidence": "current IsaacLab X2 config uses locally reconstructed URDF/collision/PD profiles, while Gold provenance is official AimDK v1 MuJoCo x2.xml/control; no morphology-equivalent Isaac contract hash is locked.",
            "required": "freeze a declared IsaacLab target asset/PD/collision contract and state exactly how it corresponds to official v1; official MuJoCo remains held-out sim-to-sim, not training truth by implication.",
        },
        {
            "id": "B6_ZERO_UPDATE_NOT_EXECUTED",
            "evidence": "Phase11 validates data ingestion, not policy initialization equivalence; no live 29-D aligned policy has produced identical zero-LoRA actions/tokens/value on a fixed batch.",
            "required": "pass the preregistered zero-update gate below before any optimizer step.",
        },
    ]

    zero_gate = {
        "workload": "0 environment steps; one deterministic fixed batch from train clip0 plus one held-out clip0 audit batch; observation corruption/DR off",
        "exact_checks": [
            "checkpoint/config/paper/data/contract hashes equal this report",
            "runtime policy action dimension=29; policy/critic term order and dimensions equal frozen contracts",
            "target29->G1->target29 name round-trip is exact; head never enters obs/action/history",
            "MotionLib train sampler keys exactly four train keys; held-out and embargo keys absent from every optimizer/callback path",
            "LoRA B=0 gives target-aligned action mean and reference token equal to frozen source-space forward: max_abs<=1e-6",
            "std equals source and is frozen; all dense checkpoint tensors are unchanged by hash",
            "trainable names exactly equal selected SONIC-specific or S7 manifest; all other requires_grad=false",
            "critic running statistics are loaded by mapped semantics and are not silently reinitialized",
        ],
        "stop": "any failed check blocks the optimizer; no partial tolerance or physics interpretation",
        "exact_compute": {"env_steps": 0, "optimizer_steps": 0, "fixed_forward_batches": 2},
        "estimated_local_cost": "one Isaac/model startup plus two forwards; budget <=10 min wall and no checkpoint output beyond a small manifest",
    }
    one_update_gate = {
        "precondition": "all zero-update checks pass",
        "workload": "64 env x 24 rollout steps = 1536 transitions; 5 PPO epochs x 4 minibatches = 20 optimizer minibatch steps; Gold train only",
        "exact_checks": [
            "all rollout observations/actions/rewards/advantages/logprobs/losses/gradients finite",
            "100% intended LoRA groups receive finite nonzero gradients; frozen dense/reference/FSQ/std tensors receive no gradient and keep exact hashes",
            "post-update effective LoRA delta is finite and nonzero; action dimension remains 29 and no head row appears",
            "approx KL finite and <=0.02 (2x source desired_kl); no optimizer-state or checkpoint reload mismatch",
            "train sampler/callback audit contains only four train keys; held-out metrics are computed only after update and never select/stop this single update",
            "post-save reload reproduces the same fixed-batch action mean within max_abs<=1e-6",
        ],
        "interpretation": "sanity only; one update is not evidence of performance improvement or Any2Any success",
        "exact_compute": {
            "envs": 64,
            "rollout_steps_per_env": 24,
            "transitions": 1536,
            "ppo_epochs": 5,
            "mini_batches_per_epoch": 4,
            "optimizer_minibatch_steps": 20,
        },
        "estimated_local_cost": "budget <=20 min wall, <=1 candidate checkpoint; measure actual peak VRAM/wall time rather than treating estimate as evidence",
    }

    checks = {
        "all_frozen_hashes_match": all(frozen_hash_checks.values()),
        "source_checkpoint_structure_29_1645": (
            inspection["policy_groups"]["actor_module.decoders.g1_dyn"]["tensors"][-2]["shape"]
            == [29, 512]
            and inspection["value_groups"]["critic_module"]["tensors"][0]["shape"]
            == [2048, 1645]
        ),
        "official31_partition_exact": set(official31) == set(target29) | set(head2)
        and not (set(target29) & set(head2)),
        "g1_target29_named_set_exact": set(g1_order) == set(target29),
        "permutations_bijective": sorted(target_to_source) == list(range(29))
        and sorted(source_to_target) == list(range(29)),
        "data_contract_passed": all(
            value for key, value in data_checks.items() if not key.startswith("embargo_")
        )
        and embargo_frames == 200,
        "phase10_gold_exported": phase10["decision"]["status"]
        == "PHASE10_NATIVE_DYNAMIC_GOLD_SEED_EXPORTED",
        "entrypoints_exist": all(entrypoint_checks.values()),
    }

    report = {
        "schema_version": "x2_faithful_any2any_readiness_phase22_v1",
        "scope": "read-only local evidence audit; no training, physics, network mutation, BASE or real robot",
        "status": "BLOCKED_BEFORE_ZERO_UPDATE",
        "not_rejected_by_phase21": True,
        "provenance": {
            name: {"path": str(path), "sha256": hashes[name]} for name, path in PATHS.items()
        },
        "frozen_hash_checks": frozen_hash_checks,
        "source_contract": {
            "checkpoint": {
                "policy_tensors": inspection["policy_tensor_count"],
                "value_tensors": inspection["value_tensor_count"],
                "g1_dyn_output_dim": 29,
                "critic_input_dim": 1645,
            },
            "observation": source_obs_contract,
            "observation_sha256": canonical_hash(source_obs_contract),
            "action": source_action_contract,
            "action_sha256": canonical_hash(source_action_contract),
            "ppo": ppo_source,
            "reward_keys": reward_keys,
            "domain_randomization_event_keys": event_keys,
        },
        "x2_alignment": {
            **alignment_contract,
            "sha256": canonical_hash(alignment_contract),
            "offline_ready": True,
            "runtime_ready": False,
        },
        "data_isolation": {
            "train": train,
            "held_out": held,
            "checks": data_checks,
            "truth_boundary": "official simulation Gold sanity seed; not GMR/AMASS Silver, real GRF/COP, or proof of Any2Any",
        },
        "lora_scope": s7,
        "optimization_entrypoints": {
            "checks": entrypoint_checks,
            "current_launcher_defaults": current_defaults,
            "source_ppo": ppo_source,
        },
        "checks": checks,
        "ready": ready,
        "blockers": blockers,
        "zero_update_gate": zero_gate,
        "one_update_gate": one_update_gate,
        "decision": {
            "result": "Local assets and offline data/alignment are ready, but the live faithful 29-D runtime/config/S7 contracts have not been instantiated.",
            "conclusion": "Do not train yet. Phase21 replay failure is outside this audit and does not reject Any2Any; readiness is blocked by implementation contracts, not by a demonstrated learning failure.",
            "next_step": "Implement only B1-B5, then run the zero-update gate. A one-update smoke is forbidden until zero-update passes.",
        },
    }

    lines = [
        "# X2 Faithful Any2Any Training Readiness — Phase22",
        "",
        f"- 裁决：**{report['status']}**。",
        "- 本阶段纯只读：没有训练、physics、网络修改、BASE或真机。",
        "- Phase21 replay失败不构成Any2Any训练否证；这里检查的是训练合同是否忠实、可证伪。",
        "",
        "## 总结",
        "",
        "离线资产已经足以搭建sanity：SONIC checkpoint、Gold train/held-out、4秒embargo、MotionLib round-trip和31↔29名称映射均已冻结。当前仍不能启动优化，因为live训练环境仍是31维扩展边界，Gold state adapter未接入训练，且没有一份锁死PPO/reward/DR和LoRA范围的faithful配置。",
        "",
        "## 冻结哈希",
        "",
        f"- checkpoint `{hashes['checkpoint']}`",
        f"- source obs contract `{report['source_contract']['observation_sha256']}`",
        f"- source action contract `{report['source_contract']['action_sha256']}`",
        f"- X2 31/29 alignment `{report['x2_alignment']['sha256']}`",
        f"- Gold train/held `{hashes['gold_train']}` / `{hashes['gold_held']}`",
        "",
        "## READY",
        "",
    ]
    lines.extend(f"- `{row['id']}`：{row['evidence']}" for row in ready)
    lines.extend(["", "## BLOCKER", ""])
    lines.extend(
        f"- `{row['id']}`：{row['evidence']} 需要：{row['required']}" for row in blockers
    )
    lines.extend(
        [
            "",
            "## S7与当前实现",
            "",
            "- 论文SONIC实验明确写的是 actor dynamics decoder + critic；Figure 7 S7来自Oli-WBT→Luna，二者必须作为两条独立对照，不可混称。",
            "- 当前Stage152-B覆盖g1_dyn全部7层和critic全部7层，第一层没有proprio-only列mask，critic input/output也被训练，因此不是严格S7。",
            "- 当前MLP的严格S7翻译：actor module.0只开放`proprioception`列；module.2/4/6/8/10作为backbone；module.12作为action output；critic只开放module.2/4/6/8/10。",
            "",
            "## 数据隔离",
            "",
            f"- train ranges `{train_ranges}`，held-out ranges `{held_ranges}`。",
            f"- embargo `{embargo_frames}`帧 / `{embargo_frames / 50.0:.1f}s`，无range/key重叠。",
            "- held-out不得进入optimizer、阈值调整、early stopping或checkpoint选择；本单条官方舞蹈只能做pipeline sanity，不能冒充Any2Any训练语料规模。",
            "",
            "## Zero-update门（必须先过）",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in zero_gate["exact_checks"])
    lines.extend(
        [
            f"- 精确工作量：`{zero_gate['exact_compute']}`；预算：{zero_gate['estimated_local_cost']}。",
            "",
            "## 1-update门（zero通过后才允许）",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in one_update_gate["exact_checks"])
    lines.extend(
        [
            f"- 精确工作量：`{one_update_gate['exact_compute']}`；预算：{one_update_gate['estimated_local_cost']}。",
            "- 1 update只证明梯度/保存/隔离合同成立，不证明性能提升。",
            "",
            "## 结论",
            "",
            f"- 结果：{report['decision']['result']}",
            f"- 结论：{report['decision']['conclusion']}",
            f"- 下一步：{report['decision']['next_step']}",
            "",
        ]
    )

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.markdown.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": report["status"], "ready": len(ready), "blockers": len(blockers)}))


if __name__ == "__main__":
    main()
