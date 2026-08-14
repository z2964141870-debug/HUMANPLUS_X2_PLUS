#!/usr/bin/env python3
"""One bounded DAgger round for the fixed-latent X2 scratch actor."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--round2", action="store_true")
parser.add_argument("--observation-contract-v3", action="store_true")
parser.add_argument("--joint-contract-v4", action="store_true")
parser.add_argument("--joint-contract-v4-round2", action="store_true")
parser.add_argument("--response-contract-v5", action="store_true")
parser.add_argument("--response-contract-v5-round2", action="store_true")
parser.add_argument("--phase-contract-v6", action="store_true")
parser.add_argument("--phase-response-v6", action="store_true")
parser.add_argument(
    "--phase-response-teacher-v7",
    action="store_true",
    help=(
        "collect one full response-domain trajectory with the Stage219 teacher "
        "executing the actions, then fit the scratch actor without another "
        "student-policy DAgger round"
    ),
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.modules import ActorCritic  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from gear_sonic.envs.x2_velocity import X2LowerVelocityFlatEnvCfg_PLAY  # noqa: E402
from gear_sonic.envs.x2_velocity.gait import gait_phase_observation  # noqa: E402
from gear_sonic.envs.manager_env.modular_tracking_env_cfg import (  # noqa: E402
    _apply_x2_actuator_response,
)
from gear_sonic.envs.manager_env.robots.x2 import (  # noqa: E402
    X2_URDF_BY_COLLISION_PROFILE,
)

from humanoidverse.agents.envs.x2_isaaclab import X2IsaacLabVectorEnv  # noqa: E402
from humanoidverse.agents.load_utils import load_agent_from_checkpoint_dir  # noqa: E402
from humanoidverse.x2_scratch import (  # noqa: E402
    ACTION_DIM,
    X2_LOWER_JOINTS_15,
    X2_SCRATCH_ACTION_SCALE_15,
)
from scripts.train_x2_scratch_closed_loop_bc import (  # noqa: E402
    BATCH_SIZE,
    BUNDLE,
    BUNDLE_SHA256,
    TEMPLATE,
    TEMPLATE_SHA256,
    actor_mean,
    build_dataset,
    evaluate_mse,
    file_hash,
    model_hash,
    reconstruct_stage219_direct_action_labels,
    tree_hash,
)


NUM_ENVS = 512
teacher_response_v7 = args.phase_response_teacher_v7
COLLECT_STEPS = 400 if teacher_response_v7 else 128
if sum((args.round2, args.observation_contract_v3, args.joint_contract_v4, args.joint_contract_v4_round2, args.response_contract_v5, args.response_contract_v5_round2, args.phase_contract_v6, args.phase_response_v6, teacher_response_v7)) > 1:
    raise ValueError("select one DAgger contract")
response_v5 = args.response_contract_v5 or args.response_contract_v5_round2
joint_v4 = args.joint_contract_v4 or args.joint_contract_v4_round2 or response_v5
phase_v6 = args.phase_contract_v6 or args.phase_response_v6 or teacher_response_v7
phase_response_v6 = args.phase_response_v6
modern_contract = args.observation_contract_v3 or joint_v4 or phase_v6
SEED = (
    (770361 if teacher_response_v7 else (770341 if phase_response_v6 else 770321))
    if phase_v6
    else (((770261 if args.response_contract_v5_round2 else 770241) if response_v5 else (770211 if args.joint_contract_v4_round2 else 770201))
    if joint_v4
    else (770181 if args.observation_contract_v3 else (770131 if args.round2 else 770121))
    )
)
EPOCHS = 2 if teacher_response_v7 else 5
SOURCE = Path(
    (
        "artifacts/x2_scratch_dagger_phase_response_v6_seed770341"
        if teacher_response_v7
        else "artifacts/x2_scratch_dagger_phase_v6_seed770321"
        if phase_response_v6
        else "artifacts/x2_scratch_phase_bc_v6_seed770301"
    )
    if phase_v6
    else (
        "artifacts/x2_scratch_dagger_response_v5_seed770241"
        if args.response_contract_v5_round2
        else ("artifacts/x2_scratch_dagger_joint_v4_round2_seed770211"
        if args.response_contract_v5
        else ("artifacts/x2_scratch_dagger_joint_v4_seed770201"
        if args.joint_contract_v4_round2
        else "artifacts/x2_scratch_closed_loop_bc_joint_v4_seed770191"
        ))
    )
    if joint_v4
    else ("artifacts/x2_scratch_closed_loop_bc_observation_v3_seed770171"
    if args.observation_contract_v3
    else (
        "artifacts/x2_scratch_dagger1_seed770121"
        if args.round2
        else "artifacts/x2_scratch_closed_loop_bc_fixed_z_seed770111"
    ))
)
SOURCE_SHA256 = (
    (
        "5252ea30061124430e25b566089ed1aaf2f1c7816b46a2e389585c942d2162b9"
        if teacher_response_v7
        else "4a48e4940e704c6dbcd1df336481ab64ba3f5613ed1cc90b3b30130fc27e4703"
        if phase_response_v6
        else "0ef2c74dedbda264fb885a434c8732aa268b3221b3a7a62d10d409ee61c191c7"
    )
    if phase_v6
    else (
        "1843dd53c80ede2d82be95c307be0c7a21709d14e3bd12444e0f3d693fd2c006"
        if args.response_contract_v5_round2
        else ("a0e91b7dc7d706099fa306fe1253077f50243c5e934b79f35b7f126afdcb9bb1"
        if args.response_contract_v5
        else ("c3d9ff0ad08022eadc03a04cc5df3b8a4db3a7dafaddd8b1f3fd2342b129120a"
        if args.joint_contract_v4_round2
        else "8c16bf2c240162d1b7d571370011f7ea93312b235b035b6cc64945c0ebfedc02"
        ))
    )
    if joint_v4
    else ("59ef6c7205d9dbe4e5dc05ceaf8583dada15524df72734c6b6db8aeb84b25603"
    if args.observation_contract_v3
    else (
        "977333c44f70d6c2d79f3ca67e93c9dd7d108aaa7bf68eda83281b3711983388"
        if args.round2
        else "bd185935d181823b31cfaa5f98553b4e1a8f352a75c4f434d1b2f4aaacd72f28"
    ))
)
CANONICAL_REPORT = Path(
    "reports/x2_scratch_phase_bc_v6.json"
    if phase_v6
    else "reports/x2_scratch_closed_loop_bc_joint_v4.json"
    if joint_v4
    else ("reports/x2_scratch_closed_loop_bc_observation_v3.json"
    if args.observation_contract_v3
    else "reports/x2_scratch_closed_loop_bc_fixed_z.json"
    )
)
CANONICAL_REPORT_SHA256 = (
    "9771b6b92c0c5c66faa64d23e0786f58fcba98fa4daedcf2aae1cc851c85d1fe"
    if phase_v6
    else "de9a49ea27d44e55a27bf1fb0f33be63d880ef242cdf0c166b4cb5c5bef300bc"
    if joint_v4
    else ("dd1a4d123ac0ad62861adb4372b29cae0cdaae4ea0782530bd315766583726a1"
    if args.observation_contract_v3
    else "00247c7d84b6654b47945bed90638ae16fc52ad5fe164692b0f0bb16a5b6e097"
    )
)
STAGE219 = Path(
    "/home/yu/x2_teleop_final/x2_sonic/logs/rsl_rl/x2_lower_velocity_flat/"
    "2026-07-22_10-19-57_stage219_cleanreset_yawcoverage25_sole12_selfoff_"
    "resume2550_to2650_v1/model_2600.pt"
)
STAGE219_SHA256 = "abcd49a823385bcd02e00480c9476ad5bc381e68252ad6930c85d731178f49bb"
OUTPUT = Path(
    (
        "artifacts/x2_scratch_response_teacher_v7_seed770361"
        if teacher_response_v7
        else "artifacts/x2_scratch_dagger_phase_response_v6_seed770341"
        if phase_response_v6
        else "artifacts/x2_scratch_dagger_phase_v6_seed770321"
    )
    if phase_v6
    else (
        "artifacts/x2_scratch_dagger_response_v5_round2_seed770261"
        if args.response_contract_v5_round2
        else ("artifacts/x2_scratch_dagger_response_v5_seed770241"
        if args.response_contract_v5
        else ("artifacts/x2_scratch_dagger_joint_v4_round2_seed770211"
        if args.joint_contract_v4_round2
        else "artifacts/x2_scratch_dagger_joint_v4_seed770201"
        ))
    )
    if joint_v4
    else ("artifacts/x2_scratch_dagger_observation_v3_seed770181"
    if args.observation_contract_v3
    else (
        "artifacts/x2_scratch_dagger2_seed770131"
        if args.round2
        else "artifacts/x2_scratch_dagger1_seed770121"
    ))
)
REPORT = Path(
    (
        "reports/x2_scratch_response_teacher_v7.json"
        if teacher_response_v7
        else "reports/x2_scratch_dagger_phase_response_v6.json"
        if phase_response_v6
        else "reports/x2_scratch_dagger_phase_v6.json"
    )
    if phase_v6
    else (
        "reports/x2_scratch_dagger_response_v5_round2.json"
        if args.response_contract_v5_round2
        else ("reports/x2_scratch_dagger_response_v5.json"
        if args.response_contract_v5
        else ("reports/x2_scratch_dagger_joint_v4_round2.json"
        if args.joint_contract_v4_round2
        else "reports/x2_scratch_dagger_joint_v4.json"
        ))
    )
    if joint_v4
    else ("reports/x2_scratch_dagger_observation_v3.json"
    if args.observation_contract_v3
    else ("reports/x2_scratch_dagger2.json" if args.round2 else "reports/x2_scratch_dagger1.json")
    )
)


def build_cfg():
    cfg = X2LowerVelocityFlatEnvCfg_PLAY()
    cfg.seed = SEED
    cfg.sim.device = args.device
    cfg.scene.num_envs = NUM_ENVS
    if modern_contract:
        cfg.scene.robot.spawn.asset_path = X2_URDF_BY_COLLISION_PROFILE["sole12"]
        cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = False
    cfg.observations.policy.enable_corruption = False
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.35, 0.35)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    cfg.actions.joint_pos.scale = {
        name: float(scale)
        for name, scale in zip(X2_LOWER_JOINTS_15, X2_SCRATCH_ACTION_SCALE_15)
    }
    if modern_contract:
        _apply_x2_actuator_response(
            cfg.scene.robot,
            {
                "enabled": True,
                "profile": "session03_session04_group",
                "randomize": False,
                "strength": 1.0,
                "filter_strength": 1.0,
                "delay_strength": 1.0,
                "include_ideal_endpoint": False,
                "ideal_env_fraction": 0.0 if (response_v5 or phase_response_v6 or teacher_response_v7) else 1.0,
                "filter_only_env_fraction": 0.0,
            },
            physics_dt_sec=cfg.sim.dt,
        )
    return cfg


def build_teacher(obs):
    teacher = ActorCritic(
        obs=obs,
        obs_groups={"policy": ["policy"], "critic": ["critic"]},
        num_actions=ACTION_DIM,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
        init_noise_std=0.4,
        noise_std_type="scalar",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
    ).to("cuda")
    payload = torch.load(STAGE219, map_location="cuda", weights_only=False)
    teacher.load_state_dict(payload["model_state_dict"], strict=True)
    return teacher.eval(), int(payload["iter"])


def teacher_observation(env, previous_raw):
    data = env.scene["robot"].data
    gait = gait_phase_observation(
        env,
        command_name="base_velocity",
        cycle_time_s=0.8,
        double_support_fraction=0.30,
    )
    value = torch.cat(
        (
            data.root_lin_vel_b,
            data.root_ang_vel_b,
            data.projected_gravity_b,
            env.command_manager.get_command("base_velocity"),
            data.joint_pos - data.default_joint_pos,
            data.joint_vel,
            previous_raw,
            gait,
        ),
        dim=-1,
    )
    if value.shape != (NUM_ENVS, 93) or not torch.isfinite(value).all():
        raise RuntimeError("reconstructed Stage219 teacher observation is invalid")
    return {"policy": value, "critic": value}


def main() -> dict:
    if (
        tree_hash(SOURCE)[0] != SOURCE_SHA256
        or file_hash(CANONICAL_REPORT) != CANONICAL_REPORT_SHA256
        or file_hash(STAGE219) != STAGE219_SHA256
        or file_hash(BUNDLE) != BUNDLE_SHA256
        or file_hash(TEMPLATE) != TEMPLATE_SHA256
    ):
        raise RuntimeError("DAgger immutable input guard failed")
    sidecar = REPORT.with_name(f"{REPORT.name}.sha256")
    temporary = OUTPUT.with_name(f".{OUTPUT.name}.tmp")
    if any(path.exists() for path in (OUTPUT, temporary, REPORT, sidecar)):
        raise FileExistsError("refusing to overwrite DAgger evidence")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    template = np.load(TEMPLATE, allow_pickle=False)
    template_q = torch.as_tensor(
        template["q_cycle_zero_mean_rad"], device="cuda"
    )
    action_scale = torch.as_tensor(template["action_scale_rad"], device="cuda")
    period_s = float(template["period_s"])
    canonical = torch.as_tensor(
        json.loads(CANONICAL_REPORT.read_text())["canonical_latent"],
        dtype=torch.float32,
        device="cuda",
    ).reshape(1, 64)
    if not torch.isfinite(canonical).all() or not math.isclose(
        float(canonical.norm()), 8.0, rel_tol=0.0, abs_tol=1.0e-4
    ):
        raise RuntimeError("canonical DAgger latent is invalid")

    student = load_agent_from_checkpoint_dir(SOURCE, device="cuda")
    source_model_hash = model_hash(student._model)
    env = ManagerBasedRLEnv(cfg=build_cfg())
    wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
    adapter = X2IsaacLabVectorEnv(
        env,
        wrapped,
        history_length=4,
        to_numpy=False,
        include_command_phase=phase_v6,
    )
    observation, _ = adapter.reset(seed=SEED)
    teacher_previous_raw = torch.zeros(NUM_ENVS, ACTION_DIM, device="cuda")
    live_obs = teacher_observation(env, teacher_previous_raw)
    teacher, teacher_iteration = build_teacher(live_obs)
    fixed_z = canonical.expand(NUM_ENVS, -1).clone()
    collected = {key: [] for key in observation if key != "time"}
    collected_actions = []
    terminated_count = 0
    safe_count = 0
    for _ in range(COLLECT_STEPS):
        current = {key: value for key, value in observation.items() if key != "time"}
        teacher_obs = teacher_observation(env, teacher_previous_raw)
        with torch.inference_mode():
            teacher_raw = teacher.act_inference(teacher_obs).clamp(-1.0, 1.0)
            phase = torch.remainder(
                env.episode_length_buf.float() * env.step_dt / period_s, 1.0
            )
            phase_position = phase * template_q.shape[0] - 0.5
            lower_unwrapped = torch.floor(phase_position)
            blend = phase_position - lower_unwrapped
            lower = lower_unwrapped.long() % template_q.shape[0]
            upper = (lower + 1) % template_q.shape[0]
            q_bias = (
                (1.0 - blend).unsqueeze(-1) * template_q[lower]
                + blend.unsqueeze(-1) * template_q[upper]
            )
            stage219_combined = torch.clamp(
                teacher_raw + 0.15 * q_bias / action_scale, -1.0, 1.0
            )
            teacher_action = (
                stage219_combined
                * action_scale
                / torch.as_tensor(X2_SCRATCH_ACTION_SCALE_15, device="cuda")
            ).clamp(-1.0, 1.0)
            student_action = student.act(current, fixed_z, mean=True).clamp(-1.0, 1.0)
        gravity = env.scene["robot"].data.projected_gravity_b
        tilt = torch.acos(torch.clamp(-gravity[:, 2], -1.0, 1.0))
        height = env.scene["robot"].data.root_pos_w[:, 2]
        safe = (height > 0.55) & (tilt < 0.50)
        safe_count += int(safe.sum())
        if safe.any():
            for key, value in current.items():
                collected[key].append(value[safe].detach().cpu())
            collected_actions.append(teacher_action[safe].detach().cpu())
        executed_action = teacher_action if teacher_response_v7 else student_action
        observation, _, terminated, truncated, _ = adapter.step(executed_action)
        done = terminated | truncated
        terminated_count += int(terminated.sum())
        teacher_previous_raw.copy_(teacher_raw)
        teacher_previous_raw[done] = 0.0
    if safe_count < NUM_ENVS * (350 if teacher_response_v7 else 32):
        raise RuntimeError("DAgger collected too few safe off-policy labels")
    if teacher_response_v7 and terminated_count != 0:
        raise RuntimeError("response-domain teacher demonstration terminated")
    dagger_observation = {key: torch.cat(parts) for key, parts in collected.items()}
    dagger_action = torch.cat(collected_actions)
    dagger_latent = canonical.cpu().expand(safe_count, -1).clone()

    payload = torch.load(BUNDLE, map_location="cpu", weights_only=False)
    direct_labels = reconstruct_stage219_direct_action_labels(
        payload["critic_observation"].numpy(),
        template["q_cycle_zero_mean_rad"],
        template["action_scale_rad"],
        template_scale=0.15,
        period_s=period_s,
    )
    offline_obs, offline_latent, offline_action, env_id, _, _ = build_dataset(
        student,
        payload["critic_observation"],
        direct_labels,
        include_command_phase=phase_v6,
    )
    offline_latent = canonical.cpu().expand_as(offline_latent).clone()
    offline_train = torch.nonzero(env_id < 48).reshape(-1)
    offline_holdout = torch.nonzero(env_id >= 48).reshape(-1)
    before = {
        "offline_holdout": evaluate_mse(
            student, offline_obs, offline_latent, offline_action, offline_holdout
        ),
        "dagger": evaluate_mse(
            student,
            dagger_observation,
            dagger_latent,
            dagger_action,
            torch.arange(safe_count),
        ),
    }

    student._model.requires_grad_(False)
    student._model._actor.requires_grad_(True)
    student._model._actor.train(True)
    optimizer = torch.optim.Adam(
        student._model._actor.parameters(),
        lr=2.0e-5 if teacher_response_v7 else 5.0e-5,
    )
    student.actor_optimizer = optimizer
    trace = []
    steps_per_epoch = int(math.ceil(max(offline_train.numel(), safe_count) / 256))
    for epoch in range(1, EPOCHS + 1):
        losses = []
        for _ in range(steps_per_epoch):
            offline_selection = offline_train[
                torch.randint(offline_train.numel(), (256,))
            ]
            dagger_selection = torch.randint(safe_count, (256,))
            batch_obs = {
                key: torch.cat(
                    (offline_obs[key][offline_selection], dagger_observation[key][dagger_selection])
                ).cuda()
                for key in offline_obs
            }
            batch_z = canonical.expand(BATCH_SIZE, -1)
            target = torch.cat(
                (offline_action[offline_selection], dagger_action[dagger_selection])
            ).cuda()
            prediction = actor_mean(student, batch_obs, batch_z)
            loss = torch.nn.functional.smooth_l1_loss(
                prediction, target, beta=0.05
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student._model._actor.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        trace.append({"epoch": epoch, "smooth_l1_mean": float(np.mean(losses))})
    after = {
        "offline_holdout": evaluate_mse(
            student, offline_obs, offline_latent, offline_action, offline_holdout
        ),
        "dagger": evaluate_mse(
            student,
            dagger_observation,
            dagger_latent,
            dagger_action,
            torch.arange(safe_count),
        ),
    }
    final_model_hash = model_hash(student._model)
    student.save(temporary)
    loaded = student.__class__.load(temporary, device="cuda")
    strict_roundtrip = model_hash(loaded._model) == final_model_hash
    if not strict_roundtrip:
        raise RuntimeError("DAgger checkpoint strict round-trip failed")
    os.replace(temporary, OUTPUT)
    checkpoint_hash, checkpoint_bytes, checkpoint_files = tree_hash(OUTPUT)
    return {
        "schema": (
            (
                "x2_bfm_zero_scratch_response_teacher_dataset_v7"
                if teacher_response_v7
                else "x2_bfm_zero_scratch_dagger_phase_response_contract_v6"
                if phase_response_v6
                else "x2_bfm_zero_scratch_dagger_phase_contract_v6"
            )
            if phase_v6
            else "x2_bfm_zero_scratch_dagger_response_contract_v5"
            if response_v5
            else
            "x2_bfm_zero_scratch_dagger_joint_contract_v4"
            if joint_v4
            else
            "x2_bfm_zero_scratch_dagger_observation_contract_v3"
            if args.observation_contract_v3
            else "x2_bfm_zero_scratch_dagger_v1"
        ),
        "decision": (
            "PASS_RESPONSE_TEACHER_DATASET_FIT_ONLY"
            if teacher_response_v7
            else "PASS_DAGGER_FIT_ONLY"
        ),
        "scratch_lineage": True,
        "stage219_weights_loaded_into_student": False,
        "stage219_teacher_used_for_labels": True,
        "dagger_round": 0 if teacher_response_v7 else (1 if phase_v6 else (2 if args.response_contract_v5_round2 else (1 if args.response_contract_v5 else (2 if args.joint_contract_v4_round2 else (1 if (args.observation_contract_v3 or joint_v4) else (2 if args.round2 else 1)))))),
        "dataset_expansion_kind": (
            "teacher_executed_full_response_trajectory"
            if teacher_response_v7
            else "student_executed_dagger_states"
        ),
        "source_domain_contract": bool(modern_contract),
        "physical_action_scale_conversion": bool(modern_contract),
        "joint_order_reindexed": bool(joint_v4),
        "command_phase_observation_contract": bool(phase_v6),
        "actuator_domain": (
            "response_only" if (response_v5 or phase_response_v6 or teacher_response_v7) else "ideal_only"
        ),
        "teacher_iteration": teacher_iteration,
        "source_checkpoint_tree_sha256": SOURCE_SHA256,
        "source_model_hash": source_model_hash,
        "final_model_hash": final_model_hash,
        "collection": {
            "num_envs": NUM_ENVS,
            "control_steps": COLLECT_STEPS,
            "attempted_states": NUM_ENVS * COLLECT_STEPS,
            "safe_labeled_states": safe_count,
            "terminated": terminated_count,
            "teacher_actions_executed": NUM_ENVS * COLLECT_STEPS if teacher_response_v7 else 0,
            "student_actions_executed": 0 if teacher_response_v7 else NUM_ENVS * COLLECT_STEPS,
        },
        "training": {
            "epochs": EPOCHS,
            "steps_per_epoch": steps_per_epoch,
            "optimizer_steps": EPOCHS * steps_per_epoch,
            "offline_fraction_per_batch": 0.5,
            "dagger_fraction_per_batch": 0.5,
            "before": before,
            "after": after,
            "trace": trace,
        },
        "checkpoint": {
            "path": str(OUTPUT),
            "tree_sha256": checkpoint_hash,
            "bytes": checkpoint_bytes,
            "files": checkpoint_files,
            "strict_roundtrip": strict_roundtrip,
        },
        "performance_claim": False,
        "long_training_unlocked": False,
    }


try:
    report = main()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary_report = REPORT.with_name(f".{REPORT.name}.tmp")
    temporary_sidecar = REPORT.with_name(f".{REPORT.name}.sha256.tmp")
    temporary_report.write_text(serialized, encoding="utf-8")
    digest = hashlib.sha256(serialized.encode()).hexdigest()
    temporary_sidecar.write_text(f"{digest}  {REPORT.name}\n", encoding="utf-8")
    os.replace(temporary_report, REPORT)
    os.replace(temporary_sidecar, REPORT.with_name(f"{REPORT.name}.sha256"))
    print(json.dumps(report, sort_keys=True), flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
except BaseException:
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)
