#!/usr/bin/env python3
"""CEM search for a bounded state-feedback governor on the frozen teacher."""

from __future__ import annotations

import hashlib
from pathlib import Path
import textwrap


BASE = Path(__file__).with_name("run_x2_privileged_teacher_panel_v2d.py")
EXPECTED_BASE_SHA256 = "d8fb6b08f1ab82a2c1ae13a26b9139f1a918ea40189a28800a0ece3696a5b45e"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if sha256(BASE) != EXPECTED_BASE_SHA256:
    raise RuntimeError("frozen v2d runner drifted")

source = BASE.read_text(encoding="utf-8")
source = source.replace(
    '"x2_privileged_teacher_command_panel_prereg_v1"',
    '"x2_privileged_feedback_teacher_search_prereg_v1"',
    1,
)
source = source.replace(
    "args.num_envs != 128 or args.steps != 512",
    "args.num_envs != 256 or args.steps != 512",
    1,
)
start = source.index("def run() -> dict:\n")
end = source.index("\ndef fail(error: BaseException) -> None:\n", start)
replacement = textwrap.dedent(r'''
def optional_mean(value: torch.Tensor):
    finite = value[torch.isfinite(value)]
    return float(finite.mean()) if finite.numel() else None


def optional_quantile(value: torch.Tensor, q: float):
    finite = value[torch.isfinite(value)]
    return float(torch.quantile(finite, q)) if finite.numel() else None


def run() -> dict:
    import numpy as np

    from cwi_x2.privileged_feedback_teacher import (
        PARAMETER_COUNT, feedback_scale, feedback_search_score,
        feedback_validation_gates, slew_limited_residual,
    )

    started = time.monotonic()
    env = ManagerBasedRLEnv(cfg=build_cfg())
    wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
    obs = wrapped.get_observations()
    if tuple(obs["policy"].shape) != (256, 93):
        raise RuntimeError("policy observation contract changed")
    term = env.action_manager._terms["joint_pos"]
    if tuple(term._joint_names) != LOWER15 or not bool(term._cwi_upper_zero_mask.all()):
        raise RuntimeError("action/fixed-upper contract changed")
    source_model = build_source(obs, env.device)
    source_sha = tensor_hash(torch.cat([value.reshape(-1) for value in source_model.state_dict().values()]))
    direction_result = json.loads(Path(PREREG["immutable_evidence"]["v2b_result"]["path"]).read_text())
    direction = torch.as_tensor(
        direction_result["search"]["best_parameters_15x5"],
        device=env.device, dtype=torch.float32,
    )
    group_ids = torch.div(torch.arange(256, device=env.device), 16, rounding_mode="floor")
    robot = env.scene["robot"]
    foot_ids = robot.find_bodies(
        ["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
    )[0]
    fingerprints = []

    def rollout(parameter_batch: torch.Tensor, candidate_groups: torch.Tensor, label: str):
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        observation, _ = wrapped.reset()
        role_ids = env.command_manager.get_term("base_velocity").command_role()
        expected_roles = torch.div(torch.arange(256, device=env.device), 2, rounding_mode="floor").remainder(8)
        if not torch.equal(role_ids, expected_roles):
            raise RuntimeError("role allocation changed")
        initial_policy = observation["policy"].detach().clone()
        initial_root = torch.cat(
            (robot.data.root_pos_w, robot.data.root_quat_w, robot.data.root_lin_vel_w, robot.data.root_ang_vel_w),
            dim=-1,
        ).detach().clone()
        initial_joint = torch.cat((robot.data.joint_pos, robot.data.joint_vel), dim=-1).detach().clone()
        fingerprints.append({
            "label": label,
            "policy": tensor_hash(initial_policy),
            "root": tensor_hash(initial_root),
            "joint": tensor_hash(initial_joint),
        })
        parameters = parameter_batch[group_ids]
        candidate = candidate_groups[group_ids]
        alive = torch.ones(256, dtype=torch.bool, device=env.device)
        survival = torch.full((256,), args.steps * env.step_dt, device=env.device)
        terminated = torch.zeros_like(alive)
        timed_out = torch.zeros_like(alive)
        root_height_min = robot.data.root_pos_w[:, 2].clone()
        tilt_max = torch.zeros(256, device=env.device)
        residual_abs_max = torch.zeros(256, device=env.device)
        residual_step_abs_max = torch.zeros(256, device=env.device)
        previous_residual = torch.zeros((256, 15), device=env.device)
        scale_sum = torch.zeros(256, device=env.device)
        scale_max = torch.zeros(256, device=env.device)
        scale_count = torch.zeros(256, device=env.device)
        samples = {name: [] for name in (
            "pitch", "support", "slip", "velocity_sq", "lateral_sq", "yaw_sq",
            "flight", "terminal_speed", "terminal_double_support",
        )}
        with torch.no_grad():
            for _step in range(args.steps):
                policy_obs = observation["policy"]
                source_action = source_model.act_inference(observation).clamp(-1.0, 1.0)
                gait = policy_obs[:, 89:93]
                phase_features = torch.stack(
                    (torch.ones(256, device=env.device), gait[:, 0], gait[:, 1], gait[:, 2] - 0.5, gait[:, 3] - 0.5),
                    dim=-1,
                )
                direction_signal = torch.einsum("af,nf->na", direction, phase_features) / math.sqrt(5.0)
                command = env.command_manager.get_command("base_velocity")
                moving = torch.linalg.vector_norm(command[:, :2], dim=-1) > 0.10
                terminal = env.command_manager.get_term("base_velocity").terminal_stop_mask()
                pre_pitch = signed_root_pitch_rad(robot)
                pre_support, pre_contact_count = actual_support_com_outside_distance(env, force_threshold_n=10.0)
                quat = robot.data.root_quat_w
                pre_tilt = torch.acos(torch.clamp(1.0 - 2.0 * (quat[:, 1].square() + quat[:, 2].square()), -1.0, 1.0))
                velocity_error = torch.linalg.vector_norm(robot.data.root_lin_vel_b[:, :2] - command[:, :2], dim=-1)
                scale = feedback_scale(
                    parameters, role_ids,
                    pitch_rad=pre_pitch,
                    pitch_rate_radps=robot.data.root_ang_vel_b[:, 1],
                    support_outside_m=pre_support,
                    velocity_error_mps=velocity_error,
                    tilt_rad=pre_tilt,
                    contact_count=pre_contact_count,
                    moving=moving,
                    terminal=terminal,
                    maximum_scale=float(PREREG["feedback"]["maximum_scale"]),
                )
                scale = torch.where(candidate, scale, torch.zeros_like(scale))
                desired = scale.unsqueeze(-1) * 0.20 * torch.tanh(direction_signal)
                residual = slew_limited_residual(
                    desired, previous_residual,
                    maximum_abs=float(PREREG["gates"]["teacher_residual_abs_max"]),
                    maximum_step=float(PREREG["feedback"]["maximum_residual_step"]),
                )
                enabled = candidate & moving & ~terminal & (pre_contact_count > 0)
                residual = torch.where(enabled.unsqueeze(-1), residual, torch.zeros_like(residual))
                action = torch.clamp(source_action + residual, -1.0, 1.0)
                realized = action - source_action
                next_observation, _reward, done, extras = wrapped.step(action)
                done = done.reshape(-1).bool()
                time_out = extras.get("time_outs", torch.zeros_like(done)).reshape(-1).bool()
                command_after = env.command_manager.get_command("base_velocity")
                pitch = signed_root_pitch_rad(robot)
                support, contact_count = actual_support_com_outside_distance(env, force_threshold_n=10.0)
                forces = torch.stack(tuple(
                    env.scene[name].data.force_matrix_w[..., 2].abs().reshape(256, -1).amax(dim=-1)
                    for name in ("left_foot_ground_contact", "right_foot_ground_contact")
                ), dim=-1)
                contact = forces > 10.0
                foot_speed = robot.data.body_lin_vel_w[:, foot_ids, :2].norm(dim=-1)
                slip = (foot_speed * contact).sum(-1) / contact_count.clamp_min(1)
                velocity_sq = torch.square(robot.data.root_lin_vel_b[:, :2] - command_after[:, :2]).sum(-1)
                lateral_sq = torch.square(robot.data.root_lin_vel_b[:, 1])
                yaw_sq = torch.square(robot.data.root_ang_vel_b[:, 2] - command_after[:, 2])
                terminal_after = env.command_manager.get_term("base_velocity").terminal_stop_mask()
                quat = robot.data.root_quat_w
                tilt = torch.acos(torch.clamp(1.0 - 2.0 * (quat[:, 1].square() + quat[:, 2].square()), -1.0, 1.0))
                valid_moving = alive & (torch.linalg.vector_norm(command_after[:, :2], dim=-1) > 0.10)

                def masked(value, mask):
                    return torch.where(mask, value, torch.full_like(value, torch.nan)).detach().cpu()

                samples["pitch"].append(masked(pitch, valid_moving))
                samples["support"].append(masked(support, valid_moving))
                samples["slip"].append(masked(slip, alive))
                samples["velocity_sq"].append(masked(velocity_sq, alive))
                samples["lateral_sq"].append(masked(lateral_sq, alive))
                samples["yaw_sq"].append(masked(yaw_sq, alive))
                samples["flight"].append(masked((contact_count == 0).float(), alive))
                samples["terminal_speed"].append(masked(torch.linalg.vector_norm(robot.data.root_lin_vel_b[:, :2], dim=-1), alive & terminal_after))
                samples["terminal_double_support"].append(masked((contact_count == 2).float(), alive & terminal_after))
                residual_abs_max = torch.maximum(residual_abs_max, realized.abs().amax(dim=-1))
                residual_step_abs_max = torch.maximum(residual_step_abs_max, (realized - previous_residual).abs().amax(dim=-1))
                scale_sum += torch.where(alive, scale, torch.zeros_like(scale))
                scale_max = torch.maximum(scale_max, scale)
                scale_count += alive
                previous_residual = realized
                root_height_min = torch.where(alive, torch.minimum(root_height_min, robot.data.root_pos_w[:, 2]), root_height_min)
                tilt_max = torch.where(alive, torch.maximum(tilt_max, tilt), tilt_max)
                newly_done = alive & done
                survival[newly_done] = (_step + 1) * env.step_dt
                terminated |= newly_done & ~time_out
                timed_out |= newly_done & time_out
                alive &= ~done
                observation = next_observation
        stacked = {name: torch.stack(values, dim=0) for name, values in samples.items()}
        rows = []
        for env_id in range(256):
            role_id = int(role_ids[env_id])
            rows.append({
                "label": label, "env_id": env_id, "group_id": int(group_ids[env_id]),
                "role_id": role_id, "role": ROLE_NAMES[role_id],
                "treatment": "candidate" if bool(candidate[env_id]) else "source",
                "signed_pitch_mean_rad": optional_mean(stacked["pitch"][:, env_id]),
                "signed_pitch_p05_rad": optional_quantile(stacked["pitch"][:, env_id], 0.05),
                "support_mean_m": optional_mean(stacked["support"][:, env_id]),
                "stance_slip_p95_mps": optional_quantile(stacked["slip"][:, env_id], 0.95),
                "velocity_mse": optional_mean(stacked["velocity_sq"][:, env_id]),
                "lateral_mse": optional_mean(stacked["lateral_sq"][:, env_id]),
                "yaw_mse": optional_mean(stacked["yaw_sq"][:, env_id]),
                "flight_fraction": optional_mean(stacked["flight"][:, env_id]),
                "terminal_speed_mean_mps": optional_mean(stacked["terminal_speed"][:, env_id]) if role_id == 7 else 0.0,
                "terminal_double_support_mean": optional_mean(stacked["terminal_double_support"][:, env_id]) if role_id == 7 else 1.0,
                "survival_s": float(survival[env_id]), "terminated": bool(terminated[env_id]),
                "time_out": bool(timed_out[env_id]), "root_height_min_m": float(root_height_min[env_id]),
                "root_tilt_max_rad": float(tilt_max[env_id]),
                "teacher_residual_abs_max": float(residual_abs_max[env_id]),
                "teacher_residual_step_abs_max": float(residual_step_abs_max[env_id]),
                "feedback_scale_mean": float(scale_sum[env_id] / scale_count[env_id].clamp_min(1)),
                "feedback_scale_max": float(scale_max[env_id]),
            })
        return rows

    search = PREREG["search"]
    rng = np.random.default_rng(int(search["parameter_seed"]))
    mean = np.asarray(search["initial_mean"], dtype=np.float64)
    std = np.asarray(search["initial_std"], dtype=np.float64)
    if mean.shape != (PARAMETER_COUNT,) or std.shape != (PARAMETER_COUNT,):
        raise RuntimeError("feedback search parameter contract changed")
    best = {"cost": float("inf"), "parameters": mean.copy(), "roles": None}
    iterations = []
    for iteration in range(int(search["iterations"])):
        population = np.clip(
            rng.normal(mean, std, size=(15, PARAMETER_COUNT)),
            float(search["lower_bound"]), float(search["upper_bound"]),
        )
        population[0] = mean
        batch = np.zeros((16, PARAMETER_COUNT), dtype=np.float32)
        batch[1:] = population.astype(np.float32)
        candidate_groups = torch.arange(16, device=env.device) > 0
        rows = rollout(
            torch.as_tensor(batch, device=env.device), candidate_groups,
            f"search_{iteration}",
        )
        source_rows = [row for row in rows if row["group_id"] == 0]
        scored = []
        for group_id in range(1, 16):
            candidate_rows = [row for row in rows if row["group_id"] == group_id]
            cost, roles = feedback_search_score(source_rows, candidate_rows, PREREG["gates"])
            scored.append((cost, population[group_id - 1].copy(), roles, group_id))
            if cost < best["cost"]:
                best = {"cost": float(cost), "parameters": population[group_id - 1].copy(), "roles": roles}
        scored.sort(key=lambda item: item[0])
        elite = np.stack([item[1] for item in scored[: int(search["elites"])]])
        mean = 0.25 * mean + 0.75 * elite.mean(axis=0)
        std = np.maximum(
            float(search["std_floor"]),
            0.25 * std + 0.75 * elite.std(axis=0),
        )
        iterations.append({
            "iteration": iteration,
            "best_cost": float(scored[0][0]),
            "best_group_id": int(scored[0][3]),
            "global_best_cost": float(best["cost"]),
        })

    best_batch = torch.as_tensor(
        np.repeat(np.asarray(best["parameters"])[None, :], 16, axis=0),
        device=env.device, dtype=torch.float32,
    )
    validation_rows = []
    for lane in ("A", "B"):
        candidate_groups = torch.arange(16, device=env.device) >= 8
        if lane == "B":
            candidate_groups = ~candidate_groups
        validation_rows.extend(rollout(best_batch, candidate_groups, f"validation_{lane}"))
    source_rows = [row for row in validation_rows if row["treatment"] == "source"]
    candidate_rows = [row for row in validation_rows if row["treatment"] == "candidate"]
    validation_passed, validation = feedback_validation_gates(
        source_rows, candidate_rows, PREREG["gates"], expected_per_role=32,
    )
    source_after = tensor_hash(torch.cat([value.reshape(-1) for value in source_model.state_dict().values()]))
    technical = {
        "reset_policy_replay": len({item["policy"] for item in fingerprints}) == 1,
        "reset_root_replay": len({item["root"] for item in fingerprints}) == 1,
        "reset_joint_replay": len({item["joint"] for item in fingerprints}) == 1,
        "search_iterations_complete": len(iterations) == int(search["iterations"]),
        "validation_rows_complete": len(validation_rows) == 512,
        "source_model_immutable": source_sha == source_after,
        "finite_search": all(math.isfinite(item["best_cost"]) for item in iterations),
        "optimizer_steps_zero": True,
        "checkpoint_writes_zero": True,
    }
    passed = bool(all(technical.values()) and validation_passed)
    return {
        "schema": "x2_privileged_feedback_teacher_search_result_v1",
        "preregistration_sha256": PREREG_SHA,
        "decision": "PASS_FEEDBACK_TEACHER_SEARCH_LOCAL_ONLY" if passed else "FAIL_FEEDBACK_TEACHER_SEARCH_STOP",
        "seed": args.seed,
        "search": {
            "iterations": iterations,
            "best_cost": float(best["cost"]),
            "best_parameters_13": np.asarray(best["parameters"]).tolist(),
            "best_search_roles": best["roles"],
        },
        "validation": {"passed": validation_passed, "by_role": validation},
        "technical_checks": technical,
        "validation_per_env": validation_rows,
        "evidence_boundary": {
            "privileged_state_used": True,
            "optimizer_steps": 0, "backward_calls": 0, "checkpoint_writes": 0,
            "fresh_multiseed_validation_preregistration_unlocked": passed,
            "teacher_dataset_unlocked": False, "training_unlocked": False,
            "deployment_unlocked": False,
        },
        "resource": {
            "wall_time_s": time.monotonic() - started,
            "control_steps": (int(search["iterations"]) + 2) * args.steps,
        },
    }
''').lstrip()

source = source[:start] + replacement + source[end:]
source = source.replace(
    '"schema": "x2_privileged_teacher_panel_failure_v1",',
    '"schema": "x2_privileged_feedback_teacher_search_failure_v1",',
    1,
)
source = source.replace(
    'os._exit(0 if result["decision"] == "PANEL_LAUNCH_FINITE" else 1)',
    'os._exit(0 if result["decision"] in {"PASS_FEEDBACK_TEACHER_SEARCH_LOCAL_ONLY", "FAIL_FEEDBACK_TEACHER_SEARCH_STOP"} else 1)',
    1,
)
if (
    source.count("def run() -> dict:") != 1
    or "feedback_validation_gates" not in source
    or "args.num_envs != 128" in source
    or 'result["decision"] == "PANEL_LAUNCH_FINITE"' in source
):
    raise RuntimeError("feedback teacher transform failed")

namespace = {"__file__": str(Path(__file__).resolve()), "__name__": "__main__", "__package__": None}
exec(compile(source, str(BASE), "exec"), namespace, namespace)
