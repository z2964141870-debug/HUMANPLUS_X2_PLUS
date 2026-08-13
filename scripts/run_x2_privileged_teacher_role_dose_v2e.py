#!/usr/bin/env python3
"""One-seed cyclic role-by-dose screen of the frozen 15D teacher direction."""

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
    '"x2_privileged_teacher_role_dose_prereg_v1"',
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
    from cwi_x2.privileged_teacher_role_dose import cell_gates, summarize_cell

    started = time.monotonic()
    env = ManagerBasedRLEnv(cfg=build_cfg())
    wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
    obs = wrapped.get_observations()
    if tuple(obs["policy"].shape) != (128, 93):
        raise RuntimeError("policy observation contract changed")
    term = env.action_manager._terms["joint_pos"]
    if tuple(term._joint_names) != LOWER15 or not bool(term._cwi_upper_zero_mask.all()):
        raise RuntimeError("action/fixed-upper contract changed")
    source_model = build_source(obs, env.device)
    source_sha = tensor_hash(torch.cat([value.reshape(-1) for value in source_model.state_dict().values()]))
    direction_result = json.loads(Path(PREREG["immutable_evidence"]["v2b_result"]["path"]).read_text())
    parameters = torch.as_tensor(direction_result["search"]["best_parameters_15x5"], device=env.device, dtype=torch.float32)
    scales = torch.as_tensor(PREREG["dose_sweep"]["scales"], device=env.device, dtype=torch.float32)
    if tuple(scales.shape) != (8,) or float(scales[0]) != 0.0:
        raise RuntimeError("role dose grid changed")
    pair_ids = torch.arange(64, device=env.device)
    pair_roles = pair_ids.remainder(8)
    pair_blocks = torch.div(pair_ids, 8, rounding_mode="floor")
    robot = env.scene["robot"]
    foot_ids = robot.find_bodies(["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True)[0]
    rows = []
    fingerprints = []
    for pass_index in range(8):
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        obs, _ = wrapped.reset()
        role_ids = env.command_manager.get_term("base_velocity").command_role()
        expected_roles = torch.repeat_interleave(pair_roles, 2)
        if not torch.equal(role_ids, expected_roles):
            raise RuntimeError("role allocation changed")
        pair_scale_ids = (pair_blocks + pass_index).remainder(8)
        scale_ids = torch.repeat_interleave(pair_scale_ids, 2)
        scale_by_env = scales[scale_ids].unsqueeze(-1)
        initial_policy = obs["policy"].detach().clone()
        initial_root = torch.cat((robot.data.root_pos_w, robot.data.root_quat_w, robot.data.root_lin_vel_w, robot.data.root_ang_vel_w), dim=-1).detach().clone()
        initial_joint = torch.cat((robot.data.joint_pos, robot.data.joint_vel), dim=-1).detach().clone()
        fingerprints.append({
            "policy": tensor_hash(initial_policy), "root": tensor_hash(initial_root),
            "joint": tensor_hash(initial_joint),
        })
        alive = torch.ones(128, dtype=torch.bool, device=env.device)
        survival = torch.full((128,), args.steps * env.step_dt, device=env.device)
        terminated = torch.zeros_like(alive)
        timed_out = torch.zeros_like(alive)
        root_height_min = robot.data.root_pos_w[:, 2].clone()
        tilt_max = torch.zeros(128, device=env.device)
        residual_abs_max = torch.zeros(128, device=env.device)
        residual_step_abs_max = torch.zeros(128, device=env.device)
        previous_residual = torch.zeros((128, 15), device=env.device)
        samples = {name: [] for name in (
            "pitch", "support", "slip", "velocity_sq", "lateral_sq", "yaw_sq",
            "flight", "terminal_speed", "terminal_double_support",
        )}
        with torch.no_grad():
            for step in range(args.steps):
                source_action = source_model.act_inference(obs).clamp(-1.0, 1.0)
                policy_obs = obs["policy"]
                gait = policy_obs[:, 89:93]
                features = torch.stack((torch.ones(128, device=env.device), gait[:, 0], gait[:, 1], gait[:, 2] - 0.5, gait[:, 3] - 0.5), dim=-1)
                signal = torch.einsum("af,nf->na", parameters, features) / math.sqrt(5.0)
                command = env.command_manager.get_command("base_velocity")
                moving = torch.linalg.vector_norm(command[:, :2], dim=-1) > 0.10
                requested = scale_by_env * 0.20 * torch.tanh(signal) * moving.unsqueeze(-1)
                action = torch.clamp(source_action + requested, -1.0, 1.0)
                realized = action - source_action
                next_obs, _reward, done, extras = wrapped.step(action)
                done = done.reshape(-1).bool()
                time_out = extras.get("time_outs", torch.zeros_like(done)).reshape(-1).bool()
                pitch = signed_root_pitch_rad(robot)
                support, contact_count = actual_support_com_outside_distance(env, force_threshold_n=10.0)
                forces = torch.stack(tuple(
                    env.scene[name].data.force_matrix_w[..., 2].abs().reshape(128, -1).amax(dim=-1)
                    for name in ("left_foot_ground_contact", "right_foot_ground_contact")
                ), dim=-1)
                contact = forces > 10.0
                foot_speed = robot.data.body_lin_vel_w[:, foot_ids, :2].norm(dim=-1)
                slip = (foot_speed * contact).sum(-1) / contact_count.clamp_min(1)
                velocity_sq = torch.square(robot.data.root_lin_vel_b[:, :2] - command[:, :2]).sum(-1)
                lateral_sq = torch.square(robot.data.root_lin_vel_b[:, 1])
                yaw_sq = torch.square(robot.data.root_ang_vel_b[:, 2] - command[:, 2])
                terminal = env.command_manager.get_term("base_velocity").terminal_stop_mask()
                quat = robot.data.root_quat_w
                tilt = torch.acos(torch.clamp(1.0 - 2.0 * (quat[:, 1].square() + quat[:, 2].square()), -1.0, 1.0))
                valid_moving = alive & moving

                def masked(value, mask):
                    return torch.where(mask, value, torch.full_like(value, torch.nan)).detach().cpu()

                samples["pitch"].append(masked(pitch, valid_moving))
                samples["support"].append(masked(support, valid_moving))
                samples["slip"].append(masked(slip, alive))
                samples["velocity_sq"].append(masked(velocity_sq, alive))
                samples["lateral_sq"].append(masked(lateral_sq, alive))
                samples["yaw_sq"].append(masked(yaw_sq, alive))
                samples["flight"].append(masked((contact_count == 0).float(), alive))
                samples["terminal_speed"].append(masked(torch.linalg.vector_norm(robot.data.root_lin_vel_b[:, :2], dim=-1), alive & terminal))
                samples["terminal_double_support"].append(masked((contact_count == 2).float(), alive & terminal))
                residual_abs_max = torch.maximum(residual_abs_max, realized.abs().amax(dim=-1))
                residual_step_abs_max = torch.maximum(residual_step_abs_max, (realized - previous_residual).abs().amax(dim=-1))
                previous_residual = realized
                root_height_min = torch.where(alive, torch.minimum(root_height_min, robot.data.root_pos_w[:, 2]), root_height_min)
                tilt_max = torch.where(alive, torch.maximum(tilt_max, tilt), tilt_max)
                newly_done = alive & done
                survival[newly_done] = (step + 1) * env.step_dt
                terminated |= newly_done & ~time_out
                timed_out |= newly_done & time_out
                alive &= ~done
                obs = next_obs
        stacked = {name: torch.stack(values, dim=0) for name, values in samples.items()}
        for env_id in range(128):
            role_id = int(role_ids[env_id])
            rows.append({
                "pass_index": pass_index, "env_id": env_id, "pair_id": env_id // 2,
                "role_id": role_id, "role": ROLE_NAMES[role_id],
                "scale_id": int(scale_ids[env_id]), "scale": float(scale_by_env[env_id]),
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
            })

    cell_results = {}
    selected = {}
    for role_id, role_name in ROLE_NAMES.items():
        role_cells = {}
        source_rows = [row for row in rows if row["role_id"] == role_id and row["scale_id"] == 0]
        try:
            source_summary = summarize_cell(source_rows, transition_role=role_id == 7)
            source_error = None
        except ValueError as error:
            source_summary = None; source_error = str(error)
        passing = []
        for scale_id, scale in enumerate(PREREG["dose_sweep"]["scales"]):
            cell_rows = [row for row in rows if row["role_id"] == role_id and row["scale_id"] == scale_id]
            try:
                summary = summarize_cell(cell_rows, transition_role=role_id == 7)
                gates = cell_gates(source_summary, summary, PREREG["gates"], transition_role=role_id == 7) if source_summary is not None else {}
                error = None
                passed = bool(scale_id != 0 and gates and all(gates.values()))
            except ValueError as exc:
                summary = None; gates = {}; error = str(exc); passed = False
            if passed:
                passing.append(scale_id)
            role_cells[str(scale_id)] = {
                "scale": float(scale), "summary": summary, "gates": gates,
                "error": error, "passed": passed,
            }
        selected_id = min(passing) if passing else None
        selected[role_name] = selected_id
        cell_results[role_name] = {
            "source_error": source_error, "cells": role_cells,
            "passing_scale_ids": passing, "selected_scale_id": selected_id,
        }

    source_after = tensor_hash(torch.cat([value.reshape(-1) for value in source_model.state_dict().values()]))
    technical = {
        "reset_policy_replay": len({item["policy"] for item in fingerprints}) == 1,
        "reset_root_replay": len({item["root"] for item in fingerprints}) == 1,
        "reset_joint_replay": len({item["joint"] for item in fingerprints}) == 1,
        "rows_complete": len(rows) == 1024,
        "cells_complete": all(
            sum(row["role_id"] == role_id and row["scale_id"] == scale_id for row in rows) == 16
            for role_id in range(8) for scale_id in range(8)
        ),
        "source_model_immutable": source_sha == source_after,
        "optimizer_steps_zero": True, "checkpoint_writes_zero": True,
    }
    passed = bool(all(technical.values()) and all(value is not None for value in selected.values()))
    return {
        "schema": "x2_privileged_teacher_role_dose_result_v1",
        "preregistration_sha256": PREREG_SHA,
        "decision": "PASS_ROLE_DOSE_WINDOWS_LOCAL_ONLY" if passed else "FAIL_NO_COMPLETE_ROLE_DOSE_WINDOWS_STOP",
        "seed": args.seed, "technical_checks": technical,
        "dose_sweep": {"scales": PREREG["dose_sweep"]["scales"], "by_role": cell_results, "selected_scale_ids": selected},
        "per_env": rows,
        "evidence_boundary": {
            "optimizer_steps": 0, "backward_calls": 0, "checkpoint_writes": 0,
            "fresh_validation_preregistration_unlocked": passed,
            "teacher_dataset_unlocked": False, "training_unlocked": False,
            "deployment_unlocked": False,
        },
        "resource": {"wall_time_s": time.monotonic() - started, "control_steps": 8 * args.steps},
    }
''').lstrip()

source = source[:start] + replacement + source[end:]
source = source.replace(
    '"schema": "x2_privileged_teacher_panel_failure_v1",',
    '"schema": "x2_privileged_teacher_role_dose_failure_v1",',
    1,
)
source = source.replace(
    'os._exit(0 if result["decision"] == "PANEL_LAUNCH_FINITE" else 1)',
    'os._exit(0 if result["decision"] in {"PASS_ROLE_DOSE_WINDOWS_LOCAL_ONLY", "FAIL_NO_COMPLETE_ROLE_DOSE_WINDOWS_STOP"} else 1)',
    1,
)
if (
    source.count("def run() -> dict:") != 1
    or "def optional_mean" not in source
    or "PANEL_LAUNCH_FINITE\" else 1" in source
):
    raise RuntimeError("role-dose transform failed")
namespace = {"__file__": str(Path(__file__).resolve()), "__name__": "__main__", "__package__": None}
exec(compile(source, str(BASE), "exec"), namespace, namespace)
