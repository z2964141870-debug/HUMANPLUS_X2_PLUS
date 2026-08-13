#!/usr/bin/env python3
"""Cyclic exact-dose sweep of the frozen privileged-teacher v2b direction."""

from __future__ import annotations

import hashlib
from pathlib import Path
import textwrap


BASE = Path(__file__).with_name("run_x2_privileged_teacher_reachability_v2_attempt0.py")
EXPECTED_BASE_SHA256 = "c221fdfb75de51dd99e1dd77af4da8fc485d9aa017a7a4ffea39d381f31b5d06"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if sha256(BASE) != EXPECTED_BASE_SHA256:
    raise RuntimeError("frozen v2 attempt0 runner drifted")

source = BASE.read_text(encoding="utf-8")
inference_needle = "    with torch.inference_mode():\n"
if source.count(inference_needle) != 1:
    raise RuntimeError("no-grad repair anchor changed")
source = source.replace(inference_needle, "    with torch.no_grad():\n", 1)

start = source.index("def run() -> dict:\n")
end = source.index("\ndef fail(error: BaseException) -> None:\n", start)
replacement = textwrap.dedent(r'''
def run_scaled_population(wrapped, env, source_model, parameters, scale_by_env):
    obs = reset_observation(wrapped, env)
    initial = obs["policy"].detach().clone()
    initial_spread = float(torch.max(torch.abs(initial - initial[-1:])))
    robot = env.scene["robot"]
    initial_root = torch.cat(
        (robot.data.root_pos_w, robot.data.root_quat_w,
         robot.data.root_lin_vel_w, robot.data.root_ang_vel_w), dim=-1,
    ).detach().clone()
    foot_ids = robot.find_bodies(
        ["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
    )[0]
    names = (
        "pitch", "velocity_sq", "lateral_sq", "yaw_sq", "support", "slip",
        "root_height", "tilt", "residual", "done", "contact_count",
    )
    storage = {name: [] for name in names}
    parameter_batch = parameters.unsqueeze(0).expand(64, -1, -1)
    scale_by_env = scale_by_env.reshape(64, 1)
    with torch.no_grad():
        for _ in range(args.steps):
            policy_obs = obs["policy"]
            source_action = source_model.act_inference(obs).clamp(-1.0, 1.0)
            gait = policy_obs[:, 89:93]
            features = torch.stack(
                (torch.ones(64, device=env.device), gait[:, 0], gait[:, 1],
                 gait[:, 2] - 0.5, gait[:, 3] - 0.5), dim=-1,
            )
            signal = torch.einsum("naf,nf->na", parameter_batch, features) / math.sqrt(5.0)
            full_direction = float(PREREG["experiment"]["teacher_residual_abs_max"]) * torch.tanh(signal)
            requested = scale_by_env * full_direction
            action = torch.clamp(source_action + requested, -1.0, 1.0)
            realized = action - source_action
            next_obs, _reward, done, _extras = wrapped.step(action)
            command = env.command_manager.get_command("base_velocity")
            pitch = signed_root_pitch_rad(robot)
            support, contact_count = actual_support_com_outside_distance(env, force_threshold_n=10.0)
            forces = torch.stack(
                tuple(
                    env.scene[name].data.force_matrix_w[..., 2].abs().reshape(64, -1).amax(dim=-1)
                    for name in ("left_foot_ground_contact", "right_foot_ground_contact")
                ), dim=-1,
            )
            contact = forces > 10.0
            foot_speed = robot.data.body_lin_vel_w[:, foot_ids, :2].norm(dim=-1)
            slip = (foot_speed * contact).sum(-1) / contact_count.clamp_min(1)
            quat = robot.data.root_quat_w
            tilt = torch.acos(torch.clamp(
                1.0 - 2.0 * (quat[:, 1].square() + quat[:, 2].square()), -1.0, 1.0
            ))
            values = {
                "pitch": pitch,
                "velocity_sq": torch.square(robot.data.root_lin_vel_b[:, :2] - command[:, :2]).sum(-1),
                "lateral_sq": torch.square(robot.data.root_lin_vel_b[:, 1]),
                "yaw_sq": torch.square(robot.data.root_ang_vel_b[:, 2] - command[:, 2]),
                "support": support,
                "slip": slip,
                "root_height": robot.data.root_pos_w[:, 2],
                "tilt": tilt,
                "residual": realized,
                "done": done.reshape(-1).bool(),
                "contact_count": contact_count,
            }
            for name, value in values.items():
                storage[name].append(value.detach().cpu())
            obs = next_obs
    arrays = {name: torch.stack(values, dim=0).numpy() for name, values in storage.items()}
    return arrays, {
        "initial_policy_observation_sha256": tensor_hash(initial),
        "initial_env_spread_max_abs": initial_spread,
        "initial_root_state_sha256": tensor_hash(initial_root),
    }


def dose_gates(source_summary, candidate_summary):
    limits = PREREG["gates"]
    return {
        "candidate_zero_termination": candidate_summary["termination_count"] == 0,
        "termination_not_worse": candidate_summary["termination_count"] <= source_summary["termination_count"],
        "pitch_mean": candidate_summary["signed_pitch_rad"]["mean"] - source_summary["signed_pitch_rad"]["mean"] >= limits["pitch_mean_delta_rad_min"],
        "pitch_p05": candidate_summary["signed_pitch_rad"]["p05"] - source_summary["signed_pitch_rad"]["p05"] >= limits["pitch_p05_delta_rad_min"],
        "velocity": candidate_summary["velocity_rmse_mps"] <= source_summary["velocity_rmse_mps"] + limits["velocity_rmse_regression_mps_max"],
        "lateral": candidate_summary["lateral_rms_mps"] <= source_summary["lateral_rms_mps"] + limits["lateral_rms_regression_mps_max"],
        "yaw": candidate_summary["yaw_rmse_radps"] <= source_summary["yaw_rmse_radps"] + limits["yaw_rmse_regression_radps_max"],
        "support": candidate_summary["support_outside_mean_m"] <= source_summary["support_outside_mean_m"] + limits["support_regression_m_max"],
        "slip": candidate_summary["stance_slip_p95_mps"] <= source_summary["stance_slip_p95_mps"] + limits["slip_regression_mps_max"],
        "flight": candidate_summary["flight_fraction"] <= source_summary["flight_fraction"] + limits["flight_regression_max"],
        "root_height": candidate_summary["root_z_min_m"] >= max(limits["root_z_absolute_min_m"], source_summary["root_z_min_m"] - limits["root_z_regression_m_max"]),
        "tilt": candidate_summary["root_tilt_max_rad"] <= min(limits["tilt_absolute_max_rad"], source_summary["root_tilt_max_rad"] + limits["tilt_regression_rad_max"]),
        "residual_bound": candidate_summary["teacher_residual_abs_max"] <= limits["teacher_residual_abs_max"],
        "residual_slew": candidate_summary["teacher_residual_step_abs_max"] <= limits["teacher_residual_step_abs_max"],
    }


def metric_deltas(source_summary, candidate_summary):
    return {
        "signed_pitch_mean_rad": candidate_summary["signed_pitch_rad"]["mean"] - source_summary["signed_pitch_rad"]["mean"],
        "signed_pitch_p05_rad": candidate_summary["signed_pitch_rad"]["p05"] - source_summary["signed_pitch_rad"]["p05"],
        "velocity_rmse_mps": candidate_summary["velocity_rmse_mps"] - source_summary["velocity_rmse_mps"],
        "lateral_rms_mps": candidate_summary["lateral_rms_mps"] - source_summary["lateral_rms_mps"],
        "yaw_rmse_radps": candidate_summary["yaw_rmse_radps"] - source_summary["yaw_rmse_radps"],
        "support_outside_mean_m": candidate_summary["support_outside_mean_m"] - source_summary["support_outside_mean_m"],
        "stance_slip_p95_mps": candidate_summary["stance_slip_p95_mps"] - source_summary["stance_slip_p95_mps"],
        "termination_count": candidate_summary["termination_count"] - source_summary["termination_count"],
    }


def run() -> dict:
    started = time.monotonic()
    env = ManagerBasedRLEnv(cfg=build_cfg())
    wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
    initial_obs = wrapped.get_observations()
    source_model = build_source(initial_obs, env.device)
    source_state_sha = hashlib.sha256(b"".join(
        name.encode() + value.detach().cpu().contiguous().numpy().tobytes()
        for name, value in sorted(source_model.state_dict().items())
    )).hexdigest()

    direction_result = json.loads(Path(PREREG["immutable_evidence"]["v2b_result"]["path"]).read_text())
    parameters = torch.as_tensor(
        direction_result["search"]["best_parameters_15x5"],
        device=env.device, dtype=torch.float32,
    )
    scales = np.asarray(PREREG["dose_sweep"]["scales"], dtype=np.float32)
    if scales.shape != (8,) or scales[0] != 0.0:
        raise RuntimeError("dose schedule must contain eight values beginning with zero")
    ids = np.arange(64)
    buckets = {
        scale_id: {name: [] for name in (
            "pitch", "velocity_sq", "lateral_sq", "yaw_sq", "support", "slip",
            "root_height", "tilt", "residual", "done", "contact_count",
        )}
        for scale_id in range(8)
    }
    bucket_env_ids = {scale_id: [] for scale_id in range(8)}
    fingerprints = []
    assignments = []
    for pass_index in range(8):
        scale_ids = (ids + pass_index) % 8
        records, fingerprint = run_scaled_population(
            wrapped, env, source_model, parameters,
            torch.as_tensor(scales[scale_ids], device=env.device),
        )
        fingerprints.append(fingerprint)
        assignments.append(scale_ids.tolist())
        for scale_id in range(8):
            mask = scale_ids == scale_id
            bucket_env_ids[scale_id].extend(ids[mask].tolist())
            for name in buckets[scale_id]:
                buckets[scale_id][name].append(records[name][:, mask])

    joined = {
        scale_id: {name: np.concatenate(parts, axis=1) for name, parts in values.items()}
        for scale_id, values in buckets.items()
    }
    summaries = {
        scale_id: summarize_arrays(values, np.ones(64, dtype=bool))
        for scale_id, values in joined.items()
    }
    source_summary = summaries[0]
    records_by_scale = []
    passing = []
    for scale_id, scale in enumerate(scales):
        summary = summaries[scale_id]
        gates = dose_gates(source_summary, summary)
        env_order = np.asarray(bucket_env_ids[scale_id])
        parity = {}
        parity_consistency = True
        for parity_name, parity_value in (("even", 0), ("odd", 1)):
            mask = env_order % 2 == parity_value
            source_order = np.asarray(bucket_env_ids[0])
            source_parity = summarize_arrays(joined[0], source_order % 2 == parity_value)
            candidate_parity = summarize_arrays(joined[scale_id], mask)
            parity_gates = dose_gates(source_parity, candidate_parity)
            pitch_ok = (
                candidate_parity["signed_pitch_rad"]["mean"] - source_parity["signed_pitch_rad"]["mean"] >= PREREG["gates"]["parity_pitch_mean_delta_rad_min"]
                and candidate_parity["signed_pitch_rad"]["p05"] - source_parity["signed_pitch_rad"]["p05"] >= PREREG["gates"]["parity_pitch_p05_delta_rad_min"]
            )
            safety_ok = all(
                value for name, value in parity_gates.items()
                if name not in {"pitch_mean", "pitch_p05"}
            )
            parity_consistency = parity_consistency and pitch_ok and safety_ok
            parity[parity_name] = {
                "source": source_parity,
                "candidate": candidate_parity,
                "gates": parity_gates,
                "pitch_thresholds_pass": pitch_ok,
                "safety_pass": safety_ok,
            }
        passed = bool(scale_id != 0 and all(gates.values()) and parity_consistency)
        if passed:
            passing.append(scale_id)
        records_by_scale.append({
            "scale_id": scale_id,
            "scale": float(scale),
            "summary": summary,
            "deltas": metric_deltas(source_summary, summary),
            "gates": gates,
            "parity": parity,
            "parity_consistency": parity_consistency,
            "passed": passed,
        })

    if passing:
        selected_id = min(passing, key=lambda index: float(scales[index]))
    else:
        selected_id = max(
            range(1, 8),
            key=lambda index: (
                sum(records_by_scale[index]["gates"].values())
                + int(records_by_scale[index]["parity_consistency"]),
                records_by_scale[index]["deltas"]["signed_pitch_mean_rad"],
                -float(scales[index]),
            ),
        )
    selected = records_by_scale[selected_id]
    technical = {
        "initial_env_spread": all(item["initial_env_spread_max_abs"] <= 1.0e-6 for item in fingerprints),
        "reset_observation_replay": len({item["initial_policy_observation_sha256"] for item in fingerprints}) == 1,
        "reset_root_replay": len({item["initial_root_state_sha256"] for item in fingerprints}) == 1,
        "cyclic_assignment_complete": all(sorted(bucket_env_ids[index]) == list(range(64)) for index in range(8)),
        "source_model_immutable": source_state_sha == hashlib.sha256(b"".join(
            name.encode() + value.detach().cpu().contiguous().numpy().tobytes()
            for name, value in sorted(source_model.state_dict().items())
        )).hexdigest(),
        "finite": all(
            math.isfinite(float(value))
            for record in records_by_scale
            for value in record["deltas"].values()
        ),
    }
    passed = bool(all(technical.values()) and passing)
    return {
        "schema": "x2_privileged_teacher_dose_v2c_result_v1",
        "preregistration_sha256": PREREG_SHA,
        "decision": "PASS_SAFE_TEACHER_DOSE_LOCAL_ONLY" if passed else "FAIL_NO_SAFE_TEACHER_DOSE_STOP",
        "technical_checks": technical,
        "dose_sweep": {
            "scales": scales.tolist(),
            "assignments": assignments,
            "records": records_by_scale,
            "passing_scale_ids": passing,
            "selected_scale_id": selected_id if passed else None,
            "diagnostic_reference_scale_id": selected_id,
        },
        "validation": {
            "source": source_summary,
            "candidate": selected["summary"],
            "deltas": selected["deltas"],
            "gates": selected["gates"],
            "parity_consistency": selected["parity_consistency"],
        },
        "evidence_boundary": {
            "privileged_oracle_not_deployable": True,
            "torch_optimizer_objects": 0,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "checkpoint_writes": 0,
            "cem_distribution_updates": 0,
            "teacher_panel_preregistration_unlocked": passed,
            "bc_or_dagger_unlocked": False,
            "ppo_or_long_training_unlocked": False,
            "deployment_unlocked": False,
        },
        "resource": {
            "wall_time_s": time.monotonic() - started,
            "isaac_launches": 1,
            "control_steps": 8 * args.steps,
            "physics_substeps_per_environment": 8 * args.steps * int(env.cfg.decimation),
        },
    }
''').lstrip()

source = source[:start] + replacement + source[end:]
source = source.replace(
    '"schema": "x2_privileged_teacher_isaac_reachability_failure_v1",',
    '"schema": "x2_privileged_teacher_dose_v2c_failure_v1",',
    1,
)
if "def run_scaled_population" not in source or source.count("def run() -> dict:") != 1:
    raise RuntimeError("dose-sweep transform failed")

namespace = {
    "__file__": str(Path(__file__).resolve()),
    "__name__": "__main__",
    "__package__": None,
}
exec(compile(source, str(BASE), "exec"), namespace, namespace)
