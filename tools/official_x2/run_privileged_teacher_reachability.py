#!/usr/bin/env python3
"""Search one zero-training privileged 15-D teacher in official direct MuJoCo.

This is a reachability screen, not a deployable controller.  It reuses the
Stage250 action/observation contract, forks source and candidate from the same
integration/controller snapshot, and uses CEM only to ask whether coordinated
phase/contact-conditioned action residuals can materially reduce backward
pitch without sacrificing the existing locomotion envelope.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import mujoco
import numpy as np
import onnxruntime as ort

from cwi_x2.privileged_teacher_reachability import (
    PARAMETER_SHAPE,
    reachability_gates,
    rollout_cost,
    summarize_rollout,
    support_outside_distance,
    teacher_residual,
)
from official_x2.audit_stage250_native_dynamic_seed import (
    LOWER_JOINTS,
    LOWER_SCALE,
    projected_gravity,
    rotation_body_to_world,
    sha256_file,
)
from official_x2.replay_official_trace_direct_mujoco import JOINTS, yaw_tilt
from official_x2.run_phase28_stage250_extended_direct import (
    CONTROL_DT,
    DirectRunner,
    fork_preflight,
    historical_action_preflight,
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    data = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with temporary.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _verify_sidecar(path: Path) -> str:
    digest = sha256_file(path)
    sidecar = path.with_name(path.name + ".sha256")
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    if fields != [digest, path.name]:
        raise RuntimeError(f"invalid sidecar for {path}")
    return digest


def _load_prereg(path: Path, output: Path) -> tuple[dict[str, Any], str]:
    prereg_sha = _verify_sidecar(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["schema"] != "x2_privileged_teacher_reachability_prereg_v1":
        raise RuntimeError("unexpected prereg schema")
    if Path(payload["output"]).resolve() != output.resolve():
        raise RuntimeError("output path does not match preregistration")
    for section in ("immutable_code", "immutable_inputs"):
        for name, record in payload[section].items():
            candidate = Path(record["path"])
            if not candidate.is_file() or sha256_file(candidate) != record["sha256"]:
                raise RuntimeError(f"{section} mismatch: {name}")
    if output.exists() or output.with_name(f".{output.name}.tmp").exists():
        raise FileExistsError(output)
    return payload, prereg_sha


def _wrapped(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


class TeacherRunner(DirectRunner):
    """Stage250 direct runner with a bounded actor-residual oracle."""

    def __init__(self, args: argparse.Namespace, initial_row: dict[str, Any]):
        super().__init__(args, initial_row)
        self.foot_body_ids = np.asarray(
            [int(self.model.body("left_ankle_roll_link").id), int(self.model.body("right_ankle_roll_link").id)],
            dtype=np.int64,
        )
        self.pelvis_body_id = int(self.model.body("pelvis").id)

    def _candidate_action(
        self,
        result: dict[str, Any],
        delta: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        actor_residual = np.clip(
            np.asarray(result["first_clip"], dtype=np.float32) + delta,
            -1.0,
            1.0,
        ).astype(np.float32)
        preclip = actor_residual + np.asarray(result["template"], dtype=np.float32)
        preclip[[5, 11]] += 0.20
        preclip[[2, 8]] += np.asarray(result["lateral_bias"], dtype=np.float32)
        final = np.clip(preclip, -1.0, 1.0).astype(np.float32)
        realized = actor_residual - np.asarray(result["first_clip"], dtype=np.float32)
        return actor_residual, final, realized

    def step_teacher(
        self,
        parameters: np.ndarray,
        *,
        command_vx: float,
        residual_bound: float,
    ) -> dict[str, Any]:
        elapsed = (self.tick - 100) * CONTROL_DT
        obs, yaw_before, _command_wz = self.observation("move", elapsed, command_vx)
        result = self.contract.action_from_obs(
            obs=obs,
            stage="move",
            stage_elapsed=elapsed,
            root_xy=(float(self.data.qpos[0]), float(self.data.qpos[1])),
            root_yaw=yaw_before,
        )
        requested = teacher_residual(parameters, obs[89:93], bound=residual_bound)
        actor_residual, final_action, realized = self._candidate_action(result, requested)
        zero_reconstruction_exact = bool(
            np.any(parameters) or np.array_equal(final_action, np.asarray(result["final"], dtype=np.float32))
        )
        self.contract.previous["main"] = actor_residual.copy()
        self.contract.issued["main"] = final_action.copy()

        targets = dict(self.default)
        for index, name in enumerate(LOWER_JOINTS):
            targets[name] += float(final_action[index] * LOWER_SCALE[index])
        ctrl = np.zeros(self.model.nu, dtype=np.float64)
        for name in JOINTS:
            kp, kd = self.gains[name]
            actuator = self.actuator_id[name]
            torque = kp * (targets[name] - self.data.qpos[self.qpos_adr[name]]) - kd * self.data.qvel[self.dof_adr[name]]
            low, high = self.model.actuator_ctrlrange[actuator]
            ctrl[actuator] = float(np.clip(torque, low, high))

        block_contact = np.zeros(2, dtype=bool)
        block_slip: list[list[float]] = [[], []]
        for _ in range(self.substeps):
            self.data.ctrl[:] = ctrl
            mujoco.mj_step(self.model, self.data)
            contacts, side_forces = self._contact_rows()
            for side_index, side in enumerate(("left", "right")):
                block_contact[side_index] |= bool(side_forces[side])
            for contact in contacts:
                side = contact["side"]
                slip = contact["foot_point_horizontal_slip_mps"]
                if side is not None and slip is not None:
                    block_slip[0 if side == "left" else 1].append(float(slip))

        quaternion = self.data.qpos[3:7].copy()
        rotation = rotation_body_to_world(quaternion)
        body_velocity = rotation.T @ self.data.qvel[:3]
        yaw_after, tilt = yaw_tilt(quaternion)
        pitch = float(np.arcsin(np.clip(projected_gravity(quaternion)[0], -1.0, 1.0)))
        foot_xy = self.data.xpos[self.foot_body_ids, :2].copy()
        foot_yaw = np.asarray([yaw_tilt(self.data.xquat[index])[0] for index in self.foot_body_ids])
        com_xy = self.data.subtree_com[self.pelvis_body_id, :2].copy()
        support = support_outside_distance(
            com_xy,
            foot_xy,
            foot_yaw,
            block_contact,
            root_yaw=yaw_after,
        )
        heading_target = yaw_after if self.contract.heading_target is None else self.contract.heading_target
        row = {
            "tick": int(self.tick),
            "signed_pitch_rad": pitch,
            "root_vx_b_mps": float(body_velocity[0]),
            "root_vy_b_mps": float(body_velocity[1]),
            "heading_error_rad": _wrapped(float(yaw_after - heading_target)),
            "support_outside_m": support,
            "root_z_m": float(self.data.qpos[2]),
            "root_tilt_rad": float(tilt),
            "contact_count": int(block_contact.sum()),
            "slip_mps": [max(values) if values else None for values in block_slip],
            "teacher_requested": requested.tolist(),
            "teacher_residual": realized.tolist(),
            "actor_residual": actor_residual.tolist(),
            "final_action": final_action.tolist(),
            "zero_reconstruction_exact": zero_reconstruction_exact,
            "physical_state_sha256": hashlib.sha256(
                self.data.qpos.tobytes() + self.data.qvel.tobytes()
            ).hexdigest(),
        }
        self.tick += 1
        return row


def _rollout(
    runner: TeacherRunner,
    snapshot: tuple[np.ndarray, dict[str, Any], int],
    parameters: np.ndarray,
    *,
    ticks: int,
    command_vx: float,
    residual_bound: float,
) -> tuple[list[dict[str, Any]], dict[str, Any], float]:
    runner.restore(snapshot)
    rows = [
        runner.step_teacher(parameters, command_vx=command_vx, residual_bound=residual_bound)
        for _ in range(ticks)
    ]
    summary = summarize_rollout(rows, command_vx=command_vx)
    summary["zero_reconstruction_exact"] = bool(all(row["zero_reconstruction_exact"] for row in rows))
    return rows, summary, rollout_cost(summary)


def _rows_exact(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> bool:
    keys = ("physical_state_sha256", "final_action", "signed_pitch_rad", "root_vx_b_mps")
    return len(left) == len(right) and all(a[key] == b[key] for a, b in zip(left, right) for key in keys)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def run(prereg: dict[str, Any], prereg_sha: str) -> dict[str, Any]:
    started = time.monotonic()
    config = prereg["experiment"]
    paths = {name: Path(record["path"]) for name, record in prereg["immutable_inputs"].items()}
    args = argparse.Namespace(
        scene=paths["scene"],
        actor=paths["actor"],
        stationary_actor=paths["stationary_actor"],
        template=paths["template"],
    )
    payload = json.loads(paths["historical_trace"].read_text(encoding="utf-8"))
    initial_row = next(row for row in payload["trace"] if len(row.get("obs", [])) == 93)
    main_session = ort.InferenceSession(str(args.actor.resolve()), providers=["CPUExecutionProvider"])
    stationary_session = ort.InferenceSession(str(args.stationary_actor.resolve()), providers=["CPUExecutionProvider"])
    action_preflight = historical_action_preflight(payload, main_session, stationary_session, args.template)
    fork = fork_preflight(args, initial_row)
    if not action_preflight["consistent_within_float32_5e7"] or not fork["exact"]:
        raise RuntimeError("Stage250 source contract preflight failed")

    runner = TeacherRunner(args, initial_row)
    for _ in range(int(config["warmup_ticks"])):
        runner.step()
    snapshot = runner.snapshot()
    zero = np.zeros(PARAMETER_SHAPE, dtype=np.float64)
    source_rows, source_summary, source_cost = _rollout(
        runner,
        snapshot,
        zero,
        ticks=int(config["rollout_ticks"]),
        command_vx=float(config["command_vx_mps"]),
        residual_bound=float(config["teacher_residual_abs_max"]),
    )
    source_rows_2, source_summary_2, source_cost_2 = _rollout(
        runner,
        snapshot,
        zero,
        ticks=int(config["rollout_ticks"]),
        command_vx=float(config["command_vx_mps"]),
        residual_bound=float(config["teacher_residual_abs_max"]),
    )
    same_snapshot_exact = _rows_exact(source_rows, source_rows_2)
    if not same_snapshot_exact or source_summary != source_summary_2 or source_cost != source_cost_2:
        raise RuntimeError("same-snapshot source fork is not exact")

    search = prereg["search"]
    rng = np.random.default_rng(int(search["seed"]))
    mean = np.zeros(PARAMETER_SHAPE, dtype=np.float64)
    std = np.full(PARAMETER_SHAPE, float(search["initial_std"]), dtype=np.float64)
    lower, upper = map(float, search["coefficient_bounds"])
    best = {"cost": source_cost, "parameters": zero.copy(), "summary": source_summary}
    invalid_evaluations = 0
    iterations = []
    evaluation_count = 0
    for iteration in range(int(search["iterations"])):
        population = np.clip(
            rng.normal(mean, std, size=(int(search["population"]),) + PARAMETER_SHAPE),
            lower,
            upper,
        )
        population[0] = mean
        if iteration == 0:
            population[0] = zero
        scored = []
        for parameters in population:
            evaluation_count += 1
            try:
                _rows, summary, cost = _rollout(
                    runner,
                    snapshot,
                    parameters,
                    ticks=int(config["rollout_ticks"]),
                    command_vx=float(config["command_vx_mps"]),
                    residual_bound=float(config["teacher_residual_abs_max"]),
                )
            except (FloatingPointError, ValueError):
                invalid_evaluations += 1
                cost, summary = 1.0e12, None
            scored.append((float(cost), parameters.copy(), summary))
            if summary is not None and cost < best["cost"]:
                best = {"cost": float(cost), "parameters": parameters.copy(), "summary": summary}
        scored.sort(key=lambda item: item[0])
        elites = scored[: int(search["elites"])]
        elite_values = np.stack([item[1] for item in elites])
        mean = 0.25 * mean + 0.75 * elite_values.mean(axis=0)
        std = np.maximum(float(search["std_floor"]), 0.25 * std + 0.75 * elite_values.std(axis=0))
        iterations.append(
            {
                "iteration": iteration,
                "best_cost": float(scored[0][0]),
                "global_best_cost": float(best["cost"]),
                "best_survived": bool(scored[0][2] and scored[0][2]["survived_full_horizon"]),
            }
        )

    candidate_rows, candidate_summary, candidate_cost = _rollout(
        runner,
        snapshot,
        np.asarray(best["parameters"]),
        ticks=int(config["rollout_ticks"]),
        command_vx=float(config["command_vx_mps"]),
        residual_bound=float(config["teacher_residual_abs_max"]),
    )
    gates = reachability_gates(source_summary, candidate_summary, prereg["gates"])
    passed = bool(all(gates.values()))
    return _jsonable(
        {
            "schema": "x2_privileged_teacher_reachability_result_v1",
            "preregistration_sha256": prereg_sha,
            "decision": "PASS_TEACHER_REACHABILITY_LOCAL_ONLY" if passed else "FAIL_NO_TEACHER_REACHABILITY_STOP",
            "evidence_boundary": {
                "direct_official_mujoco_not_aimdk_ros": True,
                "unchanged_vendor_scene": True,
                "privileged_oracle_not_deployable": True,
                "optimizer_steps": 0,
                "backward_calls": 0,
                "checkpoint_writes": 0,
                "policy_weight_changes": 0,
                "teacher_panel_preregistration_unlocked": passed,
                "bc_or_dagger_unlocked": False,
                "ppo_or_long_training_unlocked": False,
                "deployment_unlocked": False,
            },
            "preflight": {
                "historical_action": action_preflight,
                "fresh_fork": fork,
                "same_snapshot_rollout_exact": same_snapshot_exact,
                "source_zero_reconstruction_exact": source_summary["zero_reconstruction_exact"],
            },
            "experiment": config,
            "search": {
                "config": search,
                "evaluation_count": evaluation_count,
                "invalid_evaluations": invalid_evaluations,
                "iterations": iterations,
                "best_parameters_15x5": best["parameters"],
            },
            "source": {"cost": source_cost, "summary": source_summary, "trace": source_rows},
            "candidate": {"cost": candidate_cost, "summary": candidate_summary, "trace": candidate_rows},
            "deltas": {
                "signed_pitch_mean_rad": candidate_summary["signed_pitch_rad"]["mean"] - source_summary["signed_pitch_rad"]["mean"],
                "signed_pitch_p05_rad": candidate_summary["signed_pitch_rad"]["p05"] - source_summary["signed_pitch_rad"]["p05"],
                "velocity_rmse_mps": candidate_summary["velocity_rmse_mps"] - source_summary["velocity_rmse_mps"],
                "support_outside_mean_m": candidate_summary["support_outside_mean_m"] - source_summary["support_outside_mean_m"],
                "stance_slip_p95_mps": candidate_summary["stance_slip_p95_mps"] - source_summary["stance_slip_p95_mps"],
            },
            "gates": gates,
            "all_gates_pass": passed,
            "resource": {
                "cpu_only": True,
                "wall_time_s": time.monotonic() - started,
                "physics_substeps": (2 + evaluation_count + 1) * int(config["rollout_ticks"]) * 20 + int(config["warmup_ticks"]) * 20,
            },
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prereg, prereg_sha = _load_prereg(args.prereg.resolve(), args.output.resolve())
    result = run(prereg, prereg_sha)
    _atomic_json(args.output.resolve(), result)
    digest = sha256_file(args.output.resolve())
    sidecar = args.output.with_name(args.output.name + ".sha256")
    if sidecar.exists():
        raise FileExistsError(sidecar)
    sidecar.write_text(f"{digest}  {args.output.name}\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "deltas": result["deltas"], "gates": result["gates"], "wall_time_s": result["resource"]["wall_time_s"]}, indent=2))


if __name__ == "__main__":
    main()
