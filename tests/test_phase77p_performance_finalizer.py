from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FINALIZER = ROOT / "tools/retarget/finalize_x2_phase77p_performance_campaign.py"
SOURCE_SHA = "abcd49a823385bcd02e00480c9476ad5bc381e68252ad6930c85d731178f49bb"
SPEC = importlib.util.spec_from_file_location("phase77p_finalizer", FINALIZER)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload) + "\n")
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{sha(path)}  {path.name}\n"
    )
    return path


def make_train(
    tmp_path: Path,
    *,
    seed: int,
    stage: int,
    resume: tuple[Path, Path, Path] | None = None,
) -> Path:
    checkpoint = tmp_path / f"seed{seed}_u{stage}.pt"
    checkpoint.write_bytes(f"checkpoint-{seed}-{stage}".encode())
    start = 0 if stage == 5 else 5
    updates = []
    for index in range(start + 1, stage + 1):
        updates.append(
            {
                "update_index": index,
                "posture_reward_weight": -0.5,
                "losses": {
                    "optimizer_steps": 4,
                    "source_kl": 0.001,
                    "source_kl_max": 0.005,
                    "incremental_kl": 0.001,
                    "incremental_kl_max": 0.005,
                },
                "rollout_metrics": {"time_outs_count": 0},
                "gates": {key: True for key in MODULE.UPDATE_GATE_KEYS},
            }
        )
    row = {
        "schema": "x2_phase77p_performance_train_v1",
        "decision": "SEGMENT_VALID",
        "source_checkpoint_sha256": SOURCE_SHA,
        "seed": seed,
        "num_envs": 256,
        "steps_per_env": 48,
        "start_update": start,
        "end_update": stage,
        "optimizer_steps": 20,
        "campaign_config": MODULE.EXPECTED_CONFIG,
        "terminal_bootstrap": "installed RSL-RL policy.evaluate(s_T)",
        "checkpoint_count": 1,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha(checkpoint),
        "updates": updates,
        "resume_checkpoint_sha256": None,
        "resume_train_report_sha256": None,
        "resume_gate_sha256": None,
    }
    if resume is not None:
        prior_checkpoint, prior_report, gate = resume
        row.update(
            {
                "resume_checkpoint_sha256": sha(prior_checkpoint),
                "resume_train_report_sha256": sha(prior_report),
                "resume_gate_sha256": sha(gate),
            }
        )
    return write(tmp_path / f"seed{seed}_u{stage}_train.json", row)


def group(candidate: bool) -> dict:
    return {
        "role_counts": {
            "vx_0p20_a": 8,
            "vx_0p20_b": 8,
            "vx_0p35_a": 8,
            "vx_0p35_b": 8,
            "vx_0p50_a": 8,
            "vx_0p50_b": 8,
            "turn_vx_0p35_yaw_pm_0p15": 8,
            "transition_vx_0p35": 8,
        },
        "termination_rate": 0.0,
        "survival_s_min": 10.24,
        "time_outs_count": 0,
        "root_height_min_global": 0.70,
        "root_height_min_mean": 0.72,
        "root_tilt_max_global": 0.21,
        "root_tilt_max_mean": 0.20,
        "terminal_base_speed_mps": {"mean": 0.02},
        "terminal_double_support": {"mean": 0.99},
    }


def make_eval(
    tmp_path: Path,
    *,
    train: Path,
    train_seed: int,
    eval_seed: int,
    stage: int,
    lane: str,
    pitch: float,
) -> Path:
    train_row = json.loads(train.read_text())
    per_env = []
    for env_id in range(128):
        source = (env_id % 2 == 0) == (lane == "A")
        per_env.append(
            {
                "env_id": env_id,
                "pair_id": env_id // 2,
                "lane": lane,
                "treatment": "source" if source else "candidate",
                "role_id": (env_id // 2) % 8,
                "signed_pitch_mean_rad": -0.20 if source else -0.20 + pitch,
                "support_mean_m": 0.0100 if source else 0.0101,
                "stance_slip_p95_mps": 0.100 if source else 0.101,
                "velocity_mse": 0.010000 if source else 0.010201,
                "yaw_mse": 0.010000 if source else 0.010201,
                "terminal_speed_mean_mps": (
                    (0.020 if source else 0.021)
                    if (env_id // 2) % 8 == 7
                    else None
                ),
                "terminal_double_support_mean": (
                    0.99 if (env_id // 2) % 8 == 7 else None
                ),
                "terminated": False,
                "moving_sample_count": 100,
                "terminal_sample_count": 20 if (env_id // 2) % 8 == 7 else 0,
            }
        )
    row = {
        "schema": "x2_phase77p_performance_eval_v1",
        "decision": "EVAL_FINITE",
        "finite": True,
        "per_env_complete": True,
        "source_checkpoint_sha256": SOURCE_SHA,
        "candidate_checkpoint_sha256": train_row["checkpoint_sha256"],
        "candidate_update_index": stage,
        "candidate_train_seed": train_seed,
        "eval_seed": eval_seed,
        "lane": lane,
        "num_envs": 128,
        "eval_steps": 512,
        "optimizer_steps": 0,
        "checkpoint_writes": 0,
        "per_env": per_env,
        "candidate_source_action_drift_max": 0.01,
        "candidate_minus_source": {
            "signed_pitch_mean_rad": pitch,
            "signed_pitch_p05_rad": pitch,
            "support_outside_mean_m": 0.0001,
            "velocity_rmse_mps": 0.001,
            "yaw_rmse_radps": 0.001,
            "termination_rate": 0.0,
            "stance_slip_p95_mps": 0.001,
            "terminal_speed_mean_mps": 0.001,
            "terminal_double_support_mean": 0.0,
        },
        "groups": {"source": group(False), "candidate": group(True)},
    }
    return write(
        tmp_path / f"seed{train_seed}_u{stage}_eval{eval_seed}_lane{lane}.json",
        row,
    )


def make_inputs(tmp_path: Path, *, stage: int, pitch: float):
    stage5_trains: list[Path] = []
    gate = None
    if stage == 10:
        for seed in MODULE.TRAIN_SEEDS:
            stage5_trains.append(make_train(tmp_path, seed=seed, stage=5))
        gate = write(
            tmp_path / "stage5_gate.json",
            {
                "decision": "PASS_UPDATE5_CONTINUE_TO_UPDATE10",
                "update10_training_unlocked": True,
                "train_reports": [
                    {"path": str(path), "sha256": sha(path)} for path in stage5_trains
                ],
                "seed_summaries": {
                    str(seed): {"passed": True} for seed in MODULE.TRAIN_SEEDS
                },
            },
        )
    trains = []
    evals = []
    for index, train_seed in enumerate(MODULE.TRAIN_SEEDS):
        resume = None
        if stage == 10:
            prior = json.loads(stage5_trains[index].read_text())
            resume = (Path(prior["checkpoint"]), stage5_trains[index], gate)
        train = make_train(
            tmp_path,
            seed=train_seed,
            stage=stage,
            resume=resume,
        )
        trains.append(train)
        for eval_seed in MODULE.EVAL_SEEDS:
            for lane in MODULE.LANES:
                evals.append(
                    make_eval(
                        tmp_path,
                        train=train,
                        train_seed=train_seed,
                        eval_seed=eval_seed,
                        stage=stage,
                        lane=lane,
                        pitch=pitch,
                    )
                )
    return trains, evals, stage5_trains, gate


def run_finalizer(tmp_path: Path, *, stage: int, pitch: float):
    trains, evals, stage5_trains, gate = make_inputs(
        tmp_path, stage=stage, pitch=pitch
    )
    output = tmp_path / f"result_u{stage}.json"
    command = [sys.executable, str(FINALIZER), "--stage", str(stage)]
    for path in trains:
        command += ["--train", str(path)]
    for path in evals:
        command += ["--eval", str(path)]
    for path in stage5_trains:
        command += ["--stage5-train", str(path)]
    if gate is not None:
        command += ["--stage5-gate", str(gate)]
    command += ["--output", str(output)]
    process = subprocess.run(command, text=True, capture_output=True)
    return process, json.loads(output.read_text())


def test_update5_pass_unlocks_only_update10(tmp_path: Path) -> None:
    process, result = run_finalizer(tmp_path, stage=5, pitch=0.006)
    assert process.returncode == 0, process.stderr
    assert result["decision"] == "PASS_UPDATE5_CONTINUE_TO_UPDATE10"
    assert result["update10_training_unlocked"] is True
    assert result["official_panel_unlocked"] is False
    assert result["long_training_unlocked"] is False


def test_update10_requires_material_pitch_signal(tmp_path: Path) -> None:
    process, result = run_finalizer(tmp_path, stage=10, pitch=0.014)
    assert process.returncode == 2
    assert result["decision"] == "FAIL_UPDATE10_NO_SAFE_SIGNAL_STOP"
    assert result["official_panel_unlocked"] is False


def test_update10_pass_selects_seed_but_not_long_training(tmp_path: Path) -> None:
    process, result = run_finalizer(tmp_path, stage=10, pitch=0.016)
    assert process.returncode == 0, process.stderr
    assert result["decision"] == "PASS_UPDATE10_LOCAL_PENDING_OFFICIAL_PANEL"
    assert result["selected_train_seed"] in MODULE.TRAIN_SEEDS
    assert result["official_panel_unlocked"] is True
    assert result["long_training_unlocked"] is False


def test_missing_lane_swap_is_invalid(tmp_path: Path) -> None:
    trains, evals, _, _ = make_inputs(tmp_path, stage=5, pitch=0.006)
    output = tmp_path / "invalid.json"
    command = [sys.executable, str(FINALIZER), "--stage", "5"]
    for path in trains:
        command += ["--train", str(path)]
    for path in evals[:-1]:
        command += ["--eval", str(path)]
    command += ["--output", str(output)]
    process = subprocess.run(command, text=True, capture_output=True)
    assert process.returncode != 0
