#!/usr/bin/env python3
"""Build and validate the immutable Phase72 attempt0-failure backup inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path("/home/yu/projects/ZHY/CWI_CrossEmbodiment_Sim")
OLD = Path("/home/yu/x2_teleop_final/x2_sonic")
REMOTE_ROOT = "HUMAN+/HUMANPLUS_X2_PLUS/2026-08-12/task2_backward_pitch/phase72_attempt0_failure"
MANIFEST = ROOT / "reports/retarget/x2_phase72_attempt0_failure_backup_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def require_exact(path: Path, expected: str) -> None:
    if not path.is_file() or sha256(path) != expected:
        raise RuntimeError(f"immutable file absent or changed: {path}")


def require_sidecar(path: Path) -> None:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    expected = f"{sha256(path)}  {path.name}\n"
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8") != expected:
        raise RuntimeError(f"sidecar absent or mismatched: {sidecar}")


def shown_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def entry(path: Path, remote_name: str, role: str) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "role": role,
        "path": shown_path(path),
        "remote": f"{REMOTE_ROOT}/{remote_name}",
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def registered_outputs() -> list[Path]:
    artifact = ROOT / "artifacts/retarget/x2_phase72_antithetic"
    report = ROOT / "reports/retarget"
    log = Path("/tmp/x2_phase72_logs")
    paths = [
        report / "x2_phase72_antithetic_result.json",
        report / "x2_phase72_antithetic_result.json.sha256",
        report / "x2_phase72_antithetic.md",
    ]
    for seed in range(5):
        paths.extend(
            [
                artifact / f"seed{seed}_initial_commit.json",
                artifact / f"seed{seed}_initial_commit.json.sha256",
                artifact / f"seed{seed}_pair.pt",
                artifact / f"seed{seed}_pair.pt.sha256",
                report / f"x2_phase72_seed{seed}_pair_result.json",
                report / f"x2_phase72_seed{seed}_pair_result.json.sha256",
            ]
        )
        for sign in ("plus", "minus"):
            paths.extend(
                [
                    artifact / f"seed{seed}_{sign}_rollout.pt",
                    artifact / f"seed{seed}_{sign}_rollout.pt.sha256",
                    report / f"x2_phase72_seed{seed}_{sign}_screen.json",
                    report / f"x2_phase72_seed{seed}_{sign}_screen.json.sha256",
                    report / f"x2_phase72_seed{seed}_{sign}_resource.json",
                    report / f"x2_phase72_seed{seed}_{sign}_resource.json.sha256",
                    log / f"seed{seed}_{sign}.log",
                    log / f"seed{seed}_{sign}.log.sha256",
                ]
            )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-verified", action="store_true")
    parser.add_argument("--checked-at")
    args = parser.parse_args()

    prereg_path = ROOT / "reports/retarget/x2_phase72_antithetic_prereg.json"
    inventory_path = ROOT / "reports/retarget/x2_phase72_schedule_inventory.json"
    failure_path = ROOT / "reports/retarget/x2_phase72_attempt0_failure.json"
    init_path = ROOT / "artifacts/retarget/x2_phase72_antithetic/seed0_initial_commit.json"
    raw_path = ROOT / "artifacts/retarget/x2_phase72_antithetic/seed0_plus_rollout.pt"
    resource_path = ROOT / "reports/retarget/x2_phase72_seed0_plus_resource.json"
    log_path = Path("/tmp/x2_phase72_logs/seed0_plus.log")

    require_exact(prereg_path, "080f48322e0b5f1cbf33621a6d778c8970f594205d7ea7e1df5cba64d0b0988c")
    require_exact(inventory_path, "073b4ac6032a2aa5b9e439faeb9eec1350490454b91230ef57e5c8c0e395edcf")
    require_exact(failure_path, "fed54e82d2ebbba5f2eccae862cf27ecb26b6ed5b735a6687b105d58ea9619fc")
    require_exact(init_path, "cc2bf119b096c00191aa3320e30fe7f0fcf20f775ad92af2f955eb500416fb4d")
    require_exact(raw_path, "8cf725399803d9f6393f19122addacea00313c83fa1507f5622a5cc78fa5c431")
    require_exact(resource_path, "ee98b74fec9ed248cba738d90b3ffefb3988b2fca85956017491d7f22c8cf6fc")
    require_exact(log_path, "5cffe0ef56ffbe3ed53cfaa4ea2c752082152ce40ab1e746226809049df3f3df")
    for path in (prereg_path, inventory_path, failure_path, init_path, raw_path, resource_path, log_path):
        require_sidecar(path)

    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    resource = json.loads(resource_path.read_text(encoding="utf-8"))
    if failure.get("decision") != "FAIL_IMPLEMENTATION_STOP":
        raise RuntimeError("failure decision drift")
    if resource.get("exit_code") != 0 or "Tensor is not JSON serializable" not in log_path.read_text(encoding="utf-8"):
        raise RuntimeError("failure provenance drift")

    code_paths = {
        "runner_sha256": ROOT / "scripts/run_x2_phase72_antithetic.py",
        "helper_sha256": ROOT / "src/cwi_x2/phase72_antithetic.py",
        "run_script_sha256": ROOT / "scripts/run_x2_phase72_antithetic.sh",
        "pair_validator_sha256": ROOT / "tools/retarget/validate_x2_phase72_pair.py",
        "finalizer_sha256": ROOT / "tools/retarget/finalize_x2_phase72_antithetic.py",
        "runner_test_sha256": ROOT / "tests/test_phase72_antithetic_runner.py",
        "helper_test_sha256": ROOT / "tests/test_phase72_antithetic.py",
        "finalizer_test_sha256": ROOT / "tests/test_phase72_antithetic_finalizer.py",
        "schedule_generator_sha256": ROOT / "tools/retarget/prepare_x2_phase72_schedules.py",
        "phase60_posture_module_sha256": ROOT / "src/x2_native_locomotion_posture_phase60.py",
        "x2_flat_env_cfg_sha256": OLD / "sonic_x2_sandbox/gear_sonic/envs/x2_velocity/flat_env_cfg.py",
        "x2_reward_module_sha256": OLD / "sonic_x2_sandbox/gear_sonic/envs/x2_velocity/rewards.py",
        "x2_gait_module_sha256": OLD / "sonic_x2_sandbox/gear_sonic/envs/x2_velocity/gait.py",
        "x2_action_module_sha256": OLD / "sonic_x2_sandbox/gear_sonic/envs/manager_env/mdp/actions.py",
        "heading_command_module_sha256": OLD / "sonic_x2_sandbox/gear_sonic/envs/x2_velocity/heading_command.py",
        "modular_env_cfg_sha256": OLD / "sonic_x2_sandbox/gear_sonic/envs/manager_env/modular_tracking_env_cfg.py",
        "x2_robot_cfg_sha256": OLD / "sonic_x2_sandbox/gear_sonic/envs/manager_env/robots/x2.py",
        "phase68_interface_sha256": ROOT / "src/cwi_x2/phase68_residual_ppo.py",
        "residual_module_sha256": ROOT / "src/cwi_x2/phase_conditioned_knee_residual.py",
        "phase69_helper_sha256": ROOT / "src/cwi_x2/phase69_reward_attribution.py",
        "phase70_helper_sha256": ROOT / "src/cwi_x2/phase70_long_lookahead.py",
        "ledger_sha256": ROOT / "tools/retarget/run_with_gpu_ledger.py",
    }
    for name, path in code_paths.items():
        require_exact(path, prereg["immutable_code"][name])

    input_paths = {
        "base_checkpoint_sha256": OLD / "logs/rsl_rl/x2_lower_velocity_flat/2026-07-22_10-19-57_stage219_cleanreset_yawcoverage25_sole12_selfoff_resume2550_to2650_v1/model_2600.pt",
        "zero_residual_checkpoint_sha256": ROOT / "artifacts/retarget/x2_phase_conditioned_residual_phase68/source_stage219_zero_residual.pt",
        "environment_yaml_sha256": OLD / "logs/rsl_rl/x2_lower_velocity_flat/2026-07-22_10-19-57_stage219_cleanreset_yawcoverage25_sole12_selfoff_resume2550_to2650_v1/params/env.yaml",
        "agent_yaml_sha256": OLD / "logs/rsl_rl/x2_lower_velocity_flat/2026-07-22_10-19-57_stage219_cleanreset_yawcoverage25_sole12_selfoff_resume2550_to2650_v1/params/agent.yaml",
        "gait_template_sha256": OLD / "data/processed/x2_official_forward_gait_phase_template_15dof.npz",
        "upper_motion_sha256": ROOT / "artifacts/official_x2/x2_hybrid_phase44_upper_motion.npz",
    }
    for name, path in input_paths.items():
        require_exact(path, prereg["immutable_inputs"][name])

    schedule_paths = []
    for row in prereg["seed_pairs"]:
        path = Path(row["schedule_path"])
        require_exact(path, row["schedule_sha256"])
        require_sidecar(path)
        schedule_paths.append(path)

    present = {path for path in registered_outputs() if path.exists()}
    expected_present = {
        init_path,
        init_path.with_suffix(init_path.suffix + ".sha256"),
        raw_path,
        raw_path.with_suffix(raw_path.suffix + ".sha256"),
        resource_path,
        resource_path.with_suffix(resource_path.suffix + ".sha256"),
        log_path,
        log_path.with_suffix(log_path.suffix + ".sha256"),
    }
    if present != expected_present:
        raise RuntimeError(f"registered output inventory drift: {sorted(map(str, present ^ expected_present))}")

    files = [
        entry(failure_path, failure_path.name, "failure_summary"),
        entry(failure_path.with_suffix(failure_path.suffix + ".sha256"), failure_path.name + ".sha256", "failure_summary_sidecar"),
        entry(ROOT / "reports/retarget/x2_phase72_attempt0_failure.md", "x2_phase72_attempt0_failure.md", "failure_markdown"),
        entry(prereg_path, prereg_path.name, "prereg"),
        entry(prereg_path.with_suffix(prereg_path.suffix + ".sha256"), prereg_path.name + ".sha256", "prereg_sidecar"),
        entry(inventory_path, inventory_path.name, "schedule_inventory"),
        entry(inventory_path.with_suffix(inventory_path.suffix + ".sha256"), inventory_path.name + ".sha256", "schedule_inventory_sidecar"),
    ]
    for index, path in enumerate(schedule_paths):
        files.append(entry(path, path.name, f"schedule_seed{index}"))
        files.append(entry(path.with_suffix(path.suffix + ".sha256"), path.name + ".sha256", f"schedule_seed{index}_sidecar"))
    for path, remote_name, role in (
        (init_path, init_path.name, "initial_commit"),
        (init_path.with_suffix(init_path.suffix + ".sha256"), init_path.name + ".sha256", "initial_commit_sidecar"),
        (raw_path, raw_path.name, "raw_rollout"),
        (raw_path.with_suffix(raw_path.suffix + ".sha256"), raw_path.name + ".sha256", "raw_rollout_sidecar"),
        (resource_path, resource_path.name, "resource_ledger"),
        (resource_path.with_suffix(resource_path.suffix + ".sha256"), resource_path.name + ".sha256", "resource_ledger_sidecar"),
        (log_path, "seed0_plus.log", "immutable_log"),
        (log_path.with_suffix(log_path.suffix + ".sha256"), "seed0_plus.log.sha256", "immutable_log_sidecar"),
    ):
        files.append(entry(path, remote_name, role))

    code_remote_names = {
        "runner_sha256": "run_x2_phase72_antithetic_attempt0.py",
        "helper_sha256": "phase72_antithetic_attempt0.py",
        "run_script_sha256": "run_x2_phase72_antithetic_attempt0.sh",
        "pair_validator_sha256": "validate_x2_phase72_pair_attempt0.py",
        "finalizer_sha256": "finalize_x2_phase72_antithetic_attempt0.py",
        "runner_test_sha256": "test_phase72_antithetic_runner_attempt0.py",
        "helper_test_sha256": "test_phase72_antithetic_attempt0.py",
        "finalizer_test_sha256": "test_phase72_antithetic_finalizer_attempt0.py",
        "schedule_generator_sha256": "prepare_x2_phase72_schedules_attempt0.py",
        "phase60_posture_module_sha256": "x2_native_locomotion_posture_phase60_dependency.py",
        "x2_flat_env_cfg_sha256": "x2_flat_env_cfg_dependency.py",
        "x2_reward_module_sha256": "x2_rewards_dependency.py",
        "x2_gait_module_sha256": "x2_gait_dependency.py",
        "x2_action_module_sha256": "x2_actions_dependency.py",
        "heading_command_module_sha256": "heading_command_dependency.py",
        "modular_env_cfg_sha256": "modular_tracking_env_cfg_dependency.py",
        "x2_robot_cfg_sha256": "x2_robot_cfg_dependency.py",
        "phase68_interface_sha256": "phase68_residual_ppo_dependency.py",
        "residual_module_sha256": "phase_conditioned_knee_residual_dependency.py",
        "phase69_helper_sha256": "phase69_reward_attribution_dependency.py",
        "phase70_helper_sha256": "phase70_long_lookahead_dependency.py",
        "ledger_sha256": "run_with_gpu_ledger_attempt0.py",
    }
    for name, path in code_paths.items():
        files.append(entry(path, code_remote_names[name], f"immutable_code:{name}"))

    input_remote_names = {
        "base_checkpoint_sha256": "base_model_2600.pt",
        "zero_residual_checkpoint_sha256": "source_stage219_zero_residual.pt",
        "environment_yaml_sha256": "stage219_env.yaml",
        "agent_yaml_sha256": "stage219_agent.yaml",
        "gait_template_sha256": "x2_official_forward_gait_phase_template_15dof.npz",
        "upper_motion_sha256": "x2_hybrid_phase44_upper_motion.npz",
    }
    for name, path in input_paths.items():
        files.append(entry(path, input_remote_names[name], f"immutable_input:{name}"))

    builder_path = Path(__file__).resolve()
    files.append(entry(builder_path, builder_path.name, "manifest_builder"))
    if len(files) != 54 or len({row["remote"] for row in files}) != 54:
        raise RuntimeError("expected exactly 54 unique payload files")

    payload = {
        "schema": "x2_phase72_attempt0_failure_backup_manifest_v1",
        "date": "2026-08-12",
        "remote_root": REMOTE_ROOT,
        "decision": failure["decision"],
        "scientific_result": None,
        "launches_consumed": 1,
        "remaining_launches_forbidden": True,
        "replacement_launch_forbidden": True,
        "registered_output_paths": 113,
        "registered_outputs_present": 8,
        "registered_outputs_absent": 105,
        "raw_bundle_sha256": failure["evidence"]["raw_bundle"]["sha256"],
        "ledger_exit_code_reliable": False,
        "files": files,
        "payload_count": len(files),
        "payload_total_bytes": sum(int(row["bytes"]) for row in files),
        "remote_expected_unique_file_count_including_controls": len(files) + 2,
        "remote_listing_file_count": len(files) + 2 if args.remote_verified else None,
        "remote_listing_byte_sizes_match": bool(args.remote_verified),
        "remote_listing_checked_at": args.checked_at if args.remote_verified else None,
        "verification": (
            "All 56 unique remote basenames were listed with exact byte-size equality; SHA256 is preserved locally because bdpan does not expose remote content SHA256."
            if args.remote_verified
            else "Local immutable inventory complete; remote byte-size verification pending."
        ),
    }
    MANIFEST.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    digest = sha256(MANIFEST)
    MANIFEST.with_suffix(MANIFEST.suffix + ".sha256").write_text(
        f"{digest}  {MANIFEST.name}\n", encoding="utf-8"
    )
    print(json.dumps({"payloads": len(files), "bytes": payload["payload_total_bytes"], "manifest_sha256": digest, "remote_verified": args.remote_verified}))


if __name__ == "__main__":
    main()
