#!/usr/bin/env python3
"""Build and validate the immutable Phase76 pairing-lifecycle PASS backup."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path("/home/yu/projects/ZHY/CWI_CrossEmbodiment_Sim")
OLD = Path("/home/yu/x2_teleop_final/x2_sonic")
REMOTE_ROOT = "HUMAN+/HUMANPLUS_X2_PLUS/2026-08-12/task2_backward_pitch/phase76_pairing_lifecycle_pass"
MANIFEST = ROOT / "reports/retarget/x2_phase76_pairing_lifecycle_pass_backup_manifest.json"
PREREG_SHA256 = "83cf7a50eed9f8b21023e5fe0f91dff6d29873149504145e4cd95c9d2261b367"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def require_exact(path: Path, expected: str) -> None:
    if not path.is_file() or sha256(path) != expected:
        raise RuntimeError(f"immutable file absent or changed: {path}")


def require_sidecar(path: Path) -> Path:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    expected = f"{sha256(path)}  {path.name}\n"
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8") != expected:
        raise RuntimeError(f"sidecar absent or mismatched: {sidecar}")
    return sidecar


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
    artifact = ROOT / "artifacts/retarget/x2_phase76_pairing_lifecycle_preflight"
    report = ROOT / "reports/retarget"
    log = Path("/tmp/x2_phase76_logs")
    paths = [
        report / "x2_phase76_pairing_lifecycle_preflight_result.json",
        report / "x2_phase76_pairing_lifecycle_preflight_result.json.sha256",
        report / "x2_phase76_pairing_lifecycle_preflight.md",
    ]
    for seed in range(3):
        paths.extend(
            [
                artifact / f"seed{seed}_commit.json",
                artifact / f"seed{seed}_commit.json.sha256",
                report / f"x2_phase76_seed{seed}_screen.json",
                report / f"x2_phase76_seed{seed}_screen.json.sha256",
                report / f"x2_phase76_seed{seed}_failure.json",
                report / f"x2_phase76_seed{seed}_failure.json.sha256",
                report / f"x2_phase76_seed{seed}_resource.json",
                report / f"x2_phase76_seed{seed}_resource.json.sha256",
                log / f"seed{seed}.log",
                log / f"seed{seed}.log.sha256",
            ]
        )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-verified", action="store_true")
    parser.add_argument("--checked-at")
    args = parser.parse_args()

    report = ROOT / "reports/retarget"
    artifact = ROOT / "artifacts/retarget/x2_phase76_pairing_lifecycle_preflight"
    log_dir = Path("/tmp/x2_phase76_logs")
    prereg = report / "x2_phase76_pairing_lifecycle_preflight_prereg.json"
    result = report / "x2_phase76_pairing_lifecycle_preflight_result.json"
    result_md = report / "x2_phase76_pairing_lifecycle_preflight.md"
    require_exact(prereg, PREREG_SHA256)
    require_sidecar(prereg)
    require_sidecar(result)
    prereg_payload = json.loads(prereg.read_text(encoding="utf-8"))
    result_payload = json.loads(result.read_text(encoding="utf-8"))

    if result_payload.get("decision") != "PASS_INITIAL_PAIRING_PREFLIGHT_ONLY":
        raise RuntimeError("Phase76 final decision is not PASS")
    if result_payload.get("prereg_sha256") != PREREG_SHA256:
        raise RuntimeError("final-to-prereg provenance drift")
    if result_payload.get("seed_indices") != [0, 1, 2] or result_payload.get("launch_count") != 3:
        raise RuntimeError("final launch inventory drift")
    if result_payload.get("source_initial_hashes_distinct") is not True:
        raise RuntimeError("source initial hashes were not distinct")
    for key in ("scientific_metrics_present", "phase77_scientific_preregistration_unlocked", "phase77_launch_unlocked", "training_unlocked", "deployment_unlocked"):
        if result_payload.get(key) is not False:
            raise RuntimeError(f"unexpected authority in final result: {key}")
    for key in ("physics_rollout_steps", "optimizer_steps", "backward_calls", "checkpoint_count"):
        if result_payload.get(key) != 0:
            raise RuntimeError(f"unexpected work in final result: {key}")

    evidence: list[tuple[Path, str]] = []
    cumulative_positive_delta = 0
    initial_hashes: list[str] = []
    screen_by_seed = {int(Path(row["path"]).stem.split("seed", 1)[1].split("_", 1)[0]): row for row in result_payload["screens"]}
    resource_by_seed = {int(Path(row["path"]).stem.split("seed", 1)[1].split("_", 1)[0]): row for row in result_payload["resources"]}
    for seed, env_seed in enumerate((760041, 760042, 760043)):
        commit = artifact / f"seed{seed}_commit.json"
        screen = report / f"x2_phase76_seed{seed}_screen.json"
        resource = report / f"x2_phase76_seed{seed}_resource.json"
        log = log_dir / f"seed{seed}.log"
        for path in (commit, screen, resource, log):
            require_sidecar(path)
        screen_payload = json.loads(screen.read_text(encoding="utf-8"))
        resource_payload = json.loads(resource.read_text(encoding="utf-8"))
        if screen_payload.get("seed_index") != seed or screen_payload.get("env_seed") != env_seed:
            raise RuntimeError(f"seed identity drift: {seed}")
        gates = {
            "decision": screen_payload.get("decision") == "PASS_INITIAL_PAIRING_LAUNCH",
            "all_pairs_pass": screen_payload.get("all_pairs_pass") is True,
            "unknown_fields": screen_payload.get("unknown_mutable_tensor_fields_empty") is True,
            "mixed_device": screen_payload.get("mixed_device_evidence_path_exercised") is True,
            "cpu_manifest": len(screen_payload.get("cpu_manifest_fields", [])) > 0,
            "nondegenerate": screen_payload.get("source_donor_initial", {}).get("nondegenerate") is True,
            "observation": screen_payload.get("observation_recomputed_after_clone") is True,
            "readback": screen_payload.get("physx_low_level_readback_present") is True,
            "no_science": screen_payload.get("scientific_metrics_present") is False,
            "commit": screen_payload.get("commit_sha256") == sha256(commit),
        }
        if not all(gates.values()):
            raise RuntimeError(f"screen gate failure seed={seed}: {gates}")
        if screen_by_seed[seed].get("sha256") != sha256(screen):
            raise RuntimeError(f"final screen provenance drift seed={seed}")
        resource_gates = {
            "schema": resource_payload.get("schema") == "x2_gpu_deadline_ledger_phase76_v1",
            "label": resource_payload.get("label") == f"phase76_seed{seed}_single_process_pairing_preflight",
            "exit": resource_payload.get("exit_code") == 0 and resource_payload.get("raw_returncode") == 0,
            "autonomous": resource_payload.get("autonomous_exit") is True,
            "timeout": resource_payload.get("timed_out") is False,
            "signals": resource_payload.get("term_sent") is False and resource_payload.get("kill_sent") is False,
            "cleanup": resource_payload.get("forced_cleanup") is False,
            "elapsed": float(resource_payload.get("elapsed_s", 121)) <= 120,
            "gpu": resource_payload.get("gpu") is not None and int(resource_payload["gpu"]["memory_used_peak_mib"]) <= 8192,
            "disk": int(resource_payload.get("disk_used_delta_bytes", 536870913)) <= 536870912,
            "free": int(resource_payload.get("disk_after", {}).get("free_bytes", 0)) >= 28 * 1024**3,
            "log": resource_payload.get("log_sha256") == sha256(log),
        }
        if not all(resource_gates.values()):
            raise RuntimeError(f"resource gate failure seed={seed}: {resource_gates}")
        if resource_by_seed[seed].get("sha256") != sha256(resource):
            raise RuntimeError(f"final resource provenance drift seed={seed}")
        initial_hashes.append(screen_payload["source_donor_initial"]["combined_sha256"])
        cumulative_positive_delta += max(0, int(resource_payload["disk_used_delta_bytes"]))
        evidence.extend(((commit, f"technical_commit_seed{seed}"), (screen, f"technical_screen_seed{seed}"), (resource, f"resource_ledger_seed{seed}"), (log, f"immutable_log_seed{seed}")))
    if len(set(initial_hashes)) != 3 or initial_hashes != result_payload["source_initial_hashes"]:
        raise RuntimeError("cross-seed initial-hash provenance drift")
    if cumulative_positive_delta > 1610612736:
        raise RuntimeError("cumulative disk delta exceeded preregistered limit")

    code_paths = {
        "runner_sha256": ROOT / "scripts/run_x2_phase76_pairing_lifecycle_preflight.py",
        "helper_sha256": ROOT / "src/cwi_x2/phase76_pairing_preflight.py",
        "run_script_sha256": ROOT / "scripts/run_x2_phase76_pairing_lifecycle_preflight.sh",
        "test_sha256": ROOT / "tests/test_phase76_pairing_lifecycle_preflight.py",
        "finalizer_sha256": ROOT / "tools/retarget/finalize_x2_phase76_pairing_lifecycle_preflight.py",
        "finalizer_test_sha256": ROOT / "tests/test_phase76_pairing_lifecycle_finalizer.py",
        "phase60_posture_module_sha256": ROOT / "src/x2_native_locomotion_posture_phase60.py",
        "x2_flat_env_cfg_sha256": OLD / "sonic_x2_sandbox/gear_sonic/envs/x2_velocity/flat_env_cfg.py",
        "x2_reward_module_sha256": OLD / "sonic_x2_sandbox/gear_sonic/envs/x2_velocity/rewards.py",
        "x2_gait_module_sha256": OLD / "sonic_x2_sandbox/gear_sonic/envs/x2_velocity/gait.py",
        "x2_action_module_sha256": OLD / "sonic_x2_sandbox/gear_sonic/envs/manager_env/mdp/actions.py",
        "x2_gait_action_module_sha256": OLD / "sonic_x2_sandbox/gear_sonic/envs/x2_velocity/actions.py",
        "upper_hook_sha256": ROOT / "hooks/sitecustomize.py",
        "heading_command_module_sha256": OLD / "sonic_x2_sandbox/gear_sonic/envs/x2_velocity/heading_command.py",
        "modular_env_cfg_sha256": OLD / "sonic_x2_sandbox/gear_sonic/envs/manager_env/modular_tracking_env_cfg.py",
        "x2_robot_cfg_sha256": OLD / "sonic_x2_sandbox/gear_sonic/envs/manager_env/robots/x2.py",
        "phase68_interface_sha256": ROOT / "src/cwi_x2/phase68_residual_ppo.py",
        "residual_module_sha256": ROOT / "src/cwi_x2/phase_conditioned_knee_residual.py",
        "phase69_helper_sha256": ROOT / "src/cwi_x2/phase69_reward_attribution.py",
        "ledger_sha256": ROOT / "tools/retarget/run_with_gpu_deadline_ledger_phase76.py",
        "supervisor_test_sha256": ROOT / "tests/test_phase76_deadline_supervisor.py",
    }
    code_names = {
        "runner_sha256": "run_x2_phase76_pairing_lifecycle_preflight.py",
        "helper_sha256": "phase76_pairing_preflight.py",
        "run_script_sha256": "run_x2_phase76_pairing_lifecycle_preflight.sh",
        "test_sha256": "test_phase76_pairing_lifecycle_preflight.py",
        "finalizer_sha256": "finalize_x2_phase76_pairing_lifecycle_preflight.py",
        "finalizer_test_sha256": "test_phase76_pairing_lifecycle_finalizer.py",
        "phase60_posture_module_sha256": "x2_native_locomotion_posture_phase60_dependency.py",
        "x2_flat_env_cfg_sha256": "x2_flat_env_cfg_dependency.py",
        "x2_reward_module_sha256": "x2_rewards_dependency.py",
        "x2_gait_module_sha256": "x2_gait_dependency.py",
        "x2_action_module_sha256": "x2_manager_actions_dependency.py",
        "x2_gait_action_module_sha256": "x2_gait_actions_dependency.py",
        "upper_hook_sha256": "sitecustomize_upper_hook.py",
        "heading_command_module_sha256": "heading_command_dependency.py",
        "modular_env_cfg_sha256": "modular_tracking_env_cfg_dependency.py",
        "x2_robot_cfg_sha256": "x2_robot_cfg_dependency.py",
        "phase68_interface_sha256": "phase68_residual_ppo_dependency.py",
        "residual_module_sha256": "phase_conditioned_knee_residual_dependency.py",
        "phase69_helper_sha256": "phase69_reward_attribution_dependency.py",
        "ledger_sha256": "run_with_gpu_deadline_ledger_phase76.py",
        "supervisor_test_sha256": "test_phase76_deadline_supervisor.py",
    }
    for name, path in code_paths.items():
        require_exact(path, prereg_payload["immutable_code"][name])

    dynamic_sources = {
        "phase75_runner_source.py": (ROOT / "scripts/run_x2_phase75_pairing_preflight.py", prereg_payload["phase75_disclosure"]["frozen_runner_sha256"]),
        "phase75_finalizer_source.py": (ROOT / "tools/retarget/finalize_x2_phase75_pairing_preflight.py", prereg_payload["phase75_disclosure"]["frozen_finalizer_sha256"]),
        "phase74_runner_source.py": (ROOT / "scripts/run_x2_phase74_pairing_preflight.py", "3430a5130f8c04ea5083e932f387f69a91e781564333568c2de060e6187effcc"),
        "phase74_finalizer_source.py": (ROOT / "tools/retarget/finalize_x2_phase74_pairing_preflight.py", "48ea1b3fc8de1500c13f371b110a901c162d457227d43e3421f75d3927b16955"),
    }
    for path, digest in dynamic_sources.values():
        require_exact(path, digest)

    input_paths = {
        "base_checkpoint_sha256": OLD / "logs/rsl_rl/x2_lower_velocity_flat/2026-07-22_10-19-57_stage219_cleanreset_yawcoverage25_sole12_selfoff_resume2550_to2650_v1/model_2600.pt",
        "zero_residual_checkpoint_sha256": ROOT / "artifacts/retarget/x2_phase_conditioned_residual_phase68/source_stage219_zero_residual.pt",
        "environment_yaml_sha256": OLD / "logs/rsl_rl/x2_lower_velocity_flat/2026-07-22_10-19-57_stage219_cleanreset_yawcoverage25_sole12_selfoff_resume2550_to2650_v1/params/env.yaml",
        "agent_yaml_sha256": OLD / "logs/rsl_rl/x2_lower_velocity_flat/2026-07-22_10-19-57_stage219_cleanreset_yawcoverage25_sole12_selfoff_resume2550_to2650_v1/params/agent.yaml",
        "gait_template_sha256": OLD / "data/processed/x2_official_forward_gait_phase_template_15dof.npz",
        "upper_motion_sha256": ROOT / "artifacts/official_x2/x2_hybrid_phase44_upper_motion.npz",
    }
    input_names = {
        "base_checkpoint_sha256": "base_model_2600.pt",
        "zero_residual_checkpoint_sha256": "source_stage219_zero_residual.pt",
        "environment_yaml_sha256": "stage219_env.yaml",
        "agent_yaml_sha256": "stage219_agent.yaml",
        "gait_template_sha256": "x2_official_forward_gait_phase_template_15dof.npz",
        "upper_motion_sha256": "x2_hybrid_phase44_upper_motion.npz",
    }
    for name, path in input_paths.items():
        require_exact(path, prereg_payload["immutable_inputs"][name])

    phase75_controls = {
        "phase75_failure_freeze.json": (report / "x2_phase75_seed0_resource_failure_freeze.json", prereg_payload["phase75_disclosure"]["freeze_sha256"]),
        "phase75_failure_backup_manifest.json": (report / "x2_phase75_seed0_resource_failure_backup_manifest.json", prereg_payload["phase75_disclosure"]["backup_manifest_sha256"]),
    }
    for path, digest in phase75_controls.values():
        require_exact(path, digest)
        require_sidecar(path)

    expected_present = {result, result.with_suffix(result.suffix + ".sha256"), result_md}
    for path, _ in evidence:
        expected_present.add(path)
        expected_present.add(path.with_suffix(path.suffix + ".sha256"))
    present = {path for path in registered_outputs() if path.exists()}
    if present != expected_present or len(present) != 27:
        raise RuntimeError(f"registered output inventory drift: {sorted(map(str, present ^ expected_present))}")

    files = [
        entry(prereg, prereg.name, "prereg"),
        entry(require_sidecar(prereg), prereg.name + ".sha256", "prereg_sidecar"),
        entry(result, result.name, "final_result"),
        entry(require_sidecar(result), result.name + ".sha256", "final_result_sidecar"),
        entry(result_md, result_md.name, "final_markdown"),
    ]
    for path, role in evidence:
        files.append(entry(path, path.name, role))
        files.append(entry(require_sidecar(path), path.name + ".sha256", role + "_sidecar"))
    for name, path in code_paths.items():
        files.append(entry(path, code_names[name], f"immutable_code:{name}"))
    for remote_name, (path, _) in dynamic_sources.items():
        files.append(entry(path, remote_name, "dynamic_wrapper_source"))
    for name, path in input_paths.items():
        files.append(entry(path, input_names[name], f"immutable_input:{name}"))
    for remote_name, (path, _) in phase75_controls.items():
        files.append(entry(path, remote_name, "phase75_provenance_control"))
        files.append(entry(require_sidecar(path), remote_name + ".sha256", "phase75_provenance_control_sidecar"))
    builder = Path(__file__).resolve()
    files.append(entry(builder, builder.name, "manifest_builder"))
    if len(files) != 65 or len({row["remote"] for row in files}) != 65:
        raise RuntimeError("expected exactly 65 unique payload files")

    elapsed = [json.loads((report / f"x2_phase76_seed{seed}_resource.json").read_text(encoding="utf-8"))["elapsed_s"] for seed in range(3)]
    gpu_peaks = [json.loads((report / f"x2_phase76_seed{seed}_resource.json").read_text(encoding="utf-8"))["gpu"]["memory_used_peak_mib"] for seed in range(3)]
    payload = {
        "schema": "x2_phase76_pairing_lifecycle_pass_backup_manifest_v1",
        "date": "2026-08-12",
        "remote_root": REMOTE_ROOT,
        "decision": "PASS_INITIAL_PAIRING_PREFLIGHT_ONLY",
        "scientific_result": None,
        "prereg_sha256": PREREG_SHA256,
        "launches_started": 3,
        "launches_accepted": 3,
        "autonomous_exit_zero_count": 3,
        "registered_output_paths": 33,
        "registered_outputs_present": 27,
        "registered_outputs_absent": 6,
        "elapsed_s": elapsed,
        "gpu_memory_used_peak_mib": gpu_peaks,
        "cumulative_positive_disk_delta_bytes": cumulative_positive_delta,
        "physics_rollout_steps": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "checkpoint_count": 0,
        "phase77_shadow_preregistration_unlocked": True,
        "phase77_scientific_preregistration_unlocked": False,
        "phase77_launch_unlocked": False,
        "files": files,
        "payload_count": len(files),
        "payload_total_bytes": sum(int(row["bytes"]) for row in files),
        "remote_expected_unique_file_count_including_controls": len(files) + 2,
        "remote_listing_file_count": len(files) + 2 if args.remote_verified else None,
        "remote_listing_byte_sizes_match": bool(args.remote_verified),
        "remote_listing_checked_at": args.checked_at if args.remote_verified else None,
        "verification": "All 67 unique remote basenames were listed with exact byte-size equality; SHA256 is preserved in this manifest because bdpan does not expose remote content SHA256." if args.remote_verified else "Local immutable inventory complete; remote byte-size verification pending.",
    }
    MANIFEST.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    digest = sha256(MANIFEST)
    MANIFEST.with_suffix(MANIFEST.suffix + ".sha256").write_text(f"{digest}  {MANIFEST.name}\n", encoding="utf-8")
    print(json.dumps({"payloads": len(files), "bytes": payload["payload_total_bytes"], "manifest_sha256": digest, "remote_verified": args.remote_verified}))


if __name__ == "__main__":
    main()
