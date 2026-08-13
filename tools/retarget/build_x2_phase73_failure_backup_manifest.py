#!/usr/bin/env python3
"""Build and validate the immutable Phase73 failure backup inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path("/home/yu/projects/ZHY/CWI_CrossEmbodiment_Sim")
OLD = Path("/home/yu/x2_teleop_final/x2_sonic")
REMOTE_ROOT = "HUMAN+/HUMANPLUS_X2_PLUS/2026-08-12/task2_backward_pitch/phase73_failure"
MANIFEST = ROOT / "reports/retarget/x2_phase73_failure_backup_manifest.json"


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
    artifact = ROOT / "artifacts/retarget/x2_phase73_antithetic"
    report = ROOT / "reports/retarget"
    log = Path("/tmp/x2_phase73_logs")
    paths = [
        report / "x2_phase73_antithetic_result.json",
        report / "x2_phase73_antithetic_result.json.sha256",
        report / "x2_phase73_antithetic.md",
    ]
    for seed in range(5):
        paths.extend(
            [
                artifact / f"seed{seed}_initial_commit.json",
                artifact / f"seed{seed}_initial_commit.json.sha256",
                artifact / f"seed{seed}_pair.pt",
                artifact / f"seed{seed}_pair.pt.sha256",
                report / f"x2_phase73_seed{seed}_pair_result.json",
                report / f"x2_phase73_seed{seed}_pair_result.json.sha256",
            ]
        )
        for sign in ("plus", "minus"):
            paths.extend(
                [
                    artifact / f"seed{seed}_{sign}_rollout.pt",
                    artifact / f"seed{seed}_{sign}_rollout.pt.sha256",
                    report / f"x2_phase73_seed{seed}_{sign}_screen.json",
                    report / f"x2_phase73_seed{seed}_{sign}_screen.json.sha256",
                    report / f"x2_phase73_seed{seed}_{sign}_resource.json",
                    report / f"x2_phase73_seed{seed}_{sign}_resource.json.sha256",
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

    report = ROOT / "reports/retarget"
    artifact = ROOT / "artifacts/retarget/x2_phase73_antithetic"
    logs = Path("/tmp/x2_phase73_logs")
    prereg_path = report / "x2_phase73_antithetic_prereg.json"
    inventory_path = report / "x2_phase73_schedule_inventory.json"
    failure_path = report / "x2_phase73_failure.json"
    failure_md = report / "x2_phase73_failure.md"
    init_path = artifact / "seed0_initial_commit.json"
    raw_path = artifact / "seed0_plus_rollout.pt"
    plus_screen = report / "x2_phase73_seed0_plus_screen.json"
    plus_resource = report / "x2_phase73_seed0_plus_resource.json"
    minus_resource = report / "x2_phase73_seed0_minus_resource.json"
    plus_log = logs / "seed0_plus.log"
    minus_log = logs / "seed0_minus.log"

    frozen = {
        prereg_path: "c243d59f1332c940f2009a0d09c044a80d0900a27b4cc065e1c10a74cf8ec49e",
        inventory_path: "a215f7790141a6198c6d28bfb1bb10825b6634e20a206a9af2b76864cf3334e0",
        failure_path: "32a4ceef83bb8b7b02f84ccd6f64e3b3b7ab8b1ebb3747f3bea3290168c90433",
        init_path: "ac1aee4930c713cdd8e7b064484835ccab7c79591c771274a6d70ebf5dbc0471",
        raw_path: "c64b146ddb6e42015b21430dbc9aa27c0b46756b34203f2dd149d929a4cb9c9b",
        plus_screen: "236881ebd2c0a0ff20a38bd15d41423f2ce11b3b86ee2f6cc3125f21273b9c70",
        plus_resource: "87712bd5b512753f957d35a2ba861fea3a7fd8676f63161c58d1eb4a2f79c600",
        minus_resource: "0cbe73d35eba5446336b990091a566952096f52c85d01b04639f9333b52f53d3",
        plus_log: "1d75246f458b7a47c97ef39f42954aec0254875533d1f28cf46df911e0c60b23",
        minus_log: "2f5227a1a6272cd1acc173c94bf4735e7ec9c87c011822274e73c34b89ac0c64",
    }
    for path, digest in frozen.items():
        require_exact(path, digest)
        require_sidecar(path)

    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    plus = json.loads(plus_screen.read_text(encoding="utf-8"))
    plus_ledger = json.loads(plus_resource.read_text(encoding="utf-8"))
    minus_ledger = json.loads(minus_resource.read_text(encoding="utf-8"))
    if failure.get("decision") != "FAIL_INVALID_STOP" or failure.get("scientific_result") is not None:
        raise RuntimeError("failure decision drift")
    if plus.get("decision") != "VALID_PENDING_PAIR":
        raise RuntimeError("plus technical screen drift")
    if any(plus.get(name) != 0 for name in ("optimizer_steps", "backward_calls", "checkpoint_count")):
        raise RuntimeError("zero-optimizer provenance drift")
    if plus_ledger.get("exit_code") != 0 or minus_ledger.get("exit_code") != 0:
        raise RuntimeError("resource ledger provenance drift")
    if "RuntimeError: Phase73 pair initial tensors are not bitwise identical" not in minus_log.read_text(encoding="utf-8"):
        raise RuntimeError("failure traceback drift")

    code_paths = {
        "runner_sha256": ROOT / "scripts/run_x2_phase73_antithetic.py",
        "helper_sha256": ROOT / "src/cwi_x2/phase73_antithetic.py",
        "run_script_sha256": ROOT / "scripts/run_x2_phase73_antithetic.sh",
        "pair_validator_sha256": ROOT / "tools/retarget/validate_x2_phase73_pair.py",
        "finalizer_sha256": ROOT / "tools/retarget/finalize_x2_phase73_antithetic.py",
        "runner_test_sha256": ROOT / "tests/test_phase73_antithetic_runner.py",
        "helper_test_sha256": ROOT / "tests/test_phase73_antithetic.py",
        "finalizer_test_sha256": ROOT / "tests/test_phase73_antithetic_finalizer.py",
        "schedule_generator_sha256": ROOT / "tools/retarget/prepare_x2_phase73_schedules.py",
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

    wrapper_sources = {
        "phase72_runner_source": (ROOT / "scripts/run_x2_phase72_antithetic.py", "444482b29d05303b6b957b0f043d588c7f67b5bedc231912e33f7e47cccfb335"),
        "phase72_helper_source": (ROOT / "src/cwi_x2/phase72_antithetic.py", "9201a323e593215394879fe61c3d066020dbbe4a4574d33c250db26906310cb2"),
        "phase72_pair_validator_source": (ROOT / "tools/retarget/validate_x2_phase72_pair.py", "f744bbb03f1dd1ce7de4ad79c2b17aa52b76cf49225455ad219f941aadd587bf"),
        "phase72_finalizer_source": (ROOT / "tools/retarget/finalize_x2_phase72_antithetic.py", "0e7e3cbf4c292f53984d54607d382e1c4326d9da56d6ecce6af3de61f8c20197"),
    }
    for path, digest in wrapper_sources.values():
        require_exact(path, digest)

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

    schedule_paths: list[Path] = []
    for row in prereg["seed_pairs"]:
        path = Path(row["schedule_path"])
        require_exact(path, row["schedule_sha256"])
        require_sidecar(path)
        schedule_paths.append(path)

    present = {path for path in registered_outputs() if path.exists()}
    evidence_outputs = [init_path, raw_path, plus_screen, plus_resource, minus_resource, plus_log, minus_log]
    expected_present = set(evidence_outputs)
    expected_present.update(path.with_suffix(path.suffix + ".sha256") for path in evidence_outputs)
    if present != expected_present:
        raise RuntimeError(f"registered output inventory drift: {sorted(map(str, present ^ expected_present))}")

    files = [
        entry(failure_path, failure_path.name, "failure_summary"),
        entry(failure_path.with_suffix(failure_path.suffix + ".sha256"), failure_path.name + ".sha256", "failure_summary_sidecar"),
        entry(failure_md, failure_md.name, "failure_markdown"),
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
        (plus_screen, plus_screen.name, "plus_technical_screen"),
        (plus_screen.with_suffix(plus_screen.suffix + ".sha256"), plus_screen.name + ".sha256", "plus_technical_screen_sidecar"),
        (plus_resource, plus_resource.name, "plus_resource_ledger"),
        (plus_resource.with_suffix(plus_resource.suffix + ".sha256"), plus_resource.name + ".sha256", "plus_resource_ledger_sidecar"),
        (minus_resource, minus_resource.name, "minus_resource_ledger"),
        (minus_resource.with_suffix(minus_resource.suffix + ".sha256"), minus_resource.name + ".sha256", "minus_resource_ledger_sidecar"),
        (plus_log, plus_log.name, "plus_immutable_log"),
        (plus_log.with_suffix(plus_log.suffix + ".sha256"), plus_log.name + ".sha256", "plus_immutable_log_sidecar"),
        (minus_log, minus_log.name, "minus_failure_log"),
        (minus_log.with_suffix(minus_log.suffix + ".sha256"), minus_log.name + ".sha256", "minus_failure_log_sidecar"),
    ):
        files.append(entry(path, remote_name, role))

    code_remote_names = {
        "runner_sha256": "run_x2_phase73_antithetic.py",
        "helper_sha256": "phase73_antithetic.py",
        "run_script_sha256": "run_x2_phase73_antithetic.sh",
        "pair_validator_sha256": "validate_x2_phase73_pair.py",
        "finalizer_sha256": "finalize_x2_phase73_antithetic.py",
        "runner_test_sha256": "test_phase73_antithetic_runner.py",
        "helper_test_sha256": "test_phase73_antithetic.py",
        "finalizer_test_sha256": "test_phase73_antithetic_finalizer.py",
        "schedule_generator_sha256": "prepare_x2_phase73_schedules.py",
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
        "ledger_sha256": "run_with_gpu_ledger.py",
    }
    for name, path in code_paths.items():
        files.append(entry(path, code_remote_names[name], f"immutable_code:{name}"))

    wrapper_remote_names = {
        "phase72_runner_source": "phase72_frozen_runner_source.py",
        "phase72_helper_source": "phase72_frozen_helper_source.py",
        "phase72_pair_validator_source": "phase72_frozen_pair_validator_source.py",
        "phase72_finalizer_source": "phase72_frozen_finalizer_source.py",
    }
    for name, (path, _) in wrapper_sources.items():
        files.append(entry(path, wrapper_remote_names[name], f"wrapper_runtime_dependency:{name}"))

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
    if len(files) != 64 or len({row["remote"] for row in files}) != 64:
        raise RuntimeError("expected exactly 64 unique payload files")

    payload = {
        "schema": "x2_phase73_failure_backup_manifest_v1",
        "date": "2026-08-12",
        "remote_root": REMOTE_ROOT,
        "decision": failure["decision"],
        "scientific_result": None,
        "launches_started": 2,
        "launches_with_complete_rollout": 1,
        "launches_not_started": 8,
        "remaining_launches_forbidden": True,
        "replacement_launch_forbidden": True,
        "registered_output_paths": 113,
        "registered_outputs_present": 14,
        "registered_outputs_absent": 99,
        "raw_bundle_sha256": failure["evidence"]["seed0_plus_rollout"]["sha256"],
        "ledger_exit_code_reliable": False,
        "files": files,
        "payload_count": len(files),
        "payload_total_bytes": sum(int(row["bytes"]) for row in files),
        "remote_expected_unique_file_count_including_controls": len(files) + 2,
        "remote_listing_file_count": len(files) + 2 if args.remote_verified else None,
        "remote_listing_byte_sizes_match": bool(args.remote_verified),
        "remote_listing_checked_at": args.checked_at if args.remote_verified else None,
        "verification": (
            "All 66 unique remote basenames were listed with exact byte-size equality; SHA256 is preserved locally because bdpan does not expose remote content SHA256."
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
