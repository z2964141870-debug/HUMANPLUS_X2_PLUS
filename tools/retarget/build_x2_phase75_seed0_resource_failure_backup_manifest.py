#!/usr/bin/env python3
"""Build and validate the immutable Phase75 seed0 resource-failure backup."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path("/home/yu/projects/ZHY/CWI_CrossEmbodiment_Sim")
OLD = Path("/home/yu/x2_teleop_final/x2_sonic")
REMOTE_ROOT = "HUMAN+/HUMANPLUS_X2_PLUS/2026-08-12/task2_backward_pitch/phase75_seed0_resource_failure"
MANIFEST = ROOT / "reports/retarget/x2_phase75_seed0_resource_failure_backup_manifest.json"


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
    return {"role": role, "path": shown_path(path), "remote": f"{REMOTE_ROOT}/{remote_name}", "bytes": path.stat().st_size, "sha256": sha256(path)}


def registered_outputs() -> list[Path]:
    artifact = ROOT / "artifacts/retarget/x2_phase75_pairing_preflight"
    report = ROOT / "reports/retarget"
    log = Path("/tmp/x2_phase75_logs")
    paths = [report / "x2_phase75_pairing_preflight_result.json", report / "x2_phase75_pairing_preflight_result.json.sha256", report / "x2_phase75_pairing_preflight.md"]
    for seed in range(3):
        paths.extend([artifact / f"seed{seed}_commit.json", artifact / f"seed{seed}_commit.json.sha256", report / f"x2_phase75_seed{seed}_screen.json", report / f"x2_phase75_seed{seed}_screen.json.sha256", report / f"x2_phase75_seed{seed}_failure.json", report / f"x2_phase75_seed{seed}_failure.json.sha256", report / f"x2_phase75_seed{seed}_resource.json", report / f"x2_phase75_seed{seed}_resource.json.sha256", log / f"seed{seed}.log", log / f"seed{seed}.log.sha256"])
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-verified", action="store_true")
    parser.add_argument("--checked-at")
    args = parser.parse_args()
    report = ROOT / "reports/retarget"
    prereg = report / "x2_phase75_pairing_preflight_prereg.json"
    commit = ROOT / "artifacts/retarget/x2_phase75_pairing_preflight/seed0_commit.json"
    screen = report / "x2_phase75_seed0_screen.json"
    resource = report / "x2_phase75_seed0_resource.json"
    log = Path("/tmp/x2_phase75_logs/seed0.log")
    freeze = report / "x2_phase75_seed0_resource_failure_freeze.json"
    freeze_md = report / "x2_phase75_seed0_resource_failure_freeze.md"
    frozen = {
        prereg: "ab32d7b1a1d4e6a1cf8d5e00b8d2b5d224d89f426c24e5bd701bb86fade3fdb2",
        commit: "30bb7a1cd998846661c0fd737dcd6f22f110da11742fa43f970414cdbabcc757",
        screen: "ea4536d2d2b8b43a1fa7aeb166c172005d7d4b31bc43702bf00cd89ff56058e9",
        resource: "141de473eeaac127e52a4a456f6bebac1b2e734146e39ebc94eb4029ccc9cf79",
        log: "1de4cea1ea03d730d14d172a25331eba07dda53df40ecd8c5850df45c05ba48d",
        freeze: "6bfdac618c115edc5e54a4797de73f1d88d9fa422cd593fa44038117ed73bc8f",
    }
    for path, digest in frozen.items():
        require_exact(path, digest)
        require_sidecar(path)
    prereg_payload = json.loads(prereg.read_text(encoding="utf-8"))
    screen_payload = json.loads(screen.read_text(encoding="utf-8"))
    resource_payload = json.loads(resource.read_text(encoding="utf-8"))
    freeze_payload = json.loads(freeze.read_text(encoding="utf-8"))
    if screen_payload.get("decision") != "PASS_INITIAL_PAIRING_LAUNCH" or screen_payload.get("all_pairs_pass") is not True:
        raise RuntimeError("technical screen provenance drift")
    if screen_payload.get("commit_sha256") != sha256(commit):
        raise RuntimeError("screen-to-commit provenance drift")
    if resource_payload.get("exit_code") != 0 or float(resource_payload.get("elapsed_s", 0)) <= 120:
        raise RuntimeError("resource failure provenance drift")
    if freeze_payload.get("decision") != "FAIL_TECHNICAL_STOP" or freeze_payload.get("scientific_result") is not None:
        raise RuntimeError("failure freeze drift")

    code_paths = {
        "runner_sha256": ROOT / "scripts/run_x2_phase75_pairing_preflight.py", "helper_sha256": ROOT / "src/cwi_x2/phase75_pairing_preflight.py", "run_script_sha256": ROOT / "scripts/run_x2_phase75_pairing_preflight.sh", "test_sha256": ROOT / "tests/test_phase75_pairing_preflight.py", "finalizer_sha256": ROOT / "tools/retarget/finalize_x2_phase75_pairing_preflight.py", "finalizer_test_sha256": ROOT / "tests/test_phase75_pairing_finalizer.py", "phase60_posture_module_sha256": ROOT / "src/x2_native_locomotion_posture_phase60.py", "x2_flat_env_cfg_sha256": OLD / "sonic_x2_sandbox/gear_sonic/envs/x2_velocity/flat_env_cfg.py", "x2_reward_module_sha256": OLD / "sonic_x2_sandbox/gear_sonic/envs/x2_velocity/rewards.py", "x2_gait_module_sha256": OLD / "sonic_x2_sandbox/gear_sonic/envs/x2_velocity/gait.py", "x2_action_module_sha256": OLD / "sonic_x2_sandbox/gear_sonic/envs/manager_env/mdp/actions.py", "x2_gait_action_module_sha256": OLD / "sonic_x2_sandbox/gear_sonic/envs/x2_velocity/actions.py", "upper_hook_sha256": ROOT / "hooks/sitecustomize.py", "heading_command_module_sha256": OLD / "sonic_x2_sandbox/gear_sonic/envs/x2_velocity/heading_command.py", "modular_env_cfg_sha256": OLD / "sonic_x2_sandbox/gear_sonic/envs/manager_env/modular_tracking_env_cfg.py", "x2_robot_cfg_sha256": OLD / "sonic_x2_sandbox/gear_sonic/envs/manager_env/robots/x2.py", "phase68_interface_sha256": ROOT / "src/cwi_x2/phase68_residual_ppo.py", "residual_module_sha256": ROOT / "src/cwi_x2/phase_conditioned_knee_residual.py", "phase69_helper_sha256": ROOT / "src/cwi_x2/phase69_reward_attribution.py", "ledger_sha256": ROOT / "tools/retarget/run_with_gpu_ledger.py",
    }
    for name, path in code_paths.items():
        require_exact(path, prereg_payload["immutable_code"][name])
    dynamic_sources = {
        "phase74_runner_source": (ROOT / "scripts/run_x2_phase74_pairing_preflight.py", "3430a5130f8c04ea5083e932f387f69a91e781564333568c2de060e6187effcc"),
        "phase74_finalizer_source": (ROOT / "tools/retarget/finalize_x2_phase74_pairing_preflight.py", "48ea1b3fc8de1500c13f371b110a901c162d457227d43e3421f75d3927b16955"),
    }
    for path, digest in dynamic_sources.values():
        require_exact(path, digest)
    input_paths = {
        "base_checkpoint_sha256": OLD / "logs/rsl_rl/x2_lower_velocity_flat/2026-07-22_10-19-57_stage219_cleanreset_yawcoverage25_sole12_selfoff_resume2550_to2650_v1/model_2600.pt", "zero_residual_checkpoint_sha256": ROOT / "artifacts/retarget/x2_phase_conditioned_residual_phase68/source_stage219_zero_residual.pt", "environment_yaml_sha256": OLD / "logs/rsl_rl/x2_lower_velocity_flat/2026-07-22_10-19-57_stage219_cleanreset_yawcoverage25_sole12_selfoff_resume2550_to2650_v1/params/env.yaml", "agent_yaml_sha256": OLD / "logs/rsl_rl/x2_lower_velocity_flat/2026-07-22_10-19-57_stage219_cleanreset_yawcoverage25_sole12_selfoff_resume2550_to2650_v1/params/agent.yaml", "gait_template_sha256": OLD / "data/processed/x2_official_forward_gait_phase_template_15dof.npz", "upper_motion_sha256": ROOT / "artifacts/official_x2/x2_hybrid_phase44_upper_motion.npz",
    }
    for name, path in input_paths.items():
        require_exact(path, prereg_payload["immutable_inputs"][name])
    evidence = [commit, screen, resource, log]
    expected_present = set(evidence)
    expected_present.update(path.with_suffix(path.suffix + ".sha256") for path in evidence)
    present = {path for path in registered_outputs() if path.exists()}
    if present != expected_present:
        raise RuntimeError(f"registered output inventory drift: {sorted(map(str, present ^ expected_present))}")

    files = [entry(freeze, freeze.name, "failure_freeze"), entry(freeze.with_suffix(freeze.suffix + ".sha256"), freeze.name + ".sha256", "failure_freeze_sidecar"), entry(freeze_md, freeze_md.name, "failure_freeze_markdown"), entry(prereg, prereg.name, "prereg"), entry(prereg.with_suffix(prereg.suffix + ".sha256"), prereg.name + ".sha256", "prereg_sidecar")]
    for path, role in ((commit, "technical_commit"), (screen, "technical_screen"), (resource, "resource_ledger"), (log, "immutable_log")):
        files.append(entry(path, path.name, role))
        files.append(entry(path.with_suffix(path.suffix + ".sha256"), path.name + ".sha256", role + "_sidecar"))
    code_remote_names = {"runner_sha256":"run_x2_phase75_pairing_preflight.py","helper_sha256":"phase75_pairing_preflight.py","run_script_sha256":"run_x2_phase75_pairing_preflight.sh","test_sha256":"test_phase75_pairing_preflight.py","finalizer_sha256":"finalize_x2_phase75_pairing_preflight.py","finalizer_test_sha256":"test_phase75_pairing_finalizer.py","phase60_posture_module_sha256":"x2_native_locomotion_posture_phase60_dependency.py","x2_flat_env_cfg_sha256":"x2_flat_env_cfg_dependency.py","x2_reward_module_sha256":"x2_rewards_dependency.py","x2_gait_module_sha256":"x2_gait_dependency.py","x2_action_module_sha256":"x2_actions_dependency.py","x2_gait_action_module_sha256":"x2_gait_actions_dependency.py","upper_hook_sha256":"sitecustomize_upper_hook.py","heading_command_module_sha256":"heading_command_dependency.py","modular_env_cfg_sha256":"modular_tracking_env_cfg_dependency.py","x2_robot_cfg_sha256":"x2_robot_cfg_dependency.py","phase68_interface_sha256":"phase68_residual_ppo_dependency.py","residual_module_sha256":"phase_conditioned_knee_residual_dependency.py","phase69_helper_sha256":"phase69_reward_attribution_dependency.py","ledger_sha256":"run_with_gpu_ledger.py"}
    for name, path in code_paths.items():
        files.append(entry(path, code_remote_names[name], f"immutable_code:{name}"))
    for name, (path, _) in dynamic_sources.items():
        files.append(entry(path, name + ".py", f"dynamic_wrapper_source:{name}"))
    input_remote_names = {"base_checkpoint_sha256":"base_model_2600.pt","zero_residual_checkpoint_sha256":"source_stage219_zero_residual.pt","environment_yaml_sha256":"stage219_env.yaml","agent_yaml_sha256":"stage219_agent.yaml","gait_template_sha256":"x2_official_forward_gait_phase_template_15dof.npz","upper_motion_sha256":"x2_hybrid_phase44_upper_motion.npz"}
    for name, path in input_paths.items():
        files.append(entry(path, input_remote_names[name], f"immutable_input:{name}"))
    builder = Path(__file__).resolve()
    files.append(entry(builder, builder.name, "manifest_builder"))
    if len(files) != 42 or len({row["remote"] for row in files}) != 42:
        raise RuntimeError("expected exactly 42 unique payload files")
    payload = {"schema":"x2_phase75_seed0_resource_failure_backup_manifest_v1","date":"2026-08-12","remote_root":REMOTE_ROOT,"decision":"FAIL_TECHNICAL_STOP","scientific_result":None,"technical_screen_decision":"PASS_INITIAL_PAIRING_LAUNCH","launches_started":1,"launches_accepted":0,"launches_not_started":2,"remaining_launches_forbidden":True,"replacement_launch_forbidden":True,"registered_output_paths":33,"registered_outputs_present":8,"registered_outputs_absent":25,"wall_seconds_limit":120,"elapsed_s":resource_payload["elapsed_s"],"files":files,"payload_count":len(files),"payload_total_bytes":sum(int(row["bytes"]) for row in files),"remote_expected_unique_file_count_including_controls":len(files)+2,"remote_listing_file_count":len(files)+2 if args.remote_verified else None,"remote_listing_byte_sizes_match":bool(args.remote_verified),"remote_listing_checked_at":args.checked_at if args.remote_verified else None,"verification":"All 44 unique remote basenames were listed with exact byte-size equality; SHA256 is preserved locally because bdpan does not expose remote content SHA256." if args.remote_verified else "Local immutable inventory complete; remote byte-size verification pending."}
    MANIFEST.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    digest = sha256(MANIFEST)
    MANIFEST.with_suffix(MANIFEST.suffix + ".sha256").write_text(f"{digest}  {MANIFEST.name}\n", encoding="utf-8")
    print(json.dumps({"payloads":len(files),"bytes":payload["payload_total_bytes"],"manifest_sha256":digest,"remote_verified":args.remote_verified}))


if __name__ == "__main__":
    main()
