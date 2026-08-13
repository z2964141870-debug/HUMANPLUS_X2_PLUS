#!/usr/bin/env python3
"""Expand the frozen Phase76 runner and add one reward-free control-step shadow."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FROZEN_PHASE76 = ROOT / "scripts/run_x2_phase76_pairing_lifecycle_preflight.py"
EXPECTED_PHASE76_SHA256 = "e6cd670a83a76e97bca5f1efa248f11aa94c127716cff14484d8aac5fab86f47"
EXEC_LINE = 'exec(compile(source, str(Path(__file__).resolve()), "exec"), globals(), globals())'


def replace_last(text: str, old: str, new: str) -> str:
    position = text.rfind(old)
    if position < 0:
        raise RuntimeError(f"Phase77 expansion anchor absent: {old!r}")
    return text[:position] + new + text[position + len(old) :]


def expand_runner_without_launch() -> str:
    wrapper = FROZEN_PHASE76.read_text(encoding="utf-8")
    if hashlib.sha256(wrapper.encode()).hexdigest() != EXPECTED_PHASE76_SHA256:
        raise RuntimeError("frozen Phase76 runner drifted before Phase77 expansion")
    if wrapper.count(EXEC_LINE) != 1:
        raise RuntimeError("Phase76 outer runner execution anchor changed")
    first: dict[str, object] = {"__file__": str(FROZEN_PHASE76)}
    exec(
        compile(replace_last(wrapper, EXEC_LINE, "TRANSFORMED_WRAPPER = source"), str(FROZEN_PHASE76), "exec"),
        first,
    )
    transformed = first["TRANSFORMED_WRAPPER"]
    if not isinstance(transformed, str) or transformed.count(EXEC_LINE) != 1:
        raise RuntimeError("Phase76 inner runner expansion changed")
    second: dict[str, object] = {"__file__": str(FROZEN_PHASE76)}
    exec(
        compile(replace_last(transformed, EXEC_LINE, "FINAL_RUNNER = source"), str(FROZEN_PHASE76), "exec"),
        second,
    )
    final_runner = second["FINAL_RUNNER"]
    if not isinstance(final_runner, str):
        raise RuntimeError("Phase76 final runner was not captured")
    return final_runner


source = expand_runner_without_launch()
source = source.replace("Phase76", "Phase77").replace("phase76", "phase77")

path_replacements = {
    'ROOT / "src/cwi_x2/phase77_pairing_preflight.py"': (
        'ROOT / "src/cwi_x2/phase77_one_action_shadow.py"'
    ),
    'ROOT / "scripts/run_x2_phase77_pairing_lifecycle_preflight.sh"': (
        'ROOT / "scripts/run_x2_phase77_one_action_shadow.sh"'
    ),
    'ROOT / "tests/test_phase77_pairing_lifecycle_preflight.py"': (
        'ROOT / "tests/test_phase77_one_action_shadow.py"'
    ),
    'ROOT / "tools/retarget/finalize_x2_phase77_pairing_lifecycle_preflight.py"': (
        'ROOT / "tools/retarget/finalize_x2_phase77_one_action_shadow.py"'
    ),
    'ROOT / "tests/test_phase77_pairing_lifecycle_finalizer.py"': (
        'ROOT / "tests/test_phase77_one_action_shadow_finalizer.py"'
    ),
    'ROOT / "tools/retarget/run_with_gpu_deadline_ledger_phase77.py"': (
        'ROOT / "tools/retarget/run_with_gpu_deadline_ledger_phase76.py"'
    ),
    'ROOT / "tests/test_phase77_deadline_supervisor.py"': (
        'ROOT / "tests/test_phase76_deadline_supervisor.py"'
    ),
}
for old, new in path_replacements.items():
    if source.count(old) != 1:
        raise RuntimeError(f"Phase77 code-path anchor changed: {old}")
    source = source.replace(old, new)

helper_import_old = "from cwi_x2.phase77_pairing_preflight import (  # noqa: E402\n"
helper_import_new = "from cwi_x2.phase77_one_action_shadow import (  # noqa: E402\n"
if source.count(helper_import_old) != 1:
    raise RuntimeError("Phase77 helper import anchor changed")
source = source.replace(helper_import_old, helper_import_new)

permission_replacements = {
    '"phase77_shadow_preregistration_unlocked": False': (
        '"phase78_multistep_shadow_preregistration_unlocked": False'
    ),
    '"phase77_scientific_preregistration_unlocked": False': (
        '"phase78_scientific_preregistration_unlocked": False'
    ),
    '"phase77_launch_unlocked": False': '"phase78_launch_unlocked": False',
}
for old, new in permission_replacements.items():
    if source.count(old) != 1:
        raise RuntimeError(f"Phase77 permission anchor changed: {old}")
    source = source.replace(old, new)

if source.count('"PASS_INITIAL_PAIRING_LAUNCH"') != 1:
    raise RuntimeError("Phase77 per-launch decision anchor changed")
source = source.replace('"PASS_INITIAL_PAIRING_LAUNCH"', '"PASS_ONE_ACTION_SHADOW_LAUNCH"')

import_anchor = "from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402\n"
import_injection = (
    import_anchor
    + "from isaaclab.envs import ManagerBasedEnv  # noqa: E402\n"
    + "from cwi_x2.phase68_residual_ppo import PhysicalKneeResidualVecEnv  # noqa: E402\n"
    + "from cwi_x2.phase77_one_action_shadow import (  # noqa: E402\n"
    + "    native_bool, pair_shared_action, reward_state_unchanged, technical_post_tensors,\n"
    + ")\n"
)
if source.count(import_anchor) != 1:
    raise RuntimeError("Phase77 ManagerBasedRLEnv import anchor changed")
source = source.replace(import_anchor, import_injection)

ledger_entry = '        "supervisor_test_sha256": ROOT / "tests/test_phase76_deadline_supervisor.py",\n'
extra_guards = (
    ledger_entry
    + '        "manager_based_env_sha256": Path("/home/yu/IsaacLab/source/isaaclab/isaaclab/envs/manager_based_env.py"),\n'
    + '        "manager_based_rl_env_sha256": Path("/home/yu/IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py"),\n'
    + '        "action_manager_sha256": Path("/home/yu/IsaacLab/source/isaaclab/isaaclab/managers/action_manager.py"),\n'
    + '        "observation_manager_sha256": Path("/home/yu/IsaacLab/source/isaaclab/isaaclab/managers/observation_manager.py"),\n'
    + '        "reward_manager_sha256": Path("/home/yu/IsaacLab/source/isaaclab/isaaclab/managers/reward_manager.py"),\n'
    + '        "termination_manager_sha256": Path("/home/yu/IsaacLab/source/isaaclab/isaaclab/managers/termination_manager.py"),\n'
)
if source.count(ledger_entry) != 1:
    raise RuntimeError("Phase77 external code guard anchor changed")
source = source.replace(ledger_entry, extra_guards)

report_anchor = "    report = {\n"
shadow_code = r'''    if not valid:
        raise RuntimeError("Phase77 initial pairing gates failed before one-action shadow")
    initial_diagnostics = diagnostics
    initial_unknown_fields = list(unknown_fields)
    shadow = PhysicalKneeResidualVecEnv(wrapped, policy)
    action_observation = shadow.get_observations()
    with torch.inference_mode():
        raw_source_action = policy.source_action(action_observation)
        latent_mean = policy.latent_mean(action_observation)
        zero_latent = torch.zeros((128, 2), device=env.device, dtype=raw_source_action.dtype)
        requested_physical_offset = policy.physical_target_offset_from_latent(zero_latent, action_observation)
    raw_action_diagnostics = pair_diagnostics(raw_source_action, donors, recipients, atol=1.0e-6)
    executed_action = pair_shared_action(raw_source_action, donors, recipients)
    executed_action_diagnostics = pair_diagnostics(executed_action, donors, recipients, atol=0.0)
    zero_residual_contract = bool(
        torch.count_nonzero(latent_mean).item() == 0
        and torch.count_nonzero(requested_physical_offset).item() == 0
    )
    shadow._pending_offset.zero_()
    hook_calls_before = shadow._hook_calls
    sim_steps_before = int(env._sim_step_counter)
    common_steps_before = int(env.common_step_counter)
    episode_length_before = env.episode_length_buf.clone()
    reset_buf_before = env.reset_buf.clone()
    reset_terminated_before = env.reset_terminated.clone()
    reset_time_outs_before = env.reset_time_outs.clone()
    reward_state_before = {
        "env_reward_buf": env.reward_buf.clone(),
        "manager_reward_buf": env.reward_manager._reward_buf.clone(),
        "manager_step_reward": env.reward_manager._step_reward.clone(),
        **{
            f"episode_sum_{name}": value.clone()
            for name, value in env.reward_manager._episode_sums.items()
        },
    }
    forbidden_calls = {"reward_compute": 0, "termination_compute": 0, "reset": 0}
    original_reward_compute = env.reward_manager.compute
    original_termination_compute = env.termination_manager.compute
    original_reset_idx = env._reset_idx

    def forbidden_reward_compute(*_args, **_kwargs):
        forbidden_calls["reward_compute"] += 1
        raise RuntimeError("Phase77 reward compute is forbidden")

    def forbidden_termination_compute(*_args, **_kwargs):
        forbidden_calls["termination_compute"] += 1
        raise RuntimeError("Phase77 termination compute is forbidden")

    def forbidden_reset(*_args, **_kwargs):
        forbidden_calls["reset"] += 1
        raise RuntimeError("Phase77 reset is forbidden")

    env.reward_manager.compute = forbidden_reward_compute
    env.termination_manager.compute = forbidden_termination_compute
    env._reset_idx = forbidden_reset
    try:
        returned_observation, _discarded_extras = ManagerBasedEnv.step(env, executed_action)
    finally:
        env.reward_manager.compute = original_reward_compute
        env.termination_manager.compute = original_termination_compute
        env._reset_idx = original_reset_idx
    del _discarded_extras
    reward_state_after = {
        "env_reward_buf": env.reward_buf,
        "manager_reward_buf": env.reward_manager._reward_buf,
        "manager_step_reward": env.reward_manager._step_reward,
        **{
            f"episode_sum_{name}": value
            for name, value in env.reward_manager._episode_sums.items()
        },
    }
    post_all_tensors, post_unknown_fields, post_metadata = state_tensors(env, policy, residual)
    post_tensors = technical_post_tensors(post_all_tensors)
    post_tensors["returned_policy_observation"] = returned_observation["policy"].detach().clone()
    post_tensors["returned_critic_observation"] = returned_observation["critic"].detach().clone()
    post_tensors["raw_source_action"] = raw_source_action.detach().clone()
    post_tensors["executed_pair_shared_action"] = executed_action.detach().clone()
    post_tensors["shadow_source_target"] = shadow._current_source_target.detach().clone()
    post_tensors["shadow_final_target"] = shadow._current_final_target.detach().clone()
    post_tensors["shadow_effective_offset"] = shadow._current_effective_offset.detach().clone()
    post_diagnostics = {}
    for name, value in sorted(post_tensors.items()):
        if name.startswith("dynamic.sensor.") and "force" in name:
            threshold = float(tolerances["contact_force_max_abs_n"])
        elif value.dtype.is_floating_point:
            threshold = float(tolerances["derived_float_max_abs"])
        else:
            threshold = 0.0
        row = pair_diagnostics(value, donors, recipients, atol=threshold)
        finite_required = not name.endswith(".data.contact_pos_w")
        row["finite_required"] = finite_required
        row["all_finite"] = native_bool(torch.isfinite(value).all()) if value.dtype.is_floating_point else True
        if finite_required:
            row["passed"] = bool(row["passed"] and row["all_finite"])
        if name.startswith("dynamic.sensor.") and "force" in name:
            row["relative_l2_threshold"] = float(tolerances["contact_force_relative_l2"])
            row["passed"] = bool(
                row["passed"]
                and row["relative_l2"] <= float(tolerances["contact_force_relative_l2"])
            )
        post_diagnostics[name] = row
    source_final_target_exact = bool(torch.equal(shadow._current_source_target, shadow._current_final_target))
    effective_offset_exact_zero = bool(torch.count_nonzero(shadow._current_effective_offset).item() == 0)
    counter_contract = bool(
        int(env._sim_step_counter) - sim_steps_before == int(env.cfg.decimation)
        and int(env.cfg.decimation) == 4
        and abs(float(env.cfg.sim.dt) - 0.005) <= 1.0e-12
        and int(env.common_step_counter) == common_steps_before
        and torch.equal(env.episode_length_buf, episode_length_before)
    )
    no_done_or_reset = bool(
        not torch.any(reset_buf_before).item()
        and not torch.any(reset_terminated_before).item()
        and not torch.any(reset_time_outs_before).item()
        and torch.equal(env.reset_buf, reset_buf_before)
        and torch.equal(env.reset_terminated, reset_terminated_before)
        and torch.equal(env.reset_time_outs, reset_time_outs_before)
    )
    forbidden_paths_untouched = bool(all(count == 0 for count in forbidden_calls.values()))
    reward_buffers_unchanged = reward_state_unchanged(reward_state_before, reward_state_after)
    hook_exactly_once = bool(shadow._hook_calls == hook_calls_before + 1)
    post_valid = bool(
        post_diagnostics
        and not post_unknown_fields
        and raw_action_diagnostics["passed"]
        and executed_action_diagnostics["passed"]
        and zero_residual_contract
        and hook_exactly_once
        and source_final_target_exact
        and effective_offset_exact_zero
        and counter_contract
        and no_done_or_reset
        and forbidden_paths_untouched
        and reward_buffers_unchanged
        and all(bool(row["passed"]) for row in post_diagnostics.values())
    )
    diagnostics = post_diagnostics
    unknown_fields = sorted(set(initial_unknown_fields + post_unknown_fields))
    valid = bool(valid and post_valid)
'''
if source.count(report_anchor) != 1:
    raise RuntimeError("Phase77 report anchor changed")
source = source.replace(report_anchor, shadow_code + report_anchor)

diagnostics_anchor = '        "diagnostics": diagnostics,\n'
diagnostics_fields = (
    '        "initial_diagnostics": initial_diagnostics,\n'
    '        "post_step_diagnostics": diagnostics,\n'
    '        "diagnostics": diagnostics,\n'
    '        "raw_source_action_diagnostics": raw_action_diagnostics,\n'
    '        "executed_action_diagnostics": executed_action_diagnostics,\n'
    '        "zero_residual_contract": zero_residual_contract,\n'
    '        "physical_hook_calls": int(shadow._hook_calls - hook_calls_before),\n'
    '        "source_final_target_exact": source_final_target_exact,\n'
    '        "effective_offset_exact_zero": effective_offset_exact_zero,\n'
    '        "control_step_calls": 1,\n'
    '        "physics_substeps": int(env._sim_step_counter) - sim_steps_before,\n'
    '        "control_dt_seconds": float(env.cfg.decimation * env.cfg.sim.dt),\n'
    '        "counter_contract": counter_contract,\n'
    '        "no_done_or_reset": no_done_or_reset,\n'
    '        "forbidden_path_calls": forbidden_calls,\n'
    '        "forbidden_paths_untouched": forbidden_paths_untouched,\n'
    '        "reward_computed_incidentally": False,\n'
    '        "reward_inspected": False,\n'
    '        "reward_buffers_unchanged": reward_buffers_unchanged,\n'
    '        "post_metadata": post_metadata,\n'
)
if source.count(diagnostics_anchor) != 1:
    raise RuntimeError("Phase77 diagnostics report anchor changed")
source = source.replace(diagnostics_anchor, diagnostics_fields)

physics_anchor = '        "physics_rollout_steps": 0,\n'
physics_replacement = (
    '        "physics_rollout_steps": 0,\n'
    '        "technical_control_steps": 1,\n'
    '        "technical_physics_substeps": int(env.cfg.decimation),\n'
)
if source.count(physics_anchor) != 2:
    raise RuntimeError("Phase77 physics field anchor count changed")
source = source.replace(physics_anchor, physics_replacement, 1)

if "ManagerBasedRLEnv.step(" in source or "wrapped.step(" in source or "shadow.step(" in source:
    raise RuntimeError("Phase77 transformed runner retains an RL/reward step path")
if source.count("ManagerBasedEnv.step(env, executed_action)") != 1:
    raise RuntimeError("Phase77 reward-free one-action step count changed")
if "simulation_app.close()" in source:
    raise RuntimeError("Phase77 transformed runner retains blocking shutdown")

exec(compile(source, str(Path(__file__).resolve()), "exec"), globals(), globals())
