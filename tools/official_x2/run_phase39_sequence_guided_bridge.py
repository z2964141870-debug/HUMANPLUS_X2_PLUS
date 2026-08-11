#!/usr/bin/env python3
"""One frozen, sequence-guided BASE bridge from the Phase26 snapshot.

No search is performed.  The only intervention is a C2 transition from the
live brake target to a frozen 1 s Phase19-v2 success-safe physical-target
suffix selected by Phase38.  Root state is never prescribed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from official_x2.controller_snapshot_contract import capture_controller_state, restore_controller_state
from official_x2.outcome_aware_state_role_v2 import load_manifest, resolve_rows
from official_x2.run_phase26_testonly_bridge_cem import (
    BRIDGE_SECONDS,
    CONTROL_DT_S,
    ESSENTIAL_GROUPS,
    REPORT_POST_HANDOFF_SECONDS,
    SNAPSHOT_STOP_ELAPSED_S,
    STRICT_POST_HANDOFF_SECONDS,
    BridgeRunner,
    build_support,
    integration_row,
    jsonable,
    score_rollout,
    smoothstep,
    support_scores,
)
from official_x2.replay_official_trace_direct_mujoco import JOINTS, LOWER_JOINTS, LOWER_SCALE
from official_x2.run_testonly_official_mjcf_suffix_probe import (
    integration_state,
    physical_state_sha256,
    restore_integration_state,
)


def load_suffix(manifest_path: Path, phase38_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = load_manifest(manifest_path)
    phase38 = json.loads(phase38_path.read_text(encoding="utf-8"))
    indices = [int(value) for value in phase38["suffix"]["manifest_row_indices"]]
    rows = resolve_rows(manifest, indices)
    if len(rows) != 51:
        raise ValueError("Phase39 requires exactly 51 recorded rows for a 1 s transition")
    if [int(row["source_tick"]) for row in rows] != list(range(int(rows[0]["source_tick"]), int(rows[0]["source_tick"]) + 51)):
        raise ValueError("Phase38 suffix is not tick-consecutive")
    if [row["snapshot_sha256"] for row in rows] != phase38["suffix"]["snapshot_sha256"]:
        raise ValueError("Phase38 suffix snapshot hashes drifted")
    return rows, phase38


def run_sequence(
    runner: BridgeRunner,
    state: np.ndarray,
    controller_state: dict[str, Any],
    state_sha: str,
    support: dict[str, dict[str, Any]],
    suffix: list[dict[str, Any]],
) -> dict[str, Any]:
    restore_integration_state(runner.model, runner.data, state)
    restore_controller_state(runner, controller_state, expected_physical_state_sha256=state_sha)
    bridge_ticks = round(BRIDGE_SECONDS / CONTROL_DT_S)
    if bridge_ticks != len(suffix) - 1:
        raise ValueError("suffix/control duration mismatch")
    bridge_rows = []
    last_targets: dict[str, float] | None = None
    for tick in range(bridge_ticks):
        elapsed = runner.sequence_step * CONTROL_DT_S
        stop_elapsed = elapsed - runner.move_end_s
        if stop_elapsed < runner.args.stop_transition_seconds:
            command_vx = runner._move_speed(runner.args.move_seconds + stop_elapsed)
            future_vx = runner._move_speed(runner.args.move_seconds + stop_elapsed + 1.0)
            multiplier = float(np.clip(abs(command_vx) / max(abs(runner.args.vx), 1e-6), 0.0, 1.0))
        else:
            command_vx = future_vx = multiplier = 0.0
        live_targets, _obs, _issued = runner.policy_targets(
            runner.args.move_seconds + stop_elapsed,
            command_vx,
            policy_slot="main",
            force_moving=True,
            future_vx=future_vx,
            template_multiplier=multiplier,
        )
        target_row = suffix[tick + 1]
        target_lower = np.asarray(target_row["physical_lower_target_rad"], dtype=np.float64)
        blend = smoothstep((tick + 1) / bridge_ticks)
        final_targets = dict(live_targets)
        for index, name in enumerate(LOWER_JOINTS):
            final_targets[name] = (1.0 - blend) * live_targets[name] + blend * float(target_lower[index])
        actual = np.asarray(
            [(final_targets[name] - runner.default[name]) / LOWER_SCALE[index] for index, name in enumerate(LOWER_JOINTS)],
            dtype=np.float32,
        )
        if np.max(np.abs(actual)) > 1.000001:
            raise ValueError("sequence-guided physical target exceeds normalized action bounds")
        actual = np.clip(actual, -1.0, 1.0)
        runner.previous_actions["main"] = actual.copy()
        runner.issued_actions["main"] = actual.copy()
        runner.step_physics(final_targets)
        runner.sequence_step += 1
        runner.control_steps += 1
        last_targets = final_targets
        bridge_rows.append(integration_row(runner, "sequence_bridge", actual, support_scores(runner.query("main"), support)))

    if last_targets is None:
        raise RuntimeError("empty sequence bridge")
    runner.previous_actions["recovery"] = runner.previous_actions["main"].copy()
    runner.issued_actions["recovery"] = runner.issued_actions["main"].copy()
    runner.stop_hold_targets = dict(last_targets)
    runner.stop_hold_latch_s = runner.sequence_step * CONTROL_DT_S - runner.move_end_s
    endpoint = support_scores(runner.query("main"), support)

    recovery_rows = []
    for tick in range(round(REPORT_POST_HANDOFF_SECONDS / CONTROL_DT_S)):
        targets, _obs, _proposal = runner.recovery_targets()
        blend = smoothstep((tick * CONTROL_DT_S) / 0.5)
        blended = {name: (1.0 - blend) * runner.stop_hold_targets[name] + blend * targets[name] for name in JOINTS}
        actual = np.asarray(
            [(blended[name] - runner.default[name]) / LOWER_SCALE[index] for index, name in enumerate(LOWER_JOINTS)],
            dtype=np.float32,
        )
        actual = np.clip(actual, -1.0, 1.0)
        for slot in ("recovery", "stationary"):
            runner.previous_actions[slot] = actual.copy()
            runner.issued_actions[slot] = actual.copy()
        runner.step_physics(blended)
        runner.sequence_step += 1
        recovery_rows.append(integration_row(runner, "recovery", actual, support_scores(runner.query("recovery"), support)))
    return {"endpoint_scores": endpoint, "bridge_rows": bridge_rows, "recovery_rows": recovery_rows, "raw": np.zeros(1)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    suffix, phase38 = load_suffix(args.manifest, args.phase38)
    support = build_support(args.manifest)
    source = BridgeRunner(args)
    snapshot_step = round((source.move_end_s + SNAPSHOT_STOP_ELAPSED_S) / CONTROL_DT_S)
    while source.sequence_step < snapshot_step:
        source.tick()
    state = integration_state(source.model, source.data)
    state_sha = physical_state_sha256(state, args.asset_manifest_sha256)
    controller_state = capture_controller_state(source, physical_state_sha256=state_sha)
    expected_snapshot_sha = args.expected_phase26_snapshot_sha256
    if state_sha != expected_snapshot_sha:
        raise ValueError(f"Phase26 physical snapshot drift: {state_sha} != {expected_snapshot_sha}")

    defaults = np.asarray([row["default_lower_target_rad"] for row in suffix], dtype=np.float64)
    scales = np.asarray([row["lower_action_scale_rad"] for row in suffix], dtype=np.float64)
    if np.max(np.abs(defaults - defaults[0])) > 1e-8 or np.max(np.abs(scales - scales[0])) > 1e-8:
        raise ValueError("recorded suffix lower target contract is not constant")
    if np.max(np.abs(scales[0] - LOWER_SCALE)) > 1e-6:
        raise ValueError("recorded suffix scale differs from live bridge scale")

    scratch = BridgeRunner(args)
    rollout = run_sequence(scratch, state, controller_state, state_sha, support, suffix)
    cost, diagnostics = score_rollout(rollout)
    bridge_rows = rollout["bridge_rows"]
    recovery_rows = rollout["recovery_rows"]

    def first_index(rows: list[dict[str, Any]], predicate) -> int | None:
        return next((index for index, row in enumerate(rows) if predicate(row)), None)

    first_tilt = first_index(recovery_rows, lambda row: row["root_tilt_rad"] > 0.30)
    first_height = first_index(recovery_rows, lambda row: row["root_z_m"] < 0.55)
    return {
        "stage": "BASE Phase39 sequence-guided bridge",
        "pre_registration": {
            "single_candidate_no_retry": True,
            "search_or_optimizer": False,
            "bridge_seconds": BRIDGE_SECONDS,
            "target": "Phase38 frozen 51-tick success-safe physical/controller suffix",
            "transition": "C2 live brake physical target to recorded physical lower target",
            "root_prescribed_or_teleported": False,
            "upper_contract": "unchanged live fixed-upper path",
            "success": "unchanged Phase26 endpoint success support + next 1s union support + root safety",
        },
        "assets": args.asset_hashes,
        "snapshot": {"physical_state_sha256": state_sha, "controller_bound_physical_state_sha256": controller_state["physical_state_sha256"]},
        "target": {
            "phase38_selected_snapshot_sha256": phase38["selected_target"]["snapshot_sha256"],
            "source_ticks": [int(row["source_tick"]) for row in suffix],
            "snapshot_sha256": [row["snapshot_sha256"] for row in suffix],
        },
        "cost": cost,
        "diagnostics": diagnostics,
        "timeline": {
            "bridge_root_z_min_m": min(row["root_z_m"] for row in bridge_rows),
            "bridge_root_tilt_max_rad": max(row["root_tilt_rad"] for row in bridge_rows),
            "recovery_root_z_min_m": min(row["root_z_m"] for row in recovery_rows),
            "recovery_root_tilt_max_rad": max(row["root_tilt_rad"] for row in recovery_rows),
            "first_recovery_tilt_over_0p30_s": None if first_tilt is None else first_tilt * CONTROL_DT_S,
            "first_recovery_height_below_0p55_s": None if first_height is None else first_height * CONTROL_DT_S,
        },
        "rollout": rollout,
        "decision": {
            "strict_dynamic_bridge_found": bool(diagnostics["strict_bridge_success"]),
            "training_unlocked": False,
            "route": "STOP_AND_REVIEW" if not diagnostics["strict_bridge_success"] else "MECHANISM_PASS_STILL_NO_AUTOMATIC_TRAINING",
        },
    }


def parse_args() -> argparse.Namespace:
    from official_x2.run_phase26_testonly_bridge_cem import parse_args as phase26_parse_args

    # Reuse Phase26's exact live asset/controller arguments, adding only the
    # immutable target and expected snapshot contracts.
    args = phase26_parse_args()
    return args


def main() -> None:
    # Phase26 parser does not know these three flags, so keep them in a small
    # wrapper parser and forward the remaining exact Phase26 arguments.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--phase38", type=Path, required=True)
    parser.add_argument("--expected-phase26-snapshot-sha256", required=True)
    parser.add_argument("--phase39-output", type=Path, required=True)
    known, remaining = parser.parse_known_args()
    import sys
    old = sys.argv
    try:
        sys.argv = [old[0], *remaining]
        args = parse_args()
    finally:
        sys.argv = old
    args.phase38 = known.phase38
    args.expected_phase26_snapshot_sha256 = known.expected_phase26_snapshot_sha256
    result = run(args)
    known.phase39_output.parent.mkdir(parents=True, exist_ok=True)
    known.phase39_output.write_text(json.dumps(jsonable(result), indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"diagnostics": result["diagnostics"], "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()
