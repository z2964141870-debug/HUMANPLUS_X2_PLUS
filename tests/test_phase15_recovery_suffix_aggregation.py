from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from official_x2.controller_snapshot_contract import capture_controller_state
from official_x2.recovery_suffix_aggregation import (
    AGGREGATION_SCHEMA,
    SIDECAR_SCHEMA,
    SNAPSHOT_SCHEMA,
    build_aggregation_manifest,
    build_recovery_suffix_snapshot,
    build_suffix_sidecar,
    canonical_sha256,
    deduplicate_suffix_snapshots,
    load_suffix_sidecar,
    physical_state_payload,
    recovery_capture_eligible,
    sha256_file,
    suffix_source_mask,
    validate_recovery_suffix_snapshot,
    write_suffix_sidecar,
)


REPO = Path(__file__).resolve().parents[1]
STAGE335 = (
    REPO.parent
    / "x2_official_rl_deploy_v1/models/stage335_stage306_stiff_fixed_stop_recovery_states.npz"
)
STAGE335_SHA = "4d8ce06b6e8dd9b43363dadedb781e9feb75f72ac092b475de2a1e2772545013"
JOINTS = tuple(f"joint_{index:02d}" for index in range(31))


def _controller(physical_sha: str, issued: np.ndarray) -> SimpleNamespace:
    zeros = np.zeros(15, dtype=np.float32)
    return SimpleNamespace(
        args=SimpleNamespace(
            clock_mode="step",
            vx=0.3,
            stop_controller="curriculum_then_policy",
            post_handoff_snapshot_output="/results/suffix.json",
        ),
        previous_actions={slot: zeros.copy() for slot in ("main", "stationary", "recovery")},
        issued_actions={
            "main": zeros.copy(),
            "stationary": zeros.copy(),
            "recovery": issued.astype(np.float32).copy(),
        },
        sequence_step=401,
        control_steps=300,
        stop_hold_latch_s=2.0,
        stop_emergency_latch=False,
        heading_target_rad=0.0,
        heading_origin_xy=(0.0, 0.0),
        move_heading_initialized=True,
        stop_policy_initialized=True,
        lateral_recovery_state="off",
        lateral_recovery_bias=np.zeros(2, dtype=np.float32),
        heading_recovery_active=False,
        heading_action_recovery_active=False,
        heading_action_recovery_steps=0,
        last_move_targets={"joint_00": 0.0},
        stop_hold_targets={"joint_00": 0.0},
        prepare_start_q={"joint_00": 0.0},
        predicted_physical_step=400,
        previous_physical_observation=np.zeros(71, dtype=np.float32),
        predicted_physical_observation=np.zeros(71, dtype=np.float32),
        upper_previous_target=np.zeros(14, dtype=np.float32),
        upper_last_target=np.zeros(14, dtype=np.float32),
        upper_fallback_steps=0,
        upper_fallback_active=False,
        upper_fallback_first_step=None,
        finished=False,
        _physical_sha=physical_sha,
    )


def _snapshot(
    *,
    episode: str = "failed_r1",
    q_delta: float = 0.0,
    issued_delta: float = 0.0,
    root_z_delta: float = 0.0,
) -> dict:
    q = np.zeros(31, dtype=np.float64)
    q[0] = q_delta
    physical = physical_state_payload(
        joint_names=JOINTS,
        joint_position_rad=q,
        joint_velocity_radps=np.zeros(31),
        root_position_m=[4.0, -2.0, 0.62 + root_z_delta],
        root_quaternion_xyzw=[0.0, 0.0, 0.0, 1.0],
        root_linear_velocity_world_mps=[0.1, 0.0, 0.0],
        root_angular_velocity_world_radps=[0.0, 0.0, 0.0],
    )
    issued = np.zeros(15, dtype=np.float32)
    issued[0] = issued_delta
    controller = _controller(canonical_sha256(physical), issued)
    controller_state = capture_controller_state(
        controller, physical_state_sha256=canonical_sha256(physical)
    )
    obs = np.zeros(93, dtype=np.float32)
    obs[0] = 0.1
    obs[6:9] = [0.0, 0.0, -1.0]
    obs[12] = q_delta
    obs[89:93] = [0.0, 0.0, 1.0, 1.0]
    return build_recovery_suffix_snapshot(
        episode_id=episode,
        source_tick=400,
        time_after_handoff_s=0.10,
        physical_state=physical,
        observation=obs,
        previous_action_input=obs[74:89],
        issued_action=issued,
        command=obs[9:12],
        gait_phase=obs[89:93],
        controller_state=controller_state,
    )


def _sidecar(rows: list[dict]) -> dict:
    return build_suffix_sidecar(
        rows,
        source_trace_path="/results/f005_failed.json",
        source_trace_sha256="a" * 64,
        adapter_path="/repo/tools/official_x2/stage208_official_mujoco_adapter.py",
        adapter_sha256="b" * 64,
        recovery_model_path="/models/phase9_f005.onnx",
        recovery_model_sha256="c" * 64,
        episode_outcome={
            "full_gate_pass": False,
            "stand_gate_pass": True,
            "startup_gate_pass": True,
            "move_gate_pass": False,
            "stop_gate_pass": False,
        },
        horizon_s=1.5,
    )


def test_capture_gate_is_default_off_recovery_only_and_bounded() -> None:
    assert not recovery_capture_eligible(
        output_enabled=False,
        authority_slot="recovery",
        time_after_handoff_s=0.2,
    )
    assert not recovery_capture_eligible(
        output_enabled=True,
        authority_slot="main",
        time_after_handoff_s=0.2,
    )
    assert recovery_capture_eligible(
        output_enabled=True,
        authority_slot="recovery",
        time_after_handoff_s=0.0,
    )
    assert recovery_capture_eligible(
        output_enabled=True,
        authority_slot="recovery",
        time_after_handoff_s=1.5,
    )
    assert not recovery_capture_eligible(
        output_enabled=True,
        authority_slot="recovery",
        time_after_handoff_s=1.5001,
    )


def test_snapshot_binds_physics_actor_input_issued_action_and_controller() -> None:
    snapshot = _snapshot()
    assert snapshot["schema"] == SNAPSHOT_SCHEMA
    assert snapshot["capture_boundary"] == "post_inference_pre_physics"
    assert snapshot["observation_93d"][74:89] == snapshot["previous_action_input"]
    assert snapshot["controller_state"]["physical_state_sha256"] == snapshot["physical_state_sha256"]
    assert snapshot["controller_state"]["issued_actions"]["recovery"] == snapshot["actual_issued_action"]
    validate_recovery_suffix_snapshot(snapshot)

    broken = deepcopy(snapshot)
    broken["actual_issued_action"][0] = 0.2
    broken_unsigned = dict(broken)
    broken_unsigned.pop("snapshot_sha256")
    broken["snapshot_sha256"] = canonical_sha256(broken_unsigned)
    with pytest.raises(ValueError, match="controller/issued-action mismatch"):
        validate_recovery_suffix_snapshot(broken)


def test_sidecar_roundtrip_and_conservative_duplicate_suppression(tmp_path: Path) -> None:
    first = _snapshot(episode="failed_r1")
    near = _snapshot(
        episode="failed_r2", q_delta=0.001, issued_delta=0.001, root_z_delta=0.0005
    )
    sidecar = _sidecar([first])
    assert sidecar["schema"] == SIDECAR_SCHEMA
    path = tmp_path / "suffix.json"
    write_suffix_sidecar(path, sidecar)
    loaded = load_suffix_sidecar(path, sha256_file(path))
    assert loaded == sidecar
    representatives, assignments = deduplicate_suffix_snapshots([first, near])
    assert len(representatives) == 1
    assert [row["representative_index"] for row in assignments] == [0, 0]


def test_fraction_zero_is_exact_base_and_rng_noop_without_opening_sidecar() -> None:
    rng = np.random.default_rng(17)
    mask = suffix_source_mask(128, 0.0, rng)
    after = rng.random(4)
    control = np.random.default_rng(17).random(4)
    assert not mask.any()
    np.testing.assert_array_equal(after, control)

    plan = build_aggregation_manifest(
        base_dataset_path=STAGE335,
        base_dataset_sha256=STAGE335_SHA,
        base_state_count=90,
        suffix_sidecar_paths=["/this/path/must/not/be/opened.json"],
        suffix_fraction=0.0,
    )
    assert plan["schema"] == AGGREGATION_SCHEMA
    assert plan["strict_no_op"]
    assert plan["base_dataset"]["sha256"] == STAGE335_SHA
    assert plan["base_dataset"]["state_count"] == 90
    assert plan["suffix_sidecars"] == []
    assert plan["suffix_representative_state_count"] == 0


def test_nonzero_virtual_union_is_hash_bound_and_does_not_rewrite_stage335(tmp_path: Path) -> None:
    before = hashlib.sha256(STAGE335.read_bytes()).hexdigest()
    sidecar_path = tmp_path / "failed_suffix.json"
    write_suffix_sidecar(sidecar_path, _sidecar([_snapshot()]))
    plan = build_aggregation_manifest(
        base_dataset_path=STAGE335,
        base_dataset_sha256=STAGE335_SHA,
        base_state_count=90,
        suffix_sidecar_paths=[sidecar_path],
        suffix_fraction=0.05,
    )
    assert not plan["strict_no_op"]
    assert plan["suffix_raw_state_count"] == 1
    assert plan["suffix_representative_state_count"] == 1
    assert plan["materialization"].startswith("virtual union")
    assert hashlib.sha256(STAGE335.read_bytes()).hexdigest() == before == STAGE335_SHA


def test_stage208_adapter_and_runner_keep_capture_default_off() -> None:
    adapter = (REPO / "tools/official_x2/stage208_official_mujoco_adapter.py").read_text(
        encoding="utf-8"
    )
    parser = adapter[adapter.index("def parse_args()") : adapter.index("def main()")]
    assert '"--post-handoff-snapshot-output"' in parser
    assert 'default=1.5' in parser
    method_start = adapter.index("    def _maybe_record_recovery_suffix_snapshot(")
    method_end = adapter.index("    def _joint_callback", method_start)
    method = adapter[method_start:method_end]
    guard = method.index("if not recovery_capture_eligible(")
    append = method.index("self.recovery_suffix_snapshots.append(snapshot)")
    assert guard < append
    assert 'output_enabled = self.args.post_handoff_snapshot_output is not None' in method
    assert 'authority_slot = (' in method

    outer = (REPO / "tools/official_x2/run_official_gate_case.sh").read_text(encoding="utf-8")
    inner = (REPO / "tools/official_x2/run_official_gate_inner.sh").read_text(encoding="utf-8")
    assert "POST_HANDOFF_SNAPSHOT_OUTPUT" in outer
    assert 'POST_HANDOFF_SNAPSHOT_OUTPUT="${POST_HANDOFF_SNAPSHOT_OUTPUT:-}"' in inner
    assert '[[ -n "$POST_HANDOFF_SNAPSHOT_OUTPUT" ]]' in inner
    assert '--post-handoff-snapshot-output "$POST_HANDOFF_SNAPSHOT_OUTPUT"' in inner
