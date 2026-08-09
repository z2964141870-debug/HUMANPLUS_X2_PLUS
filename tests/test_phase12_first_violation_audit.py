import json
import math

import pytest

from official_x2.audit_phase12_first_violation import (
    EXPECTED_SLOT_COUNTS,
    LOWER_SCALE_RAD,
    audit_episode,
    build_report,
)


def _row(elapsed: float, action: list[float], *, z: float, tilt: float, vx: float) -> dict:
    return {
        "stage": "stop",
        "elapsed_s": elapsed,
        "action": action,
        "root_z_m": z,
        "root_tilt_rad": tilt,
        "root_pitch_rad": -tilt,
        "root_vx_w_mps": vx,
        "root_vy_w_mps": 0.0,
    }


def test_episode_audit_uses_exact_one_tick_handoff_and_existing_gates(tmp_path) -> None:
    pre = [0.0] * 15
    post = [0.0] * 15
    post[3] = 0.5
    payload = {
        "summary": {
            "policy_slot_inference_counts": EXPECTED_SLOT_COUNTS,
            "action_contract": "synthetic issued normalized action",
        },
        "trace": [
            _row(1.96, pre, z=0.62, tilt=0.10, vx=0.01),
            _row(1.98, pre, z=0.62, tilt=0.10, vx=0.01),
            _row(2.00, post, z=0.61, tilt=0.12, vx=0.02),
            _row(2.02, post, z=0.44, tilt=0.31, vx=0.04),
        ],
    }
    path = tmp_path / "episode.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = audit_episode(path, group="source_recovery")

    assert result["handoff"]["pre_elapsed_s"] == pytest.approx(1.98)
    assert result["handoff"]["post_elapsed_s"] == pytest.approx(2.0)
    assert result["handoff"]["normalized_delta_l2"] == pytest.approx(0.5)
    assert result["handoff"]["target_delta_rad_l2"] == pytest.approx(
        0.5 * LOWER_SCALE_RAD[3]
    )
    assert result["handoff"]["joint_contributions"][0]["joint"] == "left_knee_joint"
    crossing = result["first_threshold_crossing"]
    assert crossing["root_z_below_0p45_s_after_handoff"] == pytest.approx(0.02)
    assert crossing["root_tilt_above_0p30_s_after_handoff"] == pytest.approx(0.02)
    assert crossing["body_speed_above_0p03_s_after_handoff"] == pytest.approx(0.02)


def test_real_phase11_panel_is_ten_traces_with_fixed_handoff() -> None:
    from pathlib import Path

    root = Path(
        "/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/results/official_native_strict_20260807"
    )
    report = build_report(root)

    assert report["trace_count"] == 10
    assert report["contracts"]["all_slot_counts_exact"]
    assert report["contracts"]["all_handoffs_exactly_at_2s"]
    source = report["groups"]["source_recovery"]
    candidate = report["groups"]["candidate_recovery"]
    assert source["handoff"]["target_delta_rad_l2"]["median"] > 0.4
    assert candidate["handoff"]["target_delta_rad_l2"]["median"] > 0.4
    assert source["handoff"]["handoff_to_pre_1s_median_ratio"]["median"] > 8.0
    assert candidate["handoff"]["handoff_to_pre_1s_median_ratio"]["median"] > 8.0
    assert source["top_five_joints_by_median_abs_target_delta_rad"][0] == "left_knee_joint"
    assert candidate["top_five_joints_by_median_abs_target_delta_rad"][0] == "left_knee_joint"
    assert source["first_threshold_crossing"]["root_tilt_above_0p30_s_after_handoff"]["median"] > 1.0
    assert candidate["first_threshold_crossing"]["root_tilt_above_0p30_s_after_handoff"]["median"] > 1.0
