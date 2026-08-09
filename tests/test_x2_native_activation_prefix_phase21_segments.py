from __future__ import annotations

import retarget.analyze_x2_native_activation_prefix_phase21_segments as segments


def test_phase21_source_max_discontinuity_is_rl_internal_not_boundary():
    report = segments.analyze(segments.SOURCE)
    joint = report["segments"]["joint_prefix"]
    boundary = report["segments"]["mode_boundary_pm100ms"]
    rl = report["segments"]["rl_internal_after100ms"]
    assert joint["joint_step_p95_max_rad"][1] < 0.45
    assert boundary["joint_step_p95_max_rad"][1] < 0.45
    assert rl["joint_step_p95_max_rad"][1] > 0.45
    assert rl["recorded_dq_p95_max_radps"][1] > 20.0


def test_phase21_joint_prefix_is_smooth_but_physically_collapsed():
    report = segments.analyze(segments.SOURCE)
    joint = report["segments"]["joint_prefix"]
    assert joint["first_root_z_below_0p42_s"] is not None
    assert joint["root_z_min_final_m"][0] < 0.1
    assert report["replay_error_segmentation"]["available"] is False
