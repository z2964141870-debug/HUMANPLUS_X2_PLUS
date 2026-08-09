from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import retarget.run_x2_wbt_stance_xy_lock_phase8 as phase8


def test_boolean_intervals_preserve_each_contiguous_window():
    values = np.asarray([0, 1, 1, 0, 1, 0, 1, 1, 1], dtype=bool)
    assert phase8.boolean_intervals(values) == [(1, 3), (4, 5), (6, 9)]


def test_c2_boundary_weight_is_zero_at_edges_and_bounded():
    weights = [phase8.c2_boundary_weight(frame, 2, 13) for frame in range(2, 13)]
    assert weights[0] == 0.0 and weights[-1] == 0.0
    assert max(weights) == 1.0
    assert all(0.0 <= value <= 1.0 for value in weights)


def test_offline_gate_rejects_joint_or_slip_regression():
    metrics = {
        "joint_step_max_rad": {"baseline": 0.10, "candidate": 0.16},
        "stance_slip_p95_mps_model_estimate": {"baseline": 0.20, "candidate": 0.30},
        "contact_intent_agreement": {"macro": 1.0},
        "ground": {"pass": True},
        "root_xy_exact": True,
        "root_orientation_exact": True,
        "fps_exact": True,
    }
    diagnostics = {
        "boundary_joint_delta_abs_max_rad": 0.0,
        "swing_joint_delta_abs_max_rad": 0.0,
    }
    gate = phase8.offline_gate(metrics, diagnostics)
    assert gate["pass"] is False
    assert gate["checks"]["joint_step_within_tier_or_110pct_baseline"] is False
    assert gate["checks"]["model_stance_slip_not_worse"] is False


def test_checked_report_blocks_physics_and_training():
    path = REPO / "reports/retarget/x2_wbt_stance_xy_lock_phase8.json"
    if not path.exists():
        return
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["decision"]["offline_all_five"] is False
    assert report["decision"]["official_physics_ab_executed"] is False
    assert report["truth_boundary"]["prescribed_free_replay_executed"] is False
    assert report["truth_boundary"]["training_teacher_checkpoint_base_git_baidu_real_robot"] is False
    assert report["truth_boundary"]["not_hardware_grf_cop_or_foot_force"] is True
    assert report["single_structural_intervention"]["free_survival_used_for_parameter_selection"] is False
