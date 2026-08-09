from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import mujoco


REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import retarget.run_x2_wbt_canonical_root_ground_phase7 as phase7


def test_centered_moving_average_is_constant_preserving_and_length_preserving():
    values = np.full(31, 0.123)
    filtered = phase7.centered_moving_average(values)
    assert len(filtered) == len(values)
    assert np.allclose(filtered, values)


def test_centered_moving_average_rejects_non_odd_window():
    try:
        phase7.centered_moving_average(np.arange(10), 4)
    except ValueError:
        pass
    else:
        raise AssertionError("even ground window must be rejected")


def test_paired_gate_does_not_use_survival_to_select_root_z():
    baseline = {"simulated_duration_s": 2.0, "stance_slip_p95_mps": 0.05}
    accepted = phase7.paired_gate(
        baseline, {"simulated_duration_s": 1.901, "stance_slip_p95_mps": 0.070}
    )
    rejected_survival = phase7.paired_gate(
        baseline, {"simulated_duration_s": 1.899, "stance_slip_p95_mps": 0.070}
    )
    rejected_slip = phase7.paired_gate(
        baseline, {"simulated_duration_s": 2.1, "stance_slip_p95_mps": 0.071}
    )
    assert accepted["pass"]
    assert not rejected_survival["pass"]
    assert not rejected_slip["pass"]


def test_checked_report_has_truth_boundary_and_two_level_decision():
    report_path = REPO / "reports/retarget/x2_wbt_canonical_root_ground_phase7.json"
    if not report_path.exists():
        return
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["truth_boundary"]["not_hardware_grf_cop_or_foot_force"] is True
    assert report["official_robot_contract"]["root_z_selection_uses_per_clip_survival"] is False
    assert report["decision"]["geometry_contract_bronze_valid"] is True
    assert report["decision"]["dynamic_silver_promotable"] is False
    assert report["decision"]["slip_failed_roles"] == ["walk_to_stand"]
    assert report["decision"]["contact_failed_roles"] == [
        "walk_straight", "turn_left", "turn_right", "stand_to_walk", "walk_to_stand"
    ]
    assert report["root_sign_audit"]["phase4_6_distance_helper_included_visual_ankle_mesh"] is True
    permutation = report["official_robot_contract"]["official_rl_29_to_mjcf_31_indices"]
    assert len(permutation) == 29 and len(set(permutation)) == 29


def test_active_sole_contract_excludes_visual_mesh():
    # Use the authoritative scene path recorded by the checked canonical map.
    contract = json.loads(
        (REPO / "reports/retarget/x2_official_joint_body_map.json").read_text(encoding="utf-8")
    )
    model = mujoco.MjModel.from_xml_path(contract["provenance"]["scene_xml"]["path"])
    _, feet = phase7.active_sole_spheres(model)
    assert {side: len(values) for side, values in feet.items()} == {"left": 12, "right": 12}
    assert all(
        int(model.geom_type[geom]) == int(mujoco.mjtGeom.mjGEOM_SPHERE)
        and int(model.geom_contype[geom]) != 0
        for values in feet.values()
        for geom in values
    )
