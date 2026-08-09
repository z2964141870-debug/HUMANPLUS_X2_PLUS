import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "reports/retarget/x2_wbt_gate_constrained_fullbody_phase32.json"


def test_phase32_train_only_full_wbt29_contract():
    report = json.loads(REPORT.read_text())
    config = report["configuration"]
    assert config["train_only"] is True
    assert config["held_out_read"] is False
    assert config["motions"] == ["AMASS-WALK-001", "PHUMA-LUNGE-R-001", "AMASS-KICK-L-001"]
    assert config["variables"].startswith("root XYZ + all WBT29")
    assert config["soft_reward_weight_scan"] is False
    assert config["per_clip_tuning"] is False


def test_phase32_hard_projection_and_no_shortcut():
    report = json.loads(REPORT.read_text())
    config = report["configuration"]
    assert config["hard_projection"]["root_xy_correction_m"] == 0.20
    assert config["hard_projection"]["head_lock"] is True
    for row in report["motions"]:
        assert row["diagnostics"]["controlled_joint_count"] == 29
        assert row["diagnostics"]["root_xy_bound_respected"] is True
        assert row["diagnostics"]["head_lock_respected"] is True
        assert row["diagnostics"]["no_frame_or_segment_deleted"] is True


def test_phase32_truth_and_two_of_three_gate():
    report = json.loads(REPORT.read_text())
    assert report["truth_boundary"]["held_out_read"] is False
    assert report["truth_boundary"]["physics_steps"] == 0
    assert report["truth_boundary"]["ppo_updates"] == 0
    assert report["decision"]["required_train_silver"] == 2
    assert report["decision"]["heldout_reevaluation_allowed"] == (report["summary"]["silver_count"] >= 2)
