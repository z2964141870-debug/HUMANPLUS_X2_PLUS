import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "tools/retarget/run_x2_wbt_full_trajectory_repair_phase29.py"


def test_phase29_contract_is_single_config_and_offline():
    report = json.loads((REPO / "reports/retarget/x2_wbt_full_trajectory_repair_phase29.json").read_text())
    config = report["configuration"]
    assert config["motions"] == ["AMASS-WALK-001", "PHUMA-LUNGE-R-001", "AMASS-KICK-L-001"]
    assert config["per_clip_tuning"] is False
    assert config["root_xy_correction_max_m"] == 0.20
    assert "fixed_root" in config["forbidden"]
    assert "forced_double_contact" in config["forbidden"]
    assert config["stance_schedule"].endswith("none dropped/merged")


def test_phase29_preflight_or_report_truth_boundary():
    preflight = REPO / "reports/retarget/x2_wbt_full_trajectory_repair_phase29_preflight.json"
    if not preflight.exists():
        return
    value = json.loads(preflight.read_text())
    assert value["pass"] is True
    assert len(value["motions"]) == 3
    assert all(row["checks"]["train_split"] for row in value["motions"])
    assert all(row["checks"]["lower_body_15"] for row in value["motions"])


def test_phase29_report_never_unlocks_physics_or_ppo():
    path = REPO / "reports/retarget/x2_wbt_full_trajectory_repair_phase29.json"
    if not path.exists():
        return
    report = json.loads(path.read_text())
    assert report["truth_boundary"]["physics_steps"] == 0
    assert report["truth_boundary"]["ppo_updates"] == 0
    assert report["truth_boundary"].get("policy_optimizer_steps", 0) == 0
    assert report["truth_boundary"]["policy_forwards"] == 0
    assert len(report["motions"]) == 3
    for row in report["motions"]:
        assert row["diagnostics"]["root_xy_bound_respected"] is True
        assert row["diagnostics"]["no_frame_or_segment_deleted"] is True
