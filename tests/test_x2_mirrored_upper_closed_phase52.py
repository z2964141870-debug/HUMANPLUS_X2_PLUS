import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "reports/retarget/x2_mirrored_upper_closed_phase52.json"


def test_phase52_is_one_episode_causal_diagnostic_only():
    r = json.loads(REPORT.read_text())
    assert r["execution"] == {"new_physics_episodes": 1, "retries": 0, "A_reused": True, "B1_reused": True}
    assert "no retry/training" in r["scope"]


def test_phase52_mirror_and_frozen_control_contracts_hold():
    r = json.loads(REPORT.read_text())
    s = r["static"]["preregistered_checks"]
    assert s["mirror_twice_exact"] is True
    assert s["mirror_twice_max_abs_rad"] == 0.0
    assert s["bounded_joint_limits"] and s["bounded_qstep"] and s["bounded_excursion"]
    assert s["lower_waist_root_head_direct_path_unchanged"] is True
    assert r["static"]["frozen_summary_config_exact"] is True


def test_phase52_decision_follows_preregistered_signed_flip_rules():
    r = json.loads(REPORT.read_text())
    d = r["signed_causal_diagnostic"]
    if d["lateral_sign_flip"] and d["yaw_sign_flip"]:
        expected = "DIRECTIONAL_UPPER_CAUSALITY_SUPPORTED"
    elif d["lateral_sign_flip"] or d["yaw_sign_flip"]:
        expected = "MIXED_DIRECTIONAL_UPPER_EFFECT"
    else:
        expected = "DIRECTIONAL_UPPER_CAUSALITY_NOT_SUPPORTED"
    assert r["decision"]["status"] == expected
    assert r["truth_boundary"].endswith("not hardware GRF/COP")
