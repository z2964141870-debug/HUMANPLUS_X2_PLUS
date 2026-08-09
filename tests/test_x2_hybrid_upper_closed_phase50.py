import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PREREG = REPO / "reports/official_x2/phase50_hybrid_upper_closed_prereg.json"
REPORT = REPO / "reports/retarget/x2_hybrid_upper_closed_phase50.json"


def test_phase50_was_preregistered_as_one_B_episode_without_retry():
    p = json.loads(PREREG.read_text())
    assert p["status"] == "PREREGISTERED_BEFORE_PHYSICS"
    assert p["control"]["only_new_physics_episodes"] == 1
    assert p["control"]["retries"] == 0
    assert p["control"]["A"].startswith("BASE Phase34")


def test_phase50_upper_write_mask_is_structurally_arm_only():
    r = json.loads(REPORT.read_text())
    assert r["static"]["adapter_hash_exact"] is True
    assert r["static"]["upper_artifact_hash_exact"] is True
    assert r["static"]["upper_only_ast"]["pass"] is True
    assert r["static"]["upper_only_ast"]["forbidden_joint_group_references"] == []
    assert r["static"]["frozen_summary_config_exact"] is True


def test_phase50_honestly_applies_all_preregistered_gates():
    r = json.loads(REPORT.read_text())
    assert r["execution"]["A_reused_not_rerun"] is True
    assert r["execution"]["new_B_episodes"] == 1
    assert r["execution"]["B_retries"] == 0
    assert r["execution"]["prephysics_invalid_start_physics_episodes"] == 0
    expected = "PROMOTABLE_ONE_EPISODE_HYBRID_EXISTENCE" if all(r["gates"].values()) else "NOT_PROMOTABLE_UNDER_PREREGISTERED_GATE"
    assert r["decision"]["status"] == expected
    assert r["truth_boundary"]["contact"].endswith("not hardware GRF/COP")
