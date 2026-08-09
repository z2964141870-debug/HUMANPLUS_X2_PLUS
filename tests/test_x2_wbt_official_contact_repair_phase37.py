import json
from pathlib import Path

import joblib
import numpy as np


REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "reports/retarget/x2_wbt_official_contact_repair_phase37.json"


def test_phase37_frozen_minimal_variable_contract():
    report = json.loads(REPORT.read_text())
    config = report["preflight"]["configuration"]
    assert config["variables"] == ["root_z", "waist_and_lower_body_15DoF_continuous_trajectory"]
    assert config["stance_foot_orientation_variable"] is False
    assert config["official_contact_equation"].endswith("== 0 m during source-intended stance")
    assert report["truth_boundary"]["mujoco_integration_steps"] == 0
    assert report["truth_boundary"]["policy_optimizer_ppo_training"] is False


def test_phase37_root_xy_time_and_upper_body_immutable():
    report = json.loads(REPORT.read_text())
    output = joblib.load(report["candidate_cache"]["path"])["PHUMA-LUNGE-R-001"]
    source_path = report["preflight"]["provenance"]["phase30_cache"]["path"]
    source = joblib.load(source_path)["PHUMA-LUNGE-R-001"]
    assert np.array_equal(output["root_trans_offset"][:, :2], source["root_trans_offset"][:, :2])
    assert len(output["dof"]) == len(source["dof"]) == 175
    assert output["fps"] == source["fps"] == 30
    lower = set(output["phase37_official_contact_repair"]["lower_body_joint_names"])
    upper_indices = [i for i, name in enumerate(output["joint_names_mujoco"]) if name not in lower]
    assert np.array_equal(output["dof"][:, upper_indices], source["dof"][:, upper_indices])


def test_phase37_silver_claim_requires_official_equality_and_nonpenetration():
    report = json.loads(REPORT.read_text())
    decision = report["decision"]
    if decision["true_official_collision_silver"]:
        assert decision["equality_pass"] is True
        assert decision["nonpenetration_pass"] is True
        assert report["after"]["tier"] == "Silver"
    assert report["truth_boundary"]["no_bronze_tolerance_as_contact"] is True
