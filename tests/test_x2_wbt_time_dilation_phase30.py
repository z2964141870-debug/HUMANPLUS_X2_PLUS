import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "reports/retarget/x2_wbt_time_dilation_phase30.json"


def test_phase30_single_factor_and_train_only_selection():
    report = json.loads(REPORT.read_text())
    selection = report["factor_selection"]
    assert selection["selected_global_factor"] == 1.46
    assert selection["selection_split"] == "train_candidate"
    assert selection["held_out_used_for_selection"] is False
    assert selection["selected_inside_preregistered_interval"] is True


def test_phase30_preserves_full_path_contract():
    report = json.loads(REPORT.read_text())
    assert len(report["motions"]) == 3
    for row in report["motions"]:
        contract = row["phase30_contract"]
        assert contract["spatial_optimization_iterations"] == 0
        assert contract["frame_or_segment_deleted"] is False
        assert contract["root_fixed"] is False
        assert contract["forced_double_contact"] is False
        assert contract["source_phase_order_preserved"] is True
        assert row["resample"]["endpoint_dof_max_abs_error"] == 0.0
        assert row["resample"]["endpoint_root_max_abs_error_m"] == 0.0
        assert row["resample"]["endpoint_quaternion_equivalent_error"] < 1e-6
        assert row["resample"]["source_contact_phase_count_before"] == row["resample"]["source_contact_phase_count_after"]
        assert row["resample"]["source_contact_transition_count_before"] == row["resample"]["source_contact_transition_count_after"]


def test_phase30_truth_boundary_and_decision():
    report = json.loads(REPORT.read_text())
    truth = report["truth_boundary"]
    assert truth["physics_steps"] == 0
    assert truth["ppo_updates"] == 0
    assert truth["spatial_optimization_iterations"] == 0
    assert truth["silver_is_not_gold"] is True
    assert report["decision"]["freeze_candidate"] == (report["summary"]["silver_count"] >= 1)
