import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "research/dynamic_retargeting_20260811/phase18_joint_contact_generator"


def test_phase18_prereg_is_offline_fixed_and_serial() -> None:
    prereg = json.loads((DIR / "prereg_phase18_joint_contact_generator.json").read_text())
    assert prereg["templates_in_order"] == ["DS", "DS-L-DS", "DS-R-DS", "DS-L-DS-R-DS", "DS-R-DS-L-DS"]
    assert prereg["solver"]["iterations"] == 8
    assert "physics integration" in prereg["forbidden"]
    assert "GPU" in prereg["forbidden"]
    assert "parallel workers" in prereg["forbidden"]


def test_phase18_result_respects_stop_and_truth_boundary() -> None:
    result_path = DIR / "phase18_result.json"
    if not result_path.exists():
        return
    result = json.loads(result_path.read_text())
    assert result["execution"]["mj_step_calls"] == 0
    assert result["execution"]["gpu"] is False
    assert result["truth_boundary"]["source_contact_labels_are_truth"] is False
    passed = [row for row in result["templates"] if row["complete_pass"]]
    assert len(passed) <= 1
    if passed:
        assert result["templates"][-1]["template"] == passed[0]["template"]
    assert result["decision"]["physics_unlocked"] is False
    assert result["decision"]["training_unlocked"] is False
