import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "reports/official_x2/phase38_sequence_consistent_bridge_target.json"


def test_phase38_sequence_target_contract() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    assert result["execution"] == {
        "physics_steps": 0,
        "optimizer_steps": 0,
        "training": False,
        "read_only": True,
    }
    assert result["contracts"]["sequence_consistent_candidate_count"] == 215
    assert result["contracts"]["default_pose_reconstruction_max_abs_rad"] <= 1e-6
    assert result["decision"]["sequence_consistent_target_available"] is True
    assert result["decision"]["endpoint_already_inside_one_jointly_sequence_consistent_target"] is False
    assert result["decision"]["direct_bridge_training_or_physics_unlocked"] is False


def test_phase38_suffix_is_exactly_one_second_and_hash_complete() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    suffix = result["suffix"]
    assert len(suffix["manifest_row_indices"]) == 51
    assert len(suffix["snapshot_sha256"]) == 51
    assert len(suffix["controller_state_sha256"]) == 51
    assert suffix["source_ticks"] == list(range(suffix["source_ticks"][0], suffix["source_ticks"][0] + 51))
    assert suffix["all_success_safe_and_eligible"] is True
    assert all(len(value) == 64 for value in suffix["snapshot_sha256"])
    assert all(len(value) == 64 for value in suffix["controller_state_sha256"])
