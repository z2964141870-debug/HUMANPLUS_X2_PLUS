import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "research/dynamic_retargeting_20260811/phase17_next_generator_contract"


def test_phase17_capabilities_ready_without_gpu_or_physics() -> None:
    result = json.loads((DIR / "phase17_capability.json").read_text(encoding="utf-8"))
    assert result["execution"]["mj_step_calls"] == 0
    assert result["execution"]["gpu"] is False
    assert result["decision"]["offline_joint_geometry_generator_implementable_now"] is True
    assert result["decision"]["raw_mujoco_replay_implementable_now"] is True
    assert result["decision"]["faithful_ddr_or_omnitrack_upstream_available"] is False
    assert result["decision"]["long_training_unlocked"] is False


def test_phase17_contract_keeps_physics_and_training_locked() -> None:
    contract = json.loads((DIR / "phase17_generator_contract.json").read_text(encoding="utf-8"))
    assert contract["status"] == "IMPLEMENTATION_AUTHORIZED_OFFLINE_GEOMETRY_ONLY"
    assert contract["stage_a_offline_geometry"]["physics_steps"] == 0
    assert contract["stage_b_raw_physics"]["authorized"] is False
    assert contract["stage_c_training"]["authorized"] is False
    assert contract["resource_contract"]["single_process"] is True
    assert contract["resource_contract"]["cpu_threads_max"] == 4
    assert contract["resource_contract"]["gpu"] is False
    assert contract["resource_contract"]["parallel_agents"] is False
