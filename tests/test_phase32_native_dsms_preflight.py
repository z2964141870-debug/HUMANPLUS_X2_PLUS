import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "research/dynamic_retargeting_20260811/phase32_native_dsms_preflight_contract.json"
RESULT = ROOT / "research/dynamic_retargeting_20260811/phase32_native_dsms_preflight_result.json"


def test_phase32_contract_is_preflight_only_and_low_resource():
    contract = json.loads(CONTRACT.read_text())
    assert contract["status"] == "PREREGISTERED_PREFLIGHT_ONLY"
    future = contract["future_single_solve_if_qualified"]
    assert future["authorized"] is False
    assert future["cpu_threads"] == 1 and future["gpu"] is False
    assert future["max_iterations"] == 50
    assert contract["training_authorized"] is False
    assert contract["physics_rollout_authorized"] is False


def test_phase32_result_if_present_is_fail_closed():
    if not RESULT.exists():
        return
    result = json.loads(RESULT.read_text())
    assert result["execution"]["optimizer_instances"] == 0
    assert result["execution"]["optimizer_steps"] == 0
    assert result["execution"]["gpu"] is False
    assert result["decision"]["single_short_solve_authorized"] is False
    assert result["decision"]["training_unlocked"] is False
    assert result["decision"]["native_dsms_warmstart_qualified"] == all(result["checks"].values())
