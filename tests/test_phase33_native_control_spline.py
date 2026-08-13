import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "research/dynamic_retargeting_20260811/phase33_native_control_spline_contract.json"
RESULT = ROOT / "research/dynamic_retargeting_20260811/phase33_native_control_spline_result.json"


def test_phase33_contract_is_fixed_and_optimizer_free():
    contract = json.loads(CONTRACT.read_text())
    assert contract["representation"]["control_knots"] == 18
    assert contract["representation"]["spline"] == "linear least-squares fit"
    assert contract["resources"] == {"cpu_threads": 1, "gpu": False, "optimizer_instances": 0, "optimizer_steps": 0}
    assert contract["next_solve_authorized"] is False


def test_phase33_result_if_present_is_fail_closed():
    if not RESULT.exists():
        return
    result = json.loads(RESULT.read_text())
    assert result["execution"]["optimizer_instances"] == 0
    assert result["decision"]["single_short_solve_authorized"] is False
    assert result["decision"]["training_unlocked"] is False
    assert result["decision"]["control_spline_warmstart_qualified"] == all(result["checks"].values())
