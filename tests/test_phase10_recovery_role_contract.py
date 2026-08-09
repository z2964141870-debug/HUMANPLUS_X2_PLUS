from pathlib import Path

import json

from official_x2.audit_phase10_recovery_role_contract import stop_controller_branch


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = Path("/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1")


def test_branch_extractor_distinguishes_stationary_and_recovery_routing():
    # Historical Phase10 contract: preserve the blocked-audit parser without
    # requiring the live adapter to remain forever at the known-bad routing.
    text = '''
            elif self.args.stop_controller in (
                "brake_blend_to_policy",
            ):
                policy_slot=stop_policy_slot(self.args.recovery_model)
            elif self.args.stop_controller == "curriculum_then_policy":
                policy_slot="stationary"
            elif self.args.stop_controller == "ramp_policy":
                pass
    '''
    curriculum = stop_controller_branch(text, "curriculum_then_policy")
    brake_blend = stop_controller_branch(text, "brake_blend_to_policy")
    assert 'policy_slot="stationary"' in curriculum
    assert "policy_slot=stop_policy_slot" not in curriculum
    assert "policy_slot=stop_policy_slot" in brake_blend


def test_phase10_proposed_recovery_only_ab_is_detected_as_noop():
    result = json.loads(
        (ROOT / "reports/baseline/x2_recovery_phase10_role_contract.json").read_text(
            encoding="utf-8"
        )
    )
    assert result["status"] == "blocked_noop_recovery_model_under_phase9_contract"
    assert result["control"]["valid_runs"] == 5
    assert result["control"]["contract_identical_across_5"] is True
    assert result["authority_audit"]["recovery_counts"] == [0, 0, 0, 0, 0]
    assert result["authority_audit"]["stationary_counts"] == [400] * 5
    assert result["authority_audit"]["proposed_intervention_has_authority"] is False
    assert result["official_runs_started"] == 0
    assert result["training_updates"] == 0
