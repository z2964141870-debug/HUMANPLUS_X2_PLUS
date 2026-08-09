from official_x2.audit_base_recovery_suffix_inventory import episode_class


def test_episode_class_uses_existing_physical_gates_only():
    assert episode_class({"stop_gate_pass": True, "survived_stop_height_gate": True}) == "success"
    assert episode_class({"stop_gate_pass": False, "survived_stop_height_gate": True}) == "critical_nonfall_gate_failure"
    assert episode_class({"stop_gate_pass": False, "survived_stop_height_gate": False}) == "failure_with_height_collapse"


def test_episode_class_does_not_call_eventual_failure_critical_without_survival():
    assert episode_class({"stop_gate_pass": False}) == "failure_with_height_collapse"
