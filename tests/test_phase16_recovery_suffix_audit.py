import numpy as np

from official_x2.audit_phase16_recovery_suffix import _query, _relative_group


def _row(tick=470, time_s=0.0):
    obs = np.zeros(93, dtype=float)
    obs[8] = -1.0
    return {
        "observation_93d": obs.tolist(),
        "actual_issued_action": np.zeros(15).tolist(),
        "time_after_handoff_s": time_s,
        "source_tick": tick,
        "physical_state": {
            "root_position_m": [0.0, 0.0, 0.62],
            "root_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            "root_linear_velocity_world_mps": [0.1, 0.0, 0.0],
        },
    }


def test_phase16_query_preserves_93d_groups_and_root_convention():
    groups, meta = _query([_row()])
    assert groups["previous_action"].shape == (1, 15)
    assert groups["projected_gravity"].shape == (1, 3)
    assert groups["root_state"].shape == (1, 4)
    np.testing.assert_allclose(groups["root_state"][0], [0.62, 0.0, 0.0, 0.1])
    assert meta[0]["time_after_handoff_s"] == 0.0


def test_phase16_relative_group_does_not_create_a_gate():
    current = {
        "reference_loo_p95_threshold": 1.0,
        "query_ood_fraction": 0.25,
        "first_ood_s_by_episode": {"phase16": 0.24},
        "first_ood_s_median": 0.24,
        "query_distance_median": 0.5,
        "query_distance_p95": 1.5,
    }
    historical = dict(current)
    result = _relative_group(current, historical)
    assert result["threshold_identical"] is True
    assert result["phase16_first_ood_s"] == 0.24
    assert "pass" not in result
