from __future__ import annotations

from tools.retarget.finalize_x2_phase70_long_lookahead import (
    classify_scientific_outcome,
    reward_conflict_confirmed,
    stable_sign,
)


def summary(point: float, lower: float, upper: float) -> dict[str, float]:
    return {
        "point": point,
        "bootstrap_p025": lower,
        "bootstrap_p975": upper,
        "bootstrap_same_sign_fraction": 1.0,
        "leave_one_env_out_same_sign_fraction": 1.0,
    }


def test_stable_sign_requires_interval_and_cluster_stability():
    assert stable_sign(summary(1.0, 0.2, 1.4)) == 1
    assert stable_sign(summary(-1.0, -1.4, -0.2)) == -1
    assert stable_sign(summary(1.0, -0.2, 1.4)) == 0


def test_scientific_classification_is_diagnostic_only():
    positive = summary(1.0, 0.2, 1.4)
    negative = summary(-1.0, -1.4, -0.2)
    uncertain = summary(0.1, -0.2, 0.4)
    assert (
        classify_scientific_outcome(negative, negative)
        == "PASS_REWARD_CONFLICT_DIAG_ONLY"
    )
    assert (
        classify_scientific_outcome(positive, positive)
        == "PASS_LONG_LOOKAHEAD_POSITIVE_DIAG_ONLY"
    )
    assert (
        classify_scientific_outcome(positive, uncertain)
        == "INCONCLUSIVE_TOTAL_DIRECTION_STOP"
    )


def test_reward_conflict_requires_primary_reward_only_and_locomotion():
    negative = summary(-1.0, -1.4, -0.2)
    alignment = {
        name: negative
        for name in (
            "primary_total_vs_positive_pitch",
            "primary_total_vs_lower_support_outside",
            "reward_only_total_vs_positive_pitch",
            "reward_only_total_vs_lower_support_outside",
            "locomotion_reward_vs_positive_pitch",
            "locomotion_reward_vs_lower_support_outside",
        )
    }
    assert reward_conflict_confirmed(alignment)
    alignment["locomotion_reward_vs_positive_pitch"] = summary(1.0, 0.2, 1.4)
    assert not reward_conflict_confirmed(alignment)
