import torch
import pytest

from tools.retarget.finalize_x2_phase72_antithetic import (
    SUPPORT_RELATIONS,
    fold_cosines,
    relation_summary,
)


def test_balanced_folds_match_for_repeated_direction() -> None:
    value = torch.ones(5, 64, 66, dtype=torch.float64)
    summary = fold_cosines(value)
    assert summary["cross_seed_half"] == pytest.approx(1.0)
    assert summary["seed_shifted_checkerboard"] == pytest.approx(1.0)


def test_relation_summary_accepts_uniform_material_positive_signal() -> None:
    metric = torch.ones(5, 64, 66, dtype=torch.float64)
    component = 0.5 * metric
    report = relation_summary(component, metric, expected_sign=1, bootstrap_seed=72)
    assert report["point"] == 0.5
    assert all(report["checks"].values())
    assert report["exact_seed_sign_probability"] == 1.0 / 32.0


def test_relation_summary_rejects_wrong_sign() -> None:
    metric = torch.ones(5, 64, 66, dtype=torch.float64)
    component = -0.5 * metric
    report = relation_summary(component, metric, expected_sign=1, bootstrap_seed=73)
    assert not report["checks"]["five_of_five_seed_material_sign"]


def test_support_secondary_has_all_four_relations() -> None:
    assert set(SUPPORT_RELATIONS) == {
        "support_reward_vs_lower_support",
        "primary_total_vs_lower_support",
        "reward_only_vs_lower_support",
        "locomotion_vs_lower_support",
    }
