from pathlib import Path

import numpy as np

from official_x2.audit_phase14_recovery_ood import (
    ALL_GROUPS,
    build_report,
    group_pair_distance,
    robust_center_scale,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET = (
    REPO_ROOT.parent
    / "x2_official_rl_deploy_v1/models/stage335_stage306_stiff_fixed_stop_recovery_states.npz"
)
SOURCE_REPORT = REPO_ROOT / "reports/official_x2/stage335_stop_recovery_state_extraction.json"
RESULT_ROOT = (
    REPO_ROOT.parent
    / "x2_official_rl_deploy_v1/results/official_native_strict_20260807"
)


def test_group_distance_is_rms_normalized_by_dimension() -> None:
    # Duplicating an identically scaled feature must not double a semantic
    # group's distance merely because that group has more coordinates.
    reference_one = np.asarray([[0.0], [2.0]])
    query_one = np.asarray([[1.0]])
    scale_one = np.ones(1)
    one = group_pair_distance(query_one, reference_one, scale_one)
    many = group_pair_distance(
        np.repeat(query_one, 31, axis=1),
        np.repeat(reference_one, 31, axis=1),
        np.ones(31),
    )
    np.testing.assert_allclose(one, many, atol=0.0, rtol=0.0)


def test_robust_scaling_uses_iqr_and_a_finite_constant_fallback() -> None:
    reference = np.asarray([[0.0, 4.0], [1.0, 4.0], [2.0, 4.0]])
    center, scale = robust_center_scale(reference)
    np.testing.assert_allclose(center, [1.0, 4.0])
    assert scale[0] > 0.0
    assert scale[1] == 1.0


def test_phase14_real_contract_and_offline_ood_result() -> None:
    report = build_report(DATASET, SOURCE_REPORT, RESULT_ROOT)
    provenance = report["provenance"]
    assert provenance["training"]["dataset_sha256"] == (
        "4d8ce06b6e8dd9b43363dadedb781e9feb75f72ac092b475de2a1e2772545013"
    )
    assert provenance["training"]["state_count"] == 90
    assert provenance["training"]["eventual_pass"] == 63
    assert provenance["training"]["eventual_fail"] == 27
    assert provenance["phase13"]["healthy_query_episodes"] == 4
    assert provenance["phase13"]["query_rows"] == 304
    assert len(provenance["phase13"]["excluded_query_episodes"]) == 1
    assert provenance["phase13"]["stand_rows"] == 250
    assert set(report["query_window_summary"]) == {"1", "3", "4", "5"}
    assert all(row["rows"] == 76 for row in report["query_window_summary"].values())

    training = report["references"]["training_all"]
    assert training["equal_group_composite"]["query_ood_fraction"] == 0.0
    for name in ("joint_position", "joint_velocity", "current_action"):
        assert training["groups"][name]["query_ood_fraction"] == 0.0
    previous = training["groups"]["previous_action"]
    assert previous["query_ood_fraction"] > 0.20
    assert previous["first_ood_s_by_episode"]["3"] <= 0.26 + 1.0e-9
    assert previous["first_ood_s_by_episode"]["4"] <= 0.24 + 1.0e-9
    assert previous["first_ood_s_by_episode"]["5"] <= 0.24 + 1.0e-9

    stand = report["references"]["source_stand_stable"]
    assert stand["equal_group_composite"]["query_ood_fraction"] == 1.0
    for name in ALL_GROUPS:
        expected = 0.0 if name in ("command", "gait_phase") else 1.0
        assert stand["groups"][name]["query_ood_fraction"] == expected
    assert report["analysis_only"]
    assert not report["causal_claim"]


def test_phase14_report_preserves_the_frozen_phase13_adapter_digest() -> None:
    report = (REPO_ROOT / "reports/baseline/x2_recovery_phase14_ood.md").read_text(
        encoding="utf-8"
    )
    assert "a929ccec61f91af9d8492b11c4e50c7a61379d27a69b02327b984b2a9dd6759f" in report
