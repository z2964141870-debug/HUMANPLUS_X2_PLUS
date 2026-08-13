from cwi_x2.locomotion_zero_confirmation import (
    aggregate_decision,
    checkerboard_assignment,
    seed_gates,
    summarize_treatment,
    validate_assignment,
)


def test_checkerboard_is_balanced_per_offset_and_flips():
    a0 = checkerboard_assignment(256, 0)
    a1 = checkerboard_assignment(256, 1)
    assert validate_assignment(a0, 0)
    assert validate_assignment(a1, 1)
    assert all(left != right for left, right in zip(a0, a1))


def test_summary_excludes_unreached_placeholders_from_metrics():
    rows = []
    for env_id in range(128):
        for segment in ("cruise", "decelerate", "hold"):
            reached = not (env_id == 0 and segment == "hold")
            rows.append({
                "treatment": "locomotion_zero", "segment": segment,
                "sample_count": 10 if reached else 0,
                "sample_fraction": 1.0 if reached else 0.0,
                "terminated": env_id == 0 and segment == "cruise", "time_out": False,
                "pitch_mean_rad": 0.0 if reached else None,
                "pitch_p05_rad": 0.0 if reached else None,
                "velocity_mse": 0.01 if reached else None,
                "lateral_mse": 0.01 if reached else None,
                "yaw_mse": 0.01 if reached else None,
                "support_mean_m": 0.0 if reached else None,
                "slip_p95_mps": 0.0 if reached else None,
                "flight_fraction": 0.0 if reached else None,
                "root_height_min_m": 0.65 if reached else None,
                "tilt_max_rad": 0.1 if reached else None,
                "speed_p95_mps": 0.05 if reached else None,
                "double_support_fraction": 1.0 if reached else None,
                "action_slew_max": 0.1 if reached else None,
                "normalized_clip_fraction": 0.0 if reached else None,
            })
    summary = summarize_treatment(rows, "locomotion_zero")
    assert summary["hold"]["reached_count"] == 127
    assert summary["hold"]["root_z"] == 0.65
    assert summary["hold"]["reach_fraction"] == 127 / 128


def _safe_summary(hold_terms=0):
    segment = {
        "terminations": 0, "timeouts": 0, "reach_fraction": 1.0,
        "velocity_rmse": 0.05, "lateral_rms": 0.01, "yaw_rmse": 0.05,
        "support": 0.005, "slip": 0.05, "root_z": 0.65, "tilt": 0.15,
        "speed_p95": 0.05, "double_support": 1.0, "flight": 0.0,
        "action_slew": 0.2, "normalized_clip": 0.5,
    }
    return {"cruise": dict(segment), "decelerate": dict(segment),
            "hold": dict(segment, terminations=hold_terms)}


def test_seed_gates_use_relative_action_contract():
    direct = _safe_summary(100)
    candidate = _safe_summary(0)
    assert all(seed_gates(direct, candidate).values())
    candidate["hold"]["action_slew"] = 0.22
    assert not seed_gates(direct, candidate)["relative_action_slew"]


def test_aggregate_requires_all_three_seeds():
    records = []
    for _ in range(3):
        direct = _safe_summary(100)
        candidate = _safe_summary(0)
        records.append({"summaries": {"direct_mix": direct, "locomotion_zero": candidate},
                        "gates": seed_gates(direct, candidate)})
    decision, pooled = aggregate_decision(records)
    assert decision == "PASS_LOCOMOTION_ZERO_OFFICIAL_PANEL_PREREG_ONLY"
    assert all(pooled.values())
    records[1]["summaries"]["locomotion_zero"]["hold"]["terminations"] = 1
    decision, _ = aggregate_decision(records)
    assert decision == "FAIL_LOCOMOTION_ZERO_NEW_ACTOR_PREREG_ONLY"
