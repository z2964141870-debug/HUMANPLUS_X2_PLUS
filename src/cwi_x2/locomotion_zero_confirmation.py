"""Pure aggregation and gates for fresh-seed locomotion-zero confirmation."""

from __future__ import annotations

import math
from typing import Iterable


SEGMENTS = ("cruise", "decelerate", "hold")
TREATMENTS = ("direct_mix", "locomotion_zero")


def checkerboard_assignment(num_envs: int, seed_index: int) -> list[str]:
    """Balance treatment across 16 reset blocks and 16 gait offsets."""
    if num_envs != 256 or seed_index not in (0, 1, 2):
        raise ValueError("confirmation requires 256 envs and seed index 0..2")
    result = []
    for env_id in range(num_envs):
        block = env_id // 16
        offset = env_id % 16
        candidate = (block + offset + seed_index) % 2 == 1
        result.append("locomotion_zero" if candidate else "direct_mix")
    return result


def validate_assignment(assignments: list[str], seed_index: int) -> bool:
    if assignments != checkerboard_assignment(len(assignments), seed_index):
        return False
    if any(assignments.count(treatment) != 128 for treatment in TREATMENTS):
        return False
    for offset in range(16):
        lanes = [assignments[env_id] for env_id in range(offset, 256, 16)]
        if any(lanes.count(treatment) != 8 for treatment in TREATMENTS):
            return False
    return True


def summarize_treatment(rows: Iterable[dict], treatment: str) -> dict[str, dict[str, float]]:
    relevant = [row for row in rows if row["treatment"] == treatment]
    result: dict[str, dict[str, float]] = {}
    for segment in SEGMENTS:
        subset = [row for row in relevant if row["segment"] == segment]
        if len(subset) != 128:
            raise ValueError(f"expected 128 {treatment}/{segment} rows, got {len(subset)}")
        reached = [row for row in subset if int(row["sample_count"]) > 0]
        if not reached:
            raise ValueError(f"no lanes reached {treatment}/{segment}")

        def values(name: str) -> list[float]:
            output = [float(row[name]) for row in reached]
            if not output or not all(math.isfinite(value) for value in output):
                raise ValueError(f"non-finite {treatment}/{segment}/{name}")
            return output

        count = len(subset)
        result[segment] = {
            "environment_count": count,
            "reached_count": len(reached),
            "reach_fraction": len(reached) / count,
            "sample_fraction": sum(float(row["sample_fraction"]) for row in subset) / count,
            "terminations": sum(bool(row["terminated"]) for row in subset),
            "timeouts": sum(bool(row["time_out"]) for row in subset),
            "pitch_mean": sum(values("pitch_mean_rad")) / len(reached),
            "pitch_p05": sum(values("pitch_p05_rad")) / len(reached),
            "velocity_rmse": (sum(values("velocity_mse")) / len(reached)) ** 0.5,
            "lateral_rms": (sum(values("lateral_mse")) / len(reached)) ** 0.5,
            "yaw_rmse": (sum(values("yaw_mse")) / len(reached)) ** 0.5,
            "support": sum(values("support_mean_m")) / len(reached),
            "slip": max(values("slip_p95_mps")),
            "flight": sum(values("flight_fraction")) / len(reached),
            "root_z": min(values("root_height_min_m")),
            "tilt": max(values("tilt_max_rad")),
            "speed_p95": max(values("speed_p95_mps")),
            "double_support": sum(values("double_support_fraction")) / len(reached),
            "action_slew": max(values("action_slew_max")),
            "normalized_clip": max(values("normalized_clip_fraction")),
        }
    return result


def seed_gates(
    direct: dict[str, dict[str, float]],
    candidate: dict[str, dict[str, float]],
) -> dict[str, bool]:
    dc, dd, dh = (direct[name] for name in SEGMENTS)
    cc, cd, ch = (candidate[name] for name in SEGMENTS)
    return {
        "candidate_zero_timeout": sum(candidate[s]["timeouts"] for s in SEGMENTS) == 0,
        "candidate_cruise_termination": cc["terminations"] <= dc["terminations"] + 1,
        "candidate_decel_termination": cd["terminations"] <= dd["terminations"] + 1,
        "candidate_hold_zero_termination": ch["terminations"] == 0,
        "candidate_hold_reach": ch["reach_fraction"] >= 0.98,
        "cruise_velocity": cc["velocity_rmse"] <= dc["velocity_rmse"] + 0.010,
        "cruise_lateral": cc["lateral_rms"] <= dc["lateral_rms"] + 0.015,
        "cruise_yaw": cc["yaw_rmse"] <= dc["yaw_rmse"] + 0.015,
        "cruise_support": cc["support"] <= dc["support"] + 0.002,
        "cruise_slip": cc["slip"] <= dc["slip"] + 0.030,
        "cruise_root_height": cc["root_z"] >= 0.60,
        "cruise_tilt": cc["tilt"] <= 0.35,
        "decel_velocity": cd["velocity_rmse"] <= dd["velocity_rmse"] + 0.015,
        "decel_lateral": cd["lateral_rms"] <= dd["lateral_rms"] + 0.015,
        "decel_yaw": cd["yaw_rmse"] <= dd["yaw_rmse"] + 0.020,
        "decel_support": cd["support"] <= dd["support"] + 0.003,
        "decel_slip": cd["slip"] <= dd["slip"] + 0.030,
        "decel_root_height": cd["root_z"] >= 0.60,
        "decel_tilt": cd["tilt"] <= 0.35,
        "hold_speed": ch["speed_p95"] <= 0.12,
        "hold_double_support": ch["double_support"] >= 0.98,
        "hold_root_height": ch["root_z"] >= 0.60,
        "hold_tilt": ch["tilt"] <= 0.35,
        "hold_support": ch["support"] <= 0.015,
        "hold_slip": ch["slip"] <= 0.15,
        "hold_flight": ch["flight"] <= 0.01,
        "relative_action_slew": max(candidate[s]["action_slew"] for s in SEGMENTS)
        <= max(direct[s]["action_slew"] for s in SEGMENTS) + 0.01,
        "relative_normalized_clip": max(candidate[s]["normalized_clip"] for s in SEGMENTS)
        <= max(direct[s]["normalized_clip"] for s in SEGMENTS) + 0.01,
    }


def aggregate_decision(seed_records: list[dict]) -> tuple[str, dict[str, bool]]:
    if len(seed_records) != 3:
        raise ValueError("exactly three seed records are required")
    all_seed_gates = all(all(record["gates"].values()) for record in seed_records)
    direct_hold_terms = sum(record["summaries"]["direct_mix"]["hold"]["terminations"] for record in seed_records)
    candidate_hold_terms = sum(
        record["summaries"]["locomotion_zero"]["hold"]["terminations"] for record in seed_records
    )
    pooled = {
        "all_seed_gates": all_seed_gates,
        "hold_termination_reduction_90pct": candidate_hold_terms <= 0.10 * max(direct_hold_terms, 1),
        "candidate_hold_zero_all_seeds": candidate_hold_terms == 0,
        "candidate_hold_reach_all_seeds": all(
            record["summaries"]["locomotion_zero"]["hold"]["reach_fraction"] >= 0.98
            for record in seed_records
        ),
    }
    if all(pooled.values()):
        decision = "PASS_LOCOMOTION_ZERO_OFFICIAL_PANEL_PREREG_ONLY"
    else:
        hold_core = all(
            record["summaries"]["locomotion_zero"]["hold"]["terminations"] == 0
            and record["summaries"]["locomotion_zero"]["hold"]["reach_fraction"] >= 0.98
            and record["summaries"]["locomotion_zero"]["hold"]["speed_p95"] <= 0.12
            and record["summaries"]["locomotion_zero"]["hold"]["root_z"] >= 0.60
            and record["summaries"]["locomotion_zero"]["hold"]["tilt"] <= 0.35
            for record in seed_records
        )
        decision = (
            "PARTIAL_LOCOMOTION_ZERO_BRAKE_SKILL_PREREG_ONLY"
            if hold_core else "FAIL_LOCOMOTION_ZERO_NEW_ACTOR_PREREG_ONLY"
        )
    return decision, pooled
