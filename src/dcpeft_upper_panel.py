"""Analysis helpers for fixed-feet upper-body capability rollouts.

The evaluator deliberately keeps only the first episode of every environment.
Long evaluation jobs may reset short motions and continue collecting; mixing the
post-reset samples would weight short motions more heavily and hide failures.
"""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable


WRIST_BODIES = ("left_wrist_roll_link", "right_wrist_roll_link")
UPPER_BODIES = (
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_roll_link",
)
FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")


def percentile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return math.nan
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * quantile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def finite_stats(values: Iterable[float]) -> dict[str, float | int]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"count": 0, "mean": math.nan, "p50": math.nan, "p95": math.nan, "max": math.nan}
    return {
        "count": len(finite),
        "mean": fmean(finite),
        "p50": percentile(finite, 0.50),
        "p95": percentile(finite, 0.95),
        "max": max(finite),
    }


def load_first_episodes(path: str | Path) -> dict[int, list[dict[str, Any]]]:
    """Load one start-from-frame-zero episode per environment from a trace."""
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with Path(path).open() as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "env_id" not in row or "step_index" not in row:
                raise ValueError(f"trace row {line_number} lacks env_id/step_index: {path}")
            grouped[int(row["env_id"])].append(row)

    episodes: dict[int, list[dict[str, Any]]] = {}
    for env_id, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: (int(row["step_index"]), int(row.get("rollout_index", 0))))
        first_motion = rows[0].get("motion_key")
        episode: list[dict[str, Any]] = []
        for row in rows:
            if row.get("motion_key") != first_motion:
                raise ValueError(
                    f"motion changed before first episode ended in env {env_id}: "
                    f"{first_motion!r} -> {row.get('motion_key')!r}"
                )
            episode.append(row)
            if bool(row.get("done", False)):
                break
        episodes[env_id] = episode
    return episodes


def _body_values(rows: list[dict[str, Any]], bodies: tuple[str, ...]) -> list[float]:
    values: list[float] = []
    for row in rows:
        residuals = row.get("body_residuals", {})
        missing = [body for body in bodies if body not in residuals]
        if missing:
            raise ValueError(f"trace lacks body residuals {missing}")
        values.extend(float(residuals[body]["norm_m"]) for body in bodies)
    return values


def summarize_episode(
    rows: list[dict[str, Any]],
    *,
    control_dt: float = 0.02,
    root_z_floor_m: float = 0.45,
    anchor_p95_limit_m: float = 0.10,
    foot_p95_limit_m: float = 0.10,
    wrist_p95_limit_m: float = 0.15,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty episode")
    last = rows[-1]
    causes = tuple(str(value) for value in last.get("termination_causes", []))
    completed = bool(last.get("done", False)) and causes == ("motion_time_out",)
    non_timeout_done = bool(last.get("done", False)) and any(
        cause != "motion_time_out" for cause in causes
    )

    root_z = [
        float(row["structure_states"]["base_link"]["robot_pos_w_m"][2]) for row in rows
    ]
    anchor_pos = [float(row["anchor_pos_error_m"]) for row in rows]
    anchor_ori = [float(row["anchor_ori_error_rad"]) for row in rows]
    wrist = _body_values(rows, WRIST_BODIES)
    upper = _body_values(rows, UPPER_BODIES)
    foot = _body_values(rows, FOOT_BODIES)
    metrics = {
        "anchor_pos_m": finite_stats(anchor_pos),
        "anchor_ori_rad": finite_stats(anchor_ori),
        "wrist_error_m": finite_stats(wrist),
        "upper_error_m": finite_stats(upper),
        "foot_error_m": finite_stats(foot),
        "root_z_m": {"min": min(root_z), "mean": fmean(root_z)},
    }
    gates = {
        "completed": completed,
        "stable": min(root_z) >= root_z_floor_m and not non_timeout_done,
        "anchor": metrics["anchor_pos_m"]["p95"] <= anchor_p95_limit_m,
        "foot_tracking": metrics["foot_error_m"]["p95"] <= foot_p95_limit_m,
        "upper_tracking": metrics["wrist_error_m"]["p95"] <= wrist_p95_limit_m,
    }
    gates["strict"] = all(gates.values())
    return {
        "motion_key": str(rows[0].get("motion_key")),
        "env_id": int(rows[0]["env_id"]),
        "samples": len(rows),
        "duration_s": len(rows) * control_dt,
        "last_motion_time_step": int(last.get("motion_time_step", -1)),
        "termination_causes": list(causes),
        "metrics": metrics,
        "gates": gates,
    }


def summarize_trace(path: str | Path, **kwargs: Any) -> dict[str, Any]:
    episodes = [summarize_episode(rows, **kwargs) for rows in load_first_episodes(path).values()]
    all_rows = [row for rows in load_first_episodes(path).values() for row in rows]
    aggregate = {
        "motions": len(episodes),
        "samples": len(all_rows),
        "completed": sum(int(item["gates"]["completed"]) for item in episodes),
        "stable": sum(int(item["gates"]["stable"]) for item in episodes),
        "strict": sum(int(item["gates"]["strict"]) for item in episodes),
        "anchor_pos_m": finite_stats(float(row["anchor_pos_error_m"]) for row in all_rows),
        "anchor_ori_rad": finite_stats(float(row["anchor_ori_error_rad"]) for row in all_rows),
        "wrist_error_m": finite_stats(_body_values(all_rows, WRIST_BODIES)),
        "upper_error_m": finite_stats(_body_values(all_rows, UPPER_BODIES)),
        "foot_error_m": finite_stats(_body_values(all_rows, FOOT_BODIES)),
        "root_z_min_m": min(
            float(row["structure_states"]["base_link"]["robot_pos_w_m"][2])
            for row in all_rows
        ),
    }
    return {"trace": str(Path(path)), "episodes": episodes, "aggregate": aggregate}


def relative_delta(candidate: float, baseline: float) -> float:
    if baseline == 0.0:
        return 0.0 if candidate == 0.0 else math.inf
    return (candidate - baseline) / baseline


def compare_to_baseline(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    cand = candidate["aggregate"]
    base = baseline["aggregate"]
    deltas = {
        "wrist_mean": relative_delta(cand["wrist_error_m"]["mean"], base["wrist_error_m"]["mean"]),
        "wrist_p95": relative_delta(cand["wrist_error_m"]["p95"], base["wrist_error_m"]["p95"]),
        "upper_mean": relative_delta(cand["upper_error_m"]["mean"], base["upper_error_m"]["mean"]),
        "anchor_p95": relative_delta(cand["anchor_pos_m"]["p95"], base["anchor_pos_m"]["p95"]),
        "foot_p95": relative_delta(cand["foot_error_m"]["p95"], base["foot_error_m"]["p95"]),
    }
    preservation = (
        cand["completed"] >= base["completed"]
        and cand["stable"] >= base["stable"]
        and deltas["wrist_mean"] <= 0.05
        and deltas["wrist_p95"] <= 0.05
    )
    return {"relative_deltas": deltas, "upper_capability_preserved_5pct": preservation}
