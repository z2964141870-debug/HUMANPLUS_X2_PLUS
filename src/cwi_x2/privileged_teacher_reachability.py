"""Pure helpers for the X2 privileged teacher reachability screen.

The teacher is deliberately an oracle: it may use the deployable gait suffix
to generate a coordinated 15-D residual, but it never updates policy weights.
The helpers live outside MuJoCo so their numerical and decision contracts can
be tested without starting a simulator.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np


ACTION_DIM = 15
FEATURE_DIM = 5
PARAMETER_SHAPE = (ACTION_DIM, FEATURE_DIM)


def phase_contact_features(gait_suffix: np.ndarray) -> np.ndarray:
    """Return bias/clock/contact features from the frozen 4-D gait suffix."""

    gait = np.asarray(gait_suffix, dtype=np.float64)
    if gait.shape != (4,) or not np.isfinite(gait).all():
        raise ValueError("gait suffix must be a finite 4-vector")
    if np.any(gait[2:] < -1.0e-6) or np.any(gait[2:] > 1.0 + 1.0e-6):
        raise ValueError("desired contacts must lie in [0, 1]")
    return np.asarray(
        [1.0, gait[0], gait[1], gait[2] - 0.5, gait[3] - 0.5],
        dtype=np.float64,
    )


def teacher_residual(
    parameters: np.ndarray,
    gait_suffix: np.ndarray,
    *,
    bound: float,
) -> np.ndarray:
    """Map a full 15x5 parameter matrix to a smooth bounded action residual."""

    values = np.asarray(parameters, dtype=np.float64)
    if values.shape != PARAMETER_SHAPE or not np.isfinite(values).all():
        raise ValueError(f"parameters must be finite with shape {PARAMETER_SHAPE}")
    if not math.isfinite(bound) or bound <= 0.0:
        raise ValueError("bound must be finite and positive")
    features = phase_contact_features(gait_suffix)
    # The normalization keeps a coefficient at its search bound from turning
    # every joint into a saturated square wave merely because five features
    # are present.  Zero parameters remain bit-exact zero.
    signal = values @ features / math.sqrt(float(FEATURE_DIM))
    return (bound * np.tanh(signal)).astype(np.float32)


def _heading_frame(delta_xy: np.ndarray, yaw: float) -> np.ndarray:
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [cosine * delta_xy[0] + sine * delta_xy[1],
         -sine * delta_xy[0] + cosine * delta_xy[1]],
        dtype=np.float64,
    )


def support_outside_distance(
    com_xy: np.ndarray,
    foot_xy: np.ndarray,
    foot_yaw: np.ndarray,
    contact: np.ndarray,
    *,
    root_yaw: float,
    sole_bounds: tuple[float, float, float, float] = (-0.070, 0.144, -0.065, 0.065),
) -> float:
    """Conservative COM distance outside the contacted sole rectangles."""

    com = np.asarray(com_xy, dtype=np.float64)
    feet = np.asarray(foot_xy, dtype=np.float64)
    yaw = np.asarray(foot_yaw, dtype=np.float64)
    active = np.asarray(contact, dtype=bool)
    if com.shape != (2,) or feet.shape != (2, 2) or yaw.shape != (2,) or active.shape != (2,):
        raise ValueError("invalid support geometry shape")
    if not np.isfinite(com).all() or not np.isfinite(feet).all() or not np.isfinite(yaw).all():
        raise ValueError("support geometry must be finite")
    xmin, xmax, ymin, ymax = sole_bounds
    count = int(active.sum())
    if count == 0:
        return 0.0
    if count == 1:
        index = int(np.argmax(active))
        local = _heading_frame(com - feet[index], float(yaw[index]))
        dx = max(xmin - local[0], local[0] - xmax, 0.0)
        dy = max(ymin - local[1], local[1] - ymax, 0.0)
        return float(math.hypot(dx, dy))
    centers = np.stack([_heading_frame(feet[index] - com, root_yaw) for index in range(2)])
    lower_x, upper_x = float(centers[:, 0].min() + xmin), float(centers[:, 0].max() + xmax)
    lower_y, upper_y = float(centers[:, 1].min() + ymin), float(centers[:, 1].max() + ymax)
    dx = max(lower_x, -upper_x, 0.0)
    dy = max(lower_y, -upper_y, 0.0)
    return float(math.hypot(dx, dy))


def _quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def summarize_rollout(rows: Iterable[dict[str, Any]], *, command_vx: float) -> dict[str, Any]:
    records = list(rows)
    if not records:
        raise ValueError("rollout is empty")

    def array(name: str) -> np.ndarray:
        result = np.asarray([row[name] for row in records], dtype=np.float64)
        if not np.isfinite(result).all():
            raise ValueError(f"nonfinite rollout metric: {name}")
        return result

    pitch = array("signed_pitch_rad")
    vx = array("root_vx_b_mps")
    vy = array("root_vy_b_mps")
    heading = array("heading_error_rad")
    support = array("support_outside_m")
    root_z = array("root_z_m")
    tilt = array("root_tilt_rad")
    residual = np.stack([np.asarray(row["teacher_residual"], dtype=np.float64) for row in records])
    final_action = np.stack([np.asarray(row["final_action"], dtype=np.float64) for row in records])
    slip = np.asarray(
        [max((value for value in row["slip_mps"] if value is not None), default=0.0) for row in records],
        dtype=np.float64,
    )
    contact_count = np.asarray([row["contact_count"] for row in records], dtype=np.int64)
    action_delta = np.linalg.norm(np.diff(final_action, axis=0), axis=1)
    survival = bool(root_z.min() >= 0.45 and tilt.max() <= 0.80)
    return {
        "ticks": len(records),
        "signed_pitch_rad": {
            "mean": float(pitch.mean()),
            "p05": _quantile(pitch, 0.05),
            "p50": _quantile(pitch, 0.50),
        },
        "velocity_rmse_mps": float(np.sqrt(np.mean(np.square(vx - command_vx)))),
        "lateral_rms_mps": float(np.sqrt(np.mean(np.square(vy)))),
        "heading_abs_max_rad": float(np.max(np.abs(heading))),
        "support_outside_mean_m": float(support.mean()),
        "support_outside_p95_m": _quantile(support, 0.95),
        "stance_slip_p95_mps": _quantile(slip, 0.95),
        "flight_fraction": float(np.mean(contact_count == 0)),
        "root_z_min_m": float(root_z.min()),
        "root_tilt_max_rad": float(tilt.max()),
        "teacher_residual_rms": float(np.sqrt(np.mean(np.square(residual)))),
        "teacher_residual_abs_max": float(np.max(np.abs(residual))),
        "final_action_saturation_fraction": float(np.mean(np.abs(final_action) >= 0.999)),
        "final_action_delta_l2_p95": _quantile(action_delta, 0.95) if len(action_delta) else 0.0,
        "survived_full_horizon": survival,
    }


def rollout_cost(summary: dict[str, Any], *, target_pitch_rad: float = -0.05) -> float:
    """Frozen multi-objective reachability cost; lower is better."""

    pitch = summary["signed_pitch_rad"]
    backward_mean = max(0.0, target_pitch_rad - float(pitch["mean"])) / 0.10
    backward_tail = max(0.0, target_pitch_rad - float(pitch["p05"])) / 0.12
    cost = (
        4.0 * backward_mean**2
        + 2.0 * backward_tail**2
        + 2.0 * (float(summary["velocity_rmse_mps"]) / 0.15) ** 2
        + (float(summary["lateral_rms_mps"]) / 0.15) ** 2
        + (float(summary["heading_abs_max_rad"]) / 0.15) ** 2
        + 2.0 * (float(summary["support_outside_mean_m"]) / 0.04) ** 2
        + 0.5 * (float(summary["stance_slip_p95_mps"]) / 0.30) ** 2
        + 0.10 * (float(summary["teacher_residual_rms"]) / 0.10) ** 2
    )
    cost += 200.0 * max(0.0, 0.55 - float(summary["root_z_min_m"])) ** 2
    cost += 200.0 * max(0.0, float(summary["root_tilt_max_rad"]) - 0.35) ** 2
    if not summary["survived_full_horizon"]:
        cost += 1000.0
    return float(cost)


def reachability_gates(
    source: dict[str, Any], candidate: dict[str, Any], gates: dict[str, float]
) -> dict[str, bool]:
    pitch_mean_delta = candidate["signed_pitch_rad"]["mean"] - source["signed_pitch_rad"]["mean"]
    pitch_p05_delta = candidate["signed_pitch_rad"]["p05"] - source["signed_pitch_rad"]["p05"]
    return {
        "source_survival": bool(source["survived_full_horizon"]),
        "candidate_survival": bool(candidate["survived_full_horizon"]),
        "pitch_mean": pitch_mean_delta >= gates["pitch_mean_delta_rad_min"],
        "pitch_p05": pitch_p05_delta >= gates["pitch_p05_delta_rad_min"],
        "velocity": candidate["velocity_rmse_mps"] <= source["velocity_rmse_mps"] + gates["velocity_rmse_regression_mps_max"],
        "lateral": candidate["lateral_rms_mps"] <= source["lateral_rms_mps"] + gates["lateral_rms_regression_mps_max"],
        "heading": candidate["heading_abs_max_rad"] <= source["heading_abs_max_rad"] + gates["heading_regression_rad_max"],
        "support": candidate["support_outside_mean_m"] <= source["support_outside_mean_m"] + gates["support_regression_m_max"],
        "slip": candidate["stance_slip_p95_mps"] <= source["stance_slip_p95_mps"] + gates["slip_regression_mps_max"],
        "root_height": candidate["root_z_min_m"] >= max(gates["root_z_absolute_min_m"], source["root_z_min_m"] - gates["root_z_regression_m_max"]),
        "tilt": candidate["root_tilt_max_rad"] <= min(gates["tilt_absolute_max_rad"], source["root_tilt_max_rad"] + gates["tilt_regression_rad_max"]),
        "residual_bound": candidate["teacher_residual_abs_max"] <= gates["teacher_residual_abs_max"],
    }

