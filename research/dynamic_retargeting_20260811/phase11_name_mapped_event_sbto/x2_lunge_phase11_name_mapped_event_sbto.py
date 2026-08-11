#!/usr/bin/env python3
"""Low-resource name-mapped event SBTO for the X2 lunge.

The runner is fail-closed: `search` refuses to start unless `preflight` has
proved the official actuator-name partition, exact deterministic zero event,
head2 nominal targets, active12 contact geometry and the frozen resource
contract.  No RL/PPO is provided by this program.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = Path("/home/humanplus/projects/ZHY/dsms_workspace")
P8 = ROOT / "phase8_reset_sbto"
P10 = ROOT / "phase10_actuator_mapping_audit"
OUT = ROOT / "phase11_name_mapped_event_sbto"
OUT.mkdir(parents=True, exist_ok=True)
sys.path[:0] = [str(P8), str(P10)]

import mujoco
import x2_lunge_phase8_reset_sbto as phase8
from audit_x2_retarget_actuator_mapping_phase10 import CorrectEvaluator, sha


SEED = 20260812
CANDIDATES = 24
ELITES = 6
ITERATIONS = 3
WINDOWS_S = (0.34, 0.52, 0.70)
MODE_NAMES = ("lateral_load", "right_stance_flex", "left_swing_flex", "sagittal_balance")
BOUNDS = np.asarray([0.08, 0.10, 0.15, 0.06], dtype=np.float64)[:, None] * np.ones((1, 2))
NATIVE_MIN_MM = -4.487
PREFLIGHT_PATH = OUT / "phase11_preflight.json"
PREREG_PATH = OUT / "prereg_phase11_name_mapped_event_sbto.json"


def smoothstep(x: float) -> float:
    u = float(np.clip(x, 0.0, 1.0))
    return u * u * (3.0 - 2.0 * u)


def prep_envelope(t: float) -> float:
    if t <= 0.02 or t >= 0.40:
        return 0.0
    if t <= 0.18:
        return smoothstep((t - 0.02) / 0.16)
    return 1.0 - smoothstep((t - 0.18) / 0.22)


def swing_envelope(t: float) -> float:
    if t <= 0.28:
        return 0.0
    if t <= 0.40:
        return smoothstep((t - 0.28) / 0.12)
    if t <= 0.70:
        return 1.0
    if t < 1.00:
        return 1.0 - smoothstep((t - 0.70) / 0.30)
    return 0.0


def longest_true_ms(values: np.ndarray) -> int:
    best = current = 0
    for value in np.asarray(values, dtype=bool):
        current = current + 1 if value else 0
        best = max(best, current)
    return best


class EventRunner:
    def __init__(self) -> None:
        self.correct = CorrectEvaluator()
        self.r = self.correct.r
        self.names = self.correct.names
        self.head = self.correct.head
        self.body = self.correct.body
        self.modes = np.zeros((len(MODE_NAMES), len(self.names)), dtype=np.float64)

        def set_mode(mode: int, mapping: dict[str, float]) -> None:
            for name, value in mapping.items():
                self.modes[mode, self.names.index(name)] = value

        set_mode(0, {
            "left_hip_roll_joint": 1.0, "right_hip_roll_joint": 1.0,
            "left_ankle_roll_joint": 0.25, "right_ankle_roll_joint": 0.25,
            "waist_roll_joint": -0.50,
        })
        set_mode(1, {
            "right_hip_pitch_joint": -0.50, "right_knee_joint": 1.0,
            "right_ankle_pitch_joint": -0.50,
        })
        set_mode(2, {
            "left_hip_pitch_joint": -0.50, "left_knee_joint": 1.0,
            "left_ankle_pitch_joint": -0.50,
        })
        set_mode(3, {
            "left_hip_pitch_joint": 0.35, "right_hip_pitch_joint": 0.35,
            "left_ankle_pitch_joint": 0.20, "right_ankle_pitch_joint": 0.20,
            "waist_pitch_joint": -1.0,
        })
        if np.any(self.modes[:, self.head] != 0.0):
            raise RuntimeError("event modes touch head2")
        if any(np.max(np.abs(mode)) != 1.0 for mode in self.modes):
            raise RuntimeError("event modes are not normalized")

    def event_correction(self, parameters: np.ndarray, t: float) -> np.ndarray:
        parameters = np.asarray(parameters, dtype=np.float64).reshape(len(MODE_NAMES), 2)
        coefficients = parameters[:, 0] * prep_envelope(t) + parameters[:, 1] * swing_envelope(t)
        correction = coefficients @ self.modes
        correction[self.head] = 0.0
        return correction

    def run(self, parameters: np.ndarray, duration: float, stop_at_fall: bool = False) -> dict:
        r = self.r
        r.reset()
        dt = float(r.m.opt.timestep)
        steps = int(round(duration / dt))
        previous = r.d.geom_xpos.copy()
        initial = {side: np.mean(r.d.geom_xpos[sorted(geoms), :2], axis=0) for side, geoms in r.feet.items()}
        actual = {side: [] for side in r.feet}
        clearance = {side: [] for side in r.feet}
        slip = {side: [] for side in r.feet}
        excursion = {side: [] for side in r.feet}
        root_velocity = [r.d.qvel[:2].copy()]
        root_z = [float(r.d.qpos[2])]
        tilts = [phase8.physics.root_tilt(r.d.qpos[3:7])]
        tracking = []
        torque_saturation = 0
        torque_count = 0
        trace_hash = hashlib.sha256()
        fall = None

        def sample(k: int) -> None:
            contacts = phase8.actual_contacts(r.m, r.d, r.floor, r.side)
            clear = phase8.sole_clearances(r.m, r.d, r.feet, r.floor)
            for side, geoms in r.feet.items():
                actual[side].append(bool(contacts[side]))
                clearance[side].append(float(np.min(clear[side])))
                center = np.mean(r.d.geom_xpos[sorted(geoms), :2], axis=0)
                excursion[side].append(float(np.linalg.norm(center - initial[side])))
                if k > 0:
                    for geom in contacts[side]:
                        slip[side].append(float(np.linalg.norm(r.d.geom_xpos[geom, :2] - previous[geom, :2]) / dt))

        sample(0)
        for k in range(steps):
            t = k * dt
            target = self.correct.nominal(t) + self.event_correction(parameters, t)
            target[self.head] = 0.0
            q = r.d.qpos[r.aq]
            dq = r.d.qvel[r.av]
            raw = r.c.kp * (target - q) - r.c.kd * dq
            torque_saturation += int(np.sum((raw < r.c.torque_low) | (raw > r.c.torque_high)))
            torque_count += len(raw)
            r.d.ctrl[:] = np.clip(raw, r.c.torque_low, r.c.torque_high)
            mujoco.mj_step(r.m, r.d)
            sample(k + 1)
            previous[:] = r.d.geom_xpos
            root_velocity.append(r.d.qvel[:2].copy())
            root_z.append(float(r.d.qpos[2]))
            tilts.append(phase8.physics.root_tilt(r.d.qpos[3:7]))
            tracking.append(float(np.mean((r.d.qpos[r.aq] - self.correct.nominal(t)) ** 2)))
            trace_hash.update(np.asarray(r.d.qpos, dtype="<f8").tobytes())
            trace_hash.update(np.asarray(r.d.qvel, dtype="<f8").tobytes())
            if fall is None and (r.d.qpos[2] < 0.42 or tilts[-1] > 0.90):
                fall = (k + 1) * dt
                if stop_at_fall:
                    break

        count = len(root_velocity) - 1
        times = np.arange(count + 1) * dt
        act = {side: np.asarray(values, dtype=bool) for side, values in actual.items()}
        prep = (times >= 0.10) & (times <= 10 / 30 + 1e-12)
        swing = (times >= 0.40) & (times <= min(duration, 0.70) + 1e-12)
        stable = (~act["left"]) & act["right"] & swing
        acceleration = np.linalg.norm(np.diff(np.asarray(root_velocity), axis=0) / dt, axis=1)
        left_clearance = np.asarray(clearance["left"])[swing]
        right_excursion = np.asarray(excursion["right"])[swing]
        result = {
            "duration_requested_s": duration,
            "duration_simulated_s": count * dt,
            "fall_time_s": fall,
            "survive_requested": fall is None or fall >= duration - 1e-12,
            "trace_sha256": trace_hash.hexdigest(),
            "root_z_min_m": float(min(root_z)),
            "tilt_max_rad": float(max(tilts)),
            "root_horiz_accel_p95_mps2": float(np.percentile(acceleration, 95)),
            "torque_saturation_fraction": torque_saturation / max(torque_count, 1),
            "head_q_max_abs_rad": float(np.max(np.abs(r.d.qpos[r.aq[self.head]]))),
            "penetration_min_mm": {side: float(1000 * min(values)) for side, values in clearance.items()},
            "prep_both_contact_fraction": float(np.mean(act["left"][prep] & act["right"][prep])) if np.any(prep) else None,
            "swing_left_contact_fraction": float(np.mean(act["left"][swing])) if np.any(swing) else None,
            "swing_right_support_fraction": float(np.mean(act["right"][swing])) if np.any(swing) else None,
            "stable_left_off_right_on_longest_ms": longest_true_ms(stable),
            "left_swing_clearance_p50_m": float(np.percentile(left_clearance, 50)) if len(left_clearance) else None,
            "right_stance_excursion_max_m": float(np.max(right_excursion)) if len(right_excursion) else None,
            "flight_fraction": float(np.mean(~(act["left"] | act["right"]))),
            "contact": {},
            "tracking_mse": float(np.mean(tracking)) if tracking else 0.0,
        }
        for side in r.feet:
            result["contact"][side] = {
                "switches": int(np.sum(act[side][1:] != act[side][:-1])),
                "speed_p95_mps": float(np.percentile(slip[side], 95)) if slip[side] else None,
                "excursion_max_m": float(np.max(excursion[side])),
            }
        result["gates"] = {
            "survive": result["survive_requested"],
            "prep_both_contact": result["prep_both_contact_fraction"] is not None and result["prep_both_contact_fraction"] >= 0.95,
            "stable_liftoff_100ms": result["stable_left_off_right_on_longest_ms"] >= 100,
            "left_swing_contact": result["swing_left_contact_fraction"] is not None and result["swing_left_contact_fraction"] <= 0.10,
            "right_support": result["swing_right_support_fraction"] is not None and result["swing_right_support_fraction"] >= 0.95,
            "left_clearance": result["left_swing_clearance_p50_m"] is not None and result["left_swing_clearance_p50_m"] >= 0.012,
            "right_speed": result["contact"]["right"]["speed_p95_mps"] is not None and result["contact"]["right"]["speed_p95_mps"] <= 0.10,
            "right_excursion": result["right_stance_excursion_max_m"] is not None and result["right_stance_excursion_max_m"] <= 0.03,
            "flight": result["flight_fraction"] <= 0.02,
            "root_accel": result["root_horiz_accel_p95_mps2"] <= 4.0,
            "head": result["head_q_max_abs_rad"] <= 0.02,
            "native_penetration": min(result["penetration_min_mm"].values()) >= NATIVE_MIN_MM,
        }
        result["pass"] = all(result["gates"].values())

        # Search cost is event-specific and excludes penetration.
        chatter = result["contact"]["left"]["switches"] + result["contact"]["right"]["switches"]
        stable_shortfall = max(0.0, 100.0 - result["stable_left_off_right_on_longest_ms"]) / 100.0
        prep_fraction = result["prep_both_contact_fraction"] if result["prep_both_contact_fraction"] is not None else 0.0
        swing_left_contact = result["swing_left_contact_fraction"] if result["swing_left_contact_fraction"] is not None else 1.0
        swing_right_support = result["swing_right_support_fraction"] if result["swing_right_support_fraction"] is not None else 0.0
        swing_clearance = result["left_swing_clearance_p50_m"] if result["left_swing_clearance_p50_m"] is not None else 0.0
        right_speed = result["contact"]["right"]["speed_p95_mps"] if result["contact"]["right"]["speed_p95_mps"] is not None else 1.0
        right_excursion_value = result["right_stance_excursion_max_m"] if result["right_stance_excursion_max_m"] is not None else 1.0
        result["loss"] = float(
            250.0 * (0.0 if result["survive_requested"] else 1.0)
            + 30.0 * (1.0 - prep_fraction)
            + 100.0 * stable_shortfall ** 2
            + 60.0 * swing_left_contact
            + 60.0 * (1.0 - swing_right_support)
            + 40.0 * max(0.0, 0.012 - swing_clearance) ** 2 / 0.012 ** 2
            + 2.0 * chatter
            + 20.0 * max(0.0, right_speed - 0.10) ** 2 / 0.10 ** 2
            + 20.0 * max(0.0, right_excursion_value - 0.03) ** 2 / 0.03 ** 2
            + 10.0 * max(0.0, result["root_horiz_accel_p95_mps2"] - 4.0) ** 2 / 4.0 ** 2
            + 2.0 * result["tracking_mse"]
            + 0.1 * float(np.mean((np.asarray(parameters) / BOUNDS) ** 2))
        )
        return result


def preflight() -> dict:
    runner = EventRunner()
    zero = np.zeros((len(MODE_NAMES), 2), dtype=np.float64)
    first = runner.run(zero, 1.0, stop_at_fall=True)
    second = runner.run(zero, 1.0, stop_at_fall=True)
    expected = json.loads((P10 / "phase10_actuator_mapping_audit.json").read_text())["corrected_phase6_nominal_prefall"]
    samples = np.linspace(0.0, 1.0, 1001)
    max_at_bounds = max(float(np.max(np.abs(runner.event_correction(BOUNDS, t)))) for t in samples)
    output = {
        "schema": "x2_phase11_name_mapped_event_preflight_v1",
        "runner_sha256": sha(Path(__file__)),
        "official_actuator_names": runner.names,
        "body29_indices": runner.body.tolist(),
        "head2_indices": runner.head.tolist(),
        "head2_names": [runner.names[i] for i in runner.head],
        "mode_names": MODE_NAMES,
        "mode_head_max_abs": float(np.max(np.abs(runner.modes[:, runner.head]))),
        "event_zero_at_t0": float(np.max(np.abs(runner.event_correction(BOUNDS, 0.0)))) == 0.0,
        "event_max_correction_at_bounds_rad": max_at_bounds,
        "zero_trace_exact": first["trace_sha256"] == second["trace_sha256"],
        "zero_fall_time_s": first["fall_time_s"],
        "phase10_zero_fall_time_s": expected["fall_time_s"],
        "zero_fall_matches_phase10": first["fall_time_s"] == expected["fall_time_s"],
        "head_target_contract": "target[indices resolved by joint name] = 0 every tick",
        "active_spheres_per_foot": {side: len(geoms) for side, geoms in runner.r.feet.items()},
        "resource_contract": {
            "single_process": True, "threads": 1, "gpu": False,
            "candidates": CANDIDATES, "elites": ELITES, "iterations_per_window": ITERATIONS,
            "windows_s": WINDOWS_S, "total_candidate_rollouts": CANDIDATES * ITERATIONS * len(WINDOWS_S),
        },
        "search_not_started": True,
    }
    output["pass"] = bool(
        output["body29_indices"] == runner.body.tolist()
        and output["head2_indices"] == [15, 16]
        and output["mode_head_max_abs"] == 0.0
        and output["event_zero_at_t0"]
        and output["event_max_correction_at_bounds_rad"] <= 0.18
        and output["zero_trace_exact"]
        and output["zero_fall_matches_phase10"]
        and all(value == 12 for value in output["active_spheres_per_foot"].values())
    )
    PREFLIGHT_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    return output


def search() -> None:
    if not PREFLIGHT_PATH.exists():
        raise SystemExit("preflight artifact is missing")
    if not PREREG_PATH.exists():
        raise SystemExit("preregistration artifact is missing")
    if (OUT / "phase11_result.json").exists():
        raise SystemExit("the unique search result already exists; rerun is forbidden")
    gate = json.loads(PREFLIGHT_PATH.read_text())
    prereg = json.loads(PREREG_PATH.read_text())
    current_runner_sha = sha(Path(__file__))
    if not gate.get("pass") or not gate.get("search_not_started"):
        raise SystemExit("preflight gate is not eligible")
    if gate.get("runner_sha256") != current_runner_sha or prereg.get("implementation", {}).get("runner_sha256") != current_runner_sha:
        raise SystemExit("runner hash drifted after preflight/preregistration")
    if prereg.get("search", {}).get("total_candidate_rollouts") != CANDIDATES * ITERATIONS * len(WINDOWS_S):
        raise SystemExit("resource contract drifted")
    runner = EventRunner()
    rng = np.random.default_rng(SEED)
    mean = np.zeros((len(MODE_NAMES), 2), dtype=np.float64)
    std = BOUNDS * 0.60
    history = []
    start = time.perf_counter()
    best_parameters = None
    for duration in WINDOWS_S:
        for iteration in range(ITERATIONS):
            samples = mean + rng.normal(size=(CANDIDATES,) + mean.shape) * std
            samples[0] = mean
            samples = np.clip(samples, -BOUNDS, BOUNDS)
            results = [runner.run(sample, duration) for sample in samples]
            scores = np.asarray([result["loss"] for result in results])
            elite = np.argsort(scores)[:ELITES]
            mean = np.mean(samples[elite], axis=0)
            std = np.maximum(np.std(samples[elite], axis=0), BOUNDS * 0.08)
            best_parameters = samples[elite[0]].copy()
            item = {"window_s": duration, "iteration": iteration + 1, "best_loss": float(scores[elite[0]]), "mean_loss": float(np.mean(scores))}
            history.append(item)
            print(json.dumps(item), flush=True)
    candidate = runner.run(best_parameters, 0.70)
    prefall = runner.run(best_parameters, 5.8, stop_at_fall=True)
    np.savez_compressed(OUT / "phase11_event_candidate.npz", parameters=best_parameters, mode_names=np.asarray(MODE_NAMES), modes=runner.modes)
    output = {
        "schema": "x2_phase11_name_mapped_event_result_v1",
        "configuration": gate["resource_contract"],
        "wall_s": time.perf_counter() - start,
        "history": history,
        "candidate_0p70": candidate,
        "candidate_prefall": prefall,
        "parameters": best_parameters.tolist(),
        "searches": 1,
        "reruns": 0,
        "rl_ppo": 0,
        "stopped": True,
    }
    (OUT / "phase11_result.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"wall_s": output["wall_s"], "candidate_pass": candidate["pass"], "fall": prefall["fall_time_s"], "gates": candidate["gates"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "search"))
    args = parser.parse_args()
    preflight() if args.mode == "preflight" else search()
