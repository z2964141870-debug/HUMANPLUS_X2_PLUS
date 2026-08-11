#!/usr/bin/env python3
"""Zero-search audit of the Phase6/8 actuator partition mistake.

Historical Phase6/8 code assumed the last two actuators were the head.  The
official order places head_yaw/head_pitch at indices 15/16.  This script does
not rerun optimization.  It deterministically replays (a) the corrected
Phase6 nominal bridge and (b) the saved Phase8 correction projected onto the
true body29 set, with the real head targets fixed to zero.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/humanplus/projects/ZHY/dsms_workspace")
P8 = ROOT / "phase8_reset_sbto"
OUT = ROOT / "phase10_actuator_mapping_audit"
OUT.mkdir(parents=True, exist_ok=True)
sys.path[:0] = [str(P8)]

import mujoco
import x2_lunge_phase8_reset_sbto as phase8


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class CorrectEvaluator:
    def __init__(self) -> None:
        self.r = phase8.Runner()
        self.names = list(self.r.c.actuator_joint_names)
        self.head = np.asarray(
            [i for i, name in enumerate(self.names) if name in ("head_yaw_joint", "head_pitch_joint")],
            dtype=np.int64,
        )
        self.body = np.asarray([i for i in range(len(self.names)) if i not in set(self.head.tolist())], dtype=np.int64)
        self.historical_fixed = np.asarray([len(self.names) - 2, len(self.names) - 1], dtype=np.int64)
        if self.head.tolist() != [15, 16] or len(self.body) != 29:
            raise RuntimeError(f"unexpected official actuator partition: head={self.head}, body={len(self.body)}")

    def nominal(self, t: float) -> np.ndarray:
        phase = min(t / 0.02, len(self.r.ref) - 1.0)
        i = int(np.floor(phase))
        j = min(i + 1, len(self.r.ref) - 1)
        alpha = phase - i
        q = (1.0 - alpha) * self.r.ref[i] + alpha * self.r.ref[j]
        u = np.clip(t / 0.34, 0.0, 1.0)
        smooth = u * u * (3.0 - 2.0 * u)
        target = q + (1.0 - smooth) * self.r.delta
        target[self.head] = 0.0
        return target

    def correction(self, historical_cp: np.ndarray, t: float) -> np.ndarray:
        """Project saved old-index correction onto true body29.

        The saved artifact has columns for historical actuator indices 0..28.
        Corrections at real head indices are discarded.  The two body joints
        omitted by the historical representation receive zero correction.
        """
        frame = t / 0.02
        old = np.zeros(29, dtype=np.float64)
        if frame <= phase8.CP_FRAMES[-1]:
            for column in range(29):
                old[column] = np.interp(frame, phase8.CP_FRAMES, historical_cp[:, column])
        elif t <= 1.0:
            old = historical_cp[-1] * max(0.0, 1.0 - (t - phase8.HORIZON) / 0.30)
        full = np.zeros(len(self.names), dtype=np.float64)
        full[:29] = old
        full[self.head] = 0.0
        return full

    def run(self, historical_cp: np.ndarray, duration: float, stop_at_fall: bool) -> dict:
        r = self.r
        r.reset()
        dt = float(r.m.opt.timestep)
        steps = int(round(duration / dt))
        previous = r.d.geom_xpos.copy()
        initial = {side: np.mean(r.d.geom_xpos[sorted(geoms), :2], axis=0) for side, geoms in r.feet.items()}
        actual = {side: [] for side in r.feet}
        slip = {side: [] for side in r.feet}
        excursion = {side: [] for side in r.feet}
        clearance = {side: [] for side in r.feet}
        root_velocity = [r.d.qvel[:2].copy()]
        root_z = [float(r.d.qpos[2])]
        tilt = [phase8.physics.root_tilt(r.d.qpos[3:7])]
        head_q = []
        right_wrist_q = []
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
            target = self.nominal(t) + self.correction(historical_cp, t)
            target[self.head] = 0.0
            q = r.d.qpos[r.aq]
            dq = r.d.qvel[r.av]
            raw = r.c.kp * (target - q) - r.c.kd * dq
            r.d.ctrl[:] = np.clip(raw, r.c.torque_low, r.c.torque_high)
            mujoco.mj_step(r.m, r.d)
            sample(k + 1)
            previous[:] = r.d.geom_xpos
            root_velocity.append(r.d.qvel[:2].copy())
            root_z.append(float(r.d.qpos[2]))
            tilt.append(phase8.physics.root_tilt(r.d.qpos[3:7]))
            head_q.append(float(np.max(np.abs(r.d.qpos[r.aq[self.head]]))))
            right_wrist_q.append(float(np.max(np.abs(r.d.qpos[r.aq[self.historical_fixed]]))))
            if fall is None and (r.d.qpos[2] < 0.42 or tilt[-1] > 0.90):
                fall = (k + 1) * dt
                if stop_at_fall:
                    break

        nsteps = len(root_velocity) - 1
        times = np.arange(nsteps + 1) * dt
        intent = {"left": times <= 10 / 30 + 1e-12, "right": times <= 25 / 30 + 1e-12}
        act = {side: np.asarray(values, dtype=bool) for side, values in actual.items()}
        nonleft = ~intent["left"]
        stable_off = (~act["left"]) & act["right"] & nonleft
        acceleration = np.linalg.norm(np.diff(np.asarray(root_velocity), axis=0) / dt, axis=1)
        result = {
            "duration_requested_s": duration,
            "duration_simulated_s": nsteps * dt,
            "fall_time_s": fall,
            "survive_requested": fall is None or fall >= duration - 1e-12,
            "root_z_min_m": float(min(root_z)),
            "tilt_max_rad": float(max(tilt)),
            "root_horiz_accel_p95_mps2": float(np.percentile(acceleration, 95)),
            "head_q_max_abs_rad": float(max(head_q, default=0.0)),
            "historical_last2_right_wrist_q_max_abs_rad": float(max(right_wrist_q, default=0.0)),
            "penetration_min_mm": {side: float(1000 * min(values)) for side, values in clearance.items()},
            "left_nonintent_contact_fraction": float(np.mean(act["left"][nonleft])) if np.any(nonleft) else None,
            "right_support_during_left_nonintent_fraction": float(np.mean(act["right"][nonleft])) if np.any(nonleft) else None,
            "left_off_right_on_longest_ms": phase8.longest_true_ms(stable_off),
            "flight_fraction": float(np.mean(~(act["left"] | act["right"]))),
            "contact": {},
        }
        for side in r.feet:
            result["contact"][side] = {
                "switches": int(np.sum(act[side][1:] != act[side][:-1])),
                "stance_speed_p95_mps": float(np.percentile(slip[side], 95)) if slip[side] else None,
                "stance_excursion_max_m": float(np.max(np.asarray(excursion[side])[intent[side]])),
            }
        return result


def main() -> None:
    artifact = P8 / "phase8_reset_sbto_candidate.npz"
    saved = np.load(artifact, allow_pickle=False)
    candidate = np.asarray(saved["correction"], dtype=np.float64)
    evaluator = CorrectEvaluator()
    zero = np.zeros_like(candidate)
    zero_a = evaluator.run(zero, 1.0, True)
    zero_b = evaluator.run(zero, 1.0, True)
    candidate_short = evaluator.run(candidate, 0.70, False)
    candidate_prefall = evaluator.run(candidate, 5.8, True)
    output = {
        "schema": "x2_retarget_actuator_mapping_phase10_v1",
        "new_searches": 0,
        "physics_replays": 4,
        "historical_bug": {
            "assumed_fixed_indices": evaluator.historical_fixed.tolist(),
            "assumed_fixed_names": [evaluator.names[i] for i in evaluator.historical_fixed],
            "actual_head_indices": evaluator.head.tolist(),
            "actual_head_names": [evaluator.names[i] for i in evaluator.head],
            "affected_phases": ["phase4", "phase5", "phase6", "phase8"],
        },
        "source_hashes": {
            "phase8_candidate": sha(artifact),
            "phase8_runner": sha(P8 / "x2_lunge_phase8_reset_sbto.py"),
            "phase6_reset": sha(ROOT / "phase6_reset_bridge/phase6_projected_initial_state.npz"),
        },
        "corrected_phase6_nominal_prefall": zero_a,
        "corrected_zero_deterministic_exact": zero_a == zero_b,
        "corrected_phase8_projected_candidate_0p70": candidate_short,
        "corrected_phase8_projected_candidate_prefall": candidate_prefall,
        "interpretation_boundary": "The Phase8 artifact is projected by dropping saved corrections at actual head indices; omitted body corrections remain zero. This is a zero-search contract audit, not a re-optimized candidate.",
    }
    output_path = OUT / "phase10_actuator_mapping_audit.json"
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
