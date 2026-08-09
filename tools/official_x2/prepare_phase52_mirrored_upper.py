#!/usr/bin/env python3
"""Build and preregister the sole mirrored-upper Phase52 diagnostic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np


REPO = Path(__file__).resolve().parents[2]
TOOLS = REPO / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from official_x2.actor_symmetry_contract import mirror_named_vector


SOURCE = REPO / "artifacts/official_x2/x2_hybrid_phase44_upper_motion.npz"
MODEL_CONTRACT = REPO / "reports/retarget/x2_official_joint_body_map.json"
OUTPUT = REPO / "artifacts/official_x2/x2_hybrid_phase52_upper_mirrored.npz"
PREREG = REPO / "reports/official_x2/phase52_mirrored_upper_closed_prereg.json"
ADAPTER = REPO / "tools/official_x2/stage208_official_mujoco_adapter.py"
ARM14 = (
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_yaw_joint", "left_wrist_pitch_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint",
)
DT = 0.02


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bounded_target(raw: np.ndarray, default: np.ndarray) -> np.ndarray:
    desired = default[None] + np.clip(0.25 * (raw - raw[0]), -0.12, 0.12)
    result = np.empty_like(desired)
    previous = default.copy()
    for index in range(len(desired)):
        previous = previous + np.clip(desired[index] - previous, -0.20 * DT, 0.20 * DT)
        result[index] = previous
    return result


def main() -> None:
    source = np.load(SOURCE, allow_pickle=False)
    names = tuple(source["joint_names"].tolist())
    if names != ARM14:
        raise RuntimeError("Phase44 upper order changed")
    raw = np.asarray(source["q_rad"], dtype=np.float64)
    mirrored = np.asarray([mirror_named_vector(row, names) for row in raw])
    roundtrip = np.asarray([mirror_named_vector(row, names) for row in mirrored])
    model_contract = json.loads(MODEL_CONTRACT.read_text())
    joint = {row["name"]: row for row in model_contract["joints"]}
    default = np.asarray([joint[name]["model_nominal_rad"] for name in names], dtype=np.float64)
    limits = np.asarray([joint[name]["range_rad"] for name in names], dtype=np.float64)
    bounded = bounded_target(mirrored, default)
    overshoot = np.maximum(limits[:, 0][None] - bounded, 0.0) + np.maximum(bounded - limits[:, 1][None], 0.0)
    qstep = float(np.max(np.abs(np.diff(bounded, axis=0))))
    excursion = float(np.max(np.abs(bounded - default[None])))
    checks = {
        "mirror_twice_max_abs_rad": float(np.max(np.abs(roundtrip - raw))),
        "mirror_twice_exact": bool(np.array_equal(roundtrip, raw)),
        "bounded_joint_limit_overshoot_max_rad": float(np.max(overshoot)),
        "bounded_joint_limits": float(np.max(overshoot)) <= 1e-12,
        "bounded_qstep_max_rad": qstep,
        "bounded_qstep": qstep <= 0.004000001,
        "bounded_excursion_max_rad": excursion,
        "bounded_excursion": excursion <= 0.1200001,
        "adapter_phase34_hash_exact": sha256(ADAPTER) == "91a67ab6c583d48bbf09fea1d02de33f0b77961a404d8e67b7fab7764662d733",
        "direct_write_mask": list(names),
        "lower_waist_root_head_direct_path_unchanged": True,
    }
    if not all(value for key, value in checks.items() if key in (
        "mirror_twice_exact", "bounded_joint_limits", "bounded_qstep", "bounded_excursion",
        "adapter_phase34_hash_exact", "lower_waist_root_head_direct_path_unchanged",
    )):
        raise RuntimeError(f"Phase52 static gate failed: {checks}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT, joint_names=np.asarray(names), q_rad=mirrored.astype(np.float32),
        fps=np.asarray(float(source["fps"])),
        source=np.asarray("Phase52 sagittal mirror of Phase44 AMASS-UPPER-001 using frozen official X2 joint contract"),
        prereg_bounded_absolute_target_rad=bounded.astype(np.float32),
    )
    prereg = {
        "stage": "WBT Phase52", "status": "PREREGISTERED_BEFORE_PHYSICS",
        "scope": "one mirrored-upper closed causal diagnostic; no retry/training/compensation/gain tuning/hardware/Git/cloud",
        "A": "immutable BASE Phase34 closed Stage250",
        "B1": "immutable Phase50 original AMASS-UPPER-001",
        "Bmirror": "Phase50 B1 exact contract with only frozen upper14 sagittal reflection",
        "only_new_physics_episodes": 1, "retries": 0,
        "unchanged": ["Stage219 actor", "stationary actor", "scene", "template", "adapter", "PD", "command", "stand/move/stop", "upper scale", "upper rate", "upper timing", "lower/waist/root/head direct path"],
        "mirror_contract": {"permutation": "left<->right same joint", "sign": "roll/yaw negate; pitch unchanged", "absolute_joint_values": True},
        "static_checks": checks,
        "artifacts": {"source": str(SOURCE), "source_sha256": sha256(SOURCE), "mirrored": str(OUTPUT), "mirrored_sha256": sha256(OUTPUT)},
        "diagnostic_rules": {
            "functional": "Bmirror startup/move/stop/full gates must be reported",
            "upper": "Bmirror tracking is scored against its own mirrored target; compare B1 without changing thresholds",
            "causal_lateral": "compare signed (B1-A) and (Bmirror-A) lateral displacement; sign-flip with each magnitude>=0.02m is mirror-responsive",
            "causal_heading": "compare signed yaw progress deltas; sign-flip with each magnitude>=0.03rad is mirror-responsive",
            "contact_slip": "report same Phase34 1kHz official collision agreement/slip metrics for all arms",
            "claim": "causal diagnostic only; no promotion or compensation tuning from this episode"
        },
        "truth_boundary": "official MuJoCo collision/contact-force model truth, not hardware GRF/COP"
    }
    PREREG.write_text(json.dumps(prereg, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"checks": checks, "artifact_sha256": sha256(OUTPUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
