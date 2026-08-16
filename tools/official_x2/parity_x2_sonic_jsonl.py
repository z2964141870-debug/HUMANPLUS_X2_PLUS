#!/usr/bin/env python3
"""Compare two canonical X2-Sonic JSONL streams at observation/action level.

This is an offline bridge between the garment replay boundary and the official
SONIC tokenizer. It deliberately uses a deterministic synthetic qpos/qvel
trace, so it verifies field order, quaternion convention, future-frame
sampling, history layout, and ONNX action determinism without pretending to
prove closed-loop dynamics. A later robot-simulation trace can replace the
synthetic state source without changing the stream contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from typing import Iterable

import numpy as np


HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eval_official_sonic_x2 as sonic  # noqa: E402


SCHEMA = "x2_sonic_jsonl_observation_action_parity_v1"


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _vector(value, size: int, field: str, line_no: int) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"line {line_no}: {field} is not numeric") from exc
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"line {line_no}: {field} must be finite shape ({size},), got {array.shape}")
    return array


def load_jsonl(path: pathlib.Path) -> sonic.Motion:
    path = pathlib.Path(path)
    joint_rows = []
    quat_rows = []
    root_rows = []
    timestamps = []
    fps_values = []
    previous_timestamp = None
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            try:
                timestamp = float(record["timestamp_s"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_no}: timestamp_s is required") from exc
            if not np.isfinite(timestamp) or (
                previous_timestamp is not None and timestamp < previous_timestamp - 1e-12
            ):
                raise ValueError(f"{path}:{line_no}: timestamp_s is not finite or monotonic")
            previous_timestamp = timestamp
            joint_rows.append(_vector(record.get("joint_pos"), sonic.N_JOINTS, "joint_pos", line_no))
            quat = _vector(record.get("root_quat"), 4, "root_quat", line_no)
            norm = float(np.linalg.norm(quat))
            if norm <= 1e-12:
                raise ValueError(f"{path}:{line_no}: root_quat has zero norm")
            quat_rows.append(sonic.quat_normalize(quat))
            root_value = record.get("root_pos", [0.0, 0.0, 0.0])
            root_rows.append(_vector(root_value, 3, "root_pos", line_no))
            fps = float(record.get("fps", 50.0))
            if not np.isfinite(fps) or fps <= 0.0:
                raise ValueError(f"{path}:{line_no}: fps must be positive and finite")
            fps_values.append(fps)
            timestamps.append(timestamp)
    if not joint_rows:
        raise ValueError(f"{path}: no non-empty JSONL records")
    if max(fps_values) - min(fps_values) > 1e-9:
        raise ValueError(f"{path}: per-frame fps is not constant")
    return sonic.Motion(
        path.stem,
        float(fps_values[0]),
        np.asarray(joint_rows, dtype=np.float64),
        np.asarray(root_rows, dtype=np.float64),
        np.asarray(quat_rows, dtype=np.float64),
    )


def states(ticks: int) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    for tick in range(ticks):
        qpos = np.zeros(74, dtype=np.float32)
        qvel = np.zeros(72, dtype=np.float32)
        qpos[3] = 1.0
        qpos[7:38] = sonic.DEFAULT_ANGLES_MJ
        phase = 0.07 * tick
        qpos[7:38] += 0.01 * np.sin(phase + np.arange(31, dtype=np.float32))
        qvel[6:37] = 0.03 * np.cos(phase + np.arange(31, dtype=np.float32))
        qvel[3:6] = np.asarray([0.01, -0.02, 0.03], dtype=np.float32)
        yield qpos, qvel


def policy_trace(
    policy: sonic.SonicPolicy,
    motion: sonic.Motion,
    ticks: int,
) -> tuple[np.ndarray, np.ndarray]:
    policy.reset()
    observations = []
    actions = []
    for tick, (qpos, qvel) in enumerate(states(ticks)):
        obs, action = policy.infer(motion, tick / motion.fps, qpos, qvel)
        observations.append(obs.copy())
        actions.append(action.copy())
    return np.asarray(observations), np.asarray(actions)


def array_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, object]:
    delta = a.astype(np.float64) - b.astype(np.float64)
    return {
        "shape_equal": bool(a.shape == b.shape),
        "shape_a": list(a.shape),
        "shape_b": list(b.shape),
        "max_abs": float(np.max(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta**2))),
        "allclose_1e-6": bool(np.allclose(a, b, atol=1e-6, rtol=1e-6)),
        "allclose_1e-5": bool(np.allclose(a, b, atol=1e-5, rtol=1e-5)),
        "sha_a": hashlib.sha256(a.tobytes()).hexdigest(),
        "sha_b": hashlib.sha256(b.tobytes()).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=pathlib.Path, required=True)
    parser.add_argument("--candidate", type=pathlib.Path, required=True)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--ticks", type=int, default=50)
    parser.add_argument("--require-cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.ticks <= 0:
        parser.error("--ticks must be positive")

    reference = load_jsonl(args.reference)
    candidate = load_jsonl(args.candidate)
    if reference.frames != candidate.frames:
        raise ValueError(f"frame count mismatch: {reference.frames} vs {candidate.frames}")
    if reference.fps != candidate.fps:
        raise ValueError(f"fps mismatch: {reference.fps} vs {candidate.fps}")
    if args.ticks > reference.frames:
        raise ValueError(f"--ticks {args.ticks} exceeds stream frames {reference.frames}")

    policy = sonic.SonicPolicy(args.model, require_cuda=args.require_cuda)
    reference_obs, reference_action = policy_trace(policy, reference, args.ticks)
    candidate_obs, candidate_action = policy_trace(policy, candidate, args.ticks)
    result = {
        "schema": SCHEMA,
        "reference": {
            "path": str(args.reference.resolve()),
            "sha256": sha256(args.reference),
        },
        "candidate": {
            "path": str(args.candidate.resolve()),
            "sha256": sha256(args.candidate),
        },
        "model": {
            "path": str(args.model.resolve()),
            "sha256": sha256(args.model),
        },
        "providers": policy.session.get_providers(),
        "ticks": args.ticks,
        "state_source": "deterministic synthetic qpos/qvel trace; replaceable by a simulator trace",
        "observation": array_metrics(reference_obs, candidate_obs),
        "action": array_metrics(reference_action, candidate_action),
        "pass": bool(
            np.allclose(reference_obs, candidate_obs, atol=1e-6, rtol=1e-6)
            and np.allclose(reference_action, candidate_action, atol=1e-6, rtol=1e-6)
        ),
        "interpretation": (
            "Offline tokenizer/history/ONNX parity only; this does not establish "
            "closed-loop dynamic stability or hardware readiness."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
