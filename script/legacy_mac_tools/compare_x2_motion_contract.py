#!/usr/bin/env python3
"""Check source/canonical motion and SONIC observation/action parity.

The source and canonical files are loaded through the same evaluator contract,
including heading normalization and the runtime stand blend.  A deterministic
sequence of MuJoCo-shaped states is then fed to the policy so that any change
in tokenizer, future-frame sampling, quaternion handling, or history layout is
visible before a closed-loop rollout is launched.
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


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prepared(path: pathlib.Path) -> sonic.Motion:
    # This is exactly the preprocessing boundary used by evaluate_clip().
    return sonic.prepend_stand_blend(sonic.normalize_heading(sonic._load_joblib_motion(path)))


def motion_metrics(a: sonic.Motion, b: sonic.Motion) -> dict[str, object]:
    if a.joint_pos.shape != b.joint_pos.shape:
        return {
            "shape_equal": False,
            "a_joint_shape": list(a.joint_pos.shape),
            "b_joint_shape": list(b.joint_pos.shape),
        }
    joint_delta = a.joint_pos - b.joint_pos
    root_delta = a.root_pos - b.root_pos
    quat_delta = a.root_quat - b.root_quat
    return {
        "shape_equal": True,
        "fps_equal": bool(a.fps == b.fps),
        "frames_equal": bool(a.frames == b.frames),
        "joint_max_abs": float(np.max(np.abs(joint_delta))),
        "joint_rmse": float(np.sqrt(np.mean(joint_delta**2))),
        "root_max_abs": float(np.max(np.abs(root_delta))),
        "root_rmse": float(np.sqrt(np.mean(root_delta**2))),
        "quat_max_abs": float(np.max(np.abs(quat_delta))),
        "quat_rmse": float(np.sqrt(np.mean(quat_delta**2))),
        "allclose_1e-10": bool(
            np.allclose(a.joint_pos, b.joint_pos, atol=1e-10, rtol=1e-10)
            and np.allclose(a.root_pos, b.root_pos, atol=1e-10, rtol=1e-10)
            and np.allclose(a.root_quat, b.root_quat, atol=1e-10, rtol=1e-10)
        ),
    }


def states(ticks: int) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    for tick in range(ticks):
        qpos = np.zeros(74, dtype=np.float32)
        qvel = np.zeros(72, dtype=np.float32)
        qpos[3] = 1.0
        qpos[7:38] = sonic.DEFAULT_ANGLES_MJ
        phase = 0.07 * tick
        # Small, deterministic perturbations exercise the proprioception path
        # without putting a real robot or a simulator in the loop.
        qpos[7:38] += 0.01 * np.sin(phase + np.arange(31, dtype=np.float32))
        qvel[6:37] = 0.03 * np.cos(phase + np.arange(31, dtype=np.float32))
        qvel[3:6] = np.asarray([0.01, -0.02, 0.03], dtype=np.float32)
        yield qpos, qvel


def policy_trace(policy: sonic.SonicPolicy, motion: sonic.Motion, ticks: int) -> tuple[np.ndarray, np.ndarray]:
    policy.reset()
    observations, actions = [], []
    for tick, (qpos, qvel) in enumerate(states(ticks)):
        obs, action = policy.infer(motion, tick / 50.0, qpos, qvel)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--canonical", type=pathlib.Path, required=True)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--ticks", type=int, default=50)
    parser.add_argument("--require-cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.ticks <= 0:
        parser.error("--ticks must be positive")

    source_motion = prepared(args.source)
    canonical_motion = prepared(args.canonical)
    motion_result = motion_metrics(source_motion, canonical_motion)
    policy = sonic.SonicPolicy(args.model, require_cuda=args.require_cuda)
    source_obs, source_action = policy_trace(policy, source_motion, args.ticks)
    canonical_obs, canonical_action = policy_trace(policy, canonical_motion, args.ticks)
    result = {
        "schema": "x2_sonic_motion_contract_parity/v1",
        "source": str(args.source.resolve()),
        "source_sha256": sha256(args.source),
        "canonical": str(args.canonical.resolve()),
        "canonical_sha256": sha256(args.canonical),
        "model": str(args.model.resolve()),
        "model_sha256": sha256(args.model),
        "providers": policy.session.get_providers(),
        "ticks": args.ticks,
        "preprocessed_motion": motion_result,
        "observation": array_metrics(source_obs, canonical_obs),
        "action": array_metrics(source_action, canonical_action),
        "pass": bool(
            motion_result.get("allclose_1e-10", False)
            and np.allclose(source_obs, canonical_obs, atol=1e-6, rtol=1e-6)
            and np.allclose(source_action, canonical_action, atol=1e-6, rtol=1e-6)
        ),
        "interpretation": (
            "This is an offline contract test with deterministic synthetic proprioception; "
            "it is not evidence of closed-loop dynamic stability."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
