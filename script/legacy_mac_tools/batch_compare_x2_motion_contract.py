#!/usr/bin/env python3
"""Run the source/canonical parity test for every record in a manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np


HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compare_x2_motion_contract as parity  # noqa: E402
import eval_official_sonic_x2 as sonic  # noqa: E402


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--ticks", type=int, default=50)
    parser.add_argument("--require-cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = manifest.get("records", [])
    if not records:
        raise ValueError(f"manifest has no records: {args.manifest}")
    policy = sonic.SonicPolicy(args.model, require_cuda=args.require_cuda)
    results = []
    for index, record in enumerate(records, 1):
        source = pathlib.Path(record["source"])
        canonical = pathlib.Path(record["output"])
        print(f"[{index}/{len(records)}] {source.name}", flush=True)
        source_motion = parity.prepared(source)
        canonical_motion = parity.prepared(canonical)
        motion_result = parity.motion_metrics(source_motion, canonical_motion)
        source_obs, source_action = parity.policy_trace(policy, source_motion, args.ticks)
        canonical_obs, canonical_action = parity.policy_trace(policy, canonical_motion, args.ticks)
        result = {
            "source": str(source.resolve()),
            "source_sha256": sha256(source),
            "canonical": str(canonical.resolve()),
            "canonical_sha256": sha256(canonical),
            "motion": motion_result,
            "observation": parity.array_metrics(source_obs, canonical_obs),
            "action": parity.array_metrics(source_action, canonical_action),
        }
        result["pass"] = bool(
            motion_result.get("allclose_1e-10", False)
            and np.allclose(source_obs, canonical_obs, atol=1e-6, rtol=1e-6)
            and np.allclose(source_action, canonical_action, atol=1e-6, rtol=1e-6)
        )
        results.append(result)
        print(json.dumps({"pass": result["pass"], "obs_max": result["observation"]["max_abs"],
                          "action_max": result["action"]["max_abs"]}), flush=True)

    aggregate = {
        "schema": "x2_sonic_motion_contract_parity_batch/v1",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "model": str(args.model.resolve()),
        "model_sha256": sha256(args.model),
        "providers": policy.session.get_providers(),
        "ticks": args.ticks,
        "records": results,
        "summary": {
            "records": len(results),
            "passed": sum(bool(row["pass"]) for row in results),
            "failed": sum(not bool(row["pass"]) for row in results),
            "max_observation_abs": max(row["observation"]["max_abs"] for row in results),
            "max_action_abs": max(row["action"]["max_abs"] for row in results),
        },
        "interpretation": (
            "Offline deterministic contract test only; it does not establish closed-loop "
            "dynamic stability or hardware readiness."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(aggregate["summary"], indent=2))
    return 0 if aggregate["summary"]["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
