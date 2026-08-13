#!/usr/bin/env python3
"""Validate and reduce one completed Phase72 antithetic launch pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch

from cwi_x2.phase72_antithetic import (
    build_raw_signals,
    paired_direction,
    tensor_hash,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_sidecar(path: Path) -> bool:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    return sidecar.is_file() and sidecar.read_text() == f"{sha256(path)}  {path.name}\n"


def atomic_json(path: Path, payload: dict[str, object]) -> str:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    temp = path.with_name(f".{path.name}.tmp")
    if path.exists() or sidecar.exists() or temp.exists():
        raise FileExistsError(f"refusing to overwrite Phase72 pair result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temp, path)
    digest = sha256(path)
    sidecar.write_text(f"{digest}  {path.name}\n")
    return digest


def atomic_torch(path: Path, payload: dict[str, object]) -> str:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    temp = path.with_name(f".{path.name}.tmp")
    if path.exists() or sidecar.exists() or temp.exists():
        raise FileExistsError(f"refusing to overwrite Phase72 pair evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, temp)
    restored = torch.load(temp, map_location="cpu", weights_only=False)
    if restored.get("schema") != payload["schema"]:
        raise RuntimeError("Phase72 pair evidence reload schema changed")
    for name in payload["directions"]:
        if not torch.equal(restored["directions"][name], payload["directions"][name]):
            raise RuntimeError(f"Phase72 pair direction reload failed: {name}")
    os.replace(temp, path)
    digest = sha256(path)
    sidecar.write_text(f"{digest}  {path.name}\n")
    return digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--plus-screen", type=Path, required=True)
    parser.add_argument("--minus-screen", type=Path, required=True)
    parser.add_argument("--plus-resource", type=Path, required=True)
    parser.add_argument("--minus-resource", type=Path, required=True)
    parser.add_argument("--pair-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prereg = json.loads(args.prereg.read_text())
    screens = {
        "plus": json.loads(args.plus_screen.read_text()),
        "minus": json.loads(args.minus_screen.read_text()),
    }
    resources = {
        "plus": json.loads(args.plus_resource.read_text()),
        "minus": json.loads(args.minus_resource.read_text()),
    }
    bundles = {}
    for sign in ("plus", "minus"):
        screen = screens[sign]
        path = Path(screen.get("bundle", "missing"))
        if not path.is_file() or not exact_sidecar(path):
            raise RuntimeError(f"Phase72 {sign} bundle or sidecar is absent")
        if sha256(path) != screen.get("bundle_sha256"):
            raise RuntimeError(f"Phase72 {sign} bundle hash differs from screen")
        bundles[sign] = torch.load(path, map_location="cpu", weights_only=False)

    plus = bundles["plus"]
    minus = bundles["minus"]
    plus_latent = torch.as_tensor(plus["latent_action"])
    minus_latent = torch.as_tensor(minus["latent_action"])
    plus_requested = torch.as_tensor(plus["requested_knee_offset_rad"])
    minus_requested = torch.as_tensor(minus["requested_knee_offset_rad"])
    plus_effective = torch.as_tensor(plus["effective_knee_offset_rad"])
    minus_effective = torch.as_tensor(minus["effective_knee_offset_rad"])
    plus_rms = float(plus_effective.square().mean().sqrt())
    minus_rms = float(minus_effective.square().mean().sqrt())
    initial_hash = screens["plus"].get("initial_combined_sha256")
    seed_index = int(screens["plus"].get("seed_index", -1))
    seed_record = (
        prereg["seed_pairs"][seed_index]
        if 0 <= seed_index < len(prereg.get("seed_pairs", []))
        else {}
    )
    technical = {
        "prereg_sidecar": exact_sidecar(args.prereg),
        "screen_sidecars": exact_sidecar(args.plus_screen) and exact_sidecar(args.minus_screen),
        "screen_schema": all(
            screen.get("schema") == "x2_phase72_antithetic_screen_v1"
            for screen in screens.values()
        ),
        "screens_valid_pending": all(
            screen.get("decision") == "VALID_PENDING_PAIR" for screen in screens.values()
        ),
        "screen_technical_all": all(
            all(screen.get("technical_checks", {}).values()) for screen in screens.values()
        ),
        "sign_roles_exact": screens["plus"].get("sign") == "plus" and screens["minus"].get("sign") == "minus",
        "seed_index_exact": screens["plus"].get("seed_index") == screens["minus"].get("seed_index"),
        "environment_seed_exact": screens["plus"].get("env_seed") == screens["minus"].get("env_seed"),
        "latent_seed_exact": screens["plus"].get("latent_seed") == screens["minus"].get("latent_seed"),
        "seed_matches_prereg": (
            seed_record.get("seed_index") == seed_index
            and screens["plus"].get("env_seed") == seed_record.get("env_seed")
            and screens["plus"].get("latent_seed") == seed_record.get("latent_seed")
            and plus.get("schedule_sha256") == seed_record.get("schedule_sha256")
        ),
        "schedule_exact": plus.get("schedule_sha256") == minus.get("schedule_sha256"),
        "initial_state_exact": bool(initial_hash) and initial_hash == screens["minus"].get("initial_combined_sha256"),
        "latent_exact_antithetic": float((plus_latent + minus_latent).abs().max()) <= 1.0e-7,
        "requested_exact_antithetic": float((plus_requested + minus_requested).abs().max()) <= 1.0e-7,
        "effective_exact_antithetic": float((plus_effective + minus_effective).abs().max()) <= 1.0e-7,
        "effective_rms_balanced": min(plus_rms, minus_rms) > 0.0 and max(plus_rms, minus_rms) / min(plus_rms, minus_rms) <= 1.05,
        "resource_exit_zero": all(resource.get("exit_code") == 0 for resource in resources.values()),
        "resource_sidecars": exact_sidecar(args.plus_resource)
        and exact_sidecar(args.minus_resource),
        "resource_labels_bound": all(
            resources[sign].get("label")
            == f"phase72_seed{seed_index}_{sign}_antithetic_zero_optimizer"
            for sign in ("plus", "minus")
        ),
        "resource_disk_hard_cap": all(
            int(resource.get("disk_used_delta_bytes", 2**63))
            <= int(prereg["resource_limits"]["per_launch_disk_delta_bytes_hard_max"])
            for resource in resources.values()
        ),
        "resource_gpu_cap": all(
            float((resource.get("gpu") or {}).get("memory_used_peak_mib", math.inf))
            <= float(prereg["resource_limits"]["gpu_peak_memory_mib_max"])
            for resource in resources.values()
        ),
        "bundle_size_cap": all(
            int(screen.get("bundle_bytes", 2**63))
            <= int(prereg["resource_limits"]["raw_bundle_bytes_max"])
            for screen in screens.values()
        ),
        "all_finite": all(
            torch.isfinite(torch.as_tensor(bundle[name])).all()
            for bundle in bundles.values()
            for name in (
                "encoded_feature", "latent_action", "value", "total_reward",
                "reward_by_term", "signed_pitch_rad", "support_outside_m",
            )
        ),
    }
    valid = all(technical.values())
    plus_raw = build_raw_signals(plus)
    minus_raw = build_raw_signals(minus)
    directions = {}
    scales = {}
    for name in plus_raw:
        direction, scale = paired_direction(
            plus["encoded_feature"],
            plus_latent[:200],
            plus_raw[name],
            minus["encoded_feature"],
            minus_latent[:200],
            minus_raw[name],
        )
        directions[name] = direction
        scales[name] = scale
    evidence = {
        "schema": "x2_phase72_antithetic_pair_evidence_v1",
        "seed_index": int(screens["plus"]["seed_index"]),
        "env_seed": int(screens["plus"]["env_seed"]),
        "latent_seed": int(screens["plus"]["latent_seed"]),
        "initial_combined_sha256": initial_hash,
        "plus_bundle_sha256": screens["plus"]["bundle_sha256"],
        "minus_bundle_sha256": screens["minus"]["bundle_sha256"],
        "directions": directions,
        "paired_scales": scales,
    }
    evidence_sha = atomic_torch(args.pair_evidence, evidence)
    report = {
        "schema": "x2_phase72_antithetic_pair_result_v1",
        "decision": "PAIR_VALID_PENDING_FINAL" if valid else "FAIL_INVALID_STOP",
        "seed_index": evidence["seed_index"],
        "env_seed": evidence["env_seed"],
        "latent_seed": evidence["latent_seed"],
        "initial_combined_sha256": initial_hash,
        "plus_bundle_sha256": evidence["plus_bundle_sha256"],
        "minus_bundle_sha256": evidence["minus_bundle_sha256"],
        "pair_evidence": str(args.pair_evidence),
        "pair_evidence_sha256": evidence_sha,
        "direction_hashes": {name: tensor_hash(value) for name, value in directions.items()},
        "paired_scales": scales,
        "antithetic": {
            "latent_sum_max_abs": float((plus_latent + minus_latent).abs().max()),
            "requested_sum_max_abs_rad": float((plus_requested + minus_requested).abs().max()),
            "effective_sum_max_abs_rad": float((plus_effective + minus_effective).abs().max()),
            "plus_effective_rms_rad": plus_rms,
            "minus_effective_rms_rad": minus_rms,
            "rms_ratio": max(plus_rms, minus_rms) / max(min(plus_rms, minus_rms), 1.0e-30),
        },
        "technical_checks": technical,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "checkpoint_count": 0,
        "promotion_unlocked": False,
    }
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2))
    if not valid:
        raise SystemExit("Phase72 pair validity failed")


if __name__ == "__main__":
    main()
