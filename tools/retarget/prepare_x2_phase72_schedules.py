#!/usr/bin/env python3
"""Create the five immutable CPU epsilon schedules used by Phase72."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from cwi_x2.phase72_antithetic import generate_epsilon, tensor_hash


LATENT_SEEDS = (721041, 721042, 721043, 721044, 721045)
ENV_SEEDS = (720041, 720042, 720043, 720044, 720045)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    args = parser.parse_args()
    inventory_sidecar = args.inventory.with_suffix(args.inventory.suffix + ".sha256")
    if args.inventory.exists() or inventory_sidecar.exists():
        raise FileExistsError("Phase72 schedule inventory already exists")
    args.directory.mkdir(parents=True, exist_ok=True)
    records = []
    for index, (env_seed, latent_seed) in enumerate(zip(ENV_SEEDS, LATENT_SEEDS, strict=True)):
        path = args.directory / f"epsilon_seed{index}.pt"
        sidecar = path.with_suffix(path.suffix + ".sha256")
        temp = path.with_name(f".{path.name}.tmp")
        if path.exists() or sidecar.exists() or temp.exists():
            raise FileExistsError(f"Phase72 schedule already exists: {path}")
        epsilon = generate_epsilon(latent_seed)
        payload = {
            "schema": "x2_phase72_epsilon_schedule_v1",
            "seed_index": index,
            "env_seed": env_seed,
            "latent_seed": latent_seed,
            "epsilon": epsilon,
            "epsilon_tensor_sha256": tensor_hash(epsilon),
        }
        torch.save(payload, temp)
        restored = torch.load(temp, map_location="cpu", weights_only=False)
        if not torch.equal(restored["epsilon"], epsilon):
            raise RuntimeError("Phase72 schedule strict reload failed")
        os.replace(temp, path)
        digest = sha256(path)
        sidecar.write_text(f"{digest}  {path.name}\n")
        records.append(
            {
                "seed_index": index,
                "env_seed": env_seed,
                "latent_seed": latent_seed,
                "schedule_path": str(path.resolve()),
                "schedule_bytes": path.stat().st_size,
                "schedule_sha256": digest,
                "epsilon_tensor_sha256": tensor_hash(epsilon),
            }
        )
    payload = {
        "schema": "x2_phase72_schedule_inventory_v1",
        "seed_pairs": records,
        "cross_seed_epsilon_unique": len({row["epsilon_tensor_sha256"] for row in records}) == 5,
    }
    args.inventory.parent.mkdir(parents=True, exist_ok=True)
    args.inventory.write_text(json.dumps(payload, indent=2) + "\n")
    digest = sha256(args.inventory)
    inventory_sidecar.write_text(f"{digest}  {args.inventory.name}\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
