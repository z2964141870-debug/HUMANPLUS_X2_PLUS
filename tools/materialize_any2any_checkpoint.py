#!/usr/bin/env python3
"""Fold a saved Any2Any LoRA checkpoint into a compact plain-weight base."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SANDBOX = Path(
    "/home/humanplus/x2_teleop_final/x2_sonic/sonic_x2_sandbox"
)
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(SANDBOX))

from dcpeft_checkpoint_materialization import build_materialized_checkpoint  # noqa: E402
from gear_sonic.trl.utils.any2any_lora_checkpoint import (  # noqa: E402
    merge_any2any_lora_state,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report-json", required=True, type=Path)
    parser.add_argument("--fallback-alpha", type=float, default=16.0)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        raise ValueError("output must differ from input")
    if not args.input.is_file():
        raise FileNotFoundError(args.input)

    source_sha256 = sha256_file(args.input)
    source = torch.load(args.input, map_location="cpu", weights_only=False)
    output, report = build_materialized_checkpoint(
        source,
        source_path=args.input,
        source_sha256=source_sha256,
        merge_state=merge_any2any_lora_state,
        fallback_alpha=args.fallback_alpha,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    report["output_path"] = str(args.output.resolve())
    report["output_sha256"] = sha256_file(args.output)
    args.report_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("any2any_checkpoint_materialization=OK")
    print(f"source_sha256={source_sha256}")
    print(f"output_sha256={report['output_sha256']}")
    for state_key, state_report in report["states"].items():
        print(
            f"{state_key}_merged_layers="
            f"{state_report.get('merged_layer_count', 0)}"
        )


if __name__ == "__main__":
    main()
