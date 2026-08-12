#!/usr/bin/env python3
"""Run one command while recording wall time, disk space, and peak GPU use."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import threading
import time
from pathlib import Path


def gpu_sample() -> dict[str, float] | None:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--id=0",
            "--query-gpu=memory.used,memory.total,utilization.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        return None
    fields = [field.strip() for field in result.stdout.strip().split(",")]
    if len(fields) != 4:
        return None
    return {
        "memory_used_mib": float(fields[0]),
        "memory_total_mib": float(fields[1]),
        "utilization_percent": float(fields[2]),
        "power_w": float(fields[3]),
    }


def disk_snapshot(path: Path) -> dict[str, int]:
    usage = shutil.disk_usage(path)
    return {"total_bytes": usage.total, "used_bytes": usage.used, "free_bytes": usage.free}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--resource-output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--disk-path", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")
    if args.resource_output.exists() or args.log.exists():
        raise FileExistsError("resource output and log are immutable; refusing overwrite")

    args.resource_output.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, float]] = []
    sample_errors = 0
    stop = threading.Event()

    def monitor() -> None:
        nonlocal sample_errors
        while not stop.is_set():
            try:
                sample = gpu_sample()
            except (OSError, subprocess.SubprocessError, ValueError):
                sample = None
            if sample is None:
                sample_errors += 1
            else:
                sample["elapsed_s"] = time.monotonic() - start_monotonic
                samples.append(sample)
            stop.wait(0.5)

    disk_before = disk_snapshot(args.disk_path)
    start_wall_ns = time.time_ns()
    start_monotonic = time.monotonic()
    thread = threading.Thread(target=monitor, name="gpu-ledger", daemon=True)
    thread.start()
    with args.log.open("x", encoding="utf-8") as log:
        process = subprocess.run(command, check=False, stdout=log, stderr=subprocess.STDOUT)
    stop.set()
    thread.join(timeout=15)
    elapsed_s = time.monotonic() - start_monotonic
    disk_after = disk_snapshot(args.disk_path)
    report = {
        "schema": "x2_gpu_run_ledger_v1",
        "label": args.label,
        "command": command,
        "exit_code": process.returncode,
        "start_wall_ns": start_wall_ns,
        "elapsed_s": elapsed_s,
        "disk_before": disk_before,
        "disk_after": disk_after,
        "disk_used_delta_bytes": disk_after["used_bytes"] - disk_before["used_bytes"],
        "gpu_sample_count": len(samples),
        "gpu_sample_errors": sample_errors,
        "gpu": (
            {
                "memory_total_mib": max(row["memory_total_mib"] for row in samples),
                "memory_used_peak_mib": max(row["memory_used_mib"] for row in samples),
                "utilization_peak_percent": max(row["utilization_percent"] for row in samples),
                "power_peak_w": max(row["power_w"] for row in samples),
            }
            if samples
            else None
        ),
    }
    args.resource_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
