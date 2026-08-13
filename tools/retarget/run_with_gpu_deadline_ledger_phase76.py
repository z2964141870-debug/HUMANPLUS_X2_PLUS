#!/usr/bin/env python3
"""Run one command with GPU/disk accounting and a fail-closed process-group deadline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_sidecar(path: Path) -> None:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if sidecar.exists():
        raise FileExistsError(f"refusing to overwrite sidecar: {sidecar}")
    sidecar.write_text(f"{sha256(path)}  {path.name}\n", encoding="utf-8")


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    temp = path.with_name(f".{path.name}.tmp")
    if path.exists() or sidecar.exists() or temp.exists():
        raise FileExistsError(f"refusing to overwrite resource evidence: {path}")
    temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with temp.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temp, path)
    write_sidecar(path)


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


def process_group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def signal_group(pgid: int, signum: int) -> bool:
    try:
        os.killpg(pgid, signum)
    except ProcessLookupError:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--resource-output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--disk-path", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, required=True)
    parser.add_argument("--term-grace-seconds", type=float, default=5.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")
    if args.timeout_seconds <= 0 or args.term_grace_seconds < 0:
        parser.error("deadline must be positive and grace must be non-negative")
    for path in (
        args.resource_output,
        args.resource_output.with_suffix(args.resource_output.suffix + ".sha256"),
        args.log,
        args.log.with_suffix(args.log.suffix + ".sha256"),
    ):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite lifecycle evidence: {path}")

    args.resource_output.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, float]] = []
    sample_errors = 0
    stop = threading.Event()
    disk_before = disk_snapshot(args.disk_path)
    start_wall_ns = time.time_ns()
    start_monotonic = time.monotonic()

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

    thread = threading.Thread(target=monitor, name="phase76-gpu-ledger", daemon=True)
    thread.start()
    timed_out = False
    term_sent = False
    kill_sent = False
    forced_cleanup = False
    raw_returncode: int | None = None
    child_pid: int | None = None
    child_pgid: int | None = None
    with args.log.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        child_pid = process.pid
        child_pgid = os.getpgid(process.pid)
        try:
            raw_returncode = process.wait(timeout=args.timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            term_sent = signal_group(child_pgid, signal.SIGTERM)
            try:
                raw_returncode = process.wait(timeout=args.term_grace_seconds)
            except subprocess.TimeoutExpired:
                kill_sent = signal_group(child_pgid, signal.SIGKILL)
                raw_returncode = process.wait()
        if process_group_alive(child_pgid):
            forced_cleanup = True
            term_sent = signal_group(child_pgid, signal.SIGTERM) or term_sent
            deadline = time.monotonic() + args.term_grace_seconds
            while process_group_alive(child_pgid) and time.monotonic() < deadline:
                time.sleep(0.05)
            if process_group_alive(child_pgid):
                kill_sent = signal_group(child_pgid, signal.SIGKILL) or kill_sent
        log.flush()
        os.fsync(log.fileno())

    stop.set()
    thread.join(timeout=15)
    elapsed_s = time.monotonic() - start_monotonic
    disk_after = disk_snapshot(args.disk_path)
    autonomous_exit = bool(
        raw_returncode == 0
        and not timed_out
        and not term_sent
        and not kill_sent
        and not forced_cleanup
    )
    effective_exit_code = 0 if autonomous_exit else (int(raw_returncode) if raw_returncode not in (None, 0) else 124)
    write_sidecar(args.log)
    report = {
        "schema": "x2_gpu_deadline_ledger_phase76_v1",
        "label": args.label,
        "command": command,
        "log_path": str(args.log),
        "child_pid": child_pid,
        "child_pgid": child_pgid,
        "timeout_seconds": args.timeout_seconds,
        "term_grace_seconds": args.term_grace_seconds,
        "timed_out": timed_out,
        "term_sent": term_sent,
        "kill_sent": kill_sent,
        "forced_cleanup": forced_cleanup,
        "raw_returncode": raw_returncode,
        "autonomous_exit": autonomous_exit,
        "exit_code": effective_exit_code,
        "start_wall_ns": start_wall_ns,
        "elapsed_s": elapsed_s,
        "disk_before": disk_before,
        "disk_after": disk_after,
        "disk_used_delta_bytes": disk_after["used_bytes"] - disk_before["used_bytes"],
        "log_bytes": args.log.stat().st_size,
        "log_sha256": sha256(args.log),
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
    atomic_json(args.resource_output, report)
    print(json.dumps(report, indent=2), flush=True)
    raise SystemExit(effective_exit_code)


if __name__ == "__main__":
    main()
