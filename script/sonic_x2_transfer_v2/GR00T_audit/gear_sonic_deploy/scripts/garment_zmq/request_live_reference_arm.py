#!/usr/bin/env python3
"""Validate the parent/status gates, then request one live-reference arm."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Sequence

from validate_fixed_parent_manifest import ManifestError, validate_manifest


class ArmError(ValueError):
    pass


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def validate_status(path: Path) -> dict:
    try:
        status = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ArmError(f"proxy status is absent: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ArmError(f"proxy status is invalid JSON: {exc}") from exc
    if status.get("schema") != "x2-live-reference-gate-status-v1":
        raise ArmError("proxy status schema is invalid")
    if status.get("state") != "STANDSTILL_READY":
        raise ArmError(
            f"proxy state must be STANDSTILL_READY, got {status.get('state')!r}: "
            f"{status.get('reason', '')}"
        )
    if status.get("output_active") is not True:
        raise ArmError("proxy does not report active anchored StandStill output")
    if status.get("stationary_only") is not True:
        raise ArmError("first powered candidate must be stationary-only")
    if status.get("debug_dry_run") is not False:
        raise ArmError("proxy is not attached to powered x2_debug (dry_run must be 0)")
    session_id = status.get("session_id")
    if not isinstance(session_id, str) or len(session_id) < 8:
        raise ArmError("proxy session_id is invalid")
    pid = status.get("pid")
    if not isinstance(pid, int) or not _process_alive(pid):
        raise ArmError(f"proxy pid is not alive: {pid!r}")
    updated_wall_s = status.get("updated_wall_s")
    if not isinstance(updated_wall_s, (int, float)):
        raise ArmError("proxy status has no wall timestamp")
    status_age_s = time.time() - float(updated_wall_s)
    if status_age_s < -1.0 or status_age_s > 1.0:
        raise ArmError(f"proxy status file is stale: age={status_age_s:.3f}s")
    config = status.get("config")
    if not isinstance(config, dict):
        raise ArmError("proxy status has no config")
    for stream, age_key, limit_key in (
        ("source", "source_age_s", "source_stale_s"),
        ("robot", "robot_age_s", "robot_stale_s"),
    ):
        age = status.get(age_key)
        limit = config.get(limit_key)
        if not isinstance(age, (int, float)) or not isinstance(limit, (int, float)):
            raise ArmError(f"proxy {stream} freshness fields are invalid")
        effective_age = float(age) + max(0.0, status_age_s)
        if effective_age >= float(limit):
            raise ArmError(
                f"proxy {stream} is not fresh: effective_age={effective_age:.3f}s "
                f">= {float(limit):.3f}s"
            )
    arm_path_value = status.get("arm_trigger_file")
    if not isinstance(arm_path_value, str) or not arm_path_value:
        raise ArmError("proxy status has no arm_trigger_file")
    arm_path = Path(arm_path_value)
    if arm_path.parent.resolve() != Path("/tmp").resolve():
        raise ArmError(f"arm trigger must be directly under /tmp, got {arm_path}")
    if arm_path.exists():
        raise ArmError(f"arm trigger already exists: {arm_path}")
    return status


def write_trigger(status: dict) -> Path:
    arm_path = Path(status["arm_trigger_file"])
    payload = {
        "schema": "x2-live-reference-arm-v1",
        "command": "ARM_LIVE_REFERENCE",
        "session_id": status["session_id"],
        "requested_wall_s": time.time(),
        "requester_pid": os.getpid(),
    }
    descriptor = os.open(arm_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return arm_path


def wait_for_arm(status_path: Path, session_id: str, timeout_s: float) -> str:
    deadline = time.monotonic() + timeout_s
    last_state = "unknown"
    while time.monotonic() < deadline:
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            time.sleep(0.05)
            continue
        if status.get("session_id") != session_id:
            raise ArmError("proxy session changed after arm request")
        last_state = str(status.get("state"))
        if last_state in ("BLEND", "LIVE"):
            return last_state
        if last_state == "LOCKOUT":
            raise ArmError(f"proxy locked out after arm request: {status.get('reason', '')}")
        time.sleep(0.05)
    raise ArmError(f"proxy did not acknowledge arm in {timeout_s:.1f}s; state={last_state}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--status", required=True, type=Path)
    parser.add_argument("--ack-timeout-s", type=float, default=2.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validate_manifest(args.manifest.expanduser(), args.root.expanduser())
        status = validate_status(args.status.expanduser())
        trigger = write_trigger(status)
    except (OSError, ManifestError, ArmError) as exc:
        print(f"LIVE_REFERENCE_ARM_BLOCKED: {exc}")
        return 4
    try:
        state = wait_for_arm(
            args.status.expanduser(), status["session_id"], args.ack_timeout_s
        )
    except (OSError, ArmError) as exc:
        print(
            "LIVE_REFERENCE_ARM_OUTCOME_UNKNOWN: request was written and must not be "
            f"repeated; inspect proxy status. trigger={trigger} error={exc}"
        )
        return 5
    print(f"LIVE_REFERENCE_ARM_ACCEPTED state={state} trigger={trigger}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
