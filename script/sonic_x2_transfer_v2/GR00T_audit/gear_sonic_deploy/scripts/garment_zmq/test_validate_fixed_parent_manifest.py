#!/usr/bin/env python3
"""Offline tests for fixed-parent approval and proxy status validation."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
import unittest
from pathlib import Path

import request_live_reference_arm as arm
import validate_fixed_parent_manifest as manifest


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ParentManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.artifacts = []
        for role in sorted(manifest.REQUIRED_ARTIFACT_ROLES):
            path = self.root / f"{role}.bin"
            path.write_bytes(f"artifact:{role}".encode("ascii"))
            self.artifacts.append({
                "role": role,
                "path": path.name,
                "sha256": digest(path),
            })
        self.evidence = self.root / "evidence"
        self.evidence.mkdir()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def evidence_item(self, name: str, duration_s: float) -> dict:
        log_path = self.evidence / name
        log_path.mkdir()
        summary = log_path / "summary.json"
        summary.write_text(
            json.dumps({"name": name, "duration_s": duration_s}) + "\n",
            encoding="utf-8",
        )
        return {
            "result": "pass",
            "duration_s": duration_s,
            "log_path": str(log_path.relative_to(self.root)),
            "summary_path": str(summary.relative_to(self.root)),
            "summary_sha256": digest(summary),
        }

    def payload(self) -> dict:
        return {
            "schema": manifest.SCHEMA,
            "approval_status": "approved",
            "approved_scope": manifest.SCOPE,
            "approved_by": "fixed-parent-owner",
            "approved_at": "2026-08-30T12:00:00+08:00",
            "approval_note": "Accepted only for supported stationary live-reference entry.",
            "fixed_contract": dict(manifest.EXPECTED_CONTRACT),
            "artifacts": self.artifacts,
            "evidence": {
                "supported_30s_passes": [
                    self.evidence_item(f"short_{index}", 30.0 + index)
                    for index in range(3)
                ],
                "supported_300s_pass": self.evidence_item("long", 300.0),
            },
        }

    def write_manifest(self, payload: dict) -> Path:
        path = self.root / "parent.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return path

    def test_valid_manifest(self) -> None:
        payload = self.payload()
        result = manifest.validate_manifest(self.write_manifest(payload), self.root)
        self.assertEqual(result["approval_status"], "approved")

    def test_artifact_hash_mismatch_is_blocked(self) -> None:
        payload = self.payload()
        payload["artifacts"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(manifest.ManifestError, "SHA-256 mismatch"):
            manifest.validate_manifest(self.write_manifest(payload), self.root)

    def test_missing_three_short_passes_is_blocked(self) -> None:
        payload = self.payload()
        payload["evidence"]["supported_30s_passes"] = payload["evidence"][
            "supported_30s_passes"
        ][:2]
        with self.assertRaisesRegex(manifest.ManifestError, "at least three"):
            manifest.validate_manifest(self.write_manifest(payload), self.root)

    def test_unapproved_example_is_blocked(self) -> None:
        payload = self.payload()
        payload["approval_status"] = "not-approved"
        with self.assertRaisesRegex(manifest.ManifestError, "not 'approved'"):
            manifest.validate_manifest(self.write_manifest(payload), self.root)


class ProxyStatusTest(unittest.TestCase):
    def test_ready_powered_status_is_accepted(self) -> None:
        arm_path = Path(f"/tmp/x2_live_reference_gate.test.{os.getpid()}.arm.json")
        if arm_path.exists():
            arm_path.unlink()
        with tempfile.TemporaryDirectory() as directory:
            status_path = Path(directory) / "status.json"
            status = {
                "schema": "x2-live-reference-gate-status-v1",
                "state": "STANDSTILL_READY",
                "reason": "ready",
                "output_active": True,
                "stationary_only": True,
                "debug_dry_run": False,
                "session_id": "session-12345678",
                "pid": os.getpid(),
                "updated_wall_s": time.time(),
                "source_age_s": 0.01,
                "robot_age_s": 0.01,
                "config": {"source_stale_s": 0.15, "robot_stale_s": 0.10},
                "arm_trigger_file": str(arm_path),
            }
            status_path.write_text(json.dumps(status), encoding="utf-8")
            accepted = arm.validate_status(status_path)
            self.assertEqual(accepted["session_id"], status["session_id"])

    def test_dry_run_status_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            status_path = Path(directory) / "status.json"
            status_path.write_text(json.dumps({
                "schema": "x2-live-reference-gate-status-v1",
                "state": "STANDSTILL_READY",
                "output_active": True,
                "stationary_only": True,
                "debug_dry_run": True,
            }), encoding="utf-8")
            with self.assertRaisesRegex(arm.ArmError, "powered x2_debug"):
                arm.validate_status(status_path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
