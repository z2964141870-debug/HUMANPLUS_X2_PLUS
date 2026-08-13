#!/usr/bin/env python3
"""Aggregate the six immutable command-panel crossover launches."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from cwi_x2.privileged_teacher_command_panel import (
    ROLE_NAMES,
    deltas,
    pair_rows,
    panel_gates,
    summarize_rows,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sidecar(path: Path) -> None:
    sidecar = path.with_name(path.name + ".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8") != f"{sha256(path)}  {path.name}\n":
        raise RuntimeError(f"sidecar mismatch: {path}")


def atomic_json(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--prereg", type=Path, required=True)
parser.add_argument("--result", type=Path, required=True)
parser.add_argument("--markdown", type=Path, required=True)
args = parser.parse_args()

for output in (args.result, args.result.with_name(args.result.name + ".sha256"), args.markdown):
    if output.exists():
        raise FileExistsError(output)
verify_sidecar(args.prereg)
prereg_sha = sha256(args.prereg)
prereg = json.loads(args.prereg.read_text(encoding="utf-8"))
if prereg.get("schema") != "x2_privileged_teacher_command_panel_prereg_v1":
    raise RuntimeError("preregistration schema changed")

reports = []
resources = []
for launch_id, outputs in sorted(prereg["outputs"]["launches"].items()):
    report_path = Path(outputs["report"])
    resource_path = Path(outputs["resource"])
    verify_sidecar(report_path); verify_sidecar(resource_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    resource = json.loads(resource_path.read_text(encoding="utf-8"))
    if (
        report.get("schema") != "x2_privileged_teacher_panel_launch_v1"
        or report.get("preregistration_sha256") != prereg_sha
        or report.get("launch_id") != launch_id
        or report.get("decision") != "PANEL_LAUNCH_FINITE"
        or not all(report.get("technical_checks", {}).values())
    ):
        raise RuntimeError(f"invalid launch report: {launch_id}")
    if (
        resource.get("label") != f"x2_privileged_teacher_panel_v2d_{launch_id}"
        or resource.get("exit_code") != 0 or resource.get("raw_returncode") != 0
        or not resource.get("autonomous_exit") or resource.get("timed_out")
        or resource.get("term_sent") or resource.get("kill_sent")
        or resource.get("forced_cleanup")
        or float(resource["elapsed_s"]) > prereg["resource_limits"]["deadline_seconds_per_launch"]
        or float(resource["gpu"]["memory_used_peak_mib"]) > prereg["resource_limits"]["maximum_gpu_memory_mib"]
        or int(resource["disk_used_delta_bytes"]) > prereg["resource_limits"]["maximum_disk_delta_bytes_per_launch"]
    ):
        raise RuntimeError(f"invalid launch resource: {launch_id}")
    reports.append(report); resources.append(resource)

by_seed_lane = {(int(report["seed"]), report["lane"]): report for report in reports}
reset_replay = {}
for seed in prereg["panel"]["seeds"]:
    left, right = by_seed_lane[(seed, "A")], by_seed_lane[(seed, "B")]
    reset_replay[str(seed)] = {
        key: left["initial"][key] == right["initial"][key]
        for key in ("policy_observation_sha256", "root_state_sha256", "joint_state_sha256")
    }
pairs = pair_rows(reports)
if len(pairs) != len(prereg["panel"]["seeds"]) * 128:
    raise RuntimeError("crossover pair count changed")
source_rows = [pair["source"] for pair in pairs]
candidate_rows = [pair["candidate"] for pair in pairs]
source = summarize_rows(source_rows)
candidate = summarize_rows(candidate_rows)
overall_gates = panel_gates(source, candidate, prereg["gates"])

seed_results = {}
seed_limits = dict(prereg["gates"])
seed_limits["pitch_mean_delta_rad_min"] = prereg["gates"]["seed_pitch_mean_delta_rad_min"]
seed_limits["pitch_p05_delta_rad_min"] = prereg["gates"]["seed_pitch_p05_delta_rad_min"]
for seed in prereg["panel"]["seeds"]:
    selected = [pair for pair in pairs if pair["seed"] == seed]
    seed_source = summarize_rows([pair["source"] for pair in selected])
    seed_candidate = summarize_rows([pair["candidate"] for pair in selected])
    seed_gates = panel_gates(seed_source, seed_candidate, seed_limits)
    seed_results[str(seed)] = {
        "source": seed_source, "candidate": seed_candidate,
        "deltas": deltas(seed_source, seed_candidate), "gates": seed_gates,
        "passed": all(seed_gates.values()),
    }

role_results = {}
role_limits = dict(prereg["gates"])
role_limits["pitch_mean_delta_rad_min"] = prereg["gates"]["role_pitch_mean_delta_rad_min"]
role_limits["pitch_p05_delta_rad_min"] = prereg["gates"]["role_pitch_p05_delta_rad_min"]
for role_id, role_name in ROLE_NAMES.items():
    selected = [pair for pair in pairs if int(pair["source"]["role_id"]) == role_id]
    role_source = summarize_rows([pair["source"] for pair in selected])
    role_candidate = summarize_rows([pair["candidate"] for pair in selected])
    role_gates = panel_gates(role_source, role_candidate, role_limits)
    role_results[role_name] = {
        "source": role_source, "candidate": role_candidate,
        "deltas": deltas(role_source, role_candidate), "gates": role_gates,
        "passed": all(role_gates.values()),
    }

technical = {
    "six_launches_complete": len(reports) == 6,
    "reset_replay_exact": all(all(values.values()) for values in reset_replay.values()),
    "crossover_pairs_complete": len(pairs) == 384,
    "source_role_balance": source["role_counts"] == {name: 48 for name in ROLE_NAMES.values()},
    "candidate_role_balance": candidate["role_counts"] == {name: 48 for name in ROLE_NAMES.values()},
    "optimizer_steps_zero": all(report["evidence_boundary"]["optimizer_steps"] == 0 for report in reports),
    "checkpoint_writes_zero": all(report["evidence_boundary"]["checkpoint_writes"] == 0 for report in reports),
    "resource_cumulative_disk": sum(max(0, int(item["disk_used_delta_bytes"])) for item in resources) <= prereg["resource_limits"]["maximum_cumulative_disk_delta_bytes"],
}
passed = bool(
    all(technical.values()) and all(overall_gates.values())
    and all(result["passed"] for result in seed_results.values())
    and all(result["passed"] for result in role_results.values())
)
decision = "PASS_MULTI_COMMAND_TEACHER_PANEL_LOCAL_ONLY" if passed else "FAIL_TEACHER_COMMAND_GENERALIZATION_STOP"
result = {
    "schema": "x2_privileged_teacher_command_panel_result_v1",
    "preregistration_sha256": prereg_sha,
    "decision": decision,
    "technical_checks": technical,
    "reset_replay": reset_replay,
    "overall": {"source": source, "candidate": candidate, "deltas": deltas(source, candidate), "gates": overall_gates},
    "by_seed": seed_results,
    "by_role": role_results,
    "resource": {
        "isaac_launches": 6,
        "elapsed_s_total": sum(float(item["elapsed_s"]) for item in resources),
        "disk_used_delta_bytes_positive_total": sum(max(0, int(item["disk_used_delta_bytes"])) for item in resources),
        "gpu_memory_used_peak_mib": max(float(item["gpu"]["memory_used_peak_mib"]) for item in resources),
    },
    "permissions": {
        "teacher_dataset_preregistration_unlocked": passed,
        "bc_or_dagger_training_unlocked": False,
        "ppo_or_long_training_unlocked": False,
        "official_promotion_unlocked": False,
        "deployment_unlocked": False,
    },
}
atomic_json(args.result, result)
sidecar = args.result.with_name(args.result.name + ".sha256")
sidecar.write_text(f"{sha256(args.result)}  {args.result.name}\n", encoding="utf-8")
args.markdown.write_text(
    "# Privileged teacher command panel v2d\n\n"
    f"Formal decision: `{decision}`.\n\n"
    f"Overall pitch mean/p05 deltas: `{result['overall']['deltas']['signed_pitch_mean_rad']:+.6f}` / "
    f"`{result['overall']['deltas']['signed_pitch_p05_rad']:+.6f} rad`. "
    f"Candidate terminations: `{candidate['termination_count']}`; source terminations: `{source['termination_count']}`.\n\n"
    "This is a privileged, ideal-actuator, fixed-upper local teacher diagnostic. It does not authorize training, export, or deployment.\n",
    encoding="utf-8",
)
print(json.dumps({"decision": decision, "technical": technical, "overall_gates": overall_gates}, indent=2))
if not all(technical.values()):
    raise RuntimeError("panel technical gates failed")
